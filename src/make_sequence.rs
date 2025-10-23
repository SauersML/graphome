// make_sequence.rs
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::io::GfaReader;
use crate::map::{self, Coord2NodeResult};
use gbwt::GBZ;
use simple_sds::serialize;

fn complement_base(base: u8) -> u8 {
    match base {
        b'A' => b'T',
        b'a' => b't',
        b'T' => b'A',
        b't' => b'a',
        b'G' => b'C',
        b'g' => b'c',
        b'C' => b'G',
        b'c' => b'g',
        b'N' | b'n' => base,
        other => other,
    }
}

fn reverse_complement_bytes(sequence: &[u8]) -> Vec<u8> {
    let mut rc = Vec::with_capacity(sequence.len());
    for base in sequence.iter().rev() {
        rc.push(complement_base(*base));
    }
    rc
}

fn write_wrapped_segment<W: Write>(
    writer: &mut W,
    segment: &[u8],
    current_line_len: &mut usize,
) -> io::Result<()> {
    let mut index = 0;
    while index < segment.len() {
        let remaining_in_line = 60 - *current_line_len;
        let to_write = remaining_in_line.min(segment.len() - index);
        if to_write > 0 {
            writer.write_all(&segment[index..index + to_write])?;
            *current_line_len += to_write;
            index += to_write;
        }

        if *current_line_len == 60 {
            writer.write_all(b"\n")?;
            *current_line_len = 0;
        }
    }

    Ok(())
}

fn write_node_segment(
    writer: &mut BufWriter<File>,
    sequence: &str,
    orientation_forward: bool,
    clipped_start: usize,
    clipped_end: usize,
    node_len: usize,
    current_line_len: &mut usize,
) -> io::Result<usize> {
    if orientation_forward {
        let segment = &sequence[clipped_start..clipped_end];
        write_wrapped_segment(writer, segment.as_bytes(), current_line_len)?;
        Ok(segment.len())
    } else {
        let slice_start = node_len - clipped_end;
        let slice_end = node_len - clipped_start;
        let rc_segment = reverse_complement_bytes(&sequence.as_bytes()[slice_start..slice_end]);
        let len = rc_segment.len();
        write_wrapped_segment(writer, &rc_segment, current_line_len)?;
        Ok(len)
    }
}

/// Extract sequences for specific nodes directly from a GBZ graph.
fn extract_node_sequences_from_gbz(
    gbz: &GBZ,
    node_ids: &HashSet<String>,
) -> HashMap<String, String> {
    let mut node_sequences = HashMap::new();

    for node_id in node_ids {
        match node_id.parse::<usize>() {
            Ok(node_num) => {
                if let Some(seq_bytes) = gbz.sequence(node_num) {
                    let sequence = std::str::from_utf8(seq_bytes)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|_| String::from_utf8_lossy(seq_bytes).to_string());
                    if sequence != "*" {
                        node_sequences.insert(node_id.clone(), sequence);
                    }
                }
            }
            Err(_) => {
                eprintln!(
                    "[WARNING] Unable to parse node id '{}' as numeric; skipping GBZ lookup",
                    node_id
                );
            }
        }
    }

    node_sequences
}

/// Extract sequences for specific nodes from GFA using streaming reader
/// Now supports streaming from S3/HTTP without full download
fn extract_node_sequences_from_gfa(
    gfa_path: &str,
    node_ids: &HashSet<String>,
) -> io::Result<HashMap<String, String>> {
    let reader = GfaReader::new(gfa_path);
    reader.extract_sequences(node_ids)
}

/// Extract sequence for a region specified by coordinates
pub fn extract_sequence(
    graph_path: &str,
    paf_path: &str,
    region: &str,
    sample_name: &str,
    output_path: &str,
) -> Result<(), std::io::Error> {
    eprintln!(
        "[INFO] Building data structures from graph='{}' PAF='{}'",
        graph_path, paf_path
    );

    // Parse the region
    if let Some((chr, start, end)) = map::parse_region(region) {
        // Convert coordinates to nodes
        let input_is_gbz = GBZ::is_gbz(graph_path);
        let gbz_path = if input_is_gbz {
            graph_path.to_string()
        } else {
            map::make_gbz_exist(graph_path, paf_path)
        };
        let gbz: GBZ = serialize::load_from(&gbz_path).expect("Failed to load GBZ index");
        let nodes = map::coord_to_nodes_with_path(&gbz, &gbz_path, &chr, start, end);
        if nodes.is_empty() {
            eprintln!("No nodes found for region {}:{}-{}", chr, start, end);
            return Ok(());
        }

        // Get unique node IDs
        let node_ids: HashSet<String> = nodes.iter().map(|n| n.node_id.clone()).collect();

        eprintln!(
            "[INFO] Found {} unique nodes for region {}:{}-{}",
            node_ids.len(),
            chr,
            start,
            end
        );

        // Extract node sequences
        let mut node_sequences = extract_node_sequences_from_gbz(&gbz, &node_ids);

        if node_sequences.len() < node_ids.len() && !input_is_gbz {
            let missing: HashSet<String> = node_ids
                .iter()
                .filter(|id| !node_sequences.contains_key(*id))
                .cloned()
                .collect();
            if !missing.is_empty() {
                eprintln!(
                    "[INFO] Falling back to GFA to resolve {} missing node sequences",
                    missing.len()
                );
                let fallback = extract_node_sequences_from_gfa(graph_path, &missing)?;
                node_sequences.extend(fallback);
            }
        }

        // Check if we're missing any sequences
        let missing_nodes: Vec<_> = node_ids
            .iter()
            .filter(|id| !node_sequences.contains_key(*id))
            .collect();

        if !missing_nodes.is_empty() {
            eprintln!(
                "[WARNING] Missing sequences for {} nodes: {:?}...",
                missing_nodes.len(),
                missing_nodes.iter().take(5).collect::<Vec<_>>()
            );
        }

        // Group nodes by path
        let mut nodes_by_path: HashMap<String, Vec<&Coord2NodeResult>> = HashMap::new();
        for node in &nodes {
            nodes_by_path
                .entry(node.path_name.clone())
                .or_default()
                .push(node);
        }

        // Process each path
        for (path_name, path_nodes) in nodes_by_path {
            let safe_path_name = path_name.replace(['#', '/'], "_");

            // Sort nodes by path offset
            let mut sorted_nodes = path_nodes.clone();
            sorted_nodes.sort_by_key(|n| n.path_off_start);

            if sorted_nodes.is_empty() {
                continue;
            }

            eprintln!(
                "[INFO] Processing path {} with {} nodes",
                path_name,
                sorted_nodes.len()
            );

            let mut total_bases = 0usize;
            for node_result in &sorted_nodes {
                if let Some(seq) = node_sequences.get(&node_result.node_id) {
                    let node_len = seq.len();
                    let start_in_node = node_result.path_off_start.min(node_len);
                    let end_in_node = node_result.path_off_end.saturating_add(1).min(node_len);

                    if start_in_node < end_in_node {
                        total_bases += end_in_node - start_in_node;
                    } else {
                        eprintln!(
                            "[WARNING] Invalid sequence range for node {}: {}..{} (length {})",
                            node_result.node_id, start_in_node, end_in_node, node_len
                        );
                    }
                } else {
                    eprintln!(
                        "[WARNING] No sequence data for node {}",
                        node_result.node_id
                    );
                }
            }

            if total_bases == 0 {
                eprintln!(
                    "[WARNING] Generated empty sequence for path {}, skipping output",
                    path_name
                );
                continue;
            }

            // Create output directory if it doesn't exist
            let output_dir = Path::new(output_path).parent().unwrap_or(Path::new("."));
            if !output_dir.exists() {
                std::fs::create_dir_all(output_dir)?;
            }

            // Write FASTA output
            let output_file = format!(
                "{}_{}_{}_{}-{}.fa",
                output_path, sample_name, safe_path_name, start, end
            );
            let file = File::create(&output_file)?;
            let mut writer = BufWriter::new(file);

            // Write FASTA header
            writeln!(writer, ">{}_{}:{}-{}", sample_name, chr, start, end)?;

            let mut current_line_len = 0usize;
            let mut bases_written = 0usize;

            for node_result in &sorted_nodes {
                if let Some(seq) = node_sequences.get(&node_result.node_id) {
                    let node_len = seq.len();
                    let start_in_node = node_result.path_off_start.min(node_len);
                    let end_in_node = node_result.path_off_end.saturating_add(1).min(node_len);

                    if start_in_node < end_in_node {
                        bases_written += write_node_segment(
                            &mut writer,
                            seq,
                            node_result.node_orient,
                            start_in_node,
                            end_in_node,
                            node_len,
                            &mut current_line_len,
                        )?;
                    }
                }
            }

            if current_line_len > 0 {
                writer.write_all(b"\n")?;
            }

            eprintln!(
                "[INFO] Wrote sequence of length {} for path {} to {}",
                bases_written, path_name, output_file
            );
        }

        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Could not parse region format: {}", region),
        ))
    }
}

/// Command-line entry point for sequence extraction
pub fn run_make_sequence(
    graph_path: &str,
    paf_path: &str,
    region: &str,
    sample_name: &str,
    output_path: &str,
) {
    match extract_sequence(graph_path, paf_path, region, sample_name, output_path) {
        Ok(_) => {
            eprintln!(
                "[INFO] Successfully extracted sequence for region {}",
                region
            );
        }
        Err(e) => {
            eprintln!("[ERROR] Failed to extract sequence: {}", e);
            std::process::exit(1);
        }
    }
}
