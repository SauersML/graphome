// make_sequence.rs
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use memmap2::{MmapOptions};

use crate::map::{self, GlobalData, Coord2NodeResult};
use gbwt::GBZ;
use simple_sds::serialize;

/// Reverses and complements a DNA sequence
fn reverse_complement(dna: &str) -> String {
    dna.chars()
        .rev()
        .map(|c| match c {
            'A' | 'a' => 'T',
            'T' | 't' => 'A',
            'G' | 'g' => 'C',
            'C' | 'c' => 'G',
            'N' | 'n' => 'N',
            other => other, // Keep other characters as-is
        })
        .collect()
}

/// Extract sequences for specific nodes from GFA using memory mapping
fn extract_node_sequences(gfa_path: &str, node_ids: &HashSet<String>) -> HashMap<String, String> {
    let file = File::open(gfa_path).expect("Cannot open GFA file for memmap");
    
    let mmap = unsafe {
        MmapOptions::new().map(&file).expect("Failed to mmap GFA")
    };
    
    // Find line boundaries
    let mut line_indices = Vec::new();
    line_indices.push(0);
    for i in 0..mmap.len() {
        if mmap[i] == b'\n' {
            if i + 1 < mmap.len() {
                line_indices.push(i + 1);
            }
        }
    }
    
    let mut node_sequences = HashMap::new();
    
    eprintln!("[INFO] Scanning GFA for {} node sequences...", node_ids.len());
    
    for i in 0..line_indices.len() {
        let offset = line_indices[i];
        let end_offset = if i + 1 < line_indices.len() {
            line_indices[i + 1] - 1
        } else {
            mmap.len()
        };
        
        if end_offset <= offset {
            continue;
        }
        
        let line_slice = &mmap[offset..end_offset];
        if line_slice.is_empty() || line_slice[0] != b'S' {
            continue;
        }
        
        let parts: Vec<&[u8]> = line_slice.split(|&c| c == b'\t').collect();
        if parts.len() < 3 {
            continue;
        }
        
        let node_id = String::from_utf8_lossy(parts[1]).to_string();
        if node_ids.contains(&node_id) {
            let seq = String::from_utf8_lossy(parts[2]).to_string();
            if seq != "*" {
                node_sequences.insert(node_id, seq);
            }
        }
    }
    
    eprintln!("[INFO] Found sequences for {}/{} nodes", node_sequences.len(), node_ids.len());
    
    node_sequences
}

/// Extract sequence for a region specified by coordinates
pub fn extract_sequence(
    gfa_path: &str,
    paf_path: &str,
    region: &str,
    sample_name: &str,
    output_path: &str,
) -> Result<(), std::io::Error> {
    eprintln!("[INFO] Building data structures from GFA='{}' PAF='{}'", gfa_path, paf_path);

    // Initialize global data
    let mut global = GlobalData {
        node_map: HashMap::new(),
        path_map: HashMap::new(),
        node_to_paths: HashMap::new(),
        alignment_by_path: HashMap::new(),
        ref_trees: HashMap::new(),
    };

    // Parse GFA and PAF files
    map::parse_gfa_memmap(gfa_path, &mut global);
    eprintln!("[INFO] GFA parse done. node_map={} path_map={}",
        global.node_map.len(), global.path_map.len());

    map::parse_paf_parallel(paf_path, &mut global);
    eprintln!("[INFO] PAF parse done. alignment_by_path={}", global.alignment_by_path.len());

    // Build interval trees
    map::build_ref_trees(&mut global);
    eprintln!("[INFO] Interval Trees built. Ready for queries.");

    // Parse the region
    if let Some((chr, start, end)) = map::parse_region(region) {
        // Convert coordinates to nodes
        let gbz_path = map::make_gbz_exist(gfa_path, paf_path);
        let gbz: GBZ = serialize::load_from(&gbz_path).expect("Failed to load GBZ index");
        let nodes = map::coord_to_nodes(&gbz, &chr, start, end);
        if nodes.is_empty() {
            eprintln!("No nodes found for region {}:{}-{}", chr, start, end);
            return Ok(());
        }

        // Get unique node IDs
        let node_ids: HashSet<String> = nodes
            .iter()
            .map(|n| n.node_id.clone())
            .collect();

        eprintln!("[INFO] Found {} unique nodes for region {}:{}-{}", 
                 node_ids.len(), chr, start, end);

        // Extract node sequences
        let node_sequences = extract_node_sequences(gfa_path, &node_ids);
        
        // Check if we're missing any sequences
        let missing_nodes: Vec<_> = node_ids.iter()
            .filter(|id| !node_sequences.contains_key(*id))
            .collect();
        
        if !missing_nodes.is_empty() {
            eprintln!("[WARNING] Missing sequences for {} nodes: {:?}...", 
                      missing_nodes.len(), 
                      missing_nodes.iter().take(5).collect::<Vec<_>>());
        }

        // Group nodes by path
        let mut nodes_by_path: HashMap<String, Vec<&Coord2NodeResult>> = HashMap::new();
        for node in &nodes {
            nodes_by_path
                .entry(node.path_name.clone())
                .or_insert_with(Vec::new)
                .push(node);
        }

        // Process each path
        for (path_name, path_nodes) in nodes_by_path {
            let safe_path_name = path_name.replace('#', "_").replace('/', "_");
            
            // Get the path data
            let path_data = match global.path_map.get(&path_name) {
                Some(pd) => pd,
                None => {
                    eprintln!("[WARNING] Path data not found for {}, skipping", path_name);
                    continue;
                }
            };
            
            // Sort nodes by path offset
            let mut sorted_nodes = path_nodes.clone();
            sorted_nodes.sort_by_key(|n| n.path_off_start);
            
            if sorted_nodes.is_empty() {
                continue;
            }
            
            eprintln!("[INFO] Processing path {} with {} nodes", path_name, sorted_nodes.len());
            
            // Build a sequence from the nodes
            let mut final_sequence = String::new();
            
            for node_result in sorted_nodes {
                if let Some(seq) = node_sequences.get(&node_result.node_id) {
                    // Find the node in the path
                    let node_idx_opt = path_data.nodes.iter().position(|(nid, orient)| 
                        nid == &node_result.node_id && *orient == node_result.node_orient);
                    
                    let node_idx = match node_idx_opt {
                        Some(idx) => idx,
                        None => {
                            // Try finding just by node ID, ignoring orientation
                            match path_data.nodes.iter().position(|(nid, _)| nid == &node_result.node_id) {
                                Some(idx) => {
                                    eprintln!("[WARNING] Node {} found in path with different orientation", 
                                             node_result.node_id);
                                    idx
                                },
                                None => {
                                    eprintln!("[WARNING] Node {} not found in path {}, skipping", 
                                             node_result.node_id, path_name);
                                    continue;
                                }
                            }
                        }
                    };
                    
                    // Get node's path offset and length
                    let path_offset = path_data.prefix_sums[node_idx];
                    let node_len = seq.len();
                    
                    // Get the node's sequence with proper orientation
                    let oriented_seq = if node_result.node_orient {
                        seq.clone()
                    } else {
                        reverse_complement(seq)
                    };
                    
                    // Calculate which part of the node's sequence corresponds to path_off_start/end
                    let start_in_node = node_result.path_off_start.saturating_sub(path_offset);
                    let end_in_node = (node_result.path_off_end.saturating_sub(path_offset) + 1).min(node_len);
                    
                    if start_in_node < node_len && start_in_node < end_in_node {
                        final_sequence.push_str(&oriented_seq[start_in_node..end_in_node]);
                    } else {
                        eprintln!("[WARNING] Invalid sequence range for node {}: {}..{} (length {})", 
                                 node_result.node_id, start_in_node, end_in_node, node_len);
                    }
                } else {
                    eprintln!("[WARNING] No sequence data for node {}", node_result.node_id);
                }
            }
            
            if final_sequence.is_empty() {
                eprintln!("[WARNING] Generated empty sequence for path {}, skipping output", path_name);
                continue;
            }
            
            // Create output directory if it doesn't exist
            let output_dir = Path::new(output_path).parent().unwrap_or(Path::new("."));
            if !output_dir.exists() {
                std::fs::create_dir_all(output_dir)?;
            }
            
            // Write FASTA output
            let output_file = format!("{}_{}_{}_{}-{}.fa", 
                                    output_path, sample_name, safe_path_name, start, end);
            let file = File::create(&output_file)?;
            let mut writer = BufWriter::new(file);
            
            // Write FASTA header
            writeln!(writer, ">{}_{}:{}-{}", sample_name, chr, start, end)?;
            
            // Write sequence in lines of 60 characters
            for chunk in final_sequence.as_bytes().chunks(60) {
                writeln!(writer, "{}", std::str::from_utf8(chunk).unwrap())?;
            }
            
            eprintln!("[INFO] Wrote sequence of length {} for path {} to {}", 
                     final_sequence.len(), path_name, output_file);
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
pub fn run_make_sequence(gfa_path: &str, paf_path: &str, region: &str, sample_name: &str, output_path: &str) {
    match extract_sequence(gfa_path, paf_path, region, sample_name, output_path) {
        Ok(_) => {
            eprintln!("[INFO] Successfully extracted sequence for region {}", region);
        }
        Err(e) => {
            eprintln!("[ERROR] Failed to extract sequence: {}", e);
            std::process::exit(1);
        }
    }
}
