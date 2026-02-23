// src/convert.rs

// Module for converting GFA file to adjacency matrix in edge list format.

use gbz::{Orientation, GBZ};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use simple_sds::serialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn trim_ascii_whitespace(bytes: &[u8]) -> &[u8] {
    let start = match bytes.iter().position(|b| !b.is_ascii_whitespace()) {
        Some(idx) => idx,
        None => return &bytes[..0],
    };
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .expect("start guaranteed a non-whitespace byte");
    &bytes[start..=end]
}

/// Converts a graph (GFA or GBZ) to an adjacency matrix in edge list format.
///
/// This function performs a two-pass approach:
/// 1. First Pass: Parses the GFA file to collect all unique segment names and assigns them deterministic indices based on sorted order.
/// 2. Second Pass: Parses the links and writes bidirectional edges to the output file in parallel.
///
/// # Arguments
///
/// * `input_path` - Path to the input graph (GFA or GBZ).
/// * `output_path` - Path to the output adjacency matrix file.
///
/// # Errors
///
/// Returns an `io::Result` with any file or I/O errors encountered.
///
/// # Panics
///
/// This function does not explicitly panic.
pub fn convert_gfa_to_edge_list<P: AsRef<Path>>(input_path: P, output_path: P) -> io::Result<()> {
    convert_graph_to_edge_list(input_path, output_path)
}

/// Entry point for converting a graph file (GFA or GBZ) to an adjacency list.
pub fn convert_graph_to_edge_list<P: AsRef<Path>>(input_path: P, output_path: P) -> io::Result<()> {
    let input_path: PathBuf = input_path.as_ref().to_path_buf();
    let output_path: PathBuf = output_path.as_ref().to_path_buf();

    if GBZ::is_gbz(&input_path) {
        convert_gbz_to_edge_list(&input_path, &output_path)
    } else {
        convert_gfa_to_edge_list_impl(&input_path, &output_path)
    }
}

fn convert_gfa_to_edge_list_impl(gfa_path: &Path, output_path: &Path) -> io::Result<()> {
    let start_time = Instant::now();

    println!("Starting to parse GFA file: {}", gfa_path.display());

    // Step 1: Parse the GFA file to extract segments and assign deterministic indices
    let (segment_indices, num_segments) = parse_segments(gfa_path)?;
    println!("Total segments (nodes) identified: {}", num_segments);

    // Step 2: Parse links and write edges in parallel
    // Some GFA files have no `L` (Link) lines at all.
    parse_links_and_write_edges(gfa_path, &segment_indices, output_path)?;
    println!("Finished parsing links between nodes and writing edges.");

    let duration = start_time.elapsed();
    println!("Completed in {:.2?} seconds.", duration);

    Ok(())
}

fn convert_gbz_to_edge_list(gbz_path: &Path, output_path: &Path) -> io::Result<()> {
    let start_time = Instant::now();

    println!("Starting to parse GBZ file: {}", gbz_path.display());

    let gbz: GBZ = serialize::load_from(gbz_path).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to load GBZ: {e}"),
        )
    })?;

    // Collect all segment names if translation is available; otherwise, fall back to node identifiers.
    let segment_indices = if let Some(mut segment_iter) = gbz.segment_iter() {
        let mut names = Vec::new();
        while let Some(segment) = segment_iter.next() {
            names.push(segment.name.to_vec());
        }
        names.par_sort();
        let indices: HashMap<Vec<u8>, u32> = names
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, i as u32))
            .collect();
        indices
    } else {
        let mut node_ids: Vec<Vec<u8>> = gbz
            .node_iter()
            .map(|node_id| node_id.to_string().into_bytes())
            .collect();
        node_ids.par_sort();
        let indices: HashMap<Vec<u8>, u32> = node_ids
            .into_iter()
            .enumerate()
            .map(|(i, node)| (node, i as u32))
            .collect();
        indices
    };

    println!(
        "Total segments (nodes) identified: {}",
        segment_indices.len()
    );

    // Build edges by walking the GBZ topology.
    let mut edges: HashSet<(u32, u32)> = HashSet::new();

    if gbz.has_translation() {
        if let Some(mut segment_iter) = gbz.segment_iter() {
            while let Some(segment) = segment_iter.next() {
                let Some(&from_idx) = segment_indices.get(segment.name) else {
                    continue;
                };

                if let Some(mut succ_iter) = gbz.segment_successors(&segment, Orientation::Forward)
                {
                    while let Some((succ_segment, _)) = succ_iter.next() {
                        if let Some(&to_idx) = segment_indices.get(succ_segment.name) {
                            let edge = if from_idx <= to_idx {
                                (from_idx, to_idx)
                            } else {
                                (to_idx, from_idx)
                            };
                            edges.insert(edge);
                        }
                    }
                }
            }
        }
    } else {
        for node_id in gbz.node_iter() {
            let node_name = node_id.to_string();
            let Some(&from_idx) = segment_indices.get(node_name.as_bytes()) else {
                continue;
            };

            if let Some(mut succ_iter) = gbz.successors(node_id, Orientation::Forward) {
                while let Some((succ_id, _)) = succ_iter.next() {
                    let succ_name = succ_id.to_string();
                    if let Some(&to_idx) = segment_indices.get(succ_name.as_bytes()) {
                        let edge = if from_idx <= to_idx {
                            (from_idx, to_idx)
                        } else {
                            (to_idx, from_idx)
                        };
                        edges.insert(edge);
                    }
                }
            }
        }
    }

    let mut sorted_edges: Vec<(u32, u32)> = edges.into_iter().collect();
    sorted_edges.sort_unstable();

    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);
    for (from_idx, to_idx) in &sorted_edges {
        writer.write_all(&from_idx.to_le_bytes())?;
        writer.write_all(&to_idx.to_le_bytes())?;
    }
    writer.flush()?;

    println!(
        "Finished parsing GBZ edges. Unique undirected edges written: {}",
        sorted_edges.len()
    );
    let duration = start_time.elapsed();
    println!("Completed in {:.2?} seconds.", duration);

    Ok(())
}

/// Parses the GFA file to extract segments and assign unique indices deterministically.
///
/// Approach:
/// 1. Memory-map the file for fast random access.
/// 2. Identify line boundaries by scanning for newline characters.
/// 3. Estimate the number of segments for progress display by sampling lines.
/// 4. Use `fold` and `reduce` in parallel to accumulate segment names into a single `HashSet<Vec<u8>>`.
///
/// This ensures no partial lines, leverages parallelism, and avoids previous issues with map_init returning `()`.
///
/// # Arguments
///
/// * `gfa_path` - Path to the input GFA file.
///
/// # Returns
///
/// A tuple containing:
/// - A `HashMap<Vec<u8>, u32>` mapping segment names to indices
/// - A `u32` count of total segments
///
/// # Errors
///
/// Returns an `io::Result` on file or I/O errors.
///
/// # Panics
///
/// Does not explicitly panic.
fn parse_segments(gfa_path: &Path) -> io::Result<(HashMap<Vec<u8>, u32>, u32)> {
    use memchr::memchr_iter;
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // --- Memory-map the file ---
    let file = File::open(gfa_path)?;
    let file_size = file.metadata()?.len();
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

    println!("File has been memory-mapped.");

    // --- Parallel newline scanning with chunk boundary safety ---
    let chunk_size = 16 * 1024 * 1024; // 16MB chunks
    let line_ends: Vec<usize> = (0..mmap.len())
        .into_par_iter()
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(mmap.len());
            memchr_iter(b'\n', &mmap[start..end])
                .map(move |pos| start + pos)
                .collect::<Vec<_>>()
        })
        .flatten_iter()
        .collect();

    // --- Line count calculation ---
    let has_final_newline = line_ends
        .last()
        .is_some_and(|&pos| pos as u64 == file_size - 1);
    let total_lines = if has_final_newline {
        line_ends.len()
    } else {
        line_ends.len() + 1
    };

    // --- Fail-safe line slicing ---
    let get_line_slice = |i: usize| {
        let start = if i == 0 { 0 } else { line_ends[i - 1] + 1 };
        let end = if i < line_ends.len() {
            line_ends[i]
        } else {
            file_size as usize
        };
        &mmap[start..end]
    };

    // --- Parallel sampling with RNG ---
    let samples = 1000.min(total_lines);
    let seed: u64 = rand::thread_rng().gen();
    let sample_count = AtomicU32::new(0);

    (0..samples).into_par_iter().for_each(|_| {
        let mut rng =
            ChaCha8Rng::seed_from_u64(seed + rayon::current_thread_index().unwrap_or(0) as u64);
        let idx = rng.gen_range(0..total_lines);
        let line_slice = get_line_slice(idx);
        if line_slice.starts_with(b"S\t") {
            sample_count.fetch_add(1, Ordering::Relaxed);
        }
    });

    // --- Estimation logic ---
    let sample_count = sample_count.into_inner();
    let density = if samples > 0 {
        sample_count as f64 / samples as f64
    } else {
        0.0
    };
    let estimated_segments = ((total_lines as f64) * density) as u64;

    println!("🧬 Parsing segments (estimated {estimated_segments} segments)...");
    let pb = Arc::new(ProgressBar::new(estimated_segments));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {percent}% ({pos}/{len} segments)")
            .unwrap()
            .progress_chars("▰▰░"),
    );

    // --- Batched parallel processing ---
    let segment_names = (0..total_lines)
        .into_par_iter()
        .chunks(1024) // Process 1024 lines per batch, but make sure we aren't getting partial lines
        .fold(
            || {
                let mut set: HashSet<Vec<u8>> = HashSet::new();
                set.reserve(estimated_segments as usize / rayon::current_num_threads().max(1));
                set
            },
            |mut local_set, batch| {
                for i in batch {
                    let line_slice = get_line_slice(i);
                    if line_slice.starts_with(b"S\t") {
                        let mut fields = line_slice.split(|&b| b == b'\t');
                        let Some(tag) = fields.next() else {
                            continue;
                        };
                        if tag != b"S" {
                            continue;
                        }
                        let Some(name_raw) = fields.next() else {
                            continue;
                        };
                        let name = trim_ascii_whitespace(name_raw);
                        if !name.is_empty() {
                            local_set.insert(name.to_vec());
                            pb.inc(1);
                        }
                    }
                }
                local_set
            },
        )
        .reduce(HashSet::new, |mut a, b| {
            a.extend(b);
            a
        });

    pb.finish_with_message("✨ Segment parsing complete!");

    // --- Deterministic sorting and indexing ---
    let mut all_names: Vec<Vec<u8>> = segment_names.into_iter().collect();
    all_names.par_sort();

    let segment_indices: HashMap<Vec<u8>, u32> = all_names
        .into_iter()
        .enumerate()
        .map(|(i, name)| (name, i as u32))
        .collect();

    let segment_counter = segment_indices.len() as u32;
    println!("🎯 Total segments (nodes) identified: {}", segment_counter);

    Ok((segment_indices, segment_counter))
}

/// Parses the GFA file to extract links and write edges to the output file.
///
/// Writes only one edge because all edges are bidirectional.
/// Processing is done in parallel for performance.
///
/// # Arguments
///
/// * `gfa_path` - Path to the input GFA file.
/// * `segment_indices` - Mapping from segment names to indices.
/// * `output_path` - Path to the output adjacency matrix file.
///
/// # Errors
///
/// Returns an `io::Result` with any file or I/O errors encountered.
///
/// # Panics
///
/// This function may panic if writes fail (due to `.unwrap()`).
fn parse_links_and_write_edges(
    gfa_path: &Path,
    segment_indices: &HashMap<Vec<u8>, u32>,
    output_path: &Path,
) -> io::Result<()> {
    let file = File::open(gfa_path)?;
    let reader = BufReader::new(file);

    let output_file = File::create(output_path)?;
    let writer = Arc::new(Mutex::new(BufWriter::new(output_file)));

    println!("Parsing links between nodes and writing edges...");

    // Initialize a progress bar with an estimated total number of links
    let total_links_estimate = 309_511_744; // Adjust if known
    println!(
        "Note: Progress bar is based on an estimated total of {} links.",
        total_links_estimate
    );
    let pb = ProgressBar::new(total_links_estimate as u64);

    // Configure the progress bar style
    let style = ProgressStyle::default_bar()
        .template(
            "{spinner} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Links (Estimated)",
        )
        .unwrap()
        .progress_chars("#>-");
    pb.set_style(style);

    let edge_counter = Arc::new(Mutex::new(0u64));

    // Only parse "L" lines for edges; skip "C" and "P" lines to avoid introducing extra edges.
    // We will NOT write undirected edges by writing both (from->to) and (to->from).
    // Instead, we just pick one since it is known to be all undirected.
    reader
        .split(b'\n')
        .par_bridge()
        .filter_map(Result::ok)
        .for_each(|line| {
            let mut local_edges = Vec::new();

            let line = trim_ascii_whitespace(&line);
            if line.starts_with(b"L\t") {
                // GFA v1 "L" line requires 6 fields:
                //   0: "L"
                //   1: from segment
                //   2: from orientation (+/-)
                //   3: to segment
                //   4: to orientation (+/-)
                //   5: overlap/CIGAR (can be '*')
                let mut parts = line.split(|&b| b == b'\t');
                let Some(tag) = parts.next() else {
                    return;
                };
                if tag != b"L" {
                    return;
                }
                let Some(from_raw) = parts.next() else {
                    return;
                };
                let Some(_) = parts.next() else {
                    return;
                };
                let Some(to_raw) = parts.next() else {
                    return;
                };
                let Some(_) = parts.next() else {
                    return;
                };
                let Some(_) = parts.next() else {
                    return;
                };

                let from_name = trim_ascii_whitespace(from_raw);
                let to_name = trim_ascii_whitespace(to_raw);

                if from_name.is_empty() || to_name.is_empty() {
                    return;
                }

                if let (Some(&f_idx), Some(&t_idx)) =
                    (segment_indices.get(from_name), segment_indices.get(to_name))
                {
                    local_edges.push((f_idx, t_idx));
                }
            }

            // Write out edges (both directions)
            if !local_edges.is_empty() {
                let mut writer_guard = writer.lock().unwrap();
                let mut edge_count_guard = edge_counter.lock().unwrap();

                for (f, t) in local_edges {
                    // Write f->t (we understand it is undirected)
                    writer_guard.write_all(&f.to_le_bytes()).unwrap();
                    writer_guard.write_all(&t.to_le_bytes()).unwrap();
                    pb.inc(1);
                    *edge_count_guard += 2; // Since one written edge really corresponds to two edges
                }
            }
        });

    pb.finish_with_message("Finished parsing all L-type links between nodes.");

    let total_edges = *edge_counter.lock().unwrap();
    // Each actual adjacency was written twice, so total_edges is about double the count of unique connections in an undirected sense.

    if total_edges > total_links_estimate as u64 {
        println!(
            "Note: Actual number of edges written ({}) exceeded the estimated total ({}).",
            total_edges, total_links_estimate
        );
    } else {
        println!(
            "Total number of edges (counting both directions) parsed and written: {}",
            total_edges
        );
    }

    writer.lock().unwrap().flush()?;

    Ok(())
}
