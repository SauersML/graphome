// src/convert.rs

// Module for converting GFA file to adjacency matrix in edge list format.

use gbwt::{Orientation, GBZ};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use simple_sds::serialize;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Converts a graph stored as GFA or GBZ into an adjacency matrix in edge list format.
///
/// This function performs a two-pass approach:
/// 1. First Pass: Parses the GFA file to collect all unique segment names and assigns them deterministic indices based on sorted order.
/// 2. Second Pass: Parses the links and writes bidirectional edges to the output file in parallel.
///
/// # Arguments
///
/// * `gfa_path` - Path to the input GFA/GBZ file.
/// * `output_path` - Path to the output adjacency matrix file.
///
/// # Errors
///
/// Returns an `io::Result` with any file or I/O errors encountered.
///
/// # Panics
///
/// This function does not explicitly panic.
pub fn convert_gfa_to_edge_list<P: AsRef<Path>>(gfa_path: P, output_path: P) -> io::Result<()> {
    let input_path: PathBuf = gfa_path.as_ref().into();
    let output_path: PathBuf = output_path.as_ref().into();
    let start_time = Instant::now();

    let extension = input_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    match extension.as_deref() {
        Some("gbz") => convert_gbz_to_edge_list(&input_path, &output_path)?,
        _ => convert_gfa_to_edge_list_inner(&input_path, &output_path)?,
    }

    let duration = start_time.elapsed();
    println!("Completed in {:.2?} seconds.", duration);

    Ok(())
}

fn convert_gfa_to_edge_list_inner(gfa_path: &Path, output_path: &Path) -> io::Result<()> {
    println!("Starting to parse GFA file: {}", gfa_path.display());

    let (segment_indices, num_segments) = parse_segments(gfa_path)?;
    println!("Total segments (nodes) identified: {}", num_segments);

    parse_links_and_write_edges(gfa_path, &segment_indices, output_path)?;
    println!("Finished parsing links between nodes and writing edges.");

    Ok(())
}

fn convert_gbz_to_edge_list(gbz_path: &Path, output_path: &Path) -> io::Result<()> {
    println!("Starting to parse GBZ file: {}", gbz_path.display());

    let gbz: GBZ = serialize::load_from(gbz_path)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;

    let (segment_indices, num_segments) = parse_segments_from_gbz(&gbz);
    println!("Total segments (nodes) identified: {}", num_segments);

    write_edges_from_gbz(&gbz, &segment_indices, output_path)?;
    println!("Finished parsing links between nodes and writing edges.");

    Ok(())
}

/// Parses the GFA file to extract segments and assign unique indices deterministically.
///
/// Approach:
/// 1. Memory-map the file for fast random access.
/// 2. Identify line boundaries by scanning for newline characters.
/// 3. Estimate the number of segments for progress display by sampling lines.
/// 4. Use `fold` and `reduce` in parallel to accumulate segment names into a single `HashSet<String>`.
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
/// - A `HashMap<String, u32>` mapping segment names to indices
/// - A `u32` count of total segments
///
/// # Errors
///
/// Returns an `io::Result` on file or I/O errors.
///
/// # Panics
///
/// Does not explicitly panic.
fn parse_segments<P: AsRef<Path>>(gfa_path: P) -> io::Result<(HashMap<String, u32>, u32)> {
    use memchr::memchr_iter;
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // --- Memory-map the file ---
    let file = File::open(&gfa_path)?;
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
    let estimated_segments = (file_size as f64 / 100.0 * density) as u64;

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
                let mut set = HashSet::new();
                set.reserve(estimated_segments as usize / rayon::current_num_threads().max(1));
                set
            },
            |mut local_set, batch| {
                for i in batch {
                    let line_slice = get_line_slice(i);
                    if line_slice.starts_with(b"S\t") {
                        if let Ok(line_str) = std::str::from_utf8(line_slice) {
                            let parts: Vec<&str> = line_str.split('\t').collect();
                            // GFA v1 requires at least 3 fields for an S line:
                            //   0: "S"
                            //   1: segment name
                            //   2: sequence (or '*')
                            if parts.len() < 3 {
                                continue; // skip malformed lines
                            }
                            let segment_name = parts[1].trim();
                            if !segment_name.is_empty() {
                                local_set.insert(segment_name.to_string());
                                pb.inc(1);
                            }
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
    let mut all_names: Vec<String> = segment_names.into_iter().collect();
    all_names.par_sort();

    let segment_indices: HashMap<String, u32> = all_names
        .into_iter()
        .enumerate()
        .map(|(i, name)| (name, i as u32))
        .collect();

    let segment_counter = segment_indices.len() as u32;
    println!("🎯 Total segments (nodes) identified: {}", segment_counter);

    Ok((segment_indices, segment_counter))
}

fn parse_segments_from_gbz(gbz: &GBZ) -> (HashMap<String, u32>, u32) {
    let mut names: Vec<String> = Vec::new();

    if let Some(iter) = gbz.segment_iter() {
        for segment in iter {
            let key = match std::str::from_utf8(segment.name) {
                Ok(name) if !name.is_empty() => name.to_string(),
                _ => format!("node_{}", segment.id),
            };
            names.push(key);
        }
    } else {
        // Fallback to individual nodes if translation is not available.
        for node_id in gbz.node_iter() {
            names.push(node_id.to_string());
        }
    }

    names.sort();
    names.dedup();

    let mut map = HashMap::new();
    for (idx, name) in names.into_iter().enumerate() {
        map.insert(name, idx as u32);
    }

    let count = map.len() as u32;
    (map, count)
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
fn parse_links_and_write_edges<P: AsRef<Path>>(
    gfa_path: P,
    segment_indices: &HashMap<String, u32>,
    output_path: P,
) -> io::Result<()> {
    let file = File::open(&gfa_path)?;
    let reader = BufReader::new(file);

    let output_file = File::create(&output_path)?;
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
        .lines()
        .par_bridge()
        .filter_map(Result::ok)
        .for_each(|line| {
            let mut local_edges = Vec::new();

            if line.starts_with("L\t") {
                // GFA v1 "L" line requires 6 fields:
                //   0: "L"
                //   1: from segment
                //   2: from orientation (+/-)
                //   3: to segment
                //   4: to orientation (+/-)
                //   5: overlap/CIGAR (can be '*')
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() < 6 {
                    // Skip malformed L lines
                    return;
                }
                let from_name = parts[1].trim().to_string();
                let to_name = parts[3].trim().to_string();

                // Look up indices
                if let (Some(&f_idx), Some(&t_idx)) = (
                    segment_indices.get(&from_name),
                    segment_indices.get(&to_name),
                ) {
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

fn write_edges_from_gbz(
    gbz: &GBZ,
    segment_indices: &HashMap<String, u32>,
    output_path: &Path,
) -> io::Result<()> {
    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    println!("Parsing links between nodes and writing edges from GBZ...");

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} {pos} edges written")
            .unwrap(),
    );

    let mut edges_written: u64 = 0;

    let mut seen_edges: HashSet<u64> = HashSet::new();

    if let Some(segments) = gbz.segment_iter() {
        for segment in segments {
            let name_key: Cow<'_, str> = match std::str::from_utf8(segment.name) {
                Ok(name) if !name.is_empty() => Cow::Borrowed(name),
                _ => Cow::Owned(format!("node_{}", segment.id)),
            };

            let Some(&from_idx) = segment_indices.get(name_key.as_ref()) else {
                continue;
            };

            for orientation in [Orientation::Forward, Orientation::Reverse] {
                if let Some(successors) = gbz.segment_successors(&segment, orientation) {
                    for (neighbor, _) in successors {
                        let neighbor_key: Cow<'_, str> = match std::str::from_utf8(neighbor.name) {
                            Ok(name) if !name.is_empty() => Cow::Borrowed(name),
                            _ => Cow::Owned(format!("node_{}", neighbor.id)),
                        };

                        let Some(&to_idx) = segment_indices.get(neighbor_key.as_ref()) else {
                            continue;
                        };

                        let (a, b) = if from_idx <= to_idx {
                            (from_idx, to_idx)
                        } else {
                            (to_idx, from_idx)
                        };
                        let key = ((a as u64) << 32) | b as u64;
                        if !seen_edges.insert(key) {
                            continue;
                        }

                        writer.write_all(&a.to_le_bytes())?;
                        writer.write_all(&b.to_le_bytes())?;
                        edges_written += 2;
                        pb.inc(1);
                    }
                }
            }
        }
    } else {
        for node_id in gbz.node_iter() {
            let from_name = node_id.to_string();
            let Some(&from_idx) = segment_indices.get(&from_name) else {
                continue;
            };

            for orientation in [Orientation::Forward, Orientation::Reverse] {
                if let Some(successors) = gbz.successors(node_id, orientation) {
                    for (neighbor, _) in successors {
                        let neighbor_name = neighbor.to_string();
                        let Some(&to_idx) = segment_indices.get(&neighbor_name) else {
                            continue;
                        };

                        let (a, b) = if from_idx <= to_idx {
                            (from_idx, to_idx)
                        } else {
                            (to_idx, from_idx)
                        };
                        let key = ((a as u64) << 32) | b as u64;
                        if !seen_edges.insert(key) {
                            continue;
                        }

                        writer.write_all(&a.to_le_bytes())?;
                        writer.write_all(&b.to_le_bytes())?;
                        edges_written += 2;
                        pb.inc(1);
                    }
                }
            }
        }
    }

    writer.flush()?;
    pb.finish_with_message("Finished parsing links between nodes.");
    println!(
        "Total number of edges (counting both directions) parsed and written: {}",
        edges_written
    );

    Ok(())
}
