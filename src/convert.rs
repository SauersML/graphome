// src/convert.rs

// Module for converting GFA file to adjacency matrix in edge list format.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rand_chacha::ChaCha8Rng;
use rand::Rng;
use rand::SeedableRng;

use indicatif::{ProgressBar, ProgressStyle};

/// Converts a GFA file to an adjacency matrix in edge list format.
///
/// This function performs a two-pass approach:
/// 1. First Pass: Parses the GFA file to collect all unique segment names and assigns them deterministic indices based on sorted order.
/// 2. Second Pass: Parses the links and writes bidirectional edges to the output file in parallel.
///
/// # Arguments
///
/// * `gfa_path` - Path to the input GFA file.
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
    let start_time = Instant::now();

    println!(
        "Starting to parse GFA file: {}",
        gfa_path.as_ref().display()
    );

    // Step 1: Parse the GFA file to extract segments and assign deterministic indices
    let (segment_indices, num_segments) = parse_segments(&gfa_path)?;
    println!("Total segments (nodes) identified: {}", num_segments);

    // Step 2: Parse links and write edges in parallel
    parse_links_and_write_edges(&gfa_path, &segment_indices, &output_path)?;
    println!("Finished parsing links and writing edges.");

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
    use rand::Rng;

    // Step 1: Memory-map the file
    let file = File::open(&gfa_path)?;
    let file_size = file.metadata()?.len();
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

    // Step 2: Identify lines
    let mut line_ends = Vec::new();
    for (i, &byte) in mmap.iter().enumerate() {
        if byte == b'\n' {
            line_ends.push(i);
        }
    }
    let has_final_newline = line_ends.last().map_or(false, |&pos| pos as u64 == file_size - 1);
    let total_lines = if has_final_newline {
        line_ends.len()
    } else {
        line_ends.len() + 1
    };

    let get_line_slice = |i: usize| {
        let start = if i == 0 { 0 } else { line_ends[i - 1] + 1 };
        let end = if i < line_ends.len() {
            line_ends[i]
        } else {
            (file_size - 1) as usize
        };
        &mmap[start..=end]
    };

    // Step 3: Estimate segment count by sampling
    let samples = 1000.min(total_lines);
    let seed: u64 = rand::thread_rng().gen();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut sample_count = 0u32;
    for _ in 0..samples {
        let idx = rng.gen_range(0..total_lines);
        let line_slice = get_line_slice(idx);
        if line_slice.starts_with(b"S\t") {
            sample_count += 1;
        }
    }
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

    // Step 4: Parallel processing of lines using fold & reduce
    // fold: Each thread gets an initial empty HashSet, processes some lines, and returns the HashSet.
    // reduce: Combine all thread-local HashSets into one.
    let segment_names = (0..total_lines)
        .into_par_iter()
        .fold(
            || HashSet::new(),
            |mut local_set, i| {
                let line_slice = get_line_slice(i);
                if let Ok(line_str) = std::str::from_utf8(line_slice) {
                    if line_str.starts_with("S\t") {
                        let parts: Vec<&str> = line_str.split('\t').collect();
                        if parts.len() >= 2 {
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
        .reduce(
            || HashSet::new(),
            |mut a, b| {
                a.extend(b);
                a
            },
        );

    pb.finish_with_message("✨ Segment parsing complete!");

    // Directly create indices using numerical IDs from GFA
    let segment_indices: HashMap<String, u32> = segment_names
        .into_iter()
        .filter_map(|name| {
            name.parse::<u32>().ok().map(|id| (name, id - 1))  // -1 because GFA is 1-based
        })
        .collect();

    let segment_counter = segment_indices.len() as u32;
    println!("🎯 Total segments (nodes) identified: {}", segment_counter);

    Ok((segment_indices, segment_counter))
}




/// Parses the GFA file to extract links and write edges to the output file.
///
/// Writes both (from, to) and (to, from) for bidirectional edges.
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
/// This function does not explicitly panic.
fn parse_links_and_write_edges<P: AsRef<Path>>(
    gfa_path: P,
    segment_indices: &HashMap<String, u32>,
    output_path: P,
) -> io::Result<()> {
    let file = File::open(&gfa_path)?;
    let reader = BufReader::new(file);

    let output_file = File::create(&output_path)?;
    let writer = Arc::new(Mutex::new(BufWriter::new(output_file)));

    println!("Parsing links and writing edges...");

    // Initialize a progress bar with an estimated total number of links
    let total_links_estimate = 309_511_744; // Adjust based on actual data if known
    println!(
        "Note: Progress bar is based on an estimated total of {} links.",
        total_links_estimate
    );
    let pb = ProgressBar::new(total_links_estimate as u64);

    // Configure the progress bar style
    let style = ProgressStyle::default_bar()
        .template("{spinner} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Links (Estimated)")
        .unwrap() // Handle the Result here
        .progress_chars("#>-");

    pb.set_style(style);

    let edge_counter = Arc::new(Mutex::new(0u64));

    // Read lines in parallel
    reader
        .lines()
        .par_bridge()
        .filter_map(Result::ok)
        .filter(|line| line.starts_with("L\t"))
        .for_each(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 6 {
                let from_name = parts[1].trim();
                let to_name = parts[3].trim();

                if let (Some(&from_index), Some(&to_index)) =
                    (segment_indices.get(from_name), segment_indices.get(to_name))
                {
                    let mut writer = writer.lock().unwrap();
                    // Write (from, to)
                    writer.write_all(&from_index.to_le_bytes()).unwrap();
                    writer.write_all(&to_index.to_le_bytes()).unwrap();
                    // Write (to, from) for bidirectionality -- unnecessary if we understand that all edges are bidirectional
                    //writer.write_all(&to_index.to_le_bytes()).unwrap();
                    //writer.write_all(&from_index.to_le_bytes()).unwrap();

                    let mut edge_counter = edge_counter.lock().unwrap();
                    *edge_counter += 2;
                    pb.inc(2);
                }
            }
        });

    pb.finish_with_message("Finished parsing links.");

    let total_edges = *edge_counter.lock().unwrap();

    // Inform about any excess links beyond the estimate
    if total_edges > total_links_estimate as u64 {
        println!(
            "Note: Actual number of links ({}) exceeded the estimated total ({}).",
            total_edges, total_links_estimate
        );
    } else {
        println!("Total number of links parsed: {}", total_edges);
    }

    // Flush the writer
    writer.lock().unwrap().flush()?;

    Ok(())
}
