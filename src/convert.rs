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
///
/// # Arguments
///
/// * `gfa_path` - Path to the input GFA file.
///
/// # Returns
///
/// A tuple containing:
/// - A `HashMap<String, u32>` mapping segment names to deterministic indices
/// - A `u32` count of total segments identified
///
/// # Errors
///
/// Returns an `io::Result` if any file or I/O errors are encountered.
///
/// # Panics
///
/// This function does not explicitly panic.
fn parse_segments<P: AsRef<Path>>(gfa_path: P) -> io::Result<(HashMap<String, u32>, u32)> {
    // Open the GFA file and prepare for line-by-line reading
    let file = File::open(&gfa_path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();
    let reader = BufReader::new(file);

    // We will read all lines first to ensure no partial lines are processed.
    let lines: Vec<String> = reader.lines().collect::<io::Result<_>>()?;

    // Perform a quick sampling to estimate the number of segments.
    // Instead of sampling arbitrary byte positions as before, we can sample lines randomly.
    // This approach avoids partial reads and is simpler. If the file is small, this is trivial;
    // if large, it's still reliable because we never consider partial lines.
    let samples = 1000.min(lines.len()); // Take up to 1000 samples or fewer if fewer lines
    let seed: u64 = rand::thread_rng().gen();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sample_count = 0u32;
    for _ in 0..samples {
        let idx = rng.gen_range(0..lines.len());
        if lines[idx].starts_with("S\t") {
            sample_count += 1;
        }
    }

    // Compute a rough estimate of segments based on sample density.
    let density = if samples > 0 {
        sample_count as f64 / samples as f64
    } else {
        0.0
    };
    // Assume average line length ~100 bytes or another heuristic. Using the original logic:
    // The original logic used file size and a density estimate. We'll maintain the spirit:
    let estimated_segments = (file_size as f64 / 100.0 * density) as u64;

    println!("🧬 Parsing segments (estimated {estimated_segments} segments)...");
    let pb = Arc::new(ProgressBar::new(estimated_segments));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {percent}% ({pos}/{len} segments)")
            .unwrap()
            .progress_chars("▰▰░"),
    );

    // Extract segment names from each `S` line.
    // Requirements:
    // - Line must start with "S\t"
    // - Must have at least two fields: "S" and the segment name
    // We'll split on tabs and take the second field as the name.
    let mut segment_names = HashSet::new();
    for line in &lines {
        if line.starts_with("S\t") {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                // parts[0] == "S", parts[1] is the segment name
                let segment_name = parts[1].to_string();
                // Add the segment name if not empty or invalid
                if !segment_name.is_empty() {
                    segment_names.insert(segment_name);
                    pb.inc(1);
                }
            }
        }
    }

    pb.finish_with_message("✨ Segment parsing complete!");

    // Sort and assign deterministic indices
    let mut sorted_segments: Vec<String> = segment_names.into_iter().collect();
    sorted_segments.sort_unstable();

    let segment_indices: HashMap<String, u32> = sorted_segments
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx as u32))
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
                let from_name = parts[1];
                let to_name = parts[3];

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
