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
/// Returns a mapping from segment names to indices and the total number of segments.
///
/// # Arguments
///
/// * `gfa_path` - Path to the input GFA file.
///
/// # Errors
///
/// Returns an `io::Result` with any file or I/O errors encountered.
///
/// # Panics
///
/// This function does not explicitly panic.
fn parse_segments<P: AsRef<Path>>(gfa_path: P) -> io::Result<(HashMap<String, u32>, u32)> {
    let file = File::open(&gfa_path)?;
    let file_size = file.metadata()?.len();
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    
    println!("📊 Quick sampling for accurate estimation...");
    let samples = 1000;
    let seed: u64 = rand::thread_rng().gen();
    
    let counters = (0..samples).into_par_iter().map(|i| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));

        let pos = if file_size > 1000 {
            rng.gen_range(0..file_size - 1000)
        } else {
            0
        };

        let mut end = pos;
        
        while end < file_size && mmap[end as usize] != b'\n' {
            end += 1;
        }
        end += 1;
        
        let mut start = end;
        while start < file_size && mmap[start as usize] != b'\n' {
            start += 1;
        }
        
        if let Ok(line) = std::str::from_utf8(&mmap[end as usize..start as usize]) {
            (line.starts_with("S\t") as u32, 1u32)
        } else {
            (0, 1)
        }
    }).reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    let segment_density = counters.0 as f64 / counters.1 as f64;
    let estimated_segments = (file_size as f64 / 100.0 * segment_density) as u64;

    println!("🧬 Parsing segments (estimated {estimated_segments} segments)...");
    let pb = Arc::new(ProgressBar::new(estimated_segments));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {percent}% ({pos}/{len} segments)")
        .unwrap()
        .progress_chars("▰▰░"));

    const CHUNK_SIZE: usize = 10_000_000; // 10MB chunks
    let num_chunks = (file_size as usize + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    let segment_names: HashSet<String> = (0..num_chunks).into_par_iter().flat_map(|chunk| {
        let chunk_start = chunk * CHUNK_SIZE;
        let chunk_end = (chunk_start + CHUNK_SIZE).min(file_size as usize);
        
        // Find beginning of first complete line
        let real_start = if chunk == 0 {
            0
        } else {
            // Backtrack to find previous newline
            let mut start = chunk_start;
            while start > 0 && mmap[start - 1] != b'\n' {
                start -= 1;
            }
            // Now backtrack to valid UTF-8 boundary if needed
            while start > 0 && (mmap[start] & 0xC0) == 0x80 {
                start -= 1;
            }
            start
        };
        
        // Find end of last complete line
        let real_end = if chunk == num_chunks - 1 {
            chunk_end
        } else {
            // Look ahead to find next newline
            let mut end = chunk_end;
            while end < file_size as usize && mmap[end] != b'\n' {
                end += 1;
            }
            // Include the newline if we found one
            if end < file_size as usize {
                end + 1
            } else {
                end
            }
        };
        
        let mut segments = HashSet::new();
        let chunk_data = &mmap[real_start..real_end];
        let mut pos = 0;
        
        while pos < chunk_data.len() {
            // Handle empty lines explicitly
            if pos < chunk_data.len() && chunk_data[pos] == b'\n' {
                pos += 1;
                continue;
            }

            // Check for complete "S\t" prefix
            if pos + 2 <= chunk_data.len() && chunk_data[pos] == b'S' && chunk_data[pos + 1] == b'\t' {
                // Find complete line
                let remaining_data = &chunk_data[pos..];
                if let Some(line_length) = remaining_data.iter().position(|&b| b == b'\n') {
                    // Check if this line belongs to this chunk
                    let absolute_pos = pos + real_start;
                    if absolute_pos >= chunk_start && absolute_pos < chunk_end {
                        // Only process if we have a complete line
                        if let Ok(line) = std::str::from_utf8(&remaining_data[..line_length]) {
                            if let Some(name) = line.split('\t').nth(1) {
                                // Use Arc to track progress separately from segment collection
                                segments.insert(name.to_string());
                                pb.inc(1);
                            }
                        }
                    }
                    pos += line_length + 1;
                } else if chunk == num_chunks - 1 {
                    // Special handling for last line of file if it has no newline
                    let absolute_pos = pos + real_start;
                    if absolute_pos >= chunk_start {
                        if let Ok(line) = std::str::from_utf8(remaining_data) {
                            if let Some(name) = line.split('\t').nth(1) {
                                segments.insert(name.to_string());
                                pb.inc(1);
                            }
                        }
                    }
                    break;
                } else {
                    // Incomplete line and not last chunk - skip to next chunk
                    break;
                }
            } else {
                // Find next line start
                match chunk_data[pos..].iter().position(|&b| b == b'\n') {
                    Some(next_line) => pos += next_line + 1,
                    None => break, // No more newlines in this chunk
                }
            }
        }
        
        segments
    }).collect();

    pb.finish_with_message("✨ Segment parsing complete!");

    // Create deterministic index mapping
    let mut sorted_segments: Vec<String> = segment_names.into_iter().collect();
    sorted_segments.sort();
    
    let segment_indices: HashMap<String, u32> = sorted_segments.iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx as u32))
        .collect();

    let segment_counter = sorted_segments.len() as u32;
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
