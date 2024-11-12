// src/convert.rs

// Module for converting GFA file to adjacency matrix in edge list format.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rand::Rng;

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
    
    // Sample ~1000 random positions to estimate segment density
    let mut rng = rand::thread_rng();
    let samples = 1000;
    let mut segment_count = 0;
    let mut total_lines = 0;
    let mut reader = BufReader::new(file);

    println!("📊 Quick sampling for accurate estimation...");
    
    // Sample random positions in the file
    for _ in 0..samples {
        let pos = rng.gen_range(0..file_size - 1000); // Avoid end of file
        reader.seek(SeekFrom::Start(pos))?;
        
        // Read and discard partial line
        let mut partial = String::new();
        reader.read_line(&mut partial)?;
        
        // Read one full line and check if it's a segment
        let mut line = String::new();
        reader.read_line(&mut line)?;
        
        if line.starts_with("S\t") {
            segment_count += 1;
        }
        total_lines += 1;
    }

    // Calculate segment density and estimate total segments
    let segment_density = segment_count as f64 / total_lines as f64;
    let estimated_total_lines = file_size / 100; // Assume average line length
    let estimated_segments = (estimated_total_lines as f64 * segment_density) as u64;
    
    // Reset reader for actual parsing
    reader.rewind()?;
    
    // Create thread-safe vector with estimated capacity
    let segment_names = Arc::new(Mutex::new(Vec::with_capacity(estimated_segments as usize)));

    // Create progress bar with accurate estimate
    let pb = Arc::new(ProgressBar::new(estimated_segments));
    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {percent}% ({pos}/{len} segments)")
        .unwrap()
        .progress_chars("▰▰░");
    
    pb.set_style(style);
    println!("🧬 Parsing segments (estimated {estimated_segments} segments)...");

    // Parallel processing of actual segments
    reader.lines()
        .par_bridge()
        .filter_map(Result::ok)
        .filter(|line| line.starts_with("S\t"))
        .for_each(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let segment_name = parts[1].to_string();
                let mut names = segment_names.lock().unwrap();
                names.push(segment_name);
                pb.inc(1);
            }
        });

    pb.finish_with_message("✨ Segment parsing complete!");

    // Extract the Vec from Arc<Mutex>
    let segment_names = Arc::try_unwrap(segment_names)
        .unwrap()
        .into_inner()
        .unwrap();

    let unique_segments: HashSet<String> = segment_names.into_iter().collect();
    let mut sorted_segments: Vec<String> = unique_segments.into_iter().collect();
    sorted_segments.sort();

    let mut segment_indices = HashMap::new();
    for (index, segment_name) in sorted_segments.iter().enumerate() {
        segment_indices.insert(segment_name.clone(), index as u32);
    }

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
    let total_links_estimate = 990_554; // Adjust based on actual data if known
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
                    // Write (to, from) for bidirectionality
                    writer.write_all(&to_index.to_le_bytes()).unwrap();
                    writer.write_all(&from_index.to_le_bytes()).unwrap();

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
