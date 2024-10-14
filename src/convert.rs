// src/convert.rs

//! Module for converting GFA file to adjacency matrix in edge list format.

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write, BufRead};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

/// Converts a GFA file to an adjacency matrix in edge list format.
///
/// This function reads the GFA file, extracts segments and assigns indices,
/// then parses the links and writes the edges to an output file.
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

    println!("📂 Starting to parse GFA file: {}", gfa_path.as_ref().display());

    // Step 1: Parse the GFA file to extract segments and assign indices
    let (segment_indices, num_segments) = parse_segments(&gfa_path)?;
    println!("✅ Total segments (nodes) identified: {}", num_segments);

    // Step 2: Parse links and write edges
    parse_links_and_write_edges(&gfa_path, &segment_indices, &output_path)?;
    println!("🔗 Finished parsing links and writing edges.");

    let duration = start_time.elapsed();
    println!("⏰ Completed in {:.2?} seconds.", duration);

    Ok(())
}

/// Parses the GFA file to extract segments and assign unique indices.
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
    let reader = BufReader::new(file);

    let segment_indices = Arc::new(Mutex::new(HashMap::new()));
    let segment_counter = Arc::new(Mutex::new(0u32));

    println!("🔍 Parsing segments from GFA file...");

    // Initialize a progress bar with an estimated total number of segments
    let total_segments_estimate = 110_884_673;
    println!(
        "⚠️  Note: Progress bar is based on an estimated total of {} segments.",
        total_segments_estimate
    );
    let pb = ProgressBar::new(total_segments_estimate as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Segments")
            .progress_chars("#>-"),
    );

    // Read lines in parallel
    reader
        .lines()
        .par_bridge()
        .filter_map(Result::ok)
        .filter(|line| line.starts_with('S'))
        .for_each(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                let segment_name = parts[1].to_string();
                let mut indices = segment_indices.lock().unwrap();
                if !indices.contains_key(&segment_name) {
                    let mut counter = segment_counter.lock().unwrap();
                    indices.insert(segment_name, *counter);
                    *counter += 1;
                    pb.inc(1);
                }
            }
        });

    pb.finish_with_message("✅ Finished parsing segments.");

    let segment_counter = *segment_counter.lock().unwrap();

    // Inform about any excess segments beyond the estimate
    if segment_counter as u64 > total_segments_estimate as u64 {
        println!(
            "ℹ️  Note: Actual number of segments ({}) exceeded the estimated total ({}).",
            segment_counter, total_segments_estimate
        );
    } else {
        println!("ℹ️  Total number of segments parsed: {}", segment_counter);
    }

    Ok((Arc::try_unwrap(segment_indices).unwrap().into_inner().unwrap(), segment_counter))
}

/// Parses the GFA file to extract links and write edges to the output file.
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

    println!("🔍 Parsing links and writing edges...");

    // Initialize a progress bar with an estimated total number of links
    let total_links_estimate = 990_554; // This is very wrong
    println!(
        "⚠️  Note: Progress bar is based on an estimated total of {} links.",
        total_links_estimate
    );
    let pb = ProgressBar::new(total_links_estimate as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Links (Estimated)")
            .progress_chars("#>-"),
    );

    let edge_counter = Arc::new(Mutex::new(0u64));

    // Read lines in parallel
    reader
        .lines()
        .par_bridge()
        .filter_map(Result::ok)
        .filter(|line| line.starts_with('L'))
        .for_each(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 5 {
                let from_name = parts[1];
                let to_name = parts[3];

                if let (Some(&from_index), Some(&to_index)) = (
                    segment_indices.get(from_name),
                    segment_indices.get(to_name),
                ) {
                    let mut writer = writer.lock().unwrap();
                    writer.write_all(&from_index.to_le_bytes()).unwrap();
                    writer.write_all(&to_index.to_le_bytes()).unwrap();

                    let mut edge_counter = edge_counter.lock().unwrap();
                    *edge_counter += 1;
                    pb.inc(1);
                }
            }
        });

    pb.finish_with_message("✅ Finished parsing links.");

    let total_edges = *edge_counter.lock().unwrap();

    // Inform about any excess links beyond the estimate
    if total_edges > total_links_estimate as u64 {
        println!(
            "ℹ️  Note: Actual number of links ({}) exceeded the estimated total ({}).",
            total_edges, total_links_estimate
        );
    } else {
        println!("ℹ️  Total number of links parsed: {}", total_edges);
    }

    // Flush the writer
    writer.lock().unwrap().flush()?;

    Ok(())
}
