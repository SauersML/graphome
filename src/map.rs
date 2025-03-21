//   1) Parse a massive GFA v1.0 using memory-mapped I/O to handle
//      tens-of-GB files efficiently.
//   2) Parse an "untangle" PAF (also large) in parallel.
//   3) Build prefix sums for each path in the GFA so we can map
//      path offsets -> node IDs, and vice versa.
//   4) Build a reference interval tree for coordinate lookups on hg38.
//   5) Provide CLI subcommands:
//         node2coord <nodeID>
//         coord2node <chr>:<start>-<end>
//      which do node->hg38 or hg38->node queries.

use std::collections::{HashMap, HashSet};
use std::fs::{File};
use std::io::{BufReader, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use memmap2::{MmapOptions};
use tempfile::NamedTempFile;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use gbwt::{GBZ, Orientation};
use simple_sds::serialize;

// Data Structures

// We'll keep these big data structures in memory for queries.

/// Run node2coord with printing of intervals, total range, and merging by chromosome
pub fn run_node2coord(gfa_path: &str, paf_path: &str, node_id: &str) {
    // Convert node_id to numeric form
    let node_id_num = match node_id.parse::<usize>() {
        Ok(num) => num,
        Err(_) => {
            println!("No reference coords found for node {} (non-numeric node ID)", node_id);
            return;
        }
    };

    // Get or create GBZ file
    let gbz_path = make_gbz_exist(gfa_path, paf_path);
    eprintln!("[INFO] Using GBZ index from '{}'", gbz_path);

    // Load GBZ
    let gbz: GBZ = match serialize::load_from(&gbz_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading GBZ file: {}", e);
            return;
        }
    };

    if !gbz.has_node(node_id_num) {
        println!("No reference coords found for node {}", node_id);
        return;
    }

    // Get node coordinates using GBZ
    let results = node_to_coords(&gbz, node_id_num);
    if results.is_empty() {
        println!("No reference coords found for node {}", node_id);
    } else {
        // Print each interval
        let mut global_min = usize::MAX;
        let mut global_max = 0;
        let mut intervals_by_chr = std::collections::HashMap::<String, Vec<(usize, usize)>>::new();
        
        // Deduplicate coordinates for OVERALL mapping
        let mut unique_coords = std::collections::HashSet::new();

        for (chr, st, en) in &results {
            println!("{}:{}-{}", chr, st, en);
            unique_coords.insert((chr.clone(), *st, *en));
            
            if *st < global_min {
                global_min = *st;
            }
            if *en > global_max {
                global_max = *en;
            }
            intervals_by_chr.entry(chr.clone()).or_default().push((*st, *en));
        }

        // Print the TOTAL RANGE
        println!("TOTAL COORD RANGE: {}..{}", global_min, global_max);

        // Merge overlapping intervals per chromosome
        for (chr, intervals) in intervals_by_chr {
            let merged = merge_intervals(intervals);
            println!("CONTIGUOUS GROUPS for chromosome: {}", chr);
            for (start_val, end_val) in merged {
                println!("  Group range: {}..{}", start_val, end_val);
            }
        }
        
        // Print OVERALL coordinate mapping (sample-abstracted)
        println!("\n============ OVERALL MAPPING ============");
        println!("NODE: {}", node_id);
        
        // Convert unique_coords to a vector and create ranges
        let unique_coord_vec: Vec<(String, usize, usize)> = unique_coords.into_iter().collect();
        let coord_ranges = create_coord_ranges(&unique_coord_vec);
        
        println!("UNIQUE REFERENCE MAPPINGS:");
        for (chr, range) in coord_ranges {
            println!("  {}: {}", chr, range);
        }
        println!("TOTAL UNIQUE MAPPINGS: {}", unique_coord_vec.len());
        println!("================================================");
    }
}

/// Run coord2node with printing of intervals, total range, and merging by path
pub fn run_coord2node(gfa_path: &str, paf_path: &str, region: &str) {
    println!("DEBUG: Looking for region: {}", region);
    
    // Get or create GBZ file
    let gbz_path = make_gbz_exist(gfa_path, paf_path);
    eprintln!("[INFO] Using GBZ index from '{}'", gbz_path);

    // Load GBZ
    let gbz: GBZ = match serialize::load_from(&gbz_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading GBZ file: {}", e);
            return;
        }
    };

    // Parse region
    if let Some((chr, start, end)) = parse_region(region) {
        let results = coord_to_nodes(&gbz, &chr, start, end);
        if results.is_empty() {
            println!("No nodes found for region {}:{}-{}", chr, start, end);
        } else {
            // Print each overlap
            let mut global_min = usize::MAX;
            let mut global_max = 0;
            let mut intervals_by_path = std::collections::HashMap::<String, Vec<(usize, usize)>>::new();
            
            // Collect unique node IDs for OVERALL mapping
            let mut unique_nodes = std::collections::HashSet::new();

            for r in &results {
                println!(
                    "path={} node={}({}) offsets=[{}..{}]",
                    r.path_name,
                    r.node_id,
                    if r.node_orient { '+' } else { '-' },
                    r.path_off_start,
                    r.path_off_end
                );

                // Try to convert node_id to numeric form for range creation
                if let Ok(node_num) = r.node_id.parse::<usize>() {
                    unique_nodes.insert(node_num);
                }

                if r.path_off_start < global_min {
                    global_min = r.path_off_start;
                }
                if r.path_off_end > global_max {
                    global_max = r.path_off_end;
                }
                intervals_by_path
                    .entry(r.path_name.clone())
                    .or_default()
                    .push((r.path_off_start, r.path_off_end));
            }

            // Print total node offset range
            println!("TOTAL NODE OFFSET RANGE: {}..{}", global_min, global_max);

            // Merge intervals per path
            for (path, intervals) in intervals_by_path {
                let merged = merge_intervals(intervals);
                println!("CONTIGUOUS GROUPS for path: {}", path);
                for (start_val, end_val) in merged {
                    println!("  Group range: {}..{}", start_val, end_val);
                }
            }
            
            // Print OVERALL node mapping (sample-abstracted)
            println!("\n============ OVERALL MAPPING ============");
            println!("REFERENCE: {}:{}-{}", chr, start, end);
            
            // Convert unique_nodes to a vector for range creation
            let unique_node_vec: Vec<usize> = unique_nodes.into_iter().collect();
            let node_ranges = create_node_ranges(&unique_node_vec);
            
            println!("UNIQUE NODES: {}", node_ranges);
            println!("TOTAL UNIQUE NODES: {}", unique_node_vec.len());
            println!("================================================");
        }
    } else {
        eprintln!("Could not parse region format: {}", region);
    }
}

// node_to_coords
// Using GBZ index to find reference coordinates for a node
pub fn node_to_coords(gbz: &GBZ, node_id: usize) -> Vec<(String, usize, usize)> {
    let mut results = Vec::new();
    
    // Get node length
    let node_len = match gbz.sequence_len(node_id) {
        Some(len) => len,
        None => return results,
    };
    
    // Get reference sample IDs
    let ref_samples = gbz.reference_sample_ids(true);
    if ref_samples.is_empty() {
        eprintln!("Warning: No reference samples found in GBZ");
        return results;
    }
    
    if let Some(metadata) = gbz.metadata() {
        // Create a search state for the node
        let forward_state = match gbz.search_state(node_id, Orientation::Forward) {
            Some(state) => state,
            None => return results,
        };
        
        // Find all paths that contain this node
        for ref_sample in &ref_samples {
            for (path_id, path_name) in metadata.path_iter().enumerate() {
                if path_name.sample() != *ref_sample {
                    continue;
                }
                
                let contig_name = metadata.contig_name(path_name.contig());
                
                // Check if this path contains the node by looking at all paths
                if let Some(mut path_iter) = gbz.path(path_id, Orientation::Forward) {
                    let mut position = 0;
                    let mut found_positions = Vec::new();
                    
                    // Scan path to find node positions
                    while let Some((path_node, orientation)) = path_iter.next() {
                        if path_node == node_id {
                            found_positions.push((position, orientation == Orientation::Forward));
                        }
                        if let Some(len) = gbz.sequence_len(path_node) {
                            position += len;
                        }
                    }
                    
                    // For each occurrence, extract reference coordinates
                    for (pos, is_forward) in found_positions {
                        // Get reference positions for this path
                        let ref_paths = gbz.reference_positions(1000, false);
                        for ref_path in ref_paths {
                            if ref_path.id == path_id {
                                for (path_pos, gbwt_pos) in &ref_path.positions {
                                    if *path_pos <= pos && *path_pos + node_len >= pos {
                                        // We found a reference position that contains our node
                                        let offset = pos - path_pos;
                                        let ref_start = *path_pos + offset;
                                        let ref_end = ref_start + node_len - 1;
                                        
                                        results.push((contig_name.clone(), ref_start, ref_end));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    results
}


// coord_to_nodes
// Using GBZ index to find nodes at a reference position
pub fn coord_to_nodes(gbz: &GBZ, chr: &str, start: usize, end: usize) -> Vec<Coord2NodeResult> {
println!("DEBUG: Searching for region {}:{}-{}", chr, start, end);
    let mut results = Vec::new();
    
    // Get reference paths and their positions
    let ref_paths = gbz.reference_positions(1000, true);
    if ref_paths.is_empty() {
        println!("DEBUG: No reference paths found in GBZ");
        return results;
    }
    
    if let Some(metadata) = gbz.metadata() {
        // Find paths that align to this region
        for ref_path in &ref_paths {
            let path_metadata = match metadata.path(ref_path.id) {
                Some(meta) => meta,
                None => continue,
            };
            
            let contig_name = metadata.contig_name(path_metadata.contig());
            if contig_name != chr {
                continue;
            }
            
            println!("DEBUG: Found path in reference that matches chromosome {}", chr);
            
            // Find nodes in this region
            for (path_pos, pos) in &ref_path.positions {
                // Skip if position is outside our target region
                if *path_pos > end || *path_pos + 1000 < start {
                    continue;
                }
                
                // Get the node at this position
                let (node_id, orientation) = gbwt::support::decode_node(pos.node);
                if let Some(node_len) = gbz.sequence_len(node_id) {
                    let node_end = *path_pos + node_len;
                    
                    // Check if this node overlaps our region
                    if *path_pos <= end && node_end >= start {
                        let overlap_start = start.max(*path_pos);
                        let overlap_end = end.min(node_end);
                        
                        if overlap_start <= overlap_end {
                            println!("DEBUG: Found overlapping node {} with range {}..{}", 
                                     node_id, overlap_start, overlap_end);
                            
                            // Calculate offset within the node
                            let node_off_start = overlap_start - *path_pos;
                            let node_off_end = overlap_end - *path_pos;
                            
                            results.push(Coord2NodeResult {
                                path_name: contig_name.clone(),
                                node_id: node_id.to_string(),
                                node_orient: orientation == Orientation::Forward,
                                path_off_start: node_off_start,
                                path_off_end: node_off_end,
                            });
                        }
                    }
                }
            }
        }
    }
    
    results
}



/// Validates and fixes path names in a GFA file for PanSN naming compliance.
/// This function focuses on fixing "Invalid haplotype field" errors by replacing 
/// non-numeric haplotype fields with "0".
/// Returns the path to use (original if no fixes needed, new fixed file otherwise).
pub fn validate_gfa_for_gbwt(gfa_path: &str) -> Result<String, String> {
    eprintln!("[INFO] Validating and fixing GFA path names: {}", gfa_path);

    // Open and memory-map the GFA for efficient processing
    let file = File::open(gfa_path)
        .map_err(|e| format!("Failed to open GFA file: {}", e))?;
    let metadata = file.metadata()
        .map_err(|e| format!("Failed to get metadata for GFA file: {}", e))?;
    let file_size = metadata.len();

    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| format!("Could not memory-map GFA file: {}", e))?
    };

    // Setup progress tracking
    let pb = ProgressBar::new(file_size);
    pb.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {percent}% ({eta} remaining)"
    ).unwrap());

    // Prepare for parallel scanning
    let chunk_size = 100_000_000; // 100MB chunks
    let num_chunks = (file_size as usize + chunk_size - 1) / chunk_size;
    let chunks: Vec<(usize, usize)> = (0..num_chunks)
        .map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(file_size as usize);
            (start, end)
        })
        .collect();

    // First-pass: parallel scan to identify lines needing fixes
    eprintln!("[INFO] Scanning for invalid path names using {} parallel chunks...", num_chunks);
    
    // Create a mutex-protected shared progress bar
    let pb_arc = Arc::new(Mutex::new(pb));
    
    // Process chunks in parallel
    let fixes_per_chunk: Vec<HashMap<usize, (String, String)>> = chunks.par_iter()
        .map(|(start, end)| {
            let mut local_fixes = HashMap::new();
            
            // Find the true beginning of a line for this chunk
            let mut line_start = *start;
            if *start > 0 {
                // Find the previous newline or beginning of file
                for i in (0..*start).rev() {
                    if mmap[i] == b'\n' {
                        line_start = i + 1;
                        break;
                    }
                    if i == 0 {
                        line_start = 0;
                    }
                }
            }
            
            let mut line_number = 0;
            let mut prev_newline = line_start;
            
            // Count newlines before our chunk to get correct line number
            if line_start > 0 {
                let newlines_before = mmap[..line_start].iter()
                    .filter(|&&b| b == b'\n')
                    .count();
                line_number = newlines_before;
            }
            
            // Process this chunk
            for i in *start..*end {
                // Update progress periodically
                if i % 10_000_000 == 0 {
                    let mut pb_guard = pb_arc.lock().unwrap();
                    pb_guard.set_position(i as u64);
                }
                
                // Process lines when we hit a newline
                if mmap[i] == b'\n' {
                    line_number += 1;
                    
                    // Check if this is a path line (P or W) that might need fixing
                    if i > line_start && (mmap[line_start] == b'P' || mmap[line_start] == b'W') {
                        let line_bytes = &mmap[line_start..i];
                        
                        // Check if we need to process the line - fast check for tab characters
                        let tab_count = line_bytes.iter().filter(|&&b| b == b'\t').count();
                        if tab_count >= 2 {
                            // Find the path name (second column)
                            let mut tabs = 0;
                            let mut path_start = 0;
                            let mut path_end = 0;
                            
                            for (pos, &b) in line_bytes.iter().enumerate() {
                                if b == b'\t' {
                                    tabs += 1;
                                    if tabs == 1 {
                                        path_start = pos + 1;
                                    } else if tabs == 2 {
                                        path_end = pos;
                                        break;
                                    }
                                }
                            }
                            if path_end > path_start {
                                let path_name_bytes = &line_bytes[path_start..path_end];
                                let path_name = unsafe { std::str::from_utf8_unchecked(path_name_bytes) };
                                
                                // PanSN path name validation and fixing
                                if path_name.contains('#') {
                                    // Fast-path for the most common reference genome patterns
                                    // This avoids full splitting for most cases in typical pangenomes
                                    if let Some(hash_pos) = path_name.find('#') {
                                        if (path_name.starts_with("chm13#chr") && hash_pos == 5) || 
                                           (path_name.starts_with("grch38#chr") && hash_pos == 6) {
                                            if !path_name[(hash_pos+1)..].starts_with(|c: char| c.is_ascii_digit()) {
                                                // Quick fix for common reference paths
                                                let sample = &path_name[0..hash_pos];
                                                let rest = &path_name[(hash_pos+1)..];
                                                let fixed_name = format!("{}#0#{}", sample, rest);
                                                eprintln!("[FIX:ref] Line {}: '{}' → '{}'", line_number, path_name, fixed_name);
                                                local_fixes.insert(line_number, (path_name.to_string(), fixed_name));
                                                continue; // Skip to next line after handling common case
                                            }
                                        }
                                        
                                        // Count '#' symbols without full splitting
                                        let hash_count = path_name.as_bytes().iter().filter(|&&b| b == b'#').count();
                                        
                                        // Handle the by-far most common case (sample#haplotype#contig)
                                        if hash_count == 2 {
                                            // Split only once to check second part (haplotype field)
                                            let parts: Vec<&str> = path_name.splitn(3, '#').collect();
                                            // Check if haplotype field is numeric with zero-copy if possible
                                            if parts[1].parse::<u32>().is_err() || parts[1].is_empty() {
                                                // Pre-allocate result string for better performance
                                                let fixed_name = format!("{}#0#{}", parts[0], parts[2]);
                                                eprintln!("[FIX:hap] Line {}: '{}' → '{}'", line_number, path_name, fixed_name);
                                                local_fixes.insert(line_number, (path_name.to_string(), fixed_name));
                                                continue; // Skip to next line
                                            }
                                        }
                                        
                                        // Handle missing haplotype case (only one '#' symbol)
                                        if hash_count == 1 {
                                            let parts: Vec<&str> = path_name.split('#').collect();
                                            let sample = if parts[0].is_empty() { "unknown" } else { parts[0] };
                                            let contig = if parts.len() > 1 && !parts[1].is_empty() { parts[1] } else { "unknown" };
                                            let fixed_name = format!("{}#0#{}", sample, contig);
                                            eprintln!("[FIX:miss] Line {}: '{}' → '{}'", line_number, path_name, fixed_name);
                                            local_fixes.insert(line_number, (path_name.to_string(), fixed_name));
                                            continue; // Skip to next line
                                        }
                                    }
                                    
                                    // For complex or uncommon cases, fall back to robust comprehensive parsing
                                    let parts: Vec<&str> = path_name.split('#').collect();
                                    let parts_len = parts.len();
                                    
                                    // Handle complex cases with multiple '#' symbols
                                    if parts_len > 2 {
                                        let needs_fixing = if parts_len > 3 {
                                            // More than 2 '#' symbols - very likely to cause parsing issues
                                            true
                                        } else if parts[1].parse::<u32>().is_err() || parts[1].is_empty() {
                                            // Standard case where haplotype field isn't numeric
                                            true
                                        } else {
                                            false
                                        };
                                        
                                        if needs_fixing {
                                            // Handle problematic cases like "HG00438#2#JAHBCA010000258.1#MT"
                                            // where GBWT parser sees "JAHBCA010000258.1" as the haplotype field
                                            let mut fixed_name = String::new();
                                            
                                            if parts_len > 3 {
                                                // For paths with 3+ '#' symbols, completely restructure the name
                                                let sample = parts[0].to_string();
                                                let contig = parts[2..].join("_"); // Join remaining parts with underscore
                                                fixed_name = format!("{}#0#{}", sample, contig);
                                            } else {
                                                // For standard 3-part names with non-numeric haplotype
                                                let mut fixed_parts = parts.clone();
                                                fixed_parts[1] = "0";
                                                
                                                // Make sure no empty parts
                                                for i in 0..fixed_parts.len() {
                                                    if fixed_parts[i].is_empty() && i != 1 { // We already fixed position 1
                                                        fixed_parts[i] = if i == 0 { "unknown" } else if i == 2 { "unknown" } else { "0" };
                                                    }
                                                }
                                                fixed_name = fixed_parts.join("#");
                                            }
                                            
                                            eprintln!("[FIX:complex] Line {}: '{}' → '{}'", line_number, path_name, fixed_name);
                                            local_fixes.insert(line_number, (path_name.to_string(), fixed_name));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    line_start = i + 1;
                    prev_newline = i;
                }
            }
            
            local_fixes
        })
        .collect();
    
    // Combine results from all chunks
    let mut fixes = HashMap::new();
    for chunk_fixes in fixes_per_chunk {
        fixes.extend(chunk_fixes);
    }
    
    // Get the progress bar back from the Arc<Mutex>
    let pb = Arc::try_unwrap(pb_arc).unwrap().into_inner().unwrap();
    pb.finish_and_clear();
    
    // If no fixes needed, just return the original path
    if fixes.is_empty() {
        eprintln!("[INFO] No invalid path names found, GFA is compliant");
        return Ok(gfa_path.to_string());
    }
    
    // Second-pass: create fixed GFA file with parallel read/write
    eprintln!("[INFO] Found {} invalid path names, creating fixed GFA...", fixes.len());
    let fixed_gfa_path = format!("{}.fixed.gfa", gfa_path);
    
    let input_file = File::open(gfa_path)
        .map_err(|e| format!("Failed to open input GFA: {}", e))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, input_file); // 8MB buffer
    
    let output_file = File::create(&fixed_gfa_path)
        .map_err(|e| format!("Failed to create output GFA: {}", e))?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, output_file); // 8MB buffer
    
    // Stream from input to output, making fixes as needed
    let pb = ProgressBar::new(file_size);
    pb.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {percent}% ({eta} remaining)"
    ).unwrap());
    
    let mut line_buffer = String::with_capacity(10 * 1024); // Pre-allocate 10KB for line buffer
    let mut line_number = 0;
    let mut bytes_read = 0;
    
    // Create a lookup of fixed lines
    while let Ok(bytes) = reader.read_line(&mut line_buffer) {
        if bytes == 0 { break; } // EOF
        
        bytes_read += bytes as u64;
        line_number += 1;
        
        // Update progress
        if line_number % 1000 == 0 {
            pb.set_position(bytes_read);
        }
        
        if let Some((ref original, ref fixed)) = fixes.get(&line_number) {
            // Replace path name in this line - only at tab boundaries to avoid incorrect replacements
            let tab_original = format!("\t{}\t", original);
            let tab_fixed = format!("\t{}\t", fixed);
            
            // Check if we can find the pattern with tabs (more precise)
            if line_buffer.contains(&tab_original) {
                let fixed_line = line_buffer.replace(&tab_original, &tab_fixed);
                writer.write_all(fixed_line.as_bytes())
                    .map_err(|e| format!("Failed to write to output file: {}", e))?;
            } else {
                // Fallback to standard replacement if tab pattern not found
                let fixed_line = line_buffer.replace(original, fixed);
                writer.write_all(fixed_line.as_bytes())
                    .map_err(|e| format!("Failed to write to output file: {}", e))?;
            }
        } else {
            // Copy line as-is
            writer.write_all(line_buffer.as_bytes())
                .map_err(|e| format!("Failed to write to output file: {}", e))?;
        }
        
        line_buffer.clear();
    }
    
    writer.flush().map_err(|e| format!("Failed to flush output: {}", e))?;
    pb.finish_and_clear();
    
    eprintln!("[SUCCESS] Fixed GFA written to: {}", fixed_gfa_path);
    Ok(fixed_gfa_path)
}

/// Checks if a path name needs fixing and returns the fixed name if needed.
/// Specifically addresses the "Invalid haplotype field" error by ensuring
/// the haplotype part of the name is numeric.
fn fix_path_name(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('#').collect();
    
    if parts.len() >= 3 {
        // Check if haplotype field (parts[1]) is numeric
        if parts[1].parse::<u32>().is_err() {
            // Replace non-numeric haplotype with "0"
            let mut fixed_parts = parts.clone();
            fixed_parts[1] = "0";
            return Some(fixed_parts.join("#"));
        }
    } else if parts.len() == 2 {
        // Missing haplotype field - insert "0"
        return Some(format!("{}#0#{}", parts[0], parts[1]));
    }
    
    // No fix needed
    None
}

/// Ensures a GBZ file exists for given GFA and PAF files.
/// If the GBZ doesn't exist, validates and fixes the GFA, then creates the GBZ.
pub fn make_gbz_exist(gfa_path: &str, paf_path: &str) -> String {
    // Derive GBZ filename from GFA and PAF paths
    let gfa_base = Path::new(gfa_path).file_stem().unwrap_or_default().to_string_lossy();
    let paf_base = Path::new(paf_path).file_stem().unwrap_or_default().to_string_lossy();
    let gbz_path = format!("{}.{}.gbz", gfa_base, paf_base);
    
    // Check if GBZ file already exists
    if !Path::new(&gbz_path).exists() {
        eprintln!("[INFO] Creating GBZ index from GFA and PAF...");

        // Validate and fix path names in the GFA file
        let fixed_gfa_path = match validate_gfa_for_gbwt(gfa_path) {
            Ok(path) => path,
            Err(msg) => {
                eprintln!("[ERROR] GFA validation and fixing failed:");
                eprintln!("{}", msg);
                panic!("GFA validation failed - cannot create GBZ index");
            }
        };
        
        // Locate vg command
        let vg_cmd = if let Ok(output) = Command::new("vg").arg("--version").output() {
            if output.status.success() {
                "vg".to_string()
            } else if Path::new("./vg").exists() {
                "./vg".to_string()
            } else if Path::new("../vg").exists() {
                "../vg".to_string()
            } else {
                eprintln!("Error: 'vg' command not found in PATH, current directory, or parent directory.");
                panic!("vg command not found");
            }
        } else if Path::new("./vg").exists() {
            "./vg".to_string()
        } else if Path::new("../vg").exists() {
            "../vg".to_string()
        } else {
            eprintln!("Error: 'vg' command not found in PATH, current directory, or parent directory.");
            panic!("vg command not found");
        };
        
        // Create GBZ using vg gbwt with the potentially fixed GFA
        let status = Command::new(&vg_cmd)
            .args(["gbwt", "-G", &fixed_gfa_path, "--gbz-format", "-g", &gbz_path])
            .status()
            .expect("Failed to run vg gbwt");
            
        if !status.success() {
            panic!("GBZ creation failed: vg gbwt command returned non-zero exit status");
        }
        
        eprintln!("[INFO] GBZ index created at {}", gbz_path);
    } else {
        eprintln!("[INFO] Using existing GBZ index: {}", gbz_path);
    }
    
    gbz_path
}


#[derive(Debug)]
pub struct Coord2NodeResult {
    pub path_name: String,
    pub node_id:   String,
    pub node_orient: bool,
    pub path_off_start: usize,
    pub path_off_end:   usize,
}

pub fn parse_region(r: &str) -> Option<(String,usize,usize)> {
    // e.g. "grch38#chr1:120616922-120626943"
    let (chr_part, rng_part) = r.split_once(':')?;
    let (s,e) = rng_part.split_once('-')?;
    let start = s.parse::<usize>().ok()?;
    let end   = e.parse::<usize>().ok()?;
    Some((chr_part.to_string(), start, end))
}

/// Creates a human-readable representation of consecutive node IDs as ranges
/// For example: [1, 2, 3, 5, 6, 8] becomes "1-3, 5-6, 8"
fn create_node_ranges(node_ids: &[usize]) -> String {
    if node_ids.is_empty() {
        return String::new();
    }
    
    let mut sorted_ids = node_ids.to_vec();
    sorted_ids.sort();
    sorted_ids.dedup();
    
    let mut ranges = Vec::new();
    let mut start = sorted_ids[0];
    let mut end = start;
    
    for i in 1..sorted_ids.len() {
        if sorted_ids[i] == end + 1 {
            // Continuing a range
            end = sorted_ids[i];
        } else {
            // End of range, start a new one
            if start == end {
                ranges.push(format!("{}", start));
            } else {
                ranges.push(format!("{}-{}", start, end));
            }
            start = sorted_ids[i];
            end = start;
        }
    }
    
    // Add the last range
    if start == end {
        ranges.push(format!("{}", start));
    } else {
        ranges.push(format!("{}-{}", start, end));
    }
    
    ranges.join(", ")
}

/// Creates a human-readable representation of consecutive coordinate ranges
fn create_coord_ranges(ranges: &[(String, usize, usize)]) -> Vec<(String, String)> {
    let mut result = Vec::new();
    
    // Group by chromosome
    let mut by_chr = std::collections::HashMap::<String, Vec<(usize, usize)>>::new();
    for (chr, start, end) in ranges {
        by_chr.entry(chr.clone()).or_default().push((*start, *end));
    }
    
    // Process each chromosome
    for (chr, mut positions) in by_chr {
        positions.sort_by_key(|p| p.0);
        positions.dedup();
        
        let merged = merge_intervals(positions);
        let formatted = merged.iter()
            .map(|(start, end)| {
                if start == end {
                    format!("{}", start)
                } else {
                    format!("{}-{}", start, end)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        
        result.push((chr, formatted));
    }
    
    result
}

/// merge_intervals takes a vector of (start, end) pairs and merges
/// all overlapping or contiguous intervals into a minimal set of
/// non‐overlapping intervals. Two intervals a..b and c..d are considered
/// part of the same “group” if they overlap or if c <= b+1.
fn merge_intervals(mut intervals: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    // Sort by the start coordinate
    intervals.sort_by_key(|iv| iv.0);
    let mut result = Vec::new();
    if intervals.is_empty() {
        return result;
    }
    // Start with the first interval
    let mut current_start = intervals[0].0;
    let mut current_end   = intervals[0].1;

    // Sweep through the rest
    for &(s,e) in intervals.iter().skip(1) {
        if s <= current_end + 1 {
            // Overlaps or touches the current group
            if e > current_end {
                current_end = e;
            }
        } else {
            // No overlap, finalize the previous group
            result.push((current_start, current_end));
            current_start = s;
            current_end   = e;
        }
    }
    // Push the final group
    result.push((current_start, current_end));
    result
}
