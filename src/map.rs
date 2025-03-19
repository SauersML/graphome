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



/// Validates a GFA file for potential node conflicts that would cause GBWT construction to fail.
/// This function reads the GFA in a streaming manner (using memory mapping).
/// It extracts node IDs from lines that begin with 'P', writing each (node_id,line_number) to a temporary file.
/// Then it externally sorts that file by node_id to detect duplicates.
/// It prints detailed statistics on node frequencies, duplicates, and line numbers where duplicates appear.
/// If no duplicates are found, the function returns `Ok(())`. Otherwise, returns `Err(...)`.
pub fn validate_gfa_for_gbwt(gfa_path: &str) -> Result<(), String> {
    eprintln!("[INFO] Validating GFA file in a streaming manner: {}", gfa_path);

    // Open and memory-map the GFA for efficient large-file reading.
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

    // Create a progress bar to show how much of the file we've processed.
    let pb = ProgressBar::new(file_size);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}% ({eta} remaining)")
            .unwrap()
    );

    // Create a temporary file for (node_id, line_number).
    // Store its path so we can reference it after writing is complete.
    let tmp_file = NamedTempFile::new()
        .map_err(|e| format!("Could not create temp file for node IDs: {}", e))?;
    let tmp_file_path = tmp_file.path().to_path_buf();

    {
        // Use a scoped block so that `writer` is dropped before we call external sort.
        let file_mut = tmp_file
            .as_file()
            .try_clone()
            .map_err(|e| format!("Could not clone temp file: {}", e))?;
        let mut writer = BufWriter::new(file_mut);

        // We scan the memory map for newlines, building lines. If a line starts with 'P', parse it.
        let mut line_start: usize = 0;
        let mut line_number: usize = 0;
        let update_chunk = 100_000_000usize; // update progress every 100MB
        let mut next_progress_update = 0usize;

        for i in 0..mmap.len() {
            // Periodically update the progress bar
            if i >= next_progress_update {
                pb.set_position(i as u64);
                next_progress_update += update_chunk;
            }

            // Check for newline
            if mmap[i] == b'\n' {
                line_number += 1;
                if i > line_start {
                    // If this line begins with 'P', process it
                    if mmap[line_start] == b'P' {
                        let line_bytes = &mmap[line_start..i];
                        // Safe if GFA is valid UTF-8; if uncertain, handle potential errors or do a lossy conversion
                        let line_str = unsafe { std::str::from_utf8_unchecked(line_bytes) };

                        let parts: Vec<&str> = line_str.split('\t').collect();
                        if parts.len() >= 3 {
                            // The third column typically has "segmentIDs" separated by commas,
                            // each possibly followed by + or -.
                            let path_nodes = parts[2]
                                .split(',')
                                .map(|s| s.trim_end_matches('+').trim_end_matches('-'));
                            for node_id in path_nodes {
                                if !node_id.is_empty() {
                                    // Write "nodeID\tline_number\n"
                                    if let Err(e) = writeln!(writer, "{}\t{}", node_id, line_number) {
                                        return Err(format!("Error writing node list: {}", e));
                                    }
                                }
                            }
                        }
                    }
                }
                line_start = i + 1;
            }
        }

        // Handle final line if no trailing newline
        if line_start < mmap.len() {
            line_number += 1;
            if mmap[line_start] == b'P' {
                let line_bytes = &mmap[line_start..];
                let line_str = unsafe { std::str::from_utf8_unchecked(line_bytes) };

                let parts: Vec<&str> = line_str.split('\t').collect();
                if parts.len() >= 3 {
                    let path_nodes = parts[2]
                        .split(',')
                        .map(|s| s.trim_end_matches('+').trim_end_matches('-'));
                    for node_id in path_nodes {
                        if !node_id.is_empty() {
                            if let Err(e) = writeln!(writer, "{}\t{}", node_id, line_number) {
                                return Err(format!("Error writing node list: {}", e));
                            }
                        }
                    }
                }
            }
        }

        writer.flush()
            .map_err(|e| format!("Error flushing temp file: {}", e))?;
    }

    // Finish progress bar for reading
    pb.finish_and_clear();
    eprintln!("[INFO] Finished scanning GFA ({} bytes). Now sorting node IDs...", file_size);

    // Create another temp file for sorted output
    let sorted_tmp_file = NamedTempFile::new()
        .map_err(|e| format!("Could not create temp file for sorted node IDs: {}", e))?;
    let sorted_tmp_file_path = sorted_tmp_file.path().to_path_buf();

    // Sort the (node_id, line_number) file by node_id (the first column).
    let sort_status = Command::new("sort")
        .args([
            "-T", "/tmp",
            "-k1,1", // sort by the first field (node_id)
            tmp_file_path.to_str().unwrap(),
            "-o", sorted_tmp_file_path.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("Failed to run external `sort`: {}", e))?;

    if !sort_status.success() {
        return Err("External sort command failed with non-zero exit status".to_string());
    }

    // Read sorted file to detect duplicates and gather stats. 
    // Each line is "node_id\tline_number".
    let sorted_file = File::open(&sorted_tmp_file_path)
        .map_err(|e| format!("Failed to open sorted node file: {}", e))?;
    let mut sorted_reader = BufReader::new(sorted_file);

    let mut prev_node_id = String::new();
    let mut current_line_numbers: Vec<usize> = Vec::new();
    let mut found_conflict = false;
    let mut conflict_msg = String::new();

    let mut total_entries: usize = 0;     
    let mut total_unique_nodes: usize = 0; 
    let mut freq_map: HashMap<usize, usize> = HashMap::new();
    let mut freq_min = usize::MAX;
    let mut freq_max = 0usize;

    // Inlined closure to process a batch of line numbers for a node.
    // This checks if there's a duplicate and aggregates frequency stats.
    let mut handle_node = |node_id: &str, lines: &Vec<usize>| {
        if node_id.is_empty() || lines.is_empty() {
            return;
        }
        let freq = lines.len();
        total_unique_nodes += 1;
        *freq_map.entry(freq).or_insert(0) += 1;
        if freq < freq_min {
            freq_min = freq;
        }
        if freq > freq_max {
            freq_max = freq;
        }
        if freq > 1 {
            found_conflict = true;
            conflict_msg.push_str(&format!(
                "Duplicate: Node ID '{}' appears {} times.\n",
                node_id, freq
            ));
            let max_show = 100; 
            let truncated = freq > max_show;
            let shown_count = freq.min(max_show);
            conflict_msg.push_str("   Lines: ");
            for (i, &ln) in lines.iter().enumerate().take(shown_count) {
                conflict_msg.push_str(&ln.to_string());
                if i < shown_count - 1 {
                    conflict_msg.push_str(", ");
                }
            }
            if truncated {
                conflict_msg.push_str(" ... [TRUNCATED]");
            }
            conflict_msg.push('\n');
        }
    };

    let mut buffer = String::new();
    while sorted_reader.read_line(&mut buffer)
        .map_err(|e| format!("Failed to read from sorted node file: {}", e))? > 0
    {
        let line = buffer.trim_end();
        if line.is_empty() {
            buffer.clear();
            continue;
        }
        total_entries += 1;

        let mut tab_split = line.splitn(2, '\t');
        let node_id = match tab_split.next() {
            Some(x) => x,
            None => {
                buffer.clear();
                continue;
            }
        };
        let line_num_str = match tab_split.next() {
            Some(x) => x,
            None => {
                buffer.clear();
                continue;
            }
        };

        let line_num = match line_num_str.parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                buffer.clear();
                continue;
            }
        };

        if node_id == prev_node_id {
            current_line_numbers.push(line_num);
        } else {
            // finalize old node
            handle_node(&prev_node_id, &current_line_numbers);
            prev_node_id.clear();
            prev_node_id.push_str(node_id);
            current_line_numbers.clear();
            current_line_numbers.push(line_num);
        }

        buffer.clear();
    }
    // finalize last node
    handle_node(&prev_node_id, &current_line_numbers);

    if total_entries == 0 {
        eprintln!("[INFO] No node IDs found in 'P' lines. This may be an empty GFA or no 'P' lines exist.");
        // Considered successful since no duplicates found if there's nothing to compare.
        return Ok(());
    }

    let avg_freq = if total_unique_nodes > 0 {
        total_entries as f64 / total_unique_nodes as f64
    } else {
        0.0
    };

    let mut freq_pairs: Vec<(usize, usize)> = freq_map.into_iter().collect();
    freq_pairs.sort_by_key(|&(freq, _)| freq);

    let mut distribution_str = String::new();
    distribution_str.push_str("\nFrequency distribution (freq -> count_of_nodes_with_that_freq):\n");
    for (f, c) in &freq_pairs {
        distribution_str.push_str(&format!("  {:>8} -> {}\n", f, c));
    }

    let stats_msg = format!(
        "\nValidation statistics:\n\
         - Total (node_id,line) entries: {}\n\
         - Distinct node IDs: {}\n\
         - Min frequency: {}\n\
         - Max frequency: {}\n\
         - Average frequency: {:.2}\n\
         {}",
        total_entries,
        total_unique_nodes,
        if freq_min == usize::MAX { 0 } else { freq_min },
        freq_max,
        avg_freq,
        distribution_str
    );

    if found_conflict {
        let mut err_str = String::new();
        err_str.push_str("[ERROR] Duplicate node IDs detected.\n");
        err_str.push_str(&conflict_msg);
        err_str.push_str(&stats_msg);
        eprintln!("{}", err_str);
        Err(err_str)
    } else {
        eprintln!("[INFO] No duplicate node IDs found in path lines.");
        eprintln!("{}", stats_msg);
        Ok(())
    }
}


/// So that that a GBZ file exists for the given GFA and PAF files
/// If the GBZ doesn't exist, it creates it using vg
pub fn make_gbz_exist(gfa_path: &str, paf_path: &str) -> String {
    // Derive GBZ filename from GFA and PAF paths
    let gfa_base = Path::new(gfa_path).file_stem().unwrap_or_default().to_string_lossy();
    let paf_base = Path::new(paf_path).file_stem().unwrap_or_default().to_string_lossy();
    let gbz_path = format!("{}.{}.gbz", gfa_base, paf_base);
    
    // Check if GBZ file already exists
    if !Path::new(&gbz_path).exists() {
        eprintln!("[INFO] Creating GBZ index from GFA and PAF...");

        // Validate the GFA file before attempting to build the GBZ
        match validate_gfa_for_gbwt(gfa_path) {
            Ok(_) => {
                eprintln!("[INFO] GFA validation passed, proceeding with GBWT construction");
            },
            Err(msg) => {
                eprintln!("[ERROR] GFA validation failed:");
                eprintln!("{}", msg);
                eprintln!("[INFO] The GFA file contains node conflicts that would cause GBWT construction to fail");
                panic!("GFA validation failed - cannot create GBZ index");
            }
        }
        
        // Determine vg command location by checking PATH, current directory, and parent directory.
        let vg_cmd = if let Ok(output) = Command::new("vg").arg("--version").output() {
            if output.status.success() {
                "vg".to_string()
            } else if Path::new("./vg").exists() {
                "./vg".to_string()
            } else if Path::new("../vg").exists() {
                "../vg".to_string()
            } else {
                eprintln!("Error: 'vg' command not found in PATH, current directory, or parent directory. Please install vg toolkit.");
                panic!("vg command not found");
            }
        } else if Path::new("./vg").exists() {
            "./vg".to_string()
        } else if Path::new("../vg").exists() {
            "../vg".to_string()
        } else {
            eprintln!("Error: 'vg' command not found in PATH, current directory, or parent directory. Please install vg toolkit.");
            panic!("vg command not found");
        };
        
        // Create GBZ using vg gbwt
        let status = Command::new(&vg_cmd)
            .args(["gbwt", "-G", gfa_path, "--gbz-format", "-g", &gbz_path])
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
