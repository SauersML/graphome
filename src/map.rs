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

use std::fs::{File};
use std::io::{BufWriter, Write};
use std::path::{Path};
use std::process::Command;

use memmap2::{MmapOptions};
use indicatif::{ProgressBar, ProgressStyle};
use gbwt::{GBZ, Orientation};
use simple_sds::serialize;

// Data Structures

// Stats tracking for different fix types
#[derive(Default)]
struct FixStats {
    reference_paths: usize,   // Reference paths with missing haplotype (e.g., chm13#chr1)
    haplotype_fixes: usize,   // Standard haplotype fixes for non-numeric haplotype
    missing_haplotype: usize, // Missing haplotype field (only one # symbol)
    complex_paths: usize,     // Complex cases with 3+ hash symbols (e.g., MT paths)
    total_processed: usize,   // Total path lines processed
}

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
    let metadata = match gbz.metadata() {
        Some(meta) => meta,
        None => {
            eprintln!("Warning: No metadata found in GBZ");
            return results;
        }
    };

    let mut ref_samples = gbz.reference_sample_ids(true);
    if ref_samples.is_empty() {
        eprintln!("Warning: No reference samples found in GBZ; falling back to all samples");
        let mut unique_samples = std::collections::HashSet::new();
        for path_name in metadata.path_iter() {
            unique_samples.insert(path_name.sample());
        }
        if unique_samples.is_empty() {
            return results;
        }
        ref_samples = unique_samples.into_iter().collect();
    }

    let mut ref_paths_cache = gbz.reference_positions(1000, true);
    if ref_paths_cache.is_empty() {
        ref_paths_cache = gbz.reference_positions(1000, false);
    }
    let have_reference_positions = !ref_paths_cache.is_empty();

    // Create a search state for the node (ensures node exists in the index)
    let _forward_state = match gbz.search_state(node_id, Orientation::Forward) {
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
            if let Some(path_iter) = gbz.path(path_id, Orientation::Forward) {
                let mut position = 0;
                let mut found_positions = Vec::new();

                // Scan path to find node positions
                for (path_node, orientation) in path_iter {
                    if path_node == node_id {
                        found_positions.push((position, orientation == Orientation::Forward));
                    }
                    if let Some(len) = gbz.sequence_len(path_node) {
                        position += len;
                    }
                }

                // For each occurrence, extract reference coordinates
                for (pos, _is_forward) in found_positions {
                    if have_reference_positions {
                        for ref_path in &ref_paths_cache {
                            if ref_path.id == path_id {
                                for (path_pos, _gbwt_pos) in &ref_path.positions {
                                    if *path_pos <= pos && *path_pos + node_len >= pos {
                                        let offset = pos - path_pos;
                                        let ref_start = *path_pos + offset;
                                        let ref_end = ref_start + node_len - 1;
                                        results.push((contig_name.clone(), ref_start, ref_end));
                                    }
                                }
                            }
                        }
                    } else {
                        let ref_start = pos;
                        let ref_end = ref_start + node_len - 1;
                        results.push((contig_name.clone(), ref_start, ref_end));
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
    let mut ref_paths = gbz.reference_positions(1000, true);
    if ref_paths.is_empty() {
        println!("DEBUG: No reference paths found in GBZ; falling back to all paths");
        ref_paths = gbz.reference_positions(1000, false);
    }

    let metadata = match gbz.metadata() {
        Some(meta) => meta,
        None => {
            println!("DEBUG: No metadata available in GBZ");
            return results;
        }
    };

    if ref_paths.is_empty() {
        println!("DEBUG: Still no paths available in GBZ; performing manual scan");
        for (path_id, path_name) in metadata.path_iter().enumerate() {
            let contig_name = metadata.contig_name(path_name.contig());
            if contig_name != chr {
                continue;
            }

            if let Some(path_iter) = gbz.path(path_id, Orientation::Forward) {
                let mut position = 0;
                for (node_id, orientation) in path_iter {
                    if let Some(node_len) = gbz.sequence_len(node_id) {
                        let node_start = position;
                        let node_end = node_start + node_len;
                        if node_start <= end && node_end > start {
                            let overlap_start = start.max(node_start);
                            let overlap_end = end.min(node_end - 1);
                            if overlap_start <= overlap_end {
                                let node_off_start = overlap_start - node_start;
                                let node_off_end = overlap_end - node_start;
                                results.push(Coord2NodeResult {
                                    path_name: contig_name.clone(),
                                    node_id: node_id.to_string(),
                                    node_orient: orientation == Orientation::Forward,
                                    path_off_start: node_off_start,
                                    path_off_end: node_off_end,
                                });
                            }
                        }
                        position += node_len;
                    }
                }
            }
        }
        return results;
    }

    // Find paths that align to this region using reference position index
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

    results
}

/// Validates and fixes path names in a GFA file for PanSN naming compliance.
/// This function fixes "Invalid haplotype field" errors in a single pass
/// using chunk-based processing.
/// Returns the path to use (original if no fixes needed, new fixed file otherwise).
pub fn validate_gfa_for_gbwt(gfa_path: &str) -> Result<String, String> {
    eprintln!("[INFO] Fast validating and fixing GFA path names: {}", gfa_path);

    // Open and memory-map input file
    let file = File::open(gfa_path)
        .map_err(|e| format!("Failed to open GFA file: {}", e))?;
    let metadata = file.metadata()
        .map_err(|e| format!("Failed to get metadata: {}", e))?;
    let file_size = metadata.len();
    
    // Create output file path
    let fixed_gfa_path = format!("{}.fixed.gfa", gfa_path);
    
    // Create memory map for fast reading
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| format!("Could not memory-map file: {}", e))?
    };
    
    let mut stats = FixStats::default();
    
    // Create output file with large buffer
    let output_file = File::create(&fixed_gfa_path)
        .map_err(|e| format!("Failed to create output file: {}", e))?;
    let mut writer = BufWriter::with_capacity(128 * 1024 * 1024, output_file); // 128MB buffer
    
    // Setup progress bar with accurate estimation
    let pb = ProgressBar::new(file_size);
    pb.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {percent}% ({eta_precise} remaining)"
    ).unwrap());
    pb.set_position(0);
    
    // Initialize variables for chunk processing
    let chunk_size = 64 * 1024 * 1024; // 64MB chunks
    let mut position = 0;
    let mut fix_count = 0;
    
    // Process file in chunks
    while position < mmap.len() {
        // Calculate chunk boundaries with proper line handling
        let end_pos = std::cmp::min(position + chunk_size, mmap.len());
        // Find next complete line boundary
        let true_end = if end_pos < mmap.len() {
            match memchr::memchr(b'\n', &mmap[end_pos..std::cmp::min(end_pos + 1024, mmap.len())]) {
                Some(pos) => end_pos + pos + 1,
                None => end_pos,
            }
        } else {
            end_pos
        };
        
        // Create buffer for output chunk with extra space
        let mut output_buffer = Vec::with_capacity(true_end - position + 1024 * 1024);
        
        // Process this chunk line by line
        let mut line_start = position;
        while line_start < true_end {
            // Find end of current line
            let line_end = match memchr::memchr(b'\n', &mmap[line_start..true_end]) {
                Some(pos) => line_start + pos,
                None => true_end
            };
            
            // Check if this is a path line (P or W)
            if line_start < line_end && (mmap[line_start] == b'P' || mmap[line_start] == b'W') {
                // Process path line
                stats.total_processed += 1;
                
                // Find tabs to locate path name
                if let Some(first_tab) = memchr::memchr(b'\t', &mmap[line_start..line_end]) {
                    let path_start = line_start + first_tab + 1;
                    if let Some(second_tab) = memchr::memchr(b'\t', &mmap[path_start..line_end]) {
                        let path_end = path_start + second_tab;
                        
                        // Fast check if this path has a # character
                        if memchr::memchr(b'#', &mmap[path_start..path_end]).is_some() {
                            // Extract the path name for processing
                            let path_name = unsafe { std::str::from_utf8_unchecked(&mmap[path_start..path_end]) };
                            
                            // Check if fix is needed
                            let maybe_fixed_name = fix_path_name(path_name, &mut stats);
                            
                            if let Some(fixed_name) = maybe_fixed_name {
                                // Path needed fixing
                                fix_count += 1;
                                
                                // Write fixed line: first part + fixed name + rest of line
                                output_buffer.extend_from_slice(&mmap[line_start..path_start]);
                                output_buffer.extend_from_slice(fixed_name.as_bytes());
                                output_buffer.extend_from_slice(&mmap[path_end..line_end]);
                                
                                // Log occasional progress
                                if fix_count % 100_000 == 0 {
                                    eprintln!("[INFO] Fixed {} paths so far ({} ref, {} hap, {} miss, {} complex)",
                                             fix_count, stats.reference_paths, stats.haplotype_fixes,
                                             stats.missing_haplotype, stats.complex_paths);
                                }
                            } else {
                                // No fix needed, copy line as-is
                                output_buffer.extend_from_slice(&mmap[line_start..line_end]);
                            }
                        } else {
                            // No # in path name, copy line as-is
                            output_buffer.extend_from_slice(&mmap[line_start..line_end]);
                        }
                    } else {
                        // No second tab, copy line as-is
                        output_buffer.extend_from_slice(&mmap[line_start..line_end]);
                    }
                } else {
                    // No first tab, copy line as-is
                    output_buffer.extend_from_slice(&mmap[line_start..line_end]);
                }
            } else {
                // Not a path line, copy as-is
                output_buffer.extend_from_slice(&mmap[line_start..line_end]);
            }
            
            // Add newline if not at end of file
            if line_end < mmap.len() {
                output_buffer.push(b'\n');
            }
            
            // Move to next line
            line_start = line_end + 1;
        }
        
        // Write entire processed chunk at once
        writer.write_all(&output_buffer)
            .map_err(|e| format!("Failed to write chunk to output file: {}", e))?;
        
        // Update progress
        position = true_end;
        pb.set_position(position as u64);
    }
    
    // Finish writing and flush buffers
    writer.flush().map_err(|e| format!("Failed to flush output: {}", e))?;
    pb.finish_and_clear();
    
    // Show summary
    if fix_count > 0 {
        eprintln!("[SUCCESS] Fixed {} path names:", fix_count);
        eprintln!("  - Reference paths (e.g., chm13#chr1): {}", stats.reference_paths);
        eprintln!("  - Non-numeric haplotype fields: {}", stats.haplotype_fixes);
        eprintln!("  - Missing haplotype fields: {}", stats.missing_haplotype);
        eprintln!("  - Complex paths (e.g., MT paths): {}", stats.complex_paths);
        eprintln!("[SUCCESS] Fixed GFA written to: {}", fixed_gfa_path);
        Ok(fixed_gfa_path)
    } else {
        eprintln!("[INFO] No invalid path names found, using original file");
        Ok(gfa_path.to_string())
    }
}

/// Fast path name fix function that efficiently handles all cases without unnecessary allocations.
/// Returns Some(fixed_name) if a fix is needed, None otherwise.
fn fix_path_name(name: &str, stats: &mut FixStats) -> Option<String> {
    // Special fast-path for reference genomes (extremely common)
    if name.starts_with("chm13#chr") || name.starts_with("grch38#chr") {
        let hash_pos = name.find('#').unwrap();
        if !name[hash_pos+1..].contains('#') {
            // Reference path with one # - fix format to sample#0#contig
            stats.reference_paths += 1;
            return Some(format!("{}#0#{}", &name[0..hash_pos], &name[hash_pos+1..]));
        }
    }
    
    // Count # symbols for categorization
    let hash_count = memchr::memchr_iter(b'#', name.as_bytes()).count();
    
    match hash_count {
        1 => {
            // Missing haplotype field (sample#contig) - add #0#
            let parts: Vec<&str> = name.split('#').collect();
            let sample = if parts[0].is_empty() { "unknown" } else { parts[0] };
            let contig = if parts.len() > 1 && !parts[1].is_empty() { parts[1] } else { "unknown" };
            stats.missing_haplotype += 1;
            Some(format!("{}#0#{}", sample, contig))
        },
        2 => {
            // Standard PanSN (sample#haplotype#contig) - check if haplotype is numeric
            let parts: Vec<&str> = name.split('#').collect();
            if parts[1].parse::<u32>().is_err() || parts[1].is_empty() {
                // Non-numeric haplotype - replace with "0"
                let sample = if parts[0].is_empty() { "unknown" } else { parts[0] };
                let contig = if parts[2].is_empty() { "unknown" } else { parts[2] };
                stats.haplotype_fixes += 1;
                Some(format!("{}#0#{}", sample, contig))
            } else {
                // Already valid - no fix needed
                None
            }
        },
        _ if hash_count >= 3 => {
            // Complex case like "HG00438#2#JAHBCA010000258.1#MT"
            let parts: Vec<&str> = name.split('#').collect();
            
            let sample = if parts[0].is_empty() { "unknown" } else { parts[0] };
            let haplotype = parts[1];
            
            stats.complex_paths += 1;
            
            // Check if haplotype part is numeric
            if haplotype.parse::<u32>().is_ok() && !haplotype.is_empty() {
                // Fix the MT case - combine all extra parts with underscores
                let contig_parts = &parts[2..];
                let joined_contig = contig_parts.join("_");
                Some(format!("{}#{}#{}", sample, haplotype, joined_contig))
            } else {
                // Non-numeric haplotype field with multiple # symbols
                let contig_parts = &parts[2..];
                let joined_contig = contig_parts.join("_");
                Some(format!("{}#0#{}", sample, joined_contig))
            }
        },
        _ => None // No # symbols - no fix needed
    }
}

/// Ensures a GBZ file exists for given GFA and PAF files.
/// If the GBZ doesn't exist, validates and fixes the GFA, then creates the GBZ.
pub fn make_gbz_exist(gfa_path: &str, paf_path: &str) -> String {
    if GBZ::is_gbz(gfa_path) {
        eprintln!("[INFO] Using provided GBZ index: {}", gfa_path);
        return gfa_path.to_string();
    }

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
    
    for &id in &sorted_ids[1..] {
        if id == end + 1 {
            // Continuing a range
            end = id;
        } else {
            // End of range, start a new one
            if start == end {
                ranges.push(format!("{}", start));
            } else {
                ranges.push(format!("{}-{}", start, end));
            }
            start = id;
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
