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
use std::io::{BufReader, BufRead};
use std::path::{Path, PathBuf};
use std::process::Command;

use memmap2::{MmapOptions};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use gbwt::{GBZ, Orientation};
use simple_sds::serialize;

// Data Structures


/// Stores basic info about a GFA node: specifically its length.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub length: usize,
}

/// For each path in the GFA, we store:
///   - the ordered list of (nodeID, orientation)
///   - prefix_sums so we can do offset lookups
#[derive(Debug)]
#[derive(Clone)]
pub struct PathData {
    pub nodes: Vec<(String, bool)>,   // bool => orientation, true=+ / false=-
    pub prefix_sums: Vec<usize>,
    pub total_length: usize,
}

/// For alignments from the PAF (untangle). Each record:
///   - pathName  (the "query" in PAF)
///   - qStart,qEnd
///   - refChrom  (the "target" in PAF, e.g. "grch38#chr1")
///   - rStart,rEnd
///   - strand
#[derive(Debug, Clone)]
pub struct AlignmentBlock {
    pub path_name: String,
    pub q_len:     usize,
    pub q_start:   usize,
    pub q_end:     usize,
    pub ref_chrom: String,
    pub r_start:   usize,
    pub r_end:     usize,
    // Indicate reference orientation: true = '+', false = '-'
    pub ref_strand: bool,
}

/// We'll store reference intervals in a stable interval tree, so we can do
/// coordinate lookups quickly for coord->node. Each node in that tree:
///   - the [start..end] range in reference
///   - the actual alignment block
#[derive(Debug, Clone)]
pub struct Interval {
    start: usize,
    end:   usize,
    data:  AlignmentBlock,
}

// A simple interval tree node: we store intervals in a segment, and a center
#[derive(Debug)]
pub enum IntervalTree {
    Empty,
    Node {
        center: usize,
        left: Box<IntervalTree>,
        right: Box<IntervalTree>,
        overlaps: Vec<Interval>,
    }
}

impl IntervalTree {
    fn new() -> Self {
        IntervalTree::Empty
    }

    // Build a tree from a vector of intervals
    fn build(mut intervals: Vec<Interval>) -> Self {
        if intervals.is_empty() {
            return IntervalTree::Empty;
        }
        // pick a center pivot. A common approach is pick median of starts
        intervals.sort_by_key(|iv| iv.start);
        let mid = intervals.len()/2;
        // Partition intervals into left, right, and overlaps
        let mut left_vec = Vec::new();
        let mut right_vec = Vec::new();
        let mut center_vec = Vec::new();

        let pivot = intervals[mid].start;
        for iv in &intervals {
            if iv.end < pivot {
                // Completely to the left of pivot
                left_vec.push(iv.clone());
            } else if iv.start > pivot {
                // Completely to the right of pivot
                right_vec.push(iv.clone());
            } else {
                // Overlaps the pivot point
                center_vec.push(iv.clone());
            }
        }
        let left = Box::new(IntervalTree::build(left_vec));
        let right = Box::new(IntervalTree::build(right_vec));
        IntervalTree::Node {
            center: pivot,
            left,
            right,
            overlaps: center_vec,
        }
    }

    // Query all intervals that overlap [qstart..qend]
    fn query<'a>(&'a self, qstart: usize, qend: usize, results: &mut Vec<&'a Interval>) {
        match self {
            IntervalTree::Empty => {},
            IntervalTree::Node{center, left, right, overlaps} => {
                println!("DEBUG: Visiting node with center={}, overlaps.len={}", center, overlaps.len());
                
                // check left if qstart <= center
                if qstart <= *center {
                    println!("DEBUG: Searching left subtree (qstart={} <= center={})", qstart, center);
                    left.query(qstart, qend, results);
                }
                // check right if qend >= center
                if qend >= *center {
                    println!("DEBUG: Searching right subtree (qend={} >= center={})", qend, center);
                    right.query(qstart, qend, results);
                }
                // check overlaps
                for iv in overlaps {
                    // println!("DEBUG: Checking overlap: ref={}:{}-{}, path={}", 
                    //         iv.data.ref_chrom, iv.start, iv.end, iv.data.path_name);
                    if iv.end < qstart || iv.start > qend {
                        // println!("DEBUG: No overlap with query region");
                    } else {
                        // println!("DEBUG: Found overlap! Adding to results");
                        results.push(iv);
                    }
                }
            }
        }
    }
}


// We'll keep these big data structures in memory for queries.


pub struct GlobalData {
    // node information
    pub node_map: HashMap<String, NodeInfo>,

    // path information
    pub path_map: HashMap<String, PathData>,

    // node->(pathName,indexInPath)
    pub node_to_paths: HashMap<String, Vec<(String,usize)>>,

    // alignment blocks per path
    pub alignment_by_path: HashMap<String, Vec<AlignmentBlock>>,

    // for reference coords
    // e.g. "grch38#chr1" -> IntervalTree
    pub ref_trees: HashMap<String, IntervalTree>,
}

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

fn parse_cigar_overlap(cigar: &str) -> usize {
    // Example: "5M3D10M" => we count only M => 5 + 10 = 15
    // Treat '=' and 'X' as base-consuming on both sides?
    // We do not count 'I' or 'D' as overlap for the path?
    let mut total = 0usize;
    let mut num_buf = String::new();
    for ch in cigar.chars() {
        if ch.is_ascii_digit() {
            num_buf.push(ch);
        } else {
            let n = num_buf.parse::<usize>().unwrap_or(0);
            num_buf.clear();
            match ch {
                'M' | '=' | 'X' => {
                    total += n;
                }
                // 'I','D','N','S','H','P','B' => don't add to overlap
                _ => {}
            }
        }
    }
    total
}


// parse_gfa_memmap
//
// - memory-map the GFA
// - find lines by scanning for newlines
// - parse S lines to fill node_map
// - parse P lines to fill path_map, prefix sums, node_to_paths

pub fn parse_gfa_memmap(gfa_path: &str, global: &mut GlobalData) {
    // This new version writes minimal data to disk, then discards it from memory.

    #[derive(Debug)]
    struct TempPathData {
        nodes: Vec<(String, bool)>,
        overlaps: Vec<String>,
    }

    // We collect node IDs, path names, and skip storing large merges in memory.
    // Instead, we store minimal node info into global.node_map, and do not fill path_map or node_to_paths.

    let file = File::open(gfa_path).expect("Cannot open GFA file for streaming");
    let reader = BufReader::new(file);
    let pb = ProgressBar::new_spinner();
    pb.set_message("Parsing GFA for minimal disk-based approach");
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7} lines ({eta}) minimal GFA parse")
            .expect("Invalid progress style template")
            .progress_chars("##-")
    );

    use std::io::Write;
    let mut gfa_index = File::create("gfa_disk_index.txt").expect("Cannot create disk index file");
    let mut line_count = 0u64;

    for line_res in reader.lines() {
        if let Ok(line) = line_res {
            pb.inc(1);
            line_count += 1;
            if line.is_empty() {
                continue;
            }
            let first_byte = line.as_bytes()[0];
            match first_byte {
                b'S' => {
                    let parts: Vec<&str> = line.split('\t').collect();
                    if parts.len() < 3 {
                        continue;
                    }
                    let seg_name = parts[1].to_string();
                    let seq_or_star = parts[2];
                    let mut length = 0;
                    if seq_or_star == "*" {
                        for p in parts.iter().skip(3) {
                            if p.starts_with("LN:i:") {
                                if let Ok(val) = p[5..].parse::<usize>() {
                                    length = val;
                                    break;
                                }
                            }
                        }
                    } else {
                        length = seq_or_star.len();
                    }
                    global.node_map.entry(seg_name.clone()).or_insert(NodeInfo{ length });
                    writeln!(gfa_index, "S\t{}\t{}", seg_name, length).expect("write error");
                },
                b'P' => {
                    writeln!(gfa_index, "{}", line).expect("write error");
                },
                _ => {}
            }
        }
    }

    pb.finish_and_clear();
    eprintln!("Minimal GFA parse complete: wrote {} lines to gfa_disk_index.txt", line_count);

    // Clear the large in-memory data structures
    global.path_map.clear();
    global.node_to_paths.clear();
}

// parse_paf_parallel
pub fn parse_paf_parallel(paf_path: &str, global: &mut GlobalData) {
    // Writes minimal PAF data to disk, then clears from memory.
    let file = File::open(paf_path).expect("Cannot open PAF");
    let reader = BufReader::new(file);
    let pb = ProgressBar::new_spinner();
    pb.set_message("Minimal PAF parse to disk");
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.magenta/black} {pos:>7} lines ({eta}) minimal PAF parse")
            .expect("Invalid template for progress style")
            .progress_chars("##-")
    );

    use std::io::Write;
    let mut paf_index = File::create("paf_disk_index.txt").expect("Cannot create paf_disk_index.txt");
    let mut count = 0u64;
    for line_res in reader.lines() {
        if let Ok(line) = line_res {
            pb.inc(1);
            if !line.is_empty() && !line.starts_with('#') {
                writeln!(paf_index, "{}", line).expect("write error");
                count+=1;
            }
        }
    }
    pb.finish_and_clear();
    eprintln!("Minimal PAF parse complete: wrote {} lines to paf_disk_index.txt", count);

    // We do not fill alignment_by_path with the entire dataset. We keep it empty.
    global.alignment_by_path.clear();
}

// build_ref_trees
// We'll gather all alignment blocks from alignment_by_path, group by refChrom
// then build an Interval. Then build an IntervalTree for each refChrom.
pub fn build_ref_trees(global: &mut GlobalData) {
    // Print some statistics about the path names and references
    let mut path_samples = std::collections::HashSet::new();
    for path_name in global.alignment_by_path.keys() {
        // Extract sample information from path names (format is typically SAMPLE#HAPLOTYPE#CONTIG)
        if let Some(index) = path_name.find('#') {
            path_samples.insert(&path_name[0..index]);
        } else {
            path_samples.insert(path_name.as_str());
        }
    }
    println!("DEBUG: Found {} unique samples in PAF paths", path_samples.len());
    
    // Check reference chromosome formats
    let mut ref_chroms = std::collections::HashSet::new();
    for (_, blocks) in &global.alignment_by_path {
        for block in blocks {
            ref_chroms.insert(block.ref_chrom.clone());
        }
    }
    println!("DEBUG: Reference chromosomes in PAF: {:?}", ref_chroms);
    
    // Check for path availability in both datasets
    println!("DEBUG: Checking path availability between PAF and GFA");
    let mut paf_path_count = 0;
    let mut paths_in_both = 0;
    
    for path_name in global.alignment_by_path.keys() {
        paf_path_count += 1;
        if global.path_map.contains_key(path_name) {
            paths_in_both += 1;
        }
    }
    println!("DEBUG: PAF contains {} paths, {} of them found in GFA ({}%)", 
             paf_path_count, paths_in_both, 
             if paf_path_count > 0 { paths_in_both * 100 / paf_path_count } else { 0 });
    
    // gather all intervals
    use rayon::prelude::*;
    let mut by_ref = HashMap::<String, Vec<Interval>>::new();

    let pb = ProgressBar::new(global.alignment_by_path.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.yellow/black} {pos:>7}/{len:7} ({eta}) Building intervals").expect("Invalid template for progress style").progress_chars("##-"));

    let mut local_count = 0;
    for (_, blocks) in &global.alignment_by_path {
        local_count += 1;
        if local_count % 10000 == 0 {
            pb.inc(10000);
        }
        for b in blocks {
            let interval = Interval {
                start: b.r_start,
                end:   b.r_end,
                data:  b.clone()
            };
            by_ref.entry(b.ref_chrom.clone()).or_insert_with(Vec::new).push(interval);
        }
    }
    pb.inc(local_count % 10000);
    pb.finish_and_clear();

    // now build trees in parallel
    let keys:Vec<_> = by_ref.keys().cloned().collect();
    let pb2 = ProgressBar::new(keys.len() as u64);
    pb2.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/black} {pos:>7}/{len:7} ({eta}) Building IntervalTrees")
            .expect("Invalid progress style template")
            .progress_chars("##-")
    );
    
    let built: HashMap<String, IntervalTree> = keys.par_chunks(10000).flat_map(|chunk| {
        pb2.inc(chunk.len() as u64);
        chunk.iter().map(|k| {
            let intervals = by_ref.get(k).unwrap().clone();
            let tree = IntervalTree::build(intervals);
            (k.clone(), tree)
        }).collect::<Vec<(String, IntervalTree)>>()
    }).collect();
    pb2.finish_and_clear();

    global.ref_trees = built;
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
