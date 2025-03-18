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
//
// Key features / crates:
//   - clap (for CLI)
//   - memmap2 (for memory mapping the GFA file)
//   - rayon (for parallel data processing)
//   - indicatif (for progress bars)
//   - IntervalTree (we show a custom robust approach to store alignment blocks)
//   - Efficient data structures and concurrency considerations


// The code always treats query offsets as if they run from `q_start` to `q_end` in a forward orientation, even when the PAF record indicates a reverse‐strand alignment. This causes incorrect offset calculations for negative‐strand alignments. To fix it, must handle the case where `strand_char == '-'` (i.e., `b.ref_strand == false`) by inverting the offset calculations to account for the query’s reversed orientation.

use std::collections::{HashMap};
use std::fs::{File};
use std::io::{BufReader, BufRead};

use memmap2::{MmapOptions};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};


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
                    println!("DEBUG: Checking overlap: ref={}:{}-{}, path={}", 
                             iv.data.ref_chrom, iv.start, iv.end, iv.data.path_name);
                    if iv.end < qstart || iv.start > qend {
                        println!("DEBUG: No overlap with query region");
                    } else {
                        println!("DEBUG: Found overlap! Adding to results");
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
    eprintln!("[INFO] Building data structures from GFA='{}' PAF='{}'", gfa_path, paf_path);

    let mut global = GlobalData {
        node_map: HashMap::new(),
        path_map: HashMap::new(),
        node_to_paths: HashMap::new(),
        alignment_by_path: HashMap::new(),
        ref_trees: HashMap::new(),
    };

    // Step 1: parse GFA with memory mapping
    parse_gfa_memmap(gfa_path, &mut global);
    eprintln!("[INFO] GFA parse done. node_map={} path_map={}",
        global.node_map.len(), global.path_map.len());

    // Step 2: parse PAF in parallel
    parse_paf_parallel(paf_path, &mut global);
    eprintln!("[INFO] PAF parse done. alignment_by_path={} refTrees building...",
        global.alignment_by_path.len());

    // Step 3: build IntervalTrees
    build_ref_trees(&mut global);
    eprintln!("[INFO] Interval Trees built. Ready for queries.");

    // Actually do node->coord
    let results = node_to_coords(&global, node_id);
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
    eprintln!("[INFO] Building data structures from GFA='{}' PAF='{}'", gfa_path, paf_path);

    let mut global = GlobalData {
        node_map: HashMap::new(),
        path_map: HashMap::new(),
        node_to_paths: HashMap::new(),
        alignment_by_path: HashMap::new(),
        ref_trees: HashMap::new(),
    };

    // Parse GFA
    parse_gfa_memmap(gfa_path, &mut global);
    eprintln!("[INFO] GFA parse done. node_map={} path_map={}",
        global.node_map.len(), global.path_map.len());

    // Parse PAF
    parse_paf_parallel(paf_path, &mut global);
    eprintln!("[INFO] PAF parse done. alignment_by_path={} refTrees building...",
        global.alignment_by_path.len());

    // Build trees
    build_ref_trees(&mut global);
    eprintln!("[INFO] IntervalTrees built. Ready for queries.");

    // parse region
    if let Some((chr, start, end)) = parse_region(region) {
        let results = coord_to_nodes(&global, &chr, start, end);
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
    let file = File::open(gfa_path).expect("Cannot open GFA file for memmap");
    let meta = file.metadata().expect("Cannot stat GFA file");
    let filesize = meta.len();
    eprintln!("[INFO] GFA memory-mapping file of size {} bytes...", filesize);

    let mmap = unsafe {
        MmapOptions::new().map(&file).expect("Failed to mmap GFA")
    };

    eprintln!("[INFO] Finding line boundaries in GFA...");
    let mut line_indices = Vec::new();
    line_indices.push(0);
    for i in 0..mmap.len() {
        if mmap[i] == b'\n' {
            if i + 1 < mmap.len() {
                line_indices.push(i + 1);
            }
        }
    }
    let total_lines = line_indices.len();
    eprintln!("[INFO] GFA total lines found = {}", total_lines);

    let pb = ProgressBar::new(total_lines as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({eta}) {msg}")
            .expect("Invalid progress style template")
            .progress_chars("##-")
    );

    // Each thread will collect results here:
    #[derive(Debug, Clone)]
    struct TempPathData {
        nodes: Vec<(String, bool)>,
        overlaps: Vec<String>,
    }

    // We store all chunk results in these thread-local vectors. No locks on each insertion.

    // We will accumulate chunk results in a local vector, then push into a global vector after the loop.
    // This data structure:
    //   local_node_map:    HashMap<String, NodeInfo>
    //   local_path_map:    HashMap<String, Vec<TempPathData>>  (multiple P-lines can share a path_name)
    //   local_node_to_paths: HashMap<String, Vec<(String, usize)>>

    // We'll define a single container to hold them for each chunk:
    #[derive(Debug)]
    struct GfaChunkResult {
        node_map: HashMap<String, NodeInfo>,
        path_map: HashMap<String, Vec<TempPathData>>,
        node_to_paths: HashMap<String, Vec<(String, usize)>>,
        lines_processed: usize,
    }

    // We'll gather GfaChunkResult in a thread-local fashion, then do one global merge.
    let chunk_size = 50_000;
    let num_chunks = (total_lines + chunk_size - 1) / chunk_size;
    let mut all_results = Vec::new();

    (0..num_chunks).into_par_iter().map(|chunk_idx| {
        let start_line = chunk_idx * chunk_size;
        let end_line = ((chunk_idx + 1) * chunk_size).min(total_lines);

        let mut local_res = GfaChunkResult {
            node_map: HashMap::new(),
            path_map: HashMap::new(),
            node_to_paths: HashMap::new(),
            lines_processed: 0,
        };

        for li in start_line..end_line {
            local_res.lines_processed += 1;
            if local_res.lines_processed % 10000 == 0 {
                pb.inc(10000);
            }
            let offset = line_indices[li];
            let end_off = if li + 1 < total_lines {
                line_indices[li + 1] - 1
            } else {
                mmap.len()
            };
            if end_off <= offset {
                continue;
            }
            let line_slice = &mmap[offset..end_off];
            if line_slice.is_empty() {
                continue;
            }
            match line_slice[0] {
                b'S' => {
                    let parts: Vec<&[u8]> = line_slice.split(|&c| c == b'\t').collect();
                    if parts.len() < 3 {
                        continue;
                    }
                    let seg_name = String::from_utf8_lossy(parts[1]).to_string();
                    let seq_or_star = parts[2];
                    let mut length = 0;
                    if seq_or_star == b"*" {
                        for p in parts.iter().skip(3) {
                            if p.starts_with(b"LN:i:") {
                                let ln_s = &p[5..];
                                if let Ok(txt) = std::str::from_utf8(ln_s) {
                                    if let Ok(val) = txt.parse::<usize>() {
                                        length = val;
                                        break;
                                    }
                                }
                            }
                        }
                    } else {
                        length = seq_or_star.len();
                    }
                    local_res.node_map.insert(seg_name, NodeInfo { length });
                },
                b'P' => {
                    let parts: Vec<&[u8]> = line_slice.split(|&c| c == b'\t').collect();
                    if parts.len() < 3 {
                        continue;
                    }
                    let path_name = String::from_utf8_lossy(parts[1]).to_string();
                    let seg_string = String::from_utf8_lossy(parts[2]).to_string();
                    let oriented: Vec<(String, bool)> = seg_string
                        .split(',')
                        .filter_map(|x| {
                            if x.is_empty() {
                                return None;
                            }
                            let lastch = x.chars().last().unwrap();
                            let orient = match lastch {
                                '+' => true,
                                '-' => false,
                                _ => return None,
                            };
                            let nid = &x[..x.len() - 1];
                            Some((nid.to_string(), orient))
                        })
                        .collect();

                    let mut overlaps = Vec::new();
                    if parts.len() > 3 {
                        let overlap_field = String::from_utf8_lossy(parts[3]).to_string();
                        if overlap_field != "*" {
                            overlaps = overlap_field.split(',').map(|s| s.to_string()).collect();
                        }
                    }
                    if !overlaps.is_empty() && overlaps.len() + 1 != oriented.len() {
                        eprintln!("Warning: Overlap field count does not match segments for path {}", path_name);
                    }

                    // We record node->(path_name, indexInPath) now, but do NOT compute prefix sums yet
                    for (i, (nid, _orient)) in oriented.iter().enumerate() {
                        local_res
                            .node_to_paths
                            .entry(nid.clone())
                            .or_insert_with(Vec::new)
                            .push((path_name.clone(), i));
                    }

                    // Collect the nodes + overlap info for this path line
                    let temp_pd = TempPathData {
                        nodes: oriented,
                        overlaps,
                    };
                    local_res
                        .path_map
                        .entry(path_name)
                        .or_insert_with(Vec::new)
                        .push(temp_pd);
                },
                _ => {}
            }
        }
        pb.inc((local_res.lines_processed % 10000) as u64);
        local_res
    }).collect_into_vec(&mut all_results);
    
    // End the progress bar right after we finish the parallel parse
    pb.finish_and_clear();
    
    // Now do a SINGLE MERGE (no repeated locking in the loop).
    let mut merged_node_map = HashMap::new();
    let mut merged_path_map = HashMap::new();
    let mut merged_node_to_paths = HashMap::new();

    let pb_merge = ProgressBar::new(all_results.len() as u64);
    pb_merge.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.green/black} {pos:>7}/{len:7} ({eta}) Merging GFA")
            .expect("Invalid progress style template")
            .progress_chars("##-")
    );

    for mut chunk_res in all_results {
        pb_merge.inc(1);

        // Merge node_map
        for (k, v) in chunk_res.node_map.drain() {
            merged_node_map.insert(k, v);
        }
        // Merge path_map
        for (path_name, mut vec_temp) in chunk_res.path_map.drain() {
            merged_path_map.entry(path_name).or_insert_with(Vec::new).append(&mut vec_temp);
        }
        // Merge node_to_paths
        for (node_id, mut pairs) in chunk_res.node_to_paths.drain() {
            merged_node_to_paths.entry(node_id).or_insert_with(Vec::new).append(&mut pairs);
        }
    }

    pb_merge.finish_and_clear();

    // Now we have all nodes, plus path definitions in "merged_path_map" but still missing prefix sums.
    // We'll build final PathData in a parallel step:
    let final_path_map: HashMap<String, PathData> = merged_path_map
        .into_par_iter()
        .map(|(pname, temp_vec)| {
            // A path_name can appear multiple times, so we merge them into a single PathData.
            // We'll concatenate all segments from these P lines.
            // Typically GFA doesn't define the same path multiple times, but let's handle it just in case.
            let mut all_nodes = Vec::new();
            let mut all_overlaps = Vec::new();

            for tpd in temp_vec {
                all_nodes.extend(tpd.nodes);
                all_overlaps.extend(tpd.overlaps);
            }

            // Now compute prefix sums.
            let mut prefix_sums = Vec::with_capacity(all_nodes.len());
            let mut cum = 0;
            for i in 0..all_nodes.len() {
                prefix_sums.push(cum);
                if i < all_nodes.len() - 1 {
                    // add length of current node
                    let (ref_nid, _ref_or) = &all_nodes[i];
                    let node_len = merged_node_map
                        .get(ref_nid)
                        .map(|info| info.length)
                        .unwrap_or(0);
                    cum += node_len;
                    // subtract overlap if it exists
                    if i < all_overlaps.len() {
                        let overlap_len = parse_cigar_overlap(&all_overlaps[i]);
                        cum = cum.saturating_sub(overlap_len);
                    }
                }
            }
            // Add last node length
            if let Some((last_nid, _)) = all_nodes.last() {
                if let Some(n_info) = merged_node_map.get(last_nid) {
                    cum += n_info.length;
                }
            }

            let pd = PathData {
                nodes: all_nodes,
                prefix_sums,
                total_length: cum,
            };
            (pname, pd)
        })
        .collect();

    global.node_map = merged_node_map;
    global.path_map = final_path_map;
    global.node_to_paths = merged_node_to_paths;
}

// parse_paf_parallel
pub fn parse_paf_parallel(paf_path: &str, global: &mut GlobalData) {
    use std::io::BufRead;

    let file = File::open(paf_path).expect("Cannot open PAF");
    let reader = BufReader::new(file);

    // Collect lines first (you can also stream in chunks, but this is simple):
    let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();
    let pb = ProgressBar::new(lines.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.magenta/black} {pos:>7}/{len:7} ({eta}) PAF")
            .expect("Invalid template for progress style")
            .progress_chars("##-")
    );

    // Each thread accumulates local results, no lock until final merge.
    #[derive(Debug)]
    struct PafChunkResult {
        data: HashMap<String, Vec<AlignmentBlock>>,
        count: usize,
    }

    let chunk_size = 10_000;
    let mut all_results = Vec::new();
    lines
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_res = PafChunkResult {
                data: HashMap::new(),
                count: 0,
            };
            for l in chunk {
                local_res.count += 1;
                if l.is_empty() || l.starts_with('#') {
                    continue;
                }
                let parts: Vec<&str> = l.split('\t').collect();
                if parts.len() < 12 {
                    continue;
                }
                let q_name = parts[0].to_string();
                let q_len = parts[1].parse::<usize>().unwrap_or(0);
                let raw_qs = parts[2].parse::<usize>().unwrap_or(0);
                let raw_qe = parts[3].parse::<usize>().unwrap_or(0);
                let strand_char = parts[4].chars().next().unwrap_or('+');
                let t_name = parts[5].to_string();
                let t_start = parts[7].parse::<usize>().unwrap_or(0);
                let t_end = parts[8].parse::<usize>().unwrap_or(0);

                // Determine if the path is reversed with respect to the reference
                let ref_strand = strand_char == '+';
                
                // Flip q_start..q_end if it's a negative strand
                let (q_start, q_end) = if !ref_strand {
                    let flipped_start = q_len.saturating_sub(raw_qe);
                    let flipped_end   = q_len.saturating_sub(raw_qs);
                    (flipped_start, flipped_end)
                } else {
                    (raw_qs, raw_qe)
                };
                
                // Build the alignment block with the query offsets
                let ab = AlignmentBlock {
                    path_name: q_name.clone(),
                    q_len,
                    q_start,
                    q_end,
                    ref_chrom: t_name,
                    r_start: t_start,
                    r_end: t_end,
                    ref_strand: ref_strand,
                };
                
                local_res.data.entry(q_name).or_insert_with(Vec::new).push(ab);
            }
            pb.inc(local_res.count as u64);
            local_res
        })
        .collect_into_vec(&mut all_results);
    pb.finish_and_clear();

    // Single merge pass:
    let mut merged_map = HashMap::new();
    for mut chunk_res in all_results {
        for (k, mut vec_blocks) in chunk_res.data.drain() {
            merged_map.entry(k).or_insert_with(Vec::new).append(&mut vec_blocks);
        }
    }

    global.alignment_by_path = merged_map;
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
// using global data structures to do node->coords mapping
pub fn node_to_coords(global: &GlobalData, node_id: &str) -> Vec<(String,usize,usize)> {
    let mut results = Vec::new();

    let nodeinfo = match global.node_map.get(node_id) {
        Some(x) => x,
        None => return results,
    };
    let len = nodeinfo.length;

    // find all (path,index) from node_to_paths
    let pathrefs = match global.node_to_paths.get(node_id) {
        Some(v) => v,
        None => return results,
    };

    for (pname, idx) in pathrefs {
        let pd = match global.path_map.get(pname) {
            Some(x) => x,
            None => continue,
        };
        if *idx >= pd.nodes.len() {
            continue;
        }
        let node_offset_start = pd.prefix_sums[*idx];
        let node_offset_end   = node_offset_start + len.saturating_sub(1);

        // alignment blocks
        let blocks = match global.alignment_by_path.get(pname) {
            Some(b) => b,
            None => continue,
        };
        // retrieve the node orientation from pd.nodes[*idx]
        let node_or = pd.nodes[*idx].1;
        
        for b in blocks {
            let qs = b.q_start;
            let qe = b.q_end;
            if qe < node_offset_start || qs > node_offset_end {
                continue;
            }
            let ov_start = node_offset_start.max(qs);
            let ov_end   = node_offset_end.min(qe);
            if ov_start <= ov_end {
                let diff_start = ov_start - qs;
                let diff_end   = ov_end   - qs;
        
            // If the alignment itself is reversed vs. the reference, invert diff_start..diff_end
            let (rs, re) = if !b.ref_strand {
                let total_len = b.r_end.saturating_sub(b.r_start);
                let flipped_start = total_len.saturating_sub(diff_end);
                let flipped_end   = total_len.saturating_sub(diff_start);
                (flipped_start, flipped_end)
            } else {
                (diff_start, diff_end)
            };
            
            // Now apply the node's orientation in the path
            if node_or {
                // Node is forward in path
                let final_ref_start = b.r_start + rs;
                let final_ref_end   = b.r_start + re;
                results.push((b.ref_chrom.clone(), final_ref_start, final_ref_end));
            } else {
                // Node is reversed in path
                let rev_start = b.r_end.saturating_sub(re);
                let rev_end   = b.r_end.saturating_sub(rs);
                let final_ref_start = if rev_start <= rev_end { rev_start } else { rev_end };
                let final_ref_end   = if rev_start <= rev_end { rev_end } else { rev_start };
                results.push((b.ref_chrom.clone(), final_ref_start, final_ref_end));
            }
            }
        }
    }
    results
}


// coord_to_nodes
//  given e.g. "grch38#chr1", 100000, 110000
pub fn coord_to_nodes(global: &GlobalData, chr: &str, start: usize, end: usize) -> Vec<Coord2NodeResult> {
    println!("DEBUG: Searching for region {}:{}-{}", chr, start, end);
    let mut results = Vec::new();

    let tree = match global.ref_trees.get(chr) {
        Some(t) => t,
        None => {
            println!("DEBUG: No tree found for chromosome {}", chr);
            return results;
        }
    };
    // do an interval query
    let mut ivs = Vec::new();
    tree.query(start, end, &mut ivs);
    println!("DEBUG: Found {} intervals overlapping {}:{}-{}", ivs.len(), chr, start, end);

    // for each interval, we compute overlap in ref space, then convert to path offsets, then find node(s)
    for iv in ivs {
        let ab = &iv.data;
        println!("DEBUG: Processing interval: {}:{}-{} from path {}", 
                 ab.ref_chrom, iv.start, iv.end, ab.path_name);
                 
        // Even if a node only partially overlaps [start..end], we include it. We
        // compute the intersecting segment in reference coordinates, then map
        // that segment back onto the node's path offset range.
        let ov_s = start.max(iv.start);
        let ov_e = end.min(iv.end);
        if ov_s>ov_e {
            println!("DEBUG: Invalid overlap segment: {}-{}", ov_s, ov_e);
            continue;
        }
        // Compute overlap in reference space
        let diff_start = ov_s - ab.r_start;
        let diff_end   = ov_e - ab.r_start;
        
        // If ab.ref_strand == false, we flip diff_start..diff_end
        let (flip_start, flip_end) = if !ab.ref_strand {
            let total_len = ab.r_end.saturating_sub(ab.r_start);
            let fs = total_len.saturating_sub(diff_end);
            let fe = total_len.saturating_sub(diff_start);
            (fs, fe)
        } else {
            (diff_start, diff_end)
        };
        
        // Now map to path offsets
        let path_ov_start = ab.q_start + flip_start;
        let path_ov_end   = ab.q_start + flip_end;

        // now find which nodes in ab.path_name covers path_ov_start..path_ov_end
        let pd = match global.path_map.get(&ab.path_name) {
            Some(x) => x,
            None => {
                println!("DEBUG: Path {} not found in path_map (found in PAF but not in GFA)", ab.path_name);
                continue;
            }
        };
        println!("DEBUG: Found path {} in path_map with {} nodes", ab.path_name, pd.nodes.len());
        
        // do a binary search approach
        let (_start_node, mut i) = match pd.prefix_sums.binary_search_by(|&off| off.cmp(&path_ov_start)) { 
            Ok(i) => {
                println!("DEBUG: Found exact match at index {} with offset {}", i, pd.prefix_sums[i]);
                (i,i)
            },
            Err(i) => {
                if i>0 { 
                    println!("DEBUG: Found closest lower index at {} with offset {}", i-1, pd.prefix_sums[i-1]);
                    (i-1,i-1) 
                } else { 
                    println!("DEBUG: No lower bound found, starting at index 0");
                    (0,0) 
                }
            },
        };
        // we will proceed forward while offset range is in path_ov_end
        println!("DEBUG: Looking for nodes covering path offsets {}..{} in path {}", 
                 path_ov_start, path_ov_end, ab.path_name);
        
        while i<pd.nodes.len() {
            let noff_start = pd.prefix_sums[i];
            let node_id = &pd.nodes[i].0;
            let node_or = pd.nodes[i].1;
            let node_len = global.node_map.get(node_id).map(|xx|xx.length).unwrap_or(0);
            let noff_end = noff_start + node_len.saturating_sub(1);

            println!("DEBUG: Checking node {} ({}), offset range: {}..{}", 
                     node_id, if node_or {'+'}else{'-'}, noff_start, noff_end);

            if noff_start>path_ov_end {
                println!("DEBUG: Node starts after path range end, stopping search");
                break;
            }
            let o_s = noff_start.max(path_ov_start);
            let o_e = noff_end.min(path_ov_end);
            if o_s <= o_e {
                println!("DEBUG: Found overlapping node {} with effective range {}..{}", node_id, o_s, o_e);
                let final_start;
                let final_end;
                if node_or {
                    // Node is forward in the path
                    final_start = o_s;
                    final_end   = o_e;
                } else {
                    // Node is reversed in the path
                    let node_highest_offset = noff_start + node_len - 1;
                    let flipped_start = node_highest_offset.saturating_sub(o_e - noff_start);
                    let flipped_end   = node_highest_offset.saturating_sub(o_s - noff_start);
                    if flipped_start <= flipped_end {
                        final_start = flipped_start;
                        final_end   = flipped_end;
                    } else {
                        final_start = flipped_end;
                        final_end   = flipped_start;
                    }
                }
            
                results.push(Coord2NodeResult {
                    path_name: ab.path_name.clone(),
                    node_id: node_id.clone(),
                    node_orient: node_or,
                    path_off_start: final_start,
                    path_off_end:   final_end,
                });
            }
            i+=1;
        }
    }

    results
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
