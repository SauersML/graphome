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
//
// Example usage:
//   cargo run --release -- --gfa hprc-v1.0-pggb.gfa --paf untangle.paf node2coord 10127854
//   cargo run --release -- --gfa hprc-v1.0-pggb.gfa --paf untangle.paf coord2node grch38#chr1:100000-110000


use std::collections::{HashMap};
use std::fs::{File};
use std::io::{BufReader, BufRead};
use std::sync::{Arc, Mutex};

use clap::{Arg, Command};
use memmap2::{MmapOptions};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};


// Data Structures


/// Stores basic info about a GFA node: specifically its length.
#[derive(Debug, Clone)]
struct NodeInfo {
    length: usize,
}

/// For each path in the GFA, we store:
///   - the ordered list of (nodeID, orientation)
///   - prefix_sums so we can do offset lookups
#[derive(Debug)]
#[derive(Clone)]
struct PathData {
    nodes: Vec<(String, bool)>,   // bool => orientation, true=+ / false=-
    prefix_sums: Vec<usize>,
    total_length: usize,
}

/// For alignments from the PAF (untangle). Each record:
///   - pathName  (the "query" in PAF)
///   - qStart,qEnd
///   - refChrom  (the "target" in PAF, e.g. "grch38#chr1")
///   - rStart,rEnd
///   - strand
#[derive(Debug, Clone)]
struct AlignmentBlock {
    path_name: String,
    q_start:   usize,
    q_end:     usize,
    ref_chrom: String,
    r_start:   usize,
    r_end:     usize,
    strand:    bool, // + => true, - => false
}

/// We'll store reference intervals in a stable interval tree, so we can do
/// coordinate lookups quickly for coord->node. Each node in that tree:
///   - the [start..end] range in reference
///   - the actual alignment block
#[derive(Debug, Clone)]
struct Interval {
    start: usize,
    end:   usize,
    data:  AlignmentBlock,
}

// A simple interval tree node: we store intervals in a segment, and a center
#[derive(Debug)]
enum IntervalTree {
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

        // We can pick a better pivot approach, but let's do a partial approach:
        let pivot = intervals[mid].start;
        for iv in &intervals {
            // check if iv is definitely left or right or overlapping pivot
            let mid_center = pivot;
            if iv.end < mid_center {
                left_vec.push(iv.clone());
            } else if iv.start > mid_center {
                right_vec.push(iv.clone());
            } else {
                // overlaps
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
                // check left if qstart <= center
                if qstart <= *center {
                    left.query(qstart, qend, results);
                }
                // check right if qend >= center
                if qend >= *center {
                    right.query(qstart, qend, results);
                }
                // check overlaps
                for iv in overlaps {
                    if iv.end < qstart || iv.start > qend {
                        // no overlap
                    } else {
                        results.push(iv);
                    }
                }
            }
        }
    }
}


// We'll keep these big data structures in memory for queries.


struct GlobalData {
    // node information
    node_map: HashMap<String, NodeInfo>,

    // path information
    path_map: HashMap<String, PathData>,

    // node->(pathName,indexInPath)
    node_to_paths: HashMap<String, Vec<(String,usize)>>,

    // alignment blocks per path
    alignment_by_path: HashMap<String, Vec<AlignmentBlock>>,

    // for reference coords
    // e.g. "grch38#chr1" -> IntervalTree
    ref_trees: HashMap<String, IntervalTree>,
}


// Main


fn main() {
    let matches = Command::new("pangenome_lookup")
        .version("1.0")
        .about("Production-level single-file Rust code for GFA + PAF lookups with memory mapping & concurrency.")
        .arg(Arg::new("gfa")
             .long("gfa")
             .short('g')
             .help("Path to the massive GFA file")
             .required(true)
             .num_args(1))
        .arg(Arg::new("paf")
             .long("paf")
             .short('p')
             .help("Path to the untangle PAF file")
             .required(true)
             .num_args(1))
        .subcommand(Command::new("node2coord")
            .about("Given a GFA node ID, show the corresponding hg38 coords.")
            .arg(Arg::new("NODE_ID").required(true)))
        .subcommand(Command::new("coord2node")
            .about("Given <chr>:<start>-<end> in hg38, show GFA nodes that overlap.")
            .arg(Arg::new("REGION").required(true)))
        .get_matches();

    let gfa_path = matches.get_one::<String>("gfa").unwrap();
    let paf_path = matches.get_one::<String>("paf").unwrap();

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

    
    // Step 3: build IntervalTrees for each ref chrom
    
    build_ref_trees(&mut global);

    eprintln!("[INFO] IntervalTrees built. Ready for queries.");

    // handle subcommands
    if let Some(("node2coord", subm)) = matches.subcommand() {
        let node_id = subm.get_one::<String>("NODE_ID").unwrap();
        let results = node_to_coords(&global, node_id);
        if results.is_empty() {
            println!("No reference coords found for node {}", node_id);
        } else {
            for (chr, st, en) in results {
                println!("{}:{}-{}", chr, st, en);
            }
        }
    } else if let Some(("coord2node", subm)) = matches.subcommand() {
        let region = subm.get_one::<String>("REGION").unwrap();
        if let Some((chr, start, end)) = parse_region(region) {
            let results = coord_to_nodes(&global, &chr, start, end);
            if results.is_empty() {
                println!("No nodes found for region {}:{}-{}", chr, start, end);
            } else {
                for r in results {
                    println!("path={} node={}({}) offsets=[{}..{}]",
                             r.path_name, r.node_id,
                             if r.node_orient {'+'} else {'-'},
                             r.path_off_start, r.path_off_end);
                }
            }
        } else {
            eprintln!("Could not parse region format: {}", region);
        }
    } else {
        eprintln!("Please use subcommand node2coord or coord2node.");
    }
}


// parse_gfa_memmap
//
// - memory-map the GFA
// - find lines by scanning for newlines
// - parse S lines to fill node_map
// - parse P lines to fill path_map, prefix sums, node_to_paths

fn parse_gfa_memmap(gfa_path: &str, global: &mut GlobalData) {
    let file = File::open(gfa_path).expect("Cannot open GFA file for memmap");
    let meta = file.metadata().expect("Cannot stat GFA file");
    let filesize = meta.len();

    eprintln!("[INFO] GFA memory-mapping file of size {} bytes...", filesize);

    let mmap = unsafe {
        MmapOptions::new().map(&file).expect("Failed to mmap GFA")
    };

    eprintln!("[INFO] Finding line boundaries in GFA...");

    // We'll collect line start indices.
    let mut line_indices = Vec::new();
    line_indices.push(0usize);
    for i in 0..mmap.len() {
        if mmap[i] == b'\n' {
            if i+1 < mmap.len() {
                line_indices.push(i+1);
            }
        }
    }
    let total_lines = line_indices.len();
    eprintln!("[INFO] GFA total lines found = {}", total_lines);

    // progress bar
    let pb = ProgressBar::new(total_lines as u64);
    pb.set_style(ProgressStyle::default_bar()
       .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({eta}) {msg}")
       .expect("Invalid template for progress style")
       .progress_chars("##-"));

    // We'll parse lines in parallel with rayon. We'll store partial results in local thread containers, then merge.
    let chunk_size = 50_000; // lines per chunk
    let num_chunks = (total_lines + chunk_size - 1)/chunk_size;

    // for concurrency:
    let node_map_mutex = Arc::new(Mutex::new(HashMap::new()));
    let path_map_mutex = Arc::new(Mutex::new(HashMap::new()));
    let node_to_paths_mutex = Arc::new(Mutex::new(HashMap::new()));

    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
        let start_line = chunk_idx*chunk_size;
        let end_line = ((chunk_idx+1)*chunk_size).min(total_lines);

        // local temporary data
        let mut local_node_map = HashMap::new();
        let mut local_path_map = HashMap::new();
        let mut local_node_to_paths = HashMap::new();

        for li in start_line..end_line {
            pb.inc(1);
            let offset = line_indices[li];
            // next offset
            let end_off = if li+1<total_lines {
                line_indices[li+1]-1
            } else {
                mmap.len()
            };
            if end_off<=offset { continue; }
            let line_slice = &mmap[offset..end_off];
            if line_slice.is_empty() { continue; }
            let first_char = line_slice[0];
            match first_char {
                b'S' => {
                    // parse an S line: "S\t<name>\t<seq or *>..."
                    // split by tabs
                    // We'll do a quick approach
                    let parts:Vec<&[u8]> = line_slice.split(|&c| c == b'\t').collect();
                    if parts.len()<3 { continue; }
                    // parts[0] == b"S"
                    let seg_name = String::from_utf8_lossy(parts[1]).to_string();
                    let seq_or_star = parts[2];
                    let mut length = 0usize;
                    if seq_or_star == b"*" {
                        // look for LN:i: in the rest of parts
                        for p in parts.iter().skip(3) {
                            if p.starts_with(b"LN:i:") {
                                let ln_s = &p[5..];
                                if let Some(v) = std::str::from_utf8(ln_s).ok().and_then(|z| z.parse::<usize>().ok()) {
                                    length = v;
                                    break;
                                }
                            }
                        }
                    } else {
                        length = seq_or_star.len();
                    }
                    local_node_map.insert(seg_name, NodeInfo{ length });
                },
                b'P' => {
                    // "P\t<pathName>\t<segmentNames>..."
                    let parts:Vec<&[u8]> = line_slice.split(|&c| c == b'\t').collect();
                    if parts.len()<3 { continue; }
                    let path_name = String::from_utf8_lossy(parts[1]).to_string();
                    let seg_names = parts[2];
                    // we won't parse the overlaps
                    let seg_string = String::from_utf8_lossy(seg_names).to_string();
                    let oriented: Vec<(String, bool)> = seg_string
                        .split(',')
                        .filter_map(|x| {
                            if x.is_empty() {return None;}
                            let lastch = x.chars().last().unwrap();
                            let orient = match lastch {
                                '+' => true,
                                '-' => false,
                                _ => return None,
                            };
                            let nid = &x[..x.len()-1];
                            Some((nid.to_string(), orient))
                        })
                        .collect();

                    let mut prefix = Vec::with_capacity(oriented.len());
                    let mut cum = 0usize;
                    for (i,(nid,_)) in oriented.iter().enumerate() {
                        prefix.push(cum);
                        // For now, store 0. We'll fix after we merge
                        // We'll do that in a second pass
                        cum += 0; // Fix?

                        // node->(pathName,i)
                        local_node_to_paths.entry(nid.clone())
                           .or_insert_with(Vec::new)
                           .push((path_name.clone(), i));
                    }
                    let pd = PathData {
                        nodes: oriented,
                        prefix_sums: prefix,
                        total_length: 0,
                    };
                    local_path_map.insert(path_name, pd);
                },
                _ => {}
            }
        }

        // merge local data
        {
            let mut g = node_map_mutex.lock().unwrap();
            for (k,v) in local_node_map {
                g.insert(k,v);
            }
        }
        {
            let mut g = path_map_mutex.lock().unwrap();
            for (k,v) in local_path_map {
                g.insert(k,v);
            }
        }
        {
            let mut g = node_to_paths_mutex.lock().unwrap();
            for (k,v) in local_node_to_paths {
                g.entry(k).or_insert_with(Vec::new).extend(v.into_iter());
            }
        }
    });
    pb.finish_and_clear();

    // finalize
    // now we do a second pass: fill prefix sums properly
    eprintln!("[INFO] Fixing path prefix sums with actual node lengths...");

    // unify
    let nm = Arc::try_unwrap(node_map_mutex).unwrap().into_inner().unwrap();
    let pm = Arc::try_unwrap(path_map_mutex).unwrap().into_inner().unwrap();
    let np = Arc::try_unwrap(node_to_paths_mutex).unwrap().into_inner().unwrap();

    // We'll do prefix sums in parallel again
    let keys: Vec<_> = pm.keys().cloned().collect();
    let pb2 = ProgressBar::new(keys.len() as u64);
    pb2.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.green/black} {pos:>7}/{len:7} ({eta}) Fix path prefix sums").progress_chars("##-"));

    let new_pm: HashMap<_,_> = keys.into_par_iter().map(|k| {
        let mut pd = pm[&k].clone();
        let mut cum = 0usize;
        for (i,(nid,_orient)) in pd.nodes.iter().enumerate() {
            pd.prefix_sums[i] = cum;
            let len = nm.get(nid).map(|xx|xx.length).unwrap_or(0);
            cum += len;
        }
        pd.total_length = cum;
        (k, pd)
    }).inspect(|_| {
        pb2.inc(1);
    }).collect();
    pb2.finish_and_clear();

    global.node_map = nm;
    global.path_map = new_pm;
    global.node_to_paths = np;
}


// parse_paf_parallel

fn parse_paf_parallel(paf_path: &str, global: &mut GlobalData) {
    // We'll read line-by-line in parallel. Then store results in alignment_by_path + we need a container to store for ref usage
    use std::io::BufRead;

    let f = File::open(paf_path).expect("Cannot open PAF");
    let reader = BufReader::new(f);

    // We'll store partial results in thread-loc data, then merge
    let path_map_arc = Arc::new(Mutex::new(HashMap::<String, Vec<AlignmentBlock>>::new()));

    // We'll do a mem approach if we want, but let's do standard concurrency:
    let lines:Vec<_> = reader.lines().collect::<Result<_,_>>().expect("read error");
    let pb = ProgressBar::new(lines.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.magenta/black} {pos:>7}/{len:7} ({eta}) PAF").expect("Invalid template for progress style").progress_chars("##-"));

    lines.into_par_iter().for_each(|l| {
        pb.inc(1);
        if l.is_empty() || l.starts_with('#') {
            return;
        }
        let parts:Vec<&str> = l.split('\t').collect();
        if parts.len()<12 {
            return;
        }
        let q_name = parts[0].to_string();
        let q_start = parts[2].parse::<usize>().unwrap_or(0);
        let q_end   = parts[3].parse::<usize>().unwrap_or(0);
        let strand_char = parts[4].chars().next().unwrap_or('+');
        let t_name = parts[5].to_string();
        let t_start= parts[7].parse::<usize>().unwrap_or(0);
        let t_end  = parts[8].parse::<usize>().unwrap_or(0);
        let strand = strand_char=='+';

        let ab = AlignmentBlock {
            path_name: q_name.clone(),
            q_start,
            q_end,
            ref_chrom: t_name,
            r_start: t_start,
            r_end: t_end,
            strand
        };
        {
            let mut map = path_map_arc.lock().unwrap();
            map.entry(q_name).or_insert_with(Vec::new).push(ab);
        }
    });
    pb.finish_and_clear();

    let final_map = Arc::try_unwrap(path_map_arc).unwrap().into_inner().unwrap();
    global.alignment_by_path = final_map;
}


// build_ref_trees
// We'll gather all alignment blocks from alignment_by_path, group by refChrom
// then build an Interval. Then build an IntervalTree for each refChrom.
fn build_ref_trees(global: &mut GlobalData) {
    // gather all intervals
    use rayon::prelude::*;
    let mut by_ref = HashMap::<String, Vec<Interval>>::new();

    let pb = ProgressBar::new(global.alignment_by_path.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.yellow/black} {pos:>7}/{len:7} ({eta}) Building intervals").expect("Invalid template for progress style").progress_chars("##-"));

    for (pname, blocks) in &global.alignment_by_path {
        pb.inc(1);
        for b in blocks {
            // create Interval for [b.r_start..b.r_end]
            let interval = Interval {
                start: b.r_start,
                end:   b.r_end,
                data:  b.clone()
            };
            by_ref.entry(b.ref_chrom.clone()).or_insert_with(Vec::new).push(interval);
        }
    }
    pb.finish_and_clear();

    // now build trees in parallel
    let keys:Vec<_> = by_ref.keys().cloned().collect();
    let pb2 = ProgressBar::new(keys.len() as u64);
    pb2.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/black} {pos:>7}/{len:7} ({eta}) Building IntervalTrees").progress_chars("##-"));

    let built: HashMap<String, IntervalTree> = keys.into_par_iter().map(|k| {
        let intervals = by_ref.remove(&k).unwrap();
        let tree = IntervalTree::build(intervals);
        pb2.inc(1);
        (k, tree)
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
        for b in blocks {
            let qs = b.q_start;
            let qe = b.q_end;
            if qe<node_offset_start || qs>node_offset_end {
                continue;
            }
            let ov_start = node_offset_start.max(qs);
            let ov_end   = node_offset_end.min(qe);
            if ov_start <= ov_end {
                // map to reference
                let diff_start = ov_start - qs;
                let diff_end   = ov_end   - qs;
                let ref_start = b.r_start + diff_start;
                let ref_end   = b.r_start + diff_end;
                // ignoring orientation for now
                results.push((b.ref_chrom.clone(), ref_start, ref_end));
            }
        }
    }
    results
}


// coord_to_nodes
//  given e.g. "grch38#chr1", 100000, 110000
pub fn coord_to_nodes(global: &GlobalData, chr: &str, start: usize, end: usize) -> Vec<Coord2NodeResult> {
    let mut results = Vec::new();

    let tree = match global.ref_trees.get(chr) {
        Some(t) => t,
        None => return results,
    };
    // do an interval query
    let mut ivs = Vec::new();
    tree.query(start, end, &mut ivs);

    // for each interval, we compute overlap in ref space, then convert to path offsets, then find node(s)
    for iv in ivs {
        let ab = &iv.data;
        let ov_s = start.max(iv.start);
        let ov_e = end.min(iv.end);
        if ov_s>ov_e {
            continue;
        }
        let diff_start = ov_s - ab.r_start;
        let diff_end   = ov_e - ab.r_start;
        let path_ov_start = ab.q_start + diff_start;
        let path_ov_end   = ab.q_start + diff_end;

        // now find which nodes in ab.path_name covers path_ov_start..path_ov_end
        let pd = match global.path_map.get(&ab.path_name) {
            Some(x) => x,
            None => continue,
        };
        // do a binary search approach
        let (start_node, mut i) = match pd.prefix_sums.binary_search_by(|&off| off.cmp(&path_ov_start)) {
            Ok(i) => (i,i),
            Err(i) => if i>0 { (i-1,i-1) } else { (0,0) },
        };
        // we will proceed forward while offset range is in path_ov_end
        // let's define a function to get node range
        while i<pd.nodes.len() {
            let noff_start = pd.prefix_sums[i];
            let node_id = &pd.nodes[i].0;
            let node_or = pd.nodes[i].1;
            let node_len = global.node_map.get(node_id).map(|xx|xx.length).unwrap_or(0);
            let noff_end = noff_start + node_len.saturating_sub(1);

            if noff_start>path_ov_end {
                break;
            }
            let o_s = noff_start.max(path_ov_start);
            let o_e = noff_end.min(path_ov_end);
            if o_s<=o_e {
                results.push(Coord2NodeResult {
                    path_name: ab.path_name.clone(),
                    node_id: node_id.clone(),
                    node_orient: node_or,
                    path_off_start: o_s,
                    path_off_end:   o_e,
                });
            }
            i+=1;
        }
    }

    results
}

#[derive(Debug)]
struct Coord2NodeResult {
    path_name: String,
    node_id:   String,
    node_orient: bool,
    path_off_start: usize,
    path_off_end:   usize,
}

// parse_region e.g. "grch38#chr1:12345-67890"
fn parse_region(r: &str) -> Option<(String,usize,usize)> {
    // e.g. "grch38#chr1:120616922-120626943"
    let (chr_part, rng_part) = r.split_once(':')?;
    let (s,e) = rng_part.split_once('-')?;
    let start = s.parse::<usize>().ok()?;
    let end   = e.parse::<usize>().ok()?;
    Some((chr_part.to_string(), start, end))
}
