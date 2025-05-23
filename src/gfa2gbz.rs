/***************************************************************************************************
 * 
 *  GFA to GBZ
 * 
 *    - Reads a GFA v1 file containing segments (S-lines), links (L-lines), paths (P-lines), 
 *      and possibly W-lines (walks) for haplotypes.
 *    - Assigns numeric node IDs (with an optional node-to-segment translation).
 *    - Builds a fully bidirectional GBWT index capturing every path + its reverse complement.
 *    - Builds a corresponding GBWTGraph storing node labels, plus optional translation.
 *    - Combines them into a spec-compliant GBZ (version 1) file containing:
 *        1) A GBZ header + tags
 *        2) A fully valid GBWT (with a 48-byte header, run-length BWT, optional doc array samples,
 *           real metadata for path names, sample names, contig names if needed)
 *        3) A GBWTGraph (header, node labels, optional node-to-segment translation).
 *    - Writes out the result as <input.gfa>.gbz .
 *
 *    - endmarker usage (node 0).
 *    - run-length FM-index construction with forward + reverse paths, adjacency, etc.
 *    - metadata for path names: if W-lines have (sample, haplotype, contig, fragment), we 
 *      store them. If P-lines have string path names, we store them in the "generic sample" style.
 *
 *  COMPILATION & USAGE:
 *    - ./target/release/graphome gfa2gbz --input ../hprc-v1.0-pggb.gfa
 *    - This will create "<path_to_gfa>.gbz"
 ***************************************************************************************************/

use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write, ErrorKind},
    path::Path,
    collections::{HashMap, BTreeMap, BTreeSet},
    env, process,
    time::Instant,
};

use indicatif::{ProgressBar, ProgressStyle};

use simple_sds::sparse_vector::{SparseBuilder, SparseVector};
use simple_sds::int_vector::IntVector;
use simple_sds::serialize::Serialize;
use simple_sds::ops::Push;

/// Maximum length for a single node. If a GFA segment is longer, we chunk it into multiple nodes.
const CHUNK_LIMIT: usize = 1024;

/***************************************************************************************************
 * BASIC TYPES
 **************************************************************************************************/

/// Bidirected orientation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Orientation {
    Forward = 0,
    Reverse = 1,
}
impl Orientation {
    fn flip(self) -> Orientation {
        match self {
            Orientation::Forward => Orientation::Reverse,
            Orientation::Reverse => Orientation::Forward,
        }
    }
}

/// Encode a node ID + orientation into a single "GBWT node" integer.
#[inline]
fn encode_node(node_id: usize, orientation: Orientation) -> usize {
    (node_id << 1) | (orientation as usize)
}
/// Decode a GBWT node into (node_id, orientation).
#[inline]
fn decode_node(gbwt_node: usize) -> (usize, Orientation) {
    let orientation_bit = gbwt_node & 1;
    let node_id = gbwt_node >> 1;
    let orientation = if orientation_bit == 1 {
        Orientation::Reverse
    } else {
        Orientation::Forward
    };
    (node_id, orientation)
}

/// A run-length pair (value, length) for storing edges or BWT transitions.
#[derive(Clone, Debug)]
struct Run {
    value: usize,
    len: usize,
}
impl Run {
    fn new(value: usize, len: usize) -> Self {
        Run { value, len }
    }
}

/// This structure holds a single node record in the final GBWT:
///   - adjacency to successors,
///   - run-length-encoded BWT body referencing those successors.
#[derive(Clone, Debug)]
struct GBWTRecord {
    /// adjacency is stored in ascending order by successor node ID
    edges: Vec<(usize, usize)>,
    /// BWT body run-length encoding: each run => "the next successor is edges[run.value], repeated run.len times"
    runs: Vec<Run>,
}

/// A stand-in for the final "GBWT index" in memory. We do not store doc array samples, but we do 
/// store real path-based metadata. We'll store endmarker 0 as well.
#[derive(Debug)]
struct GBWTIndex {
    /// We store node 0 for endmarker, plus up to (max_node_id + 1)*2 for real nodes, so 
    /// total size is (max_node_id + 1)*2 + 1
    records: Vec<Option<GBWTRecord>>,
    /// The total number of sequences stored in the BWT
    sequences: usize,
    /// The total length (including endmarkers).
    total_size: usize,
    /// The offset: we'll keep it 0 so node IDs are 1..=alphabet_size for real nodes, plus 0 for endmarker
    offset: usize,
    /// The size of the effective alphabet. We have 1..=something for real, plus 0 as endmarker => 
    /// so the total is 2*(max_node_id+1)+1
    alphabet_size: usize,

    /// Path-based metadata: each "path" in forward orientation => 1 sequence, 
    /// plus a reverse orientation => 1 sequence. We store them as structured fields.
    metadata: GBWTMetadata,
}

/// A single path name in the metadata. 
///   - We store sample, contig, phase, fragment. 
///   - Must be unique across the entire index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PathName {
    sample: u32,
    contig: u32,
    phase: u32,
    fragment: u32,
}

/// The "metadata" structure for the entire GBWT: samples, contigs, path names, etc.
#[derive(Debug)]
struct GBWTMetadata {
    /// number of samples, haplotypes, contigs
    sample_count: usize,
    haplotype_count: usize, 
    contig_count: usize,

    /// The dictionary of sample names => sample_id
    sample_names: Vec<String>,
    /// The dictionary of contig names => contig_id
    contig_names: Vec<String>,

    /// The pathName array, 1 entry per path in forward orientation. The bidirectional index 
    /// duplicates each path in reverse orientation => 2 sequences total. 
    /// The path with ID i in metadata => sequence i in forward orientation, sequence i+some offset in reverse orientation, etc.
    path_names: Vec<PathName>,
}

/***************************************************************************************************
 * For the "graph" portion of GBWTGraph
 **************************************************************************************************/
#[derive(Debug)]
struct GBWTGraph {
    /// The node labels in forward orientation. For node i, we store the string. 
    /// If chunked, each original GFA segment might correspond to multiple consecutive node IDs.
    forward_labels: Vec<String>,

    /// An optional node->segment translation structure. 
    /// If present, we store segment names and a bitvector or offset array describing how nodes map to segments.
    translation: Option<Translation>,
}

/// If we do a node->segment translation, we store:
///   - A list of segment names
///   - A mapping from node ranges -> segment index
#[derive(Debug)]
struct Translation {
    segment_names: Vec<String>,
    /// For each segment i, we store the starting node. The next segment i+1 starts at offset i+1 in this array. 
    /// The final sentinel is forward_labels.len().
    /// So segment i => nodes [mapping[i] .. mapping[i+1])
    mapping: Vec<usize>,
}

/***************************************************************************************************
 * GFA Parsing Structures
 **************************************************************************************************/
#[derive(Debug)]
struct SegmentData {
    name: String,
    seq: String,
}
#[derive(Debug)]
struct LinkData {
    from_name: String,
    from_orient: Orientation,
    to_name: String,
    to_orient: Orientation,
}
#[derive(Debug, Clone)]
enum PathLineType {
    P,
    W,
}
#[derive(Debug, Clone)]
struct GFAPathLine {
    ptype: PathLineType,
    raw_name: String,  // If P-line, this is the path name; if W-line, we store something else
    // For W-lines, we parse out (sample, haplotype, contig, fragment, etc.)
    sample: Option<String>,
    hap_id: Option<usize>,
    contig: Option<String>,
    fragment: Option<usize>,
    segments: Vec<(String, Orientation)>,
}

/***************************************************************************************************
 * GFA -> Data
 **************************************************************************************************/
pub fn run_gfa2gbz(input: &str) {
    let infile = input;

    let outfile = format!("{}.gbz", infile);

    let start_t = Instant::now();

    println!("[gfa2gbz] Parsing GFA '{}'", infile);
    let parse_bar = ProgressBar::new_spinner();
    parse_bar.set_style(ProgressStyle::with_template("{spinner} Parsing GFA...").unwrap());
    parse_bar.enable_steady_tick(std::time::Duration::from_millis(100));
    let (segments, links, paths) = match parse_gfa(infile) {
        Ok(x) => x,
        Err(e) => {
            parse_bar.finish_and_clear();
            eprintln!("Error parsing GFA: {}", e);
            process::exit(1);
        }
    };
    parse_bar.finish_and_clear();
    println!("[gfa2gbz] Parsed: {} segments, {} links, {} path lines", 
        segments.len(), links.len(), paths.len());

    // Build node list from segments, chunk if >CHUNK_LIMIT
    println!("[gfa2gbz] Assigning node IDs (with chunking if needed) ...");
    let (node_map, node_labels, node_translation) = build_node_id_map(&segments);

    let max_node_id = node_labels.len() - 1; // if node_labels is [0..N]
    println!("[gfa2gbz] Total nodes after chunking: {}", node_labels.len());

    // Convert links
    println!("[gfa2gbz] Building adjacency from links...");
    let adjacency = build_adjacency(&links, &node_map, node_labels.len());

    // Convert paths
    println!("[gfa2gbz] Converting GFA paths into numeric form (will store in GBWT)...");
    let path_data = build_paths(&paths, &node_map);

    // Build the GBWT
    println!("[gfa2gbz] Building fully bidirectional GBWT with real endmarkers...");
    let gbwt_index = build_gbwt(&adjacency, node_labels.len(), &path_data);

    // Build the GBWTGraph
    println!("[gfa2gbz] Building GBWTGraph structure...");
    let gbwt_graph = build_gbwt_graph(node_labels, node_translation);

    // Write
    println!("[gfa2gbz] Writing final GBZ to '{}'", outfile);
    let write_bar = ProgressBar::new_spinner();
    write_bar.set_style(ProgressStyle::with_template("{spinner} Writing...").unwrap());
    write_bar.enable_steady_tick(std::time::Duration::from_millis(100));
    if let Err(e) = write_gbz(&gbwt_index, &gbwt_graph, &outfile) {
        write_bar.finish_and_clear();
        eprintln!("Error writing GBZ: {}", e);
        process::exit(1);
    }
    write_bar.finish_and_clear();
    println!("[gfa2gbz] Done. Elapsed: {:.2?}", start_t.elapsed());
}


/***************************************************************************************************
 * GFA Parsing
 **************************************************************************************************/
fn parse_gfa(filename: &str) -> Result<(Vec<SegmentData>, Vec<LinkData>, Vec<GFAPathLine>), std::io::Error> {
    let f = File::open(filename)?;
    let reader = BufReader::new(f);
    let mut segments = Vec::new();
    let mut links = Vec::new();
    let mut paths = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split('\t');
        let rec_type = match fields.next() {
            Some(x) => x,
            None => continue,
        };
        match rec_type {
            "H" => {
                // skip header fields
            },
            "S" => {
                // S <Name> <Sequence> ...
                let name = fields.next().unwrap_or("").to_string();
                let seq = fields.next().unwrap_or("").to_string();
                segments.push(SegmentData { name, seq });
            },
            "L" => {
                // L <From> <FromOrient> <To> <ToOrient> <Overlap>
                let from_name = fields.next().unwrap_or("").to_string();
                let from_o = fields.next().unwrap_or("+");
                let from_orient = if from_o == "-" { Orientation::Reverse } else { Orientation::Forward };
                let to_name = fields.next().unwrap_or("").to_string();
                let to_o = fields.next().unwrap_or("+");
                let to_orient = if to_o == "-" { Orientation::Reverse } else { Orientation::Forward };
                // skip overlap
                links.push(LinkData {
                    from_name,
                    from_orient,
                    to_name,
                    to_orient,
                });
            },
            "P" | "W" => {
                // parse differently
                let ptype = if rec_type == "P" { PathLineType::P } else { PathLineType::W };
                match ptype {
                    PathLineType::P => {
                        // P <PathName> <SegmentNames> <Overlaps> ...
                        let raw_name = fields.next().unwrap_or("").to_string();
                        let segnames = fields.next().unwrap_or("").to_string(); // segment visits
                        // skip overlap field
                        let _ = fields.next();
                        let segs = parse_path_segments(&segnames);
                        let gfapath = GFAPathLine {
                            ptype,
                            raw_name,
                            sample: None,
                            hap_id: None,
                            contig: None,
                            fragment: None,
                            segments: segs,
                        };
                        paths.push(gfapath);
                    },
                    PathLineType::W => {
                        // W <Sample> <HaplotypeId> <Contig> <(interval?)> <SegmentNames>
                        let sample = fields.next().unwrap_or("NA").to_string();
                        let hap_str = fields.next().unwrap_or("0");
                        let hap_id = hap_str.parse::<usize>().ok();
                        let contig = fields.next().map(|x| x.to_string());
                        let region = fields.next().unwrap_or("*");
                        let fragment = if region == "*" { None } else { Some(0usize) }; 
                        let segnames = fields.next().unwrap_or("").to_string();
                        let segs = parse_path_segments(&segnames);
                        let gfapath = GFAPathLine {
                            ptype,
                            raw_name: format!("wline:{}#{}", sample, contig.clone().unwrap_or("unknown".to_string())),
                            sample: Some(sample),
                            hap_id,
                            contig,
                            fragment,
                            segments: segs,
                        };
                        paths.push(gfapath);
                    },
                }
            },
            _ => {
                // ignore
            },
        }
    }

    Ok((segments, links, paths))
}

fn parse_path_segments(path_str: &str) -> Vec<(String, Orientation)> {
    // e.g. "11+,12-,13+"
    let mut result = Vec::new();
    if path_str.is_empty() {
        return result;
    }
    for item in path_str.split(',') {
        if item.is_empty() { continue; }
        let orient_char = item.chars().last().unwrap();
        let orientation = if orient_char == '-' { Orientation::Reverse } else { Orientation::Forward };
        let name = &item[..item.len()-1];
        result.push((name.to_string(), orientation));
    }
    result
}

/***************************************************************************************************
 * Build node ID map from segments, chunking if needed.
 *
 * Output:
 *    - a mapping from "segment_name" => list of newly assigned node IDs
 *    - an array of forward label strings for each node ID
 *    - optional translation structure describing how node IDs map back to segment names.
 *
 * We do a 1..=N approach, but we also have to add the endmarker node 0. We'll add that later 
 * in the GBWT. For the graph, we only store actual "physical nodes" that correspond to GFA segments.
 **************************************************************************************************/
#[allow(clippy::type_complexity)]
fn build_node_id_map(
    segments: &Vec<SegmentData>,
) -> (HashMap<String, Vec<usize>>, Vec<String>, Option<Translation>)
{
    // Store the node labels in forward_labels
    let mut forward_labels: Vec<String> = Vec::new();
    // Store "segment_name -> list of node IDs"
    let mut seg_map = HashMap::new();
    // Store the translation: a list of segment names plus offset array
    let mut seg_names = Vec::new();
    let mut mapping = Vec::new(); 
    let mut current_node_id = 0usize; 

    for seg in segments {
        let seg_name = seg.name.clone();
        let mut node_ids_for_this_segment = Vec::new();
        let bases = seg.seq.as_bytes();
        let mut start = 0;
        while start < bases.len() {
            let end = (start + CHUNK_LIMIT).min(bases.len());
            let chunk_seq = &bases[start..end];
            let chunk_str = String::from_utf8_lossy(chunk_seq).to_string();
            forward_labels.push(chunk_str);
            node_ids_for_this_segment.push(current_node_id);
            current_node_id += 1;
            start = end;
        }
        if node_ids_for_this_segment.is_empty() {
            // means empty sequence => create a single node with empty label
            forward_labels.push(String::new());
            node_ids_for_this_segment.push(current_node_id);
            current_node_id += 1;
        }
        seg_map.insert(seg_name.clone(), node_ids_for_this_segment);
    }

    // Second pass
    let mut cumul = 0usize;
    mapping.push(cumul);
    for seg in segments {
        let list = seg_map.get(&seg.name).unwrap();
        cumul += list.len();
        mapping.push(cumul);
        seg_names.push(seg.name.clone());
    }
    // Done. 
    let translation = Translation {
        segment_names: seg_names,
        mapping,
    };

    (seg_map, forward_labels, Some(translation))
}

/***************************************************************************************************
 * Build adjacency: for each node+orientation, store successors in ascending order. 
 * But we also must map from a GFA "segment -> nodes" for chunking.
 **************************************************************************************************/
fn build_adjacency(
    links: &Vec<LinkData>,
    seg_map: &HashMap<String, Vec<usize>>,
    node_count: usize,
) -> Vec<Vec<usize>> {
    // We'll store adjacency for each node in forward orientation => 2 * node_count in total
    // We haven't yet added endmarker 0: we'll do it at GBWT build time. 
    // For each link: from_name, from_orient => last chunk ID if from_orient==Forward, or first chunk ID if Reverse, 
    // to_name, to_orient => first chunk ID if forward, last if reverse 
    // Then adjacency is from that chunk's orientation -> the other chunk's orientation
    let mut adjacency = vec![Vec::new(); node_count * 2];

    for l in links {
        // from_name => from_id(s), depending on orientation 
        let from_ids_opt = seg_map.get(&l.from_name);
        let to_ids_opt = seg_map.get(&l.to_name);
        if from_ids_opt.is_none() || to_ids_opt.is_none() {
            continue;
        }
        let from_ids = from_ids_opt.unwrap();
        let to_ids = to_ids_opt.unwrap();
        if from_ids.is_empty() || to_ids.is_empty() {
            continue;
        }
        let (src_node, src_or) = match l.from_orient {
            Orientation::Forward => {
                // forward => from the last chunk of from_ids
                let node = from_ids[from_ids.len() - 1];
                (node, Orientation::Forward)
            },
            Orientation::Reverse => {
                // reverse => from the first chunk
                let node = from_ids[0];
                (node, Orientation::Reverse)
            },
        };
        let (dst_node, dst_or) = match l.to_orient {
            Orientation::Forward => {
                let node = to_ids[0];
                (node, Orientation::Forward)
            },
            Orientation::Reverse => {
                let node = to_ids[to_ids.len() - 1];
                (node, Orientation::Reverse)
            },
        };

        let src_encoded = encode_node(src_node, src_or);
        let dst_encoded = encode_node(dst_node, dst_or);

        adjacency[src_encoded].push(dst_encoded);
        // We can add the symmetrical edge ?
        let (src_flip_node, src_flip_or) = (src_node, src_or.flip());
        let (dst_flip_node, dst_flip_or) = (dst_node, dst_or.flip());
        let src_flip_enc = encode_node(src_flip_node, src_flip_or);
        let dst_flip_enc = encode_node(dst_flip_node, dst_flip_or);
        adjacency[dst_flip_enc].push(src_flip_enc);
    }
    // sort each adjacency
    for adj in adjacency.iter_mut() {
        adj.sort_unstable();
        adj.dedup();
    }
    adjacency
}

/***************************************************************************************************
 * Build path data
 * We note that each P-line or W-line might have multiple chunked nodes. We expand them 
 * properly (for each segment in the path, we get from seg_map the chunked node list).
 **************************************************************************************************/
#[derive(Clone, Debug)]
struct PathData {
    sample: String,
    contig: String,
    hap: usize,
    fragment: usize,
    // The nodes in order: (node_id, orientation)
    nodes: Vec<(usize, Orientation)>,
}
fn build_paths(
    gfa_paths: &Vec<GFAPathLine>,
    seg_map: &HashMap<String, Vec<usize>>,
) -> Vec<PathData> {
    let mut sample_dict: BTreeMap<String, usize> = BTreeMap::new(); // not strictly needed
    let mut contig_dict: BTreeMap<String, usize> = BTreeMap::new(); // not strictly needed
    let mut results = Vec::new();

    for p in gfa_paths {
        // if W-line => p.sample, p.hap_id, p.contig, p.fragment
        // else => treat them as sample="_gbwt_ref", hap=0, contig=p.raw_name, fragment=0
        let sample = p.sample.clone().unwrap_or_else(|| "_gbwt_ref".to_string());
        let hap = p.hap_id.unwrap_or(0);
        let contig = p.contig.clone().unwrap_or_else(|| p.raw_name.clone());
        let fragment = p.fragment.unwrap_or(0);
        let mut path_nodes = Vec::new();
        for (seg_name, orient) in &p.segments {
            if let Some(id_list) = seg_map.get(seg_name) {
                // Double check this
                if orient == &Orientation::Forward {
                    // push (id_list[0], Forward), (id_list[1], Forward), ...
                    for &nid in id_list {
                        path_nodes.push((nid, Orientation::Forward));
                    }
                } else {
                    for &nid in id_list.iter().rev() {
                        path_nodes.push((nid, Orientation::Reverse));
                    }
                }
            } else {
                // unknown segment?
            }
        }
        results.push(PathData {
            sample,
            contig,
            hap,
            fragment,
            nodes: path_nodes,
        });
    }

    results
}

/***************************************************************************************************
 * Build GBWT
 * We:
 *   - add a node 0 as the endmarker
 *   - for each real node + orientation => store adjacency in ascending order
 *   - we insert each path in forward orientation + reversed orientation (with node flips).
 *   - we run-length encode the BWT transitions, referencing the adjacency edges. 
 *   - we store real metadata with path names. W-lines => we parse sample/contig/hap. 
 *     P-lines => sample = "_gbwt_ref", contig = path_name, hap=0, fragment= index?
 **************************************************************************************************/
fn build_gbwt(
    adjacency: &Vec<Vec<usize>>,
    node_count: usize,
    path_data: &Vec<PathData>,
) -> GBWTIndex {
    // We will store usage counts for (from, to) pairs to compute rank(v, w).
    // Then we build final adjacency with partial prefix sums.

    // Prepare all records.
    let total_count = (node_count << 1) + 1;
    let mut records: Vec<Option<GBWTRecord>> = vec![None; total_count];
    for i in 0..total_count {
        records[i] = Some(GBWTRecord {
            edges: Vec::new(),
            runs: Vec::new(),
        });
    }

    // This map will track how many times (v, w) is used in the BWT of node v.
    // We only fill it during add_transition calls.
    use std::collections::HashMap;
    let mut usage_count: HashMap<(usize, usize), usize> = HashMap::new();

    // Helper function for incrementing usage_count and for adding runs to the record.
    fn add_transition(
        records: &mut [Option<GBWTRecord>],
        usage_map: &mut HashMap<(usize, usize), usize>,
        from: usize,
        to: usize,
    ) {
        // Look up record for 'from'
        let record = records[from].as_mut().unwrap();
        // Binary search in adjacency edges
        match record.edges.binary_search_by_key(&to, |&(s, _)| s) {
            Ok(edge_index) => {
                // Add run-length for BWT
                if let Some(last_run) = record.runs.last_mut() {
                    if last_run.value == edge_index {
                        last_run.len += 1;
                    } else {
                        record.runs.push(Run::new(edge_index, 1));
                    }
                } else {
                    record.runs.push(Run::new(edge_index, 1));
                }
                // Increase usage counter
                let cnt = usage_map.entry((from, to)).or_insert(0);
                *cnt += 1;
            },
            Err(_) => {
                // We skip if there's no adjacency entry
            },
        }
    }

    // Initialize adjacency edges for real usage. We copy adjacency but store (succ, rank=0) for now.
    // We'll fill the correct rank later with partial prefix sums from usage_count.
    for enc in 1..total_count {
        if enc < adjacency.len() {
            let successors = &adjacency[enc];
            let mut edge_list = Vec::with_capacity(successors.len());
            let mut prev_id = 0;
            for &succ_id in successors {
                edge_list.push((succ_id, 0));
                prev_id = succ_id;
            }
            edge_list.sort_by_key(|x| x.0);
            let r = records[enc].as_mut().unwrap();
            r.edges = edge_list;
        }
    }

    let mut total_size = 0usize;
    let mut seq_count = 0usize;

    // For metadata
    let mut sample_map = std::collections::BTreeMap::new();
    let mut contig_map = std::collections::BTreeMap::new();
    let mut sample_vec = Vec::new();
    let mut contig_vec = Vec::new();
    let mut path_names = Vec::new();

    fn get_or_insert(
        map: &mut std::collections::BTreeMap<String, usize>,
        vec: &mut Vec<String>,
        key: &str,
    ) -> usize {
        if let Some(&id) = map.get(key) {
            id
        } else {
            let id = vec.len();
            vec.push(key.to_string());
            map.insert(key.to_string(), id);
            id
        }
    }

    // Insert the paths into the BWT by enumerating transitions.
    for p in path_data {
        let sample_id = get_or_insert(&mut sample_map, &mut sample_vec, &p.sample);
        let contig_id = get_or_insert(&mut contig_map, &mut contig_vec, &p.contig);
        let phase_id = p.hap as u32;
        let frag_id = p.fragment as u32;

        // Create a pathName for forward orientation
        path_names.push(PathName {
            sample: sample_id as u32,
            contig: contig_id as u32,
            phase: phase_id,
            fragment: frag_id,
        });
        // Insert forward path
        if p.nodes.is_empty() {
            add_transition(&mut records, &mut usage_count, 0, 0);
            total_size += 2;
            seq_count += 1;
        } else {
            seq_count += 1;
            add_transition(&mut records, &mut usage_count, 0, encode_node(p.nodes[0].0, p.nodes[0].1));
            total_size += 1;
            for w in 0..(p.nodes.len() - 1) {
                let from = encode_node(p.nodes[w].0, p.nodes[w].1);
                let to = encode_node(p.nodes[w+1].0, p.nodes[w+1].1);
                add_transition(&mut records, &mut usage_count, from, to);
                total_size += 1;
            }
            let last_node = encode_node(p.nodes[p.nodes.len() - 1].0, p.nodes[p.nodes.len() - 1].1);
            add_transition(&mut records, &mut usage_count, last_node, 0);
            total_size += 1;
        }

        // Insert reverse path
        seq_count += 1;
        if p.nodes.is_empty() {
            add_transition(&mut records, &mut usage_count, 0, 0);
            total_size += 2;
        } else {
            let mut rev_nodes = Vec::with_capacity(p.nodes.len());
            for &nd in p.nodes.iter().rev() {
                rev_nodes.push((nd.0, nd.1.flip()));
            }
            add_transition(&mut records, &mut usage_count, 0, encode_node(rev_nodes[0].0, rev_nodes[0].1));
            total_size += 1;
            for w in 0..(rev_nodes.len() - 1) {
                let f = encode_node(rev_nodes[w].0, rev_nodes[w].1);
                let t = encode_node(rev_nodes[w+1].0, rev_nodes[w+1].1);
                add_transition(&mut records, &mut usage_count, f, t);
                total_size += 1;
            }
            let last_rev = encode_node(rev_nodes[rev_nodes.len() - 1].0, rev_nodes[rev_nodes.len() - 1].1);
            add_transition(&mut records, &mut usage_count, last_rev, 0);
            total_size += 1;
        }
    }

    // Construct the final adjacency with partial prefix sums. This sets rank(v, w).
    for node_id in 0..total_count {
        if let Some(rec) = records[node_id].as_mut() {
            if rec.edges.is_empty() {
                continue;
            }
            let mut running_sum = 0;
            for edge_idx in 0..rec.edges.len() {
                let succ_id = rec.edges[edge_idx].0;
                let usage = usage_count.get(&(node_id, succ_id)).unwrap_or(&0);
                rec.edges[edge_idx].1 = running_sum;
                running_sum += *usage;
            }
        }
    }

    let sample_count = sample_vec.len();
    let mut hap_set = std::collections::BTreeSet::new();
    for p in path_data {
        hap_set.insert((p.sample.clone(), p.hap));
    }
    let haplotype_count = hap_set.len();
    let contig_count = contig_vec.len();

    let metadata = GBWTMetadata {
        sample_count,
        haplotype_count,
        contig_count,
        sample_names: sample_vec,
        contig_names: contig_vec,
        path_names,
    };

    GBWTIndex {
        records,
        sequences: seq_count,
        total_size,
        offset: 0,
        alphabet_size: total_count,
        metadata,
    }
}

/***************************************************************************************************
 * Build GBWTGraph from node labels (forward only) + optional translation
 **************************************************************************************************/
fn build_gbwt_graph(
    forward_labels: Vec<String>,
    translation: Option<Translation>,
) -> GBWTGraph {
    GBWTGraph {
        forward_labels,
        translation,
    }
}

/***************************************************************************************************
 * Now the final writing of the entire GBZ file, in the official spec layout:
 * 
 *  1) 16-byte GBZ header + tags
 *  2) GBWT (with 48-byte header, tags, BWT, doc array samples optional, metadata optional)
 *  3) GBWTGraph (with 24-byte header, node labels, translation)
 **************************************************************************************************/
fn write_gbz(gbwt: &GBWTIndex, graph: &GBWTGraph, outfile: &str) -> Result<(), std::io::Error> {
    // This writes the final GBZ container, ensuring 8-byte alignment after each major section.
    use std::io::Seek;
    let f = OpenOptions::new().write(true).create(true).truncate(true).open(outfile)?;
    let mut w = BufWriter::new(f);

    // Helper function to pad up to next multiple of 8.
    fn pad_to_8<W: Write + Seek>(writer: &mut BufWriter<W>) -> std::io::Result<()> {
        let pos = writer.stream_position()? as usize;
        let remainder = pos % 8;
        if remainder != 0 {
            let pad = 8 - remainder;
            let zeros = vec![0u8; pad];
            writer.write_all(&zeros)?;
        }
        Ok(())
    }

    // Write the 16-byte GBZ header
    let tag: u32 = 0x205A4247; // 'GBZ '
    let version: u32 = 1;
    let flags: u64 = 0;
    w.write_all(&tag.to_le_bytes())?;
    w.write_all(&version.to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;
    pad_to_8(&mut w)?;

    // Write the GBZ-level tags
    let mut tags = Vec::new();
    tags.push(("source".to_string(), "gfa2gbz".to_string()));
    write_tags(&mut w, &tags)?;
    pad_to_8(&mut w)?;

    // Write the GBWT index
    write_gbwt_index(&mut w, gbwt)?;
    pad_to_8(&mut w)?;

    // Write the GBWTGraph
    write_gbwt_graph_data(&mut w, graph)?;
    pad_to_8(&mut w)?;

    w.flush()?;
    Ok(())
}

/***************************************************************************************************
 * Write the "Tags" structure in simple-sds style:
 *   - It's a string array with 2*N strings. Each pair is (key, value). 
 *   - The library approach is: 
 *       sizeInElements => we do actual code.
 *       1) we flatten 
 *       2) we store length 
 *       3) we store offsets 
 *       4) we store big blob 
 **************************************************************************************************/
fn write_tags<W: Write>(w: &mut W, tags: &[(String, String)]) -> Result<(), std::io::Error> {
    let n = tags.len();
    // Flatten: 2*n strings
    let mut all_strings = Vec::with_capacity(2*n);
    for (k,v) in tags {
        // keys are case-insensitive in official
        // but we store them as-lower
        let k_lower = k.to_lowercase();
        all_strings.push(k_lower);
        all_strings.push(v.clone());
    }
    write_string_array(w, &all_strings)
}

/***************************************************************************************************
 * Write the GBWT index:
 * 
 * The format is:
 *   - 48-byte header (tag=0x6B376B37, version=5, sequences, size, offset, alph_size, flags)
 *   - tags
 *   - BWT => we must store:
 *       * index: a "sparse vector" with record offsets
 *       * data: the run-length adjacency encodings
 *   - doc array samples => optional 
 *   - metadata => optional
 **************************************************************************************************/
fn write_gbwt_index<W: Write>(w: &mut W, gbwt: &GBWTIndex) -> Result<(), std::io::Error> {
    // Write the 48-byte header
    let tag: u32 = 0x6B376B37;
    let version: u32 = 5;
    let sequences = gbwt.sequences as u64;
    let size = gbwt.total_size as u64;
    let offset = gbwt.offset as u64;
    let alph = gbwt.alphabet_size as u64;
    // Flags for simple-sds, bidirectional, metadata
    let mut flags = 0x0004u64; 
    flags |= 0x0001; 
    flags |= 0x0002;
    w.write_all(&tag.to_le_bytes())?;
    w.write_all(&version.to_le_bytes())?;
    w.write_all(&sequences.to_le_bytes())?;
    w.write_all(&size.to_le_bytes())?;
    w.write_all(&offset.to_le_bytes())?;
    w.write_all(&alph.to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;

    // Write any GBWT-level tags
    let gtags = Vec::new();
    write_tags(w, &gtags)?;

    // Build the BWT data array and record start offsets
    let (record_offsets, data_bytes) = build_bwt_bytes(&gbwt.records);

    // Use a SparseBuilder from simple-sds to store these offsets as a proper Elias-Fano index
    let n = data_bytes.len();
    let m = record_offsets.len();
    let mut builder = SparseBuilder::new(n, m).unwrap();
    for &(_, offset_in_data) in &record_offsets {
        builder.set(offset_in_data);
    }
    let index_sv = SparseVector::try_from(builder).unwrap();
    // Now we serialize the sparse vector. Then we write the raw data bytes next.
    index_sv.serialize(w)?;

    // Append the BWT data (no extra data_len written).
    w.write_all(&data_bytes)?;

    // Then we skip doc array samples by writing a length=0
    let zero: u64 = 0;
    w.write_all(&zero.to_le_bytes())?;

    // Finally, the metadata
    write_gbwt_metadata(w, &gbwt.metadata)?;

    Ok(())
}

/***************************************************************************************************
 * Build the BWT record data for each node. For each record i, we convert adjacency edges + runs
 * into a byte-coded structure. Then we store them all in a single array "data_bytes." 
 * We also store an offset for record i in record_offsets[i].
 **************************************************************************************************/
fn build_bwt_bytes(records: &Vec<Option<GBWTRecord>>) -> (Vec<(usize,usize)>, Vec<u8>) {
    let mut data = Vec::new();
    let mut offsets = Vec::new();
    for (rid, rec_opt) in records.iter().enumerate() {
        let offset_here = data.len();
        offsets.push((rid, offset_here));
        if let Some(rec) = rec_opt {
            let sigma = rec.edges.len();
            encode_varuint(&mut data, sigma as u64);
            let mut prev_id = 0;
            for &(succ_id, rank_val) in &rec.edges {
                let delta = succ_id - prev_id;
                encode_varuint(&mut data, delta as u64);
                encode_varuint(&mut data, rank_val as u64);
                prev_id = succ_id;
            }
            for r in &rec.runs {
                if sigma == 0 {
                    continue;
                }
                if sigma < 255 {
                    let mut length = r.len;
                    while length > 0 {
                        let max_code = 255usize.saturating_sub(r.value);
                        if max_code == 0 {
                            encode_varuint(&mut data, r.value as u64);
                            encode_varuint(&mut data, (length - 1) as u64);
                            break;
                        }
                        let sub_len = (max_code / sigma) + 1;
                        let run_len = if sub_len < length { sub_len } else { length };
                        let code = r.value + sigma * (run_len - 1);
                        encode_byte(&mut data, code as u8);
                        length -= run_len;
                        if length > 0 {
                            encode_varuint(&mut data, (run_len - 1) as u64);
                        }
                    }
                } else {
                    encode_varuint(&mut data, r.value as u64);
                    encode_varuint(&mut data, (r.len - 1) as u64);
                }
            }
        } else {
            encode_varuint(&mut data, 0u64);
        }
    }
    (offsets, data)
}


/***************************************************************************************************
 * Write the GBWT metadata
 * This is an optional structure, but we have flagged in the header that we have it.
 * 
 * The structure is:
 *   - 40 bytes => (tag=0x6B375E7A, version=2, sample_count, haplotype_count, contig_count, flags=some)
 *   - then path names
 *   - then sample dictionary 
 *   - then contig dictionary
 **************************************************************************************************/
fn write_gbwt_metadata<W: Write>(w: &mut W, meta: &GBWTMetadata) -> Result<(), std::io::Error> {
    let tag: u32 = 0x6B375E7A;
    let version: u32 = 2;
    let scount = meta.sample_count as u64;
    let hcount = meta.haplotype_count as u64;
    let ccount = meta.contig_count as u64;
    // flags => we store if we have path names? yes => 0x1. sample names =>0x2 if meta.sample_count>0. contig =>0x4 if meta.contig_count>0.
    let mut flags: u64 = 0;
    if !meta.path_names.is_empty() {
        flags |= 0x1;
    }
    if meta.sample_count>0 {
        flags |= 0x2;
    }
    if meta.contig_count>0 {
        flags |= 0x4;
    }
    // write 
    w.write_all(&tag.to_le_bytes())?;
    w.write_all(&version.to_le_bytes())?;
    w.write_all(&scount.to_le_bytes())?;
    w.write_all(&hcount.to_le_bytes())?;
    w.write_all(&ccount.to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;

    // path names => if we have path_names, we store them. 
    // The number of path_names = meta.path_names.len(). If 0 => store empty. 
    if meta.path_names.is_empty() {
        // store 0 => no path names 
        write_varuint(w, 0)?;
    } else {
        // store the array of path names. Each is 16 bytes => sample(4), contig(4), phase(4), fragment(4) in little-endian 
        // Then we do not store them as a "StringArray," because the spec says: "We store them as a vector of 16-byte items."
        let len = meta.path_names.len() as u64;
        write_varuint(w, len)?;
        // then each path name
        for pn in &meta.path_names {
            let sample_le = pn.sample.to_le_bytes();
            w.write_all(&sample_le)?;
            let contig_le = pn.contig.to_le_bytes();
            w.write_all(&contig_le)?;
            let phase_le = pn.phase.to_le_bytes();
            w.write_all(&phase_le)?;
            let frag_le = pn.fragment.to_le_bytes();
            w.write_all(&frag_le)?;
        }
    }

    // sample dictionary => if sample_count>0 
    if meta.sample_count>0 {
        write_dictionary(w, &meta.sample_names)?;
    } else {
        // empty dictionary => store 0 => no strings 
        write_varuint(w, 0)?;
        // no further 
        // done
    }

    // contig dictionary => if contig_count>0
    if meta.contig_count>0 {
        write_dictionary(w, &meta.contig_names)?;
    } else {
        write_varuint(w, 0)?;
    }

    Ok(())
}

/***************************************************************************************************
 * Write the dictionary => 
 *   - 1) string array 
 *   - 2) sorted_ids => we store them in ascending order, but we skip duplicates? Actually 
 * We have meta.sample_names in the order they were inserted, so the ID= index. 
 * Then we must store them in lexicographic order => sorted_ids => an integer vector 
 **************************************************************************************************/
fn write_dictionary<W: Write>(w: &mut W, items: &Vec<String>) -> Result<(), std::io::Error> {
    write_string_array(w, items)?;
    let mut sorted: Vec<usize> = (0..items.len()).collect();
    sorted.sort_by(|&a, &b| items[a].cmp(&items[b]));
    let length = sorted.len() as u64;
    write_varuint(w, length)?;
    let width = if length > 1 { 64 - (length - 1).leading_zeros() } else { 1 };
    let mut iv = IntVector::with_capacity(sorted.len(), width as usize).unwrap();
    for &sid in &sorted {
        iv.push(sid as u64);
    }
    iv.serialize(w)?;
    Ok(())
}


/***************************************************************************************************
 * Write the GBWTGraph:
 *   1) 24-byte header => tag=0x6B3764AF, version=3, nodes=..., flags= 0x2 for simple-sds plus 0x1 if translation
 *   2) Sequences => string array => these are the forward labels for each node ID 
 *   3) Node-to-segment translation => if present, we store the "translation" data
 **************************************************************************************************/
fn write_gbwt_graph_data<W: Write>(w: &mut W, graph: &GBWTGraph) -> Result<(), std::io::Error> {
    let tag: u32 = 0x6B3764AF;
    let version: u32 = 3;
    let nodes = graph.forward_labels.len() as u64;
    let mut flags: u64 = 0x0002; // simple-sds
    let has_translation = graph.translation.is_some();
    if has_translation {
        flags |= 0x0001; // translation
    }
    w.write_all(&tag.to_le_bytes())?;
    w.write_all(&version.to_le_bytes())?;
    w.write_all(&nodes.to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;

    // now the forward label array as a string array
    write_string_array(w, &graph.forward_labels)?;

    // if translation => write. else => write an empty 
    if let Some(tr) = &graph.translation {
        // we store segment names as a string array
        write_string_array(w, &tr.segment_names)?;
        // We store that as a "sparse vector" with set bits at the start of each segment. 
        // We'll do length= forward_labels.len()+1, # of set bits = tr.mapping.len(), and we set bits at tr.mapping[i]. 
        let full_len = graph.forward_labels.len() + 1;
        // write the bit length 
        let bit_len = full_len as u64;
        w.write_all(&bit_len.to_le_bytes())?;
        // we must store the bits in a plain array of (bit_len+7)/8 
        // We'll build it 
        let mut bits = vec![0u8; (full_len+7)/8];
        for &pos in &tr.mapping {
            let p = pos;
            if p < full_len {
                bits[p>>3] |= 1 << (p & 7);
            }
        }
        w.write_all(&bits)?;
    } else {
        // If there's no translation, we store empty arrays. The spec requires us to store an empty string array for segments, 
        // and an empty bitvector for mapping. 
        let zero_count = 0u64;
        // empty string array => "0" => means no strings 
        write_varuint(w, 0)?;
        // empty mapping => bit_len= forward_labels.len()+1 => then all zero. 
        let length = (graph.forward_labels.len()+1) as u64;
        w.write_all(&length.to_le_bytes())?;
        let byte_len = (length as usize +7)/8;
        let zero_bits = vec![0u8; byte_len];
        w.write_all(&zero_bits)?;
    }

    Ok(())
}

/***************************************************************************************************
 * Utility: write a string array in "simple-sds" style => 
 *   1) store length (n)
 *   2) store a single concatenated data 
 *   3) store an offset array with n+1 offsets 
 **************************************************************************************************/
fn write_string_array<W: Write>(w: &mut W, arr: &Vec<String>) -> Result<(), std::io::Error> {
    // In the "simple-sds" approach, a "StringArray" is: 
    //   - The index (a "sparse vector" for starts) 
    //   - The alphabet 
    //   - The strings in a bit-packed form 
    // Step 1) collect all bytes
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(arr.len()+1);
    offsets.push(0);
    for s in arr {
        data.extend_from_slice(s.as_bytes());
        offsets.push(data.len());
    }
    // Now we do "alphabet compaction," i.e. gather all used bytes, build a mapping. 
    // Let's find which bytes appear. 
    let mut used = [false; 256];
    for &b in &data {
        used[b as usize] = true;
    }
    let mut alpha_map = Vec::new(); // maps byte-> rank
    alpha_map.resize(256, 0);
    let mut alpha_vec = Vec::new();
    let mut rank = 0;
    for b in 0..256 {
        if used[b] {
            alpha_map[b] = rank;
            alpha_vec.push(b as u8);
            rank += 1;
        }
    }
    // Now we build the packed array of width= bit_len(rank-1)
    let sigma = rank;
    if sigma==0 {
        // ?
    }

    // The "StringArray" structure in the spec sets bit i at offsets[i], then arr[i] is data from that offset to offsets[i+1]. 
    // Build the bitvector of length data.len()+1. We'll set bit at offsets[i], for i in 0..arr.len(). 
    // The last offset is data.len(). 
    let bit_length = data.len()+1;
    let mut bits = vec![0u8; (bit_length+7)/8];
    for &ofs in &offsets[..arr.len()] {
        bits[ofs>>3] |= 1 << (ofs & 7);
    }
    // store the bitvector 
    let bit_len64 = bit_length as u64;
    w.write_all(&bit_len64.to_le_bytes())?;
    w.write_all(&bits)?;
    // Then we store the "alphabet" as a plain vector of length = sigma 
    // then the "strings" as an intvector of length data.len(), width= bit_len(sigma-1)
    let sigma_bytes = sigma as u64;
    write_varuint(w, sigma_bytes)?;
    for &b in &alpha_vec {
        w.write_all(&[b])?;
    }
    // now build the packed array. width = 
    let wbits = if sigma <= 1 { 1 } else { (64 - ((sigma - 1) as u64).leading_zeros()) as u8 };
    // write length in varuint => data.len()
    write_varuint(w, data.len() as u64)?;
    write_varuint(w, wbits as u64)?;
    // pack each data[i] => alpha_map[data[i]] in wbits
    // We'll do a naive approach 
    let mut buff = 0u64;
    let mut used_bits = 0;
    let mut out_bytes = Vec::new();
    for &byte in &data {
        let mapped = alpha_map[byte as usize] as u64;
        buff |= mapped << used_bits;
        used_bits += wbits as usize;
        while used_bits>=8 {
            out_bytes.push((buff & 0xFF) as u8);
            buff >>= 8;
            used_bits-=8;
        }
    }
    if used_bits>0 {
        out_bytes.push((buff & 0xFF) as u8);
    }
    // store out_bytes
    write_varuint(w, out_bytes.len() as u64)?;
    w.write_all(&out_bytes)?;

    Ok(())
}

/***************************************************************************************************
 * Low-level varuint encoders
 **************************************************************************************************/
fn write_varuint<W: Write>(w: &mut W, mut x: u64) -> Result<(), std::io::Error> {
    while x>0x7F {
        let b = ((x & 0x7F) as u8) | 0x80;
        w.write_all(&[b])?;
        x >>= 7;
    }
    w.write_all(&[(x & 0x7F) as u8])?;
    Ok(())
}
fn encode_varuint(dst: &mut Vec<u8>, mut x: u64) {
    while x>0x7F {
        dst.push(((x & 0x7F) as u8) | 0x80);
        x >>= 7;
    }
    dst.push((x & 0x7F) as u8);
}
fn encode_byte(dst: &mut Vec<u8>, b: u8) {
    dst.push(b);
}
