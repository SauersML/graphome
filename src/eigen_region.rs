// src/eigen_region.rs
// Region-based eigendecomposition: extract subgraph from GBZ by genomic coordinates

use crate::eigen_print::{
    call_eigendecomp, compute_ngec, print_eigenvalues_heatmap, print_heatmap_normalized,
};
use crate::map::{coord_to_nodes_mapped, make_gbz_exist, parse_region};
use crate::mapped_gbz::MappedGBZ;
use faer::Mat;
use gbwt::{Orientation, GBZ};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use simple_sds::serialize;
use std::error::Error;
use std::io;
use std::time::Instant;

/// Extract edges from GBZ for a specific subset of nodes
/// Matches the behavior of convert_gbz_to_edge_list: Forward orientation only, undirected
fn extract_subgraph_edges(gbz: &GBZ, node_ids: &FxHashSet<usize>) -> Vec<(usize, usize)> {
    if node_ids.is_empty() {
        return Vec::new();
    }

    eprintln!("[INFO] Extracting edges for {} nodes...", node_ids.len());

    let nodes: Vec<usize> = node_ids.iter().copied().collect();

    // Build local edge sets in parallel and merge them, avoiding contention.
    let edges: FxHashSet<(usize, usize)> = nodes
        .par_iter()
        .fold(FxHashSet::default, |mut local_edges, &node_id| {
            if let Some(mut succ_iter) = gbz.successors(node_id, Orientation::Forward) {
                while let Some((succ_id, _)) = succ_iter.next() {
                    if node_ids.contains(&succ_id) {
                        let edge = if node_id <= succ_id {
                            (node_id, succ_id)
                        } else {
                            (succ_id, node_id)
                        };
                        local_edges.insert(edge);
                    }
                }
            }
            local_edges
        })
        .reduce(FxHashSet::default, |mut acc, set| {
            if acc.capacity() < acc.len() + set.len() {
                acc.reserve(set.len());
            }
            for edge in set {
                acc.insert(edge);
            }
            acc
        });

    let edge_count = edges.len();
    eprintln!("[INFO] Found {} unique edges in subgraph", edge_count);

    let mut edge_vec: Vec<(usize, usize)> = edges.into_iter().collect();
    edge_vec.shrink_to_fit();
    edge_vec
}

/// Build Laplacian matrix directly from the edge list without materialising the adjacency
fn build_laplacian_matrix(edges: &[(usize, usize)], node_ids: &[usize]) -> Mat<f64> {
    let n = node_ids.len();
    eprintln!(
        "[INFO] Building Laplacian matrix for {} nodes ({} edges)...",
        n,
        edges.len()
    );

    let mut laplacian = Mat::zeros(n, n);

    if n == 0 {
        return laplacian;
    }

    // Create mapping: node_id -> matrix_index
    let id_to_idx: FxHashMap<usize, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();

    let mut degrees = vec![0.0f64; n];
    let mut self_loops = vec![0.0f64; n];

    for &(u, v) in edges {
        let (Some(&i), Some(&j)) = (id_to_idx.get(&u), id_to_idx.get(&v)) else {
            continue;
        };

        if i == j {
            degrees[i] += 1.0;
            self_loops[i] += 1.0;
        } else {
            degrees[i] += 1.0;
            degrees[j] += 1.0;
            laplacian[(i, j)] -= 1.0;
            laplacian[(j, i)] -= 1.0;
        }
    }

    for (idx, (&degree, &self_loop)) in degrees.iter().zip(self_loops.iter()).enumerate() {
        laplacian[(idx, idx)] = degree - self_loop;
    }

    laplacian
}

/// Main entry point for region-based eigendecomposition
pub fn run_eigen_region(gfa_path: &str, region: &str, viz: bool) -> Result<(), Box<dyn Error>> {
    eprintln!("[INFO] Starting region-based eigendecomposition");
    eprintln!("[INFO] Region: {}", region);

    // Parse region
    let (chr, start, end) =
        parse_region(region).ok_or_else(|| format!("Invalid region format: {}", region))?;

    eprintln!("[INFO] Parsed region: {}:{}-{}", chr, start, end);

    // Get or create GBZ file (handles S3 URLs, local copies, etc.)
    let gbz_path = make_gbz_exist(gfa_path, "");
    eprintln!("[INFO] Using GBZ: {}", gbz_path);

    // Load GBZ with memory mapping for coord2node
    eprintln!("[INFO] Loading GBZ with memory mapping for coordinate lookup...");
    let mapped_gbz = MappedGBZ::new(&gbz_path)?;
    eprintln!("[INFO] GBZ loaded successfully");

    // Find nodes in the region using coord2node
    eprintln!("[INFO] Finding nodes in region...");
    let results = coord_to_nodes_mapped(&mapped_gbz, &chr, start, end);

    if results.is_empty() {
        return Err(format!("No nodes found in region {}:{}-{}", chr, start, end).into());
    }

    // Extract unique node IDs
    let mut node_ids: FxHashSet<usize> = FxHashSet::default();
    node_ids.reserve(results.len());
    node_ids.extend(
        results
            .iter()
            .filter_map(|r| r.node_id.parse::<usize>().ok()),
    );

    eprintln!("[INFO] Found {} unique nodes in region", node_ids.len());

    // Load full GBZ for edge extraction (need successors API)
    eprintln!("[INFO] Loading full GBZ for edge extraction...");
    let gbz: GBZ = serialize::load_from(&gbz_path).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to load GBZ: {}", e),
        )
    })?;
    eprintln!("[INFO] Full GBZ loaded");

    // Extract subgraph edges
    let edges = extract_subgraph_edges(&gbz, &node_ids);

    if edges.is_empty() {
        eprintln!("[WARNING] No edges found between nodes in region");
        eprintln!("[WARNING] Subgraph may be disconnected or have isolated nodes");
    }

    // Sort node IDs for consistent matrix indexing
    let mut sorted_nodes: Vec<usize> = node_ids.into_iter().collect();
    sorted_nodes.sort_unstable();
    let laplacian_start = Instant::now();
    let laplacian = build_laplacian_matrix(&edges, &sorted_nodes);
    eprintln!(
        "[INFO] Laplacian matrix built in {:.3?}",
        laplacian_start.elapsed()
    );

    // Perform eigendecomposition
    eprintln!("[INFO] Performing eigendecomposition...");
    let eig_start = Instant::now();
    let (eigenvalues, _eigenvectors) = call_eigendecomp(laplacian.as_ref())?;
    eprintln!(
        "[INFO] Eigendecomposition complete in {:.3?}",
        eig_start.elapsed()
    );

    // Compute NGEC
    let ngec = compute_ngec(&eigenvalues)?;

    // Print results
    println!("\n=== EIGENANALYSIS RESULTS ===");
    println!("Region: {}:{}-{}", chr, start, end);
    println!("Nodes: {}", sorted_nodes.len());
    println!("Edges: {}", edges.len());
    println!("Eigenvalues: {}", eigenvalues.len());
    println!("NGEC: {:.6}", ngec);

    // Print top eigenvalues
    println!("\nTop 10 Eigenvalues:");
    for (i, &val) in eigenvalues.iter().take(10).enumerate() {
        println!("  λ{} = {:.6}", i, val);
    }

    // Visualization if requested
    if viz {
        println!("\n=== LAPLACIAN HEATMAP ===");
        print_heatmap_normalized(laplacian.as_ref());

        println!("\n=== EIGENVALUE DISTRIBUTION ===");
        print_eigenvalues_heatmap(&eigenvalues);
    }

    Ok(())
}
