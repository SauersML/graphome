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
use rayon::slice::ParallelSliceMut;
use rustc_hash::FxHashMap;
use simple_sds::serialize;
use std::error::Error;
use std::io;
use std::time::Instant;

/// Extract edges from GBZ for a specific subset of nodes
/// Matches the behavior of convert_gbz_to_edge_list: Forward orientation only, undirected
fn extract_subgraph_edges(
    gbz: &GBZ,
    node_ids: &[usize],
    id_to_idx: &FxHashMap<usize, usize>,
) -> Vec<(usize, usize)> {
    if node_ids.is_empty() {
        return Vec::new();
    }

    eprintln!("[INFO] Extracting edges for {} nodes...", node_ids.len());

    let mut edges: Vec<(usize, usize)> = node_ids
        .par_iter()
        .fold(
            || Vec::new(),
            |mut local_edges, &node_id| {
                let Some(&src_idx) = id_to_idx.get(&node_id) else {
                    return local_edges;
                };

                if let Some(mut succ_iter) = gbz.successors(node_id, Orientation::Forward) {
                    while let Some((succ_id, _)) = succ_iter.next() {
                        let Some(&dst_idx) = id_to_idx.get(&succ_id) else {
                            continue;
                        };

                        let edge = if src_idx <= dst_idx {
                            (src_idx, dst_idx)
                        } else {
                            (dst_idx, src_idx)
                        };
                        local_edges.push(edge);
                    }
                }

                local_edges
            },
        )
        .reduce(
            || Vec::new(),
            |mut acc, mut local| {
                acc.append(&mut local);
                acc
            },
        );

    if edges.is_empty() {
        return edges;
    }

    edges.par_sort_unstable();
    edges.dedup();
    edges.shrink_to_fit();
    eprintln!("[INFO] Found {} unique edges in subgraph", edges.len());
    edges
}

/// Build Laplacian matrix directly from the edge list without materialising the adjacency
fn build_laplacian_matrix(edges: &[(usize, usize)], node_count: usize) -> Mat<f64> {
    let n = node_count;
    eprintln!(
        "[INFO] Building Laplacian matrix for {} nodes ({} edges)...",
        n,
        edges.len()
    );

    let mut laplacian = Mat::zeros(n, n);

    if n == 0 {
        return laplacian;
    }

    let mut degrees = vec![0.0f64; n];

    for &(u, v) in edges {
        if u == v {
            continue;
        }

        degrees[u] += 1.0;
        degrees[v] += 1.0;
        laplacian[(u, v)] -= 1.0;
        laplacian[(v, u)] -= 1.0;
    }

    for (idx, &degree) in degrees.iter().enumerate() {
        laplacian[(idx, idx)] = degree;
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
    // start and end are already 0-based half-open from parse_region
    eprintln!("[INFO] Finding nodes in region...");

    let results = coord_to_nodes_mapped(&mapped_gbz, &chr, start, end);

    if results.is_empty() {
        return Err(format!("No nodes found in region {}:{}-{}", chr, start, end).into());
    }

    // Extract unique node IDs
    let mut sorted_nodes: Vec<usize> = Vec::with_capacity(results.len());
    for record in &results {
        if let Ok(id) = record.node_id.parse::<usize>() {
            sorted_nodes.push(id);
        }
    }

    sorted_nodes.sort_unstable();
    sorted_nodes.dedup();

    eprintln!("[INFO] Found {} unique nodes in region", sorted_nodes.len());

    if sorted_nodes.is_empty() {
        return Err(format!("No nodes found in region {}:{}-{}", chr, start, end).into());
    }

    let mut id_to_idx = FxHashMap::with_capacity_and_hasher(sorted_nodes.len(), Default::default());
    for (idx, &id) in sorted_nodes.iter().enumerate() {
        id_to_idx.insert(id, idx);
    }

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
    let edges = extract_subgraph_edges(&gbz, &sorted_nodes, &id_to_idx);

    if edges.is_empty() {
        eprintln!("[WARNING] No edges found between nodes in region");
        eprintln!("[WARNING] Subgraph may be disconnected or have isolated nodes");
    }

    let laplacian_start = Instant::now();
    let laplacian = build_laplacian_matrix(&edges, sorted_nodes.len());
    eprintln!(
        "[INFO] Laplacian matrix built in {:.3?}",
        laplacian_start.elapsed()
    );

    // Perform eigendecomposition
    eprintln!("[INFO] Performing eigendecomposition...");
    let eig_start = Instant::now();
    let (eigenvalues, eigenvectors) = call_eigendecomp(laplacian.as_ref())?;
    eprintln!(
        "[INFO] Eigendecomposition complete in {:.3?}",
        eig_start.elapsed()
    );
    eprintln!(
        "[INFO] Eigenvector matrix shape: {}x{}",
        eigenvectors.nrows(),
        eigenvectors.ncols()
    );

    // Compute NGEC
    let ngec = compute_ngec(&eigenvalues)?;

    // Print results
    println!("\n=== EIGENANALYSIS RESULTS ===");
    // Convert back to 1-based coordinates for display (parse_region returns 0-based)
    println!("Region: {}:{}-{}", chr, start + 1, end);
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
