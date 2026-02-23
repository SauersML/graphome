// src/eigen_region.rs
// Region-based eigendecomposition: extract subgraph from GBZ by genomic coordinates

use crate::eigen_print::{print_eigenvalues_heatmap, print_heatmap};
use crate::map::{coord_to_nodes_with_path_filtered, make_gbz_exist, parse_region};
use crate::sparse_spectral::{estimate_ngec_hutchinson, lanczos_smallest};
use faer::Mat;
use gbz::{Orientation, GBZ};
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

/// Main entry point for region-based eigendecomposition
pub fn run_eigen_region(gfa_path: &str, region: &str, viz: bool) -> Result<(), Box<dyn Error>> {
    eprintln!("[INFO] Starting region-based eigendecomposition");
    eprintln!("[INFO] Region: {}", region);

    // Parse region
    let (chr, start, end) =
        parse_region(region).ok_or_else(|| format!("Invalid region format: {}", region))?;
    let display_range = crate::coords::format_for_user(start, end);

    eprintln!(
        "[INFO] Parsed region: {}:{} (internal 0-based [{}..{}))",
        chr, display_range, start, end
    );

    // Get or create GBZ file (handles S3 URLs, local copies, etc.)
    let gbz_path = make_gbz_exist(gfa_path, "");
    eprintln!("[INFO] Using GBZ: {}", gbz_path);

    // Load full GBZ once and reuse it for both coordinate lookup and edge extraction.
    eprintln!("[INFO] Loading GBZ...");
    let gbz: GBZ = serialize::load_from(&gbz_path).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to load GBZ: {}", e),
        )
    })?;
    eprintln!("[INFO] GBZ loaded successfully");

    // Find nodes in the region using coord2node
    // start and end are already 0-based half-open from parse_region
    eprintln!("[INFO] Finding nodes in region...");
    let mut results = Vec::new();
    for assembly in ["grch38", "chm13", "hg38", "t2t", ""] {
        let candidate = coord_to_nodes_with_path_filtered(&gbz, assembly, &chr, start, end, None);
        if !candidate.is_empty() {
            if assembly.is_empty() {
                eprintln!("[INFO] Found nodes using permissive assembly fallback");
            } else {
                eprintln!("[INFO] Found nodes using assembly '{}'", assembly);
            }
            results = candidate;
            break;
        }
    }

    if results.is_empty() {
        return Err(format!("No nodes found in region {}:{}", chr, display_range).into());
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
        return Err(format!("No nodes found in region {}:{}", chr, display_range).into());
    }

    let mut id_to_idx = FxHashMap::with_capacity_and_hasher(sorted_nodes.len(), Default::default());
    for (idx, &id) in sorted_nodes.iter().enumerate() {
        id_to_idx.insert(id, idx);
    }

    // Extract subgraph edges
    let edges = extract_subgraph_edges(&gbz, &sorted_nodes, &id_to_idx);

    if edges.is_empty() {
        eprintln!("[WARNING] No edges found between nodes in region");
        eprintln!("[WARNING] Subgraph may be disconnected or have isolated nodes");
    }

    // Build sparse adjacency list and degree vector for operator-based methods.
    let n = sorted_nodes.len();
    let mut adjacency = vec![Vec::<usize>::new(); n];
    for &(u, v) in &edges {
        if u == v {
            continue;
        }
        adjacency[u].push(v);
        adjacency[v].push(u);
    }
    let degrees: Vec<f64> = adjacency.iter().map(|nbrs| nbrs.len() as f64).collect();
    let trace_l: f64 = degrees.iter().sum();
    let max_degree = degrees.iter().copied().fold(0.0f64, f64::max);
    let lambda_max = (2.0 * max_degree).max(1.0);

    // Compute approximate NGEC with Hutchinson + Chebyshev.
    let ngec_start = Instant::now();
    let ngec = estimate_ngec_hutchinson(n, trace_l, lambda_max, 64, 24, |x, y| {
        for i in 0..n {
            let mut acc = degrees[i] * x[i];
            for &j in &adjacency[i] {
                acc -= x[j];
            }
            y[i] = acc;
        }
    })?;
    eprintln!(
        "[INFO] Approximate NGEC computed in {:.3?}",
        ngec_start.elapsed()
    );

    // Optional: approximate low eigenvalues for display.
    eprintln!("[INFO] Performing eigendecomposition...");
    let eig_start = Instant::now();
    let (eigenvalues, _) = lanczos_smallest(n, 10.min(n), 16, |x, y| {
        for i in 0..n {
            let mut acc = degrees[i] * x[i];
            for &j in &adjacency[i] {
                acc -= x[j];
            }
            y[i] = acc;
        }
    })?;
    eprintln!(
        "[INFO] Approximate low-spectrum computation complete in {:.3?}",
        eig_start.elapsed()
    );

    // Print results
    println!("\n=== EIGENANALYSIS RESULTS ===");
    println!("Region: {}:{}", chr, display_range);
    println!("Nodes: {}", sorted_nodes.len());
    println!("Edges: {}", edges.len());
    println!("Approx Eigenvalues (low modes): {}", eigenvalues.len());
    println!("Approx NGEC: {:.6}", ngec);

    // Print top eigenvalues
    println!("\nTop 10 Eigenvalues:");
    for (i, &val) in eigenvalues.iter().take(10).enumerate() {
        println!("  λ{} = {:.6}", i, val);
    }

    // Visualization if requested
    if viz {
        const HEATMAP_PREVIEW_MAX_NODES: usize = 64;
        let preview_n = n.min(HEATMAP_PREVIEW_MAX_NODES);
        let mut laplacian_preview = Mat::<f64>::zeros(preview_n, preview_n);

        for i in 0..preview_n {
            laplacian_preview[(i, i)] = degrees[i];
            for &j in &adjacency[i] {
                if j < preview_n {
                    laplacian_preview[(i, j)] = -1.0;
                }
            }
        }

        println!("\n=== LAPLACIAN HEATMAP ===");
        if n > preview_n {
            println!(
                "(showing top-left {}x{} preview of {}x{} Laplacian)",
                preview_n, preview_n, n, n
            );
        }
        print_heatmap(laplacian_preview.as_ref());

        println!("\n=== EIGENVALUE DISTRIBUTION ===");
        println!("(approximate low-spectrum values)");
        print_eigenvalues_heatmap(&eigenvalues);
    }

    Ok(())
}
