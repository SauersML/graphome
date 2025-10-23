// src/eigen_region.rs
// Region-based eigendecomposition: extract subgraph from GBZ by genomic coordinates

use crate::eigen_print::{
    call_eigendecomp, compute_ngec, print_eigenvalues_heatmap, print_heatmap_normalized,
};
use crate::map::{coord_to_nodes_mapped, make_gbz_exist, parse_region};
use crate::mapped_gbz::MappedGBZ;
use faer::Mat;
use gbwt::{Orientation, GBZ};
use rustc_hash::{FxHashMap, FxHashSet};
use simple_sds::serialize;
use std::error::Error;
use std::io;
use std::time::Instant;

/// Extract edges from GBZ for a specific subset of nodes
/// Matches the behavior of convert_gbz_to_edge_list: Forward orientation only, undirected
fn extract_subgraph_edges(gbz: &GBZ, node_ids: &FxHashSet<usize>) -> Vec<(usize, usize)> {
    let mut edges: FxHashSet<(usize, usize)> = FxHashSet::default();

    if !node_ids.is_empty() {
        edges.reserve(node_ids.len().saturating_mul(4));
    }

    eprintln!("[INFO] Extracting edges for {} nodes...", node_ids.len());

    for &node_id in node_ids {
        // Query Forward orientation only (matches existing convert.rs behavior)
        if let Some(mut succ_iter) = gbz.successors(node_id, Orientation::Forward) {
            while let Some((succ_id, _)) = succ_iter.next() {
                // Only keep edges where both nodes are in our subset
                if node_ids.contains(&succ_id) {
                    // Normalize to (min, max) for undirected graph
                    let edge = if node_id <= succ_id {
                        (node_id, succ_id)
                    } else {
                        (succ_id, node_id)
                    };
                    edges.insert(edge);
                }
            }
        }
    }

    let edge_count = edges.len();
    eprintln!("[INFO] Found {} unique edges in subgraph", edge_count);

    edges.into_iter().collect()
}

/// Build Laplacian matrix directly from the edge list without forming a dense adjacency matrix
fn build_laplacian_matrix(edges: &[(usize, usize)], node_ids: &[usize]) -> Mat<f64> {
    let n = node_ids.len();
    eprintln!("[INFO] Building {}x{} Laplacian matrix...", n, n);

    let mut id_to_idx: FxHashMap<usize, usize> = FxHashMap::default();
    id_to_idx.reserve(n);
    for (idx, &id) in node_ids.iter().enumerate() {
        id_to_idx.insert(id, idx);
    }

    // Build adjacency lists to track degrees without storing a dense adjacency matrix
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    if n > 0 {
        let average_degree = (edges.len().saturating_mul(2) / n).max(1);
        for neighbors in &mut adjacency {
            neighbors.reserve(average_degree);
        }
    }

    for &(u, v) in edges {
        if let (Some(&i), Some(&j)) = (id_to_idx.get(&u), id_to_idx.get(&v)) {
            adjacency[i].push(j);
            if i != j {
                adjacency[j].push(i);
            }
        }
    }

    let mut laplacian = Mat::zeros(n, n);

    for i in 0..n {
        let degree = adjacency[i].len() as f64;
        laplacian[(i, i)] = degree;

        for &j in &adjacency[i] {
            if i == j {
                // Self loops contribute to the degree but should not add negative weights off-diagonal
                laplacian[(i, i)] -= 1.0;
            } else {
                laplacian[(i, j)] = -1.0;
            }
        }
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
