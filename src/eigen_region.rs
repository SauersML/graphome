// src/eigen_region.rs
// Region-based eigendecomposition: extract subgraph from GBZ by genomic coordinates

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::io;
use faer::Mat;
use gbwt::{GBZ, Orientation};
use simple_sds::serialize;
use crate::map::{coord_to_nodes_mapped, parse_region, make_gbz_exist};
use crate::mapped_gbz::MappedGBZ;
use crate::eigen_print::{call_eigendecomp, compute_ngec, print_heatmap_normalized, print_eigenvalues_heatmap};

/// Extract edges from GBZ for a specific subset of nodes
/// Matches the behavior of convert_gbz_to_edge_list: Forward orientation only, undirected
fn extract_subgraph_edges(
    gbz: &GBZ,
    node_ids: &HashSet<usize>
) -> Vec<(usize, usize)> {
    let mut edges = HashSet::new();
    
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

/// Build adjacency matrix from edge list
/// Creates mapping from arbitrary node IDs to contiguous matrix indices
fn build_adjacency_matrix(
    edges: &[(usize, usize)],
    node_ids: &[usize]  // sorted list of node IDs
) -> Mat<f64> {
    let n = node_ids.len();
    eprintln!("[INFO] Building {}x{} adjacency matrix...", n, n);
    
    let mut matrix = Mat::zeros(n, n);
    
    // Create mapping: node_id -> matrix_index
    let id_to_idx: HashMap<usize, usize> = node_ids.iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();
    
    // Fill adjacency matrix (symmetric for undirected graph)
    for &(u, v) in edges {
        if let (Some(&i), Some(&j)) = (id_to_idx.get(&u), id_to_idx.get(&v)) {
            matrix[(i, j)] = 1.0;
            matrix[(j, i)] = 1.0;  // Symmetric
        }
    }
    
    matrix
}

/// Compute Laplacian matrix: L = D - A
/// where D is the degree matrix (diagonal) and A is the adjacency matrix
fn compute_laplacian(adjacency: &Mat<f64>) -> Mat<f64> {
    let n = adjacency.nrows();
    eprintln!("[INFO] Computing Laplacian matrix...");
    
    let mut laplacian = adjacency.clone();
    
    // Compute degree for each node and set diagonal
    for i in 0..n {
        let mut degree = 0.0;
        for j in 0..n {
            degree += adjacency[(i, j)];
        }
        laplacian[(i, i)] = degree - adjacency[(i, i)];
    }
    
    // Subtract adjacency from degree matrix
    for i in 0..n {
        for j in 0..n {
            if i != j {
                laplacian[(i, j)] = -adjacency[(i, j)];
            }
        }
    }
    
    laplacian
}

/// Main entry point for region-based eigendecomposition
pub fn run_eigen_region(
    gfa_path: &str,
    region: &str,
    viz: bool
) -> Result<(), Box<dyn Error>> {
    eprintln!("[INFO] Starting region-based eigendecomposition");
    eprintln!("[INFO] Region: {}", region);
    
    // Parse region
    let (chr, start, end) = parse_region(region)
        .ok_or_else(|| format!("Invalid region format: {}", region))?;
    
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
    let node_ids: HashSet<usize> = results.iter()
        .filter_map(|r| r.node_id.parse().ok())
        .collect();
    
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
    
    // Build adjacency matrix
    let adjacency = build_adjacency_matrix(&edges, &sorted_nodes);
    
    // Compute Laplacian
    let laplacian = compute_laplacian(&adjacency);
    
    // Perform eigendecomposition
    eprintln!("[INFO] Performing eigendecomposition...");
    let (eigenvalues, _eigenvectors) = call_eigendecomp(laplacian.as_ref())?;
    eprintln!("[INFO] Eigendecomposition complete");
    
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
