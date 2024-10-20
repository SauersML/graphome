// src/extract.rs

//! Module for extracting adjacency submatrix from edge list and performing analysis.

use ndarray::prelude::*;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::cmp::min;

use crate::eigen::{call_eigendecomp, save_array_to_csv_dsbevd, save_vector_to_csv_dsbevd, adjacency_matrix_to_ndarray, compute_ngec, print_heatmap, print_heatmap_ndarray, print_eigenvalues_heatmap};

/// Extracts a submatrix for a given node range from the adjacency matrix edge list,
/// computes the Laplacian, performs eigendecomposition, and saves the results.
pub fn extract_and_analyze_submatrix<P: AsRef<Path>>(
    edge_list_path: P,
    start_node: usize,
    end_node: usize,
    output_path: P,
) -> io::Result<()> {
    let start_time = Instant::now();

    // Load the adjacency matrix from the .gam file
    println!(
        "📂 Loading adjacency matrix from {:?}",
        edge_list_path.as_ref()
    );

    let adjacency_matrix = Arc::new(Mutex::new(load_adjacency_matrix(
        &edge_list_path,
        start_node,
        end_node,
    )?));

    println!("✅ Loaded adjacency matrix.");

    // Compute Laplacian and eigendecomposition
    println!("🔬 Computing Laplacian matrix and eigendecomposition...");

    let adj_matrix =
        adjacency_matrix_to_ndarray(&adjacency_matrix.lock().unwrap(), start_node, end_node);

    // Compute degree matrix
    let degrees = adj_matrix.sum_axis(Axis(1));
    let degree_matrix = Array2::<f64>::from_diag(&degrees);

    // Compute Laplacian matrix: L = D - A
    let laplacian = &degree_matrix - &adj_matrix;

    // Save Laplacian matrix to CSV
    let laplacian_csv_path = output_path.as_ref().with_extension("laplacian.csv");
    save_array_to_csv_dsbevd(&laplacian, &laplacian_csv_path)?;
    println!(
        "✅ Laplacian matrix saved to {}",
        laplacian_csv_path.display()
    );

    // Compute eigenvalues and eigenvectors
    println!("🔬 Performing eigendecomposition...");
    let (eigvals, eigvecs) = call_eigendecomp(&laplacian)?;
    
    // Save eigenvectors to CSV
    let eigen_csv_path = output_path.as_ref().with_extension("eigenvectors.csv");
    save_array_to_csv_dsbevd(&eigvecs, &eigen_csv_path)?;
    println!("✅ Eigenvectors saved to {}", eigen_csv_path.display());
    
    // Save eigenvalues to CSV
    let eigenvalues_csv_path = output_path.as_ref().with_extension("eigenvalues.csv");
    save_vector_to_csv_dsbevd(&eigvals, &eigenvalues_csv_path)?;
    println!("✅ Eigenvalues saved to {}", eigenvalues_csv_path.display());
    
    // Compute and Print NGEC
    println!("📊 Computing Normalized Global Eigen-Complexity (NGEC)...");
    let ngec = compute_ngec(&eigvals)?;
    println!("✅ NGEC: {:.4}", ngec);
    
    // Print heatmaps
    println!("🎨 Printing heatmaps:");
    println!("Laplacian Matrix:");
    print_heatmap(&laplacian.view());

    println!("Eigenvectors:");
    let eigenvecs_subset = eigvecs.slice(s![.., 0..min(500, eigvecs.ncols())]); // Display at max first 500
    print_heatmap_ndarray(&eigenvecs_subset.to_owned());

    println!("Eigenvalues:");
    print_eigenvalues_heatmap(&eigvals);

    let duration = start_time.elapsed();
    println!("⏰ Completed in {:.2?} seconds.", duration);

    Ok(())
}

/// Loads the adjacency matrix from a binary edge list file (.gam)
pub fn load_adjacency_matrix<P: AsRef<Path>>(
    path: P,
    start_node: usize,
    end_node: usize,
) -> io::Result<Vec<(u32, u32)>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0u8; 8];
    let mut edges = Vec::new();

    while let Ok(_) = reader.read_exact(&mut buffer) {
        let from = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        let to = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);

        // Only store edges between start_node and end_node
        if (start_node..=end_node).contains(&(from as usize))
            && (start_node..=end_node).contains(&(to as usize))
        {
            edges.push((from, to));
        }
    }

    Ok(edges)
}
