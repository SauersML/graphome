// src/extract.rs

//! Module for extracting adjacency submatrix from edge list and performing analysis.

use faer::Mat;
use memmap2::MmapOptions;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use rayon::prelude::*;
use std::fs::File;
use std::io;
use std::path::Path;
use std::time::Instant;

use crate::eigen_print::{
    call_eigendecomp, compute_ngec, print_eigenvalues_heatmap, print_heatmap,
    print_heatmap_normalized, save_matrix_to_csv, save_vector_to_csv,
};

/// Extracts a submatrix for a given node range from the adjacency matrix edge list,
/// computes the Laplacian, performs eigendecomposition, and saves the results.
pub fn extract_and_analyze_submatrix<P: AsRef<Path> + Send + Sync>(
    edge_list_path: P,
    start_node: usize,
    end_node: usize,
) -> io::Result<()> {
    let start_time = Instant::now();
    println!("=== extract_and_analyze_submatrix BEGIN ===");

    // Load the adjacency matrix from the .gam file
    println!(
        "📂 Loading adjacency matrix from {:?} ...",
        edge_list_path.as_ref()
    );
    let load_start = Instant::now();
    let adjacency_edges = load_adjacency_matrix(&edge_list_path, start_node, end_node)?;
    let load_duration = load_start.elapsed();
    println!("✅ Loaded adjacency matrix in {:.4?}", load_duration);

    // Compute Laplacian and eigendecomposition
    println!("🔬 Computing Laplacian matrix and eigendecomposition...");
    let lap_build_start = Instant::now();
    let laplacian = build_laplacian_from_edges(&adjacency_edges, start_node, end_node);
    let lap_build_duration = lap_build_start.elapsed();
    println!(
        "    ✅ Laplacian assembled directly from edges in {:.4?}",
        lap_build_duration
    );

    // Save Laplacian matrix to CSV
    println!("    ⏳ Saving Laplacian matrix to CSV...");
    let csv_start = Instant::now();
    let output_dir = edge_list_path.as_ref().parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(output_dir)?;
    let laplacian_csv = output_dir.join("laplacian.csv");
    println!("    📁 Output dir = {:?}", output_dir);
    println!("    📄 Laplacian CSV path = {:?}", laplacian_csv);
    save_matrix_to_csv(laplacian.as_ref(), &laplacian_csv)?;
    let csv_duration = csv_start.elapsed();
    println!("    ✅ Laplacian CSV saved in {:.4?}", csv_duration);

    // Compute eigenvalues and eigenvectors
    println!("🔬 Performing eigendecomposition...");
    let eig_start = Instant::now();
    let (eigvals, eigvecs) = call_eigendecomp(laplacian.as_ref())?;
    let eig_duration = eig_start.elapsed();
    println!("✅ Eigendecomposition done in {:.4?}", eig_duration);

    // Save eigenvectors and eigenvalues
    println!("    ⏳ Saving eigenvalues and eigenvectors...");
    let save_eig_start = Instant::now();
    let eigenvalues_csv = output_dir.join("eigenvalues.csv");
    let eigenvectors_csv = output_dir.join("eigenvectors.csv");

    let eigenvals_save_start = Instant::now();
    save_vector_to_csv(&eigvals, &eigenvalues_csv)?;
    let eigenvals_save_duration = eigenvals_save_start.elapsed();
    println!(
        "    ✅ Eigenvalues saved in {:.4?}",
        eigenvals_save_duration
    );

    let eigenvecs_save_start = Instant::now();
    save_matrix_to_csv(eigvecs.as_ref(), &eigenvectors_csv)?;
    let eigenvecs_save_duration = eigenvecs_save_start.elapsed();
    println!(
        "    ✅ Eigenvectors saved in {:.4?}",
        eigenvecs_save_duration
    );

    let save_eig_duration = save_eig_start.elapsed();
    println!(
        "    ✅ Finished saving eigen data in {:.4?}",
        save_eig_duration
    );

    // Compute and Print NGEC
    println!("📊 Computing Normalized Global Eigen-Complexity (NGEC)...");
    let ngec_start = Instant::now();
    let ngec = compute_ngec(&eigvals)?;
    let ngec_duration = ngec_start.elapsed();
    println!("✅ NGEC: {:.4} (computed in {:.4?})", ngec, ngec_duration);

    // Print heatmaps
    println!("🎨 Printing heatmaps:");
    println!("    Laplacian Matrix:");
    let lap_hm_start = Instant::now();
    print_heatmap(laplacian.as_ref());
    let lap_hm_duration = lap_hm_start.elapsed();
    println!(
        "    ✅ Laplacian heatmap printed in {:.4?}",
        lap_hm_duration
    );

    println!("    Eigenvectors:");
    let eigenvecs_hm_start = Instant::now();
    let cols_to_show = eigvecs.ncols().min(500);
    let eigenvecs_subset = Mat::from_fn(eigvecs.nrows(), cols_to_show, |i, j| eigvecs[(i, j)]);
    print_heatmap_normalized(eigenvecs_subset.as_ref());
    let eigenvecs_hm_duration = eigenvecs_hm_start.elapsed();
    println!(
        "    ✅ Eigenvector heatmap printed in {:.4?}",
        eigenvecs_hm_duration
    );

    println!("    Eigenvalues:");
    let eigenvals_hm_start = Instant::now();
    print_eigenvalues_heatmap(&eigvals);
    let eigenvals_hm_duration = eigenvals_hm_start.elapsed();
    println!(
        "    ✅ Eigenvalue heatmap printed in {:.4?}",
        eigenvals_hm_duration
    );

    let duration = start_time.elapsed();
    println!(
        "⏰ Completed extract_and_analyze_submatrix in {:.2?} seconds.",
        duration
    );
    println!("=== extract_and_analyze_submatrix END ===");

    Ok(())
}

/// Loads the adjacency matrix from a binary edge list file (.gam) in parallel.
/// Breaks the file into multiple chunk offsets and processes them concurrently.
///
/// We know each edge is exactly 8 bytes (4 bytes for `from`, 4 for `to`).
pub fn load_adjacency_matrix<P: AsRef<Path> + Send + Sync>(
    path: P,
    start_node: usize,
    end_node: usize,
) -> io::Result<Vec<(u32, u32)>> {
    println!("    === load_adjacency_matrix BEGIN (PARALLEL) ===");
    let func_start = Instant::now();

    if end_node < start_node {
        println!("    Start node is greater than end node; returning empty edge set.");
        return Ok(Vec::new());
    }

    let start_u32: u32 = start_node
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "start_node exceeds u32"))?;
    let end_u32: u32 = end_node
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "end_node exceeds u32"))?;

    let file = File::open(&path)?;
    let file_size = file.metadata()?.len();
    println!("    File size: {} bytes", file_size);

    if file_size < 8 {
        println!("    File smaller than single edge; returning empty edge set.");
        return Ok(Vec::new());
    }

    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let total_bytes = mmap.len();
    let valid_bytes = total_bytes - (total_bytes % 8);

    if valid_bytes != total_bytes {
        println!(
            "    ⚠️  Ignoring {} trailing bytes that do not form full edges.",
            total_bytes - valid_bytes
        );
    }

    let edge_width = 8usize;
    let total_edges = valid_bytes / edge_width;
    println!("    Total full edges detected: {}", total_edges);

    if valid_bytes == 0 {
        println!("    No complete edges present; returning empty edge set.");
        return Ok(Vec::new());
    }

    let chunk_bytes: usize = 8 * 262_144; // 2MB chunks keep good cache locality.
    let estimated_chunks = (valid_bytes + chunk_bytes - 1) / chunk_bytes;
    println!(
        "    Processing in approximately {} parallel chunks",
        estimated_chunks
    );

    let edges: Vec<(u32, u32)> = mmap
        .get(..valid_bytes)
        .expect("valid_bytes already checked")
        .par_chunks_exact(edge_width)
        .filter_map(|edge_bytes| {
            let from = u32::from_le_bytes(edge_bytes[0..4].try_into().unwrap());
            let to = u32::from_le_bytes(edge_bytes[4..8].try_into().unwrap());
            if from >= start_u32 && from <= end_u32 && to >= start_u32 && to <= end_u32 {
                Some((from, to))
            } else {
                None
            }
        })
        .collect();

    println!(
        "    Parallel parse complete. Total edges in range: {}",
        edges.len()
    );

    let func_duration = func_start.elapsed();
    println!(
        "    === load_adjacency_matrix END (total time: {:.4?}) ===",
        func_duration
    );

    Ok(edges)
}

fn build_laplacian_from_edges(
    edges: &[(u32, u32)],
    start_node: usize,
    end_node: usize,
) -> Mat<f64> {
    if end_node < start_node {
        return Mat::<f64>::zeros(0, 0);
    }

    let size = end_node - start_node + 1;
    let mut laplacian = Mat::<f64>::zeros(size, size);
    let mut degrees = vec![0.0f64; size];

    for &(from, to) in edges {
        let from_usize = from as usize;
        let to_usize = to as usize;
        if from_usize < start_node
            || from_usize > end_node
            || to_usize < start_node
            || to_usize > end_node
        {
            continue;
        }

        let r = from_usize - start_node;
        let c = to_usize - start_node;

        laplacian[(r, c)] -= 1.0;
        laplacian[(c, r)] -= 1.0;

        degrees[r] += 1.0;
        degrees[c] += 1.0;
    }

    for (idx, degree) in degrees.into_iter().enumerate() {
        *laplacian.get_mut(idx, idx) = degree;
    }

    laplacian
}

/// Fast Laplacian construction directly from GAM file, in parallel.
/// Each chunk accumulates a partial (dim x dim) Laplacian + degrees,
/// then we merge them in the reduce step.
pub fn fast_laplacian_from_gam<P: AsRef<Path> + Send + Sync>(
    path: P,
    start_node: usize,
    end_node: usize,
) -> io::Result<Array2<f64>> {
    println!("    === fast_laplacian_from_gam BEGIN (PARALLEL) ===");
    let func_start = Instant::now();

    if end_node < start_node {
        println!("    Start node is greater than end node; returning empty Laplacian.");
        return Ok(Array2::<f64>::zeros((0, 0)));
    }

    let start_u32: u32 = start_node
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "start_node exceeds u32"))?;
    let end_u32: u32 = end_node
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "end_node exceeds u32"))?;

    let dim = end_node - start_node + 1;
    println!("    Allocating Laplacian: {} x {}", dim, dim);

    let file = File::open(&path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();
    println!("    File size: {} bytes", file_size);

    if file_size < 8 {
        println!("    File smaller than single edge; returning zero Laplacian.");
        return Ok(Array2::<f64>::zeros((dim, dim)));
    }

    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let total_bytes = mmap.len();
    let valid_bytes = total_bytes - (total_bytes % 8);

    if valid_bytes != total_bytes {
        println!(
            "    ⚠️  Ignoring {} trailing bytes that do not form full edges.",
            total_bytes - valid_bytes
        );
    }

    let mut laplacian = Array2::<f64>::zeros((dim, dim));
    let mut degrees = vec![0.0f64; dim];

    for edge_bytes in mmap
        .get(..valid_bytes)
        .expect("valid_bytes already checked")
        .chunks_exact(8)
    {
        let from = u32::from_le_bytes(edge_bytes[0..4].try_into().unwrap());
        let to = u32::from_le_bytes(edge_bytes[4..8].try_into().unwrap());

        if from >= start_u32 && from <= end_u32 && to >= start_u32 && to <= end_u32 {
            let r = (from - start_u32) as usize;
            let c = (to - start_u32) as usize;

            unsafe {
                *laplacian.uget_mut([r, c]) -= 1.0;
                *laplacian.uget_mut([c, r]) -= 1.0;
            }
            degrees[r] += 1.0;
            degrees[c] += 1.0;
        }
    }

    println!("    Filling Laplacian diagonal with accumulated degrees...");
    for (idx, degree) in degrees.iter().enumerate() {
        unsafe {
            *laplacian.uget_mut([idx, idx]) = *degree;
        }
    }

    let func_duration = func_start.elapsed();
    println!(
        "    === fast_laplacian_from_gam END (total time: {:.4?}) ===",
        func_duration
    );

    Ok(laplacian)
}

/// Extracts and saves just the Laplacian matrix as .npy file
pub fn extract_and_save_matrices<P: AsRef<Path> + Send + Sync>(
    edge_list_path: P,
    start_node: usize,
    end_node: usize,
    output_dir: P,
) -> io::Result<()> {
    println!("=== extract_and_save_matrices BEGIN ===");
    let start_time = Instant::now();

    println!(
        "📂 Computing Laplacian directly from {:?}",
        edge_list_path.as_ref()
    );
    // Single pass construction
    let lap_start = Instant::now();
    let laplacian = fast_laplacian_from_gam(&edge_list_path, start_node, end_node)?;
    let lap_duration = lap_start.elapsed();
    println!("    ✅ Laplacian constructed in {:.4?}", lap_duration);

    // Save result
    println!("    ⏳ Saving Laplacian to .npy file...");
    let save_start = Instant::now();
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)?;

    let lap_path = output_dir.join("laplacian.npy");
    println!("    .npy path: {:?}", lap_path);
    write_npy(&lap_path, &laplacian).map_err(io::Error::other)?;
    let save_duration = save_start.elapsed();
    println!("    ✅ Laplacian .npy saved in {:.4?}", save_duration);

    let duration = start_time.elapsed();
    println!(
        "⏰ Completed extract_and_save_matrices in {:.2?} seconds.",
        duration
    );
    println!("=== extract_and_save_matrices END ===");

    Ok(())
}
