// src/extract.rs

//! Module for extracting adjacency submatrix from edge list and performing analysis.

use ndarray::prelude::*;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::cmp::min;
use nalgebra::{DMatrix, DVector};
use ndarray_npy::write_npy;

use crate::eigen_print::{
    call_eigendecomp, 
    adjacency_matrix_to_ndarray, 
    compute_ngec, 
    print_heatmap, 
    print_heatmap_ndarray, 
    print_eigenvalues_heatmap,
    save_nalgebra_matrix_to_csv,
    save_nalgebra_vector_to_csv,
};

/// Saves an `Array2<f64>` to CSV by converting it to a nalgebra DMatrix first.
fn save_ndarray_to_csv<P: AsRef<Path>>(matrix: &Array2<f64>, path: P) -> io::Result<()> {
    let save_start = Instant::now();
    println!("    ⏳ Converting ndarray to nalgebra for CSV saving...");
    let convert_start = Instant::now();
    let nalgebra_matrix = DMatrix::from_iterator(
        matrix.nrows(),
        matrix.ncols(),
        matrix.iter().cloned()
    );
    let convert_duration = convert_start.elapsed();
    println!("    ✅ Conversion to nalgebra done in {:.4?}", convert_duration);

    println!("    ⏳ Writing nalgebra matrix to CSV at {:?}", path.as_ref());
    let write_start = Instant::now();
    save_nalgebra_matrix_to_csv(&nalgebra_matrix, path)?;
    let write_duration = write_start.elapsed();
    println!("    ✅ CSV write completed in {:.4?}", write_duration);

    let save_duration = save_start.elapsed();
    println!("    📂 Total time for save_ndarray_to_csv: {:.4?}", save_duration);

    Ok(())
}

/// Extracts a submatrix for a given node range from the adjacency matrix edge list,
/// computes the Laplacian, performs eigendecomposition, and saves the results.
pub fn extract_and_analyze_submatrix<P: AsRef<Path>>(
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
    let adjacency_matrix = Arc::new(Mutex::new(load_adjacency_matrix(
        &edge_list_path,
        start_node,
        end_node,
    )?));
    let load_duration = load_start.elapsed();
    println!("✅ Loaded adjacency matrix in {:.4?}", load_duration);

    // Compute Laplacian and eigendecomposition
    println!("🔬 Computing Laplacian matrix and eigendecomposition...");

    // Convert adjacency list to ndarray
    let adjacency_conv_start = Instant::now();
    println!("    ⏳ Converting adjacency list to ndarray...");
    let adj_matrix =
        adjacency_matrix_to_ndarray(&adjacency_matrix.lock().unwrap(), start_node, end_node); // This function enforces symmetry
    let adjacency_conv_duration = adjacency_conv_start.elapsed();
    println!("    ✅ Adjacency list -> ndarray in {:.4?}", adjacency_conv_duration);

    // Compute degree matrix
    let degree_start = Instant::now();
    println!("    ⏳ Computing degrees (sum of rows)...");
    let degrees = adj_matrix.sum_axis(Axis(1));
    println!("    ⏳ Building degree diagonal matrix...");
    let degree_matrix = Array2::<f64>::from_diag(&degrees);
    let degree_duration = degree_start.elapsed();
    println!("    ✅ Degree matrix computed in {:.4?}", degree_duration);

    // Compute Laplacian matrix: L = D - A
    let lap_start = Instant::now();
    println!("    ⏳ Computing Laplacian L = D - A...");
    let laplacian = &degree_matrix - &adj_matrix;
    let lap_duration = lap_start.elapsed();
    println!("    ✅ Laplacian computed in {:.4?}", lap_duration);

    // Save Laplacian matrix to CSV
    println!("    ⏳ Saving Laplacian matrix to CSV...");
    let csv_start = Instant::now();
    let output_dir = edge_list_path.as_ref().parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(&output_dir)?;
    let laplacian_csv = output_dir.join("laplacian.csv");
    println!("    📁 Output dir = {:?}", output_dir);
    println!("    📄 Laplacian CSV path = {:?}", laplacian_csv);
    save_ndarray_to_csv(&laplacian, &laplacian_csv)?;
    let csv_duration = csv_start.elapsed();
    println!("    ✅ Laplacian CSV saved in {:.4?}", csv_duration);
    
    // Compute eigenvalues and eigenvectors
    println!("🔬 Performing eigendecomposition...");
    let eig_start = Instant::now();
    let (eigvals, eigvecs) = call_eigendecomp(&laplacian)?;
    let eig_duration = eig_start.elapsed();
    println!("✅ Eigendecomposition done in {:.4?}", eig_duration);
    
    // Save eigenvectors and eigenvalues
    println!("    ⏳ Saving eigenvalues and eigenvectors...");
    let save_eig_start = Instant::now();
    let eigenvalues_csv = output_dir.join("eigenvalues.csv");
    let eigenvectors_csv = output_dir.join("eigenvectors.csv");
    
    // Convert ndarray types to nalgebra types for saving
    let to_na_start = Instant::now();
    let nalgebra_eigvals = DVector::from_iterator(
        eigvals.len(),
        eigvals.iter().cloned()
    );
    let nalgebra_eigvecs = DMatrix::from_iterator(
        eigvecs.nrows(),
        eigvecs.ncols(),
        eigvecs.iter().cloned()
    );
    let to_na_duration = to_na_start.elapsed();
    println!("    ✅ Converted eigen results to nalgebra in {:.4?}", to_na_duration);

    let eigenvals_save_start = Instant::now();
    save_nalgebra_vector_to_csv(&nalgebra_eigvals, &eigenvalues_csv)?;
    let eigenvals_save_duration = eigenvals_save_start.elapsed();
    println!("    ✅ Eigenvalues saved in {:.4?}", eigenvals_save_duration);

    let eigenvecs_save_start = Instant::now();
    save_nalgebra_matrix_to_csv(&nalgebra_eigvecs, &eigenvectors_csv)?;
    let eigenvecs_save_duration = eigenvecs_save_start.elapsed();
    println!("    ✅ Eigenvectors saved in {:.4?}", eigenvecs_save_duration);

    let save_eig_duration = save_eig_start.elapsed();
    println!("    ✅ Finished saving eigen data in {:.4?}", save_eig_duration);
    
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
    print_heatmap(&laplacian.view());
    let lap_hm_duration = lap_hm_start.elapsed();
    println!("    ✅ Laplacian heatmap printed in {:.4?}", lap_hm_duration);

    println!("    Eigenvectors:");
    let eigenvecs_hm_start = Instant::now();
    let eigenvecs_subset = eigvecs.slice(s![.., 0..min(500, eigvecs.ncols())]); // Display at max first 500
    print_heatmap_ndarray(&eigenvecs_subset.to_owned());
    let eigenvecs_hm_duration = eigenvecs_hm_start.elapsed();
    println!("    ✅ Eigenvector heatmap printed in {:.4?}", eigenvecs_hm_duration);

    println!("    Eigenvalues:");
    let eigenvals_hm_start = Instant::now();
    print_eigenvalues_heatmap(&eigvals);
    let eigenvals_hm_duration = eigenvals_hm_start.elapsed();
    println!("    ✅ Eigenvalue heatmap printed in {:.4?}", eigenvals_hm_duration);

    let duration = start_time.elapsed();
    println!("⏰ Completed extract_and_analyze_submatrix in {:.2?} seconds.", duration);
    println!("=== extract_and_analyze_submatrix END ===");

    Ok(())
}

/// Loads the adjacency matrix from a binary edge list file (.gam)
pub fn load_adjacency_matrix<P: AsRef<Path>>(
    path: P,
    start_node: usize,
    end_node: usize,
) -> io::Result<Vec<(u32, u32)>> {
    println!("    === load_adjacency_matrix BEGIN ===");
    let func_start = Instant::now();

    // Open the file and wrap in a BufReader
    let file = File::open(&path)?;
    println!("    File opened. Using BufReader...");
    let mut reader = BufReader::new(file);

    // We'll read in larger chunks to reduce overhead
    const CHUNK_SIZE: usize = 64 * 1024; // 64 KB
    let mut chunk_buf = vec![0u8; CHUNK_SIZE];

    // We keep leftover bytes (less than 8) between reads here
    let mut leftover = Vec::with_capacity(8);

    let mut edges = Vec::new();
    println!("    Starting to read edges in larger chunks from: {:?}", path.as_ref());

    let mut edge_count = 0usize;
    let read_start = Instant::now();

    // Function to parse as many 8-byte edges as possible from `buffer`
    // If there's leftover at the end, we return how many bytes we consumed.
    fn parse_edges_in_place(
        buffer: &mut [u8],
        valid_len: usize, // how many bytes in buffer are valid
        start_node: usize,
        end_node: usize,
        edges_out: &mut Vec<(u32, u32)>,
        edge_counter: &mut usize,
    ) -> usize {
        let mut i = 0;
        while i + 7 < valid_len {
            let from = u32::from_le_bytes(buffer[i..i+4].try_into().unwrap());
            let to   = u32::from_le_bytes(buffer[i+4..i+8].try_into().unwrap());
            i += 8;

            *edge_counter += 1;
            if *edge_counter % 100_000_000 == 0 {
                println!("    ... read {} edges so far ...", edge_counter);
            }

            // Only store edges between start_node and end_node
            if (start_node..=end_node).contains(&(from as usize))
                && (start_node..=end_node).contains(&(to as usize))
            {
                edges_out.push((from, to));
            }
        }
        i // number of bytes consumed
    }

    loop {
        // Read one chunk
        let bytes_read = reader.read(&mut chunk_buf)?;
        if bytes_read == 0 {
            // EOF: parse leftover (if any) and then break
            if !leftover.is_empty() {
                let leftover_len = leftover.len();
                let consumed = parse_edges_in_place(
                    &mut leftover,
                    leftover_len,
                    start_node,
                    end_node,
                    &mut edges,
                    &mut edge_count,
                );
                leftover.drain(0..consumed);
            }
            break;
        }

        // If we had leftover from last time, prepend it
        // so we parse across the leftover boundary
        if !leftover.is_empty() {
            let mut temp = Vec::with_capacity(leftover.len() + bytes_read);
            temp.extend_from_slice(&leftover);
            temp.extend_from_slice(&chunk_buf[..bytes_read]);
            leftover.clear();

            let temp_len = temp.len(); // <--- store length to avoid conflict
            let consumed = parse_edges_in_place(
                &mut temp,
                temp_len,
                start_node,
                end_node,
                &mut edges,
                &mut edge_count,
            );
            if consumed < temp_len {
                leftover.extend_from_slice(&temp[consumed..]);
            }
        } else {
            let consumed = parse_edges_in_place(
                &mut chunk_buf,
                bytes_read,
                start_node,
                end_node,
                &mut edges,
                &mut edge_count,
            );
            if consumed < bytes_read {
                leftover.extend_from_slice(&chunk_buf[consumed..bytes_read]);
            }
        }
    }

    let read_duration = read_start.elapsed();
    println!("    Finished reading edges. Total edges read: {}", edge_count);
    println!("    Filtering & storing relevant edges took {:.4?}", read_duration);

    let func_duration = func_start.elapsed();
    println!("    === load_adjacency_matrix END (total time: {:.4?}) ===", func_duration);
    Ok(edges)
}

/// Fast Laplacian construction directly from GAM file
pub fn fast_laplacian_from_gam<P: AsRef<Path>>(
    path: P,
    start_node: usize,
    end_node: usize,
) -> io::Result<Array2<f64>> {
    println!("    === fast_laplacian_from_gam BEGIN ===");
    let func_start = Instant::now();

    let dim = end_node - start_node + 1;
    println!("    Allocating Laplacian: {} x {}", dim, dim);
    let mut laplacian = Array2::<f64>::zeros((dim, dim));
    let mut degrees = Array1::<f64>::zeros(dim);

    let file = File::open(&path)?;
    println!("    File opened for fast Laplacian construction: {:?}", path.as_ref());
    let mut reader = BufReader::new(file);

    // We'll read in larger chunks to reduce overhead
    const CHUNK_SIZE: usize = 64 * 1024; // 64 KB
    let mut chunk_buf = vec![0u8; CHUNK_SIZE];

    // We keep leftover bytes (less than 8) between reads here
    let mut leftover = Vec::with_capacity(8);

    println!("    Reading edges and accumulating into Laplacian...");
    let mut edge_count = 0usize;
    let read_start = Instant::now();

    // Helper to parse 8-byte edges in place
    fn parse_and_accumulate(
        buffer: &mut [u8],
        valid_len: usize,
        start_node: usize,
        end_node: usize,
        laplacian: &mut Array2<f64>,
        degrees: &mut Array1<f64>,
        edge_counter: &mut usize,
    ) -> usize {
        let mut i = 0;
        while i + 7 < valid_len {
            let from = u32::from_le_bytes(buffer[i..i+4].try_into().unwrap()) as usize;
            let to   = u32::from_le_bytes(buffer[i+4..i+8].try_into().unwrap()) as usize;
            i += 8;

            *edge_counter += 1;
            if *edge_counter % 100_000_000 == 0 {
                println!("    ... processed {} edges so far ...", edge_counter);
            }

            if (start_node..=end_node).contains(&from) && (start_node..=end_node).contains(&to) {
                let r = from - start_node;
                let c = to   - start_node;

                // For an undirected graph with multiple edges possibly between the same nodes,
                // decrement by 1.0 to accumulate the effect of each edge
                laplacian[[r, c]] -= 1.0;
                laplacian[[c, r]] -= 1.0;

                degrees[r] += 1.0;
                degrees[c] += 1.0;
            }
        }
        i
    }

    loop {
        // Read a chunk
        let bytes_read = reader.read(&mut chunk_buf)?;
        if bytes_read == 0 {
            // EOF: parse leftover if any
            if !leftover.is_empty() {
                let leftover_len = leftover.len();
                let consumed = parse_and_accumulate(
                    &mut leftover,
                    leftover_len,
                    start_node,
                    end_node,
                    &mut laplacian,
                    &mut degrees,
                    &mut edge_count,
                );
                leftover.drain(0..consumed);
            }
            break;
        }

        if !leftover.is_empty() {
            let mut temp = Vec::with_capacity(leftover.len() + bytes_read);
            temp.extend_from_slice(&leftover);
            temp.extend_from_slice(&chunk_buf[..bytes_read]);
            leftover.clear();

            let temp_len = temp.len();
            let consumed = parse_and_accumulate(
                &mut temp,
                temp_len,
                start_node,
                end_node,
                &mut laplacian,
                &mut degrees,
                &mut edge_count,
            );
            if consumed < temp_len {
                leftover.extend_from_slice(&temp[consumed..]);
            }
        } else {
            let consumed = parse_and_accumulate(
                &mut chunk_buf,
                bytes_read,
                start_node,
                end_node,
                &mut laplacian,
                &mut degrees,
                &mut edge_count,
            );
            if consumed < bytes_read {
                leftover.extend_from_slice(&chunk_buf[consumed..bytes_read]);
            }
        }
    }

    let read_duration = read_start.elapsed();
    println!("    Finished reading and accumulating edges. Total edges read: {}", edge_count);
    println!("    Single-pass accumulation took {:.4?}", read_duration);

    // Now fill in the diagonal using the accumulated degrees
    println!("    Filling diagonal with degrees...");
    let diag_start = Instant::now();
    for i in 0..dim {
        laplacian[[i, i]] = degrees[i];
    }
    let diag_duration = diag_start.elapsed();
    println!("    ✅ Diagonal filled in {:.4?}", diag_duration);

    let func_duration = func_start.elapsed();
    println!("    === fast_laplacian_from_gam END (total time: {:.4?}) ===", func_duration);

    Ok(laplacian)
}

/// Extracts and saves just the Laplacian matrix as .npy file
pub fn extract_and_save_matrices<P: AsRef<Path>>(
    edge_list_path: P,
    start_node: usize,
    end_node: usize,
    output_dir: P,
) -> io::Result<()> {
    println!("=== extract_and_save_matrices BEGIN ===");
    let start_time = Instant::now();

    println!("📂 Computing Laplacian directly from {:?}", edge_list_path.as_ref());
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
    write_npy(&lap_path, &laplacian).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let save_duration = save_start.elapsed();
    println!("    ✅ Laplacian .npy saved in {:.4?}", save_duration);

    let duration = start_time.elapsed();
    println!("⏰ Completed extract_and_save_matrices in {:.2?} seconds.", duration);
    println!("=== extract_and_save_matrices END ===");
    
    Ok(())
}
