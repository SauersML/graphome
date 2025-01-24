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
use rayon::prelude::*;

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

/// Loads the adjacency matrix from a binary edge list file (.gam) in parallel.
/// Breaks the file into multiple chunk offsets and processes them concurrently.
/// 
/// We know each edge is exactly 8 bytes (4 bytes for `from`, 4 for `to`).
pub fn load_adjacency_matrix<P: AsRef<Path>>(
    path: P,
    start_node: usize,
    end_node: usize,
) -> io::Result<Vec<(u32, u32)>> {
    println!("    === load_adjacency_matrix BEGIN (PARALLEL) ===");
    let func_start = Instant::now();

    // 1) Get file size
    let file_size = std::fs::metadata(&path)?.len();
    println!("    File size: {} bytes", file_size);

    // 2) Define chunk size (1 GB for example). Make sure it's multiple of 8:
    let mut chunk_size = 1_073_741_824u64; // 1 GB
    if chunk_size % 8 != 0 {
        chunk_size -= chunk_size % 8; 
    }
    println!("    Using chunk_size = {} bytes", chunk_size);

    // 3) Calculate how many chunks
    let num_chunks = if file_size == 0 {
        0
    } else {
        (file_size + chunk_size - 1) / chunk_size
    };
    println!("    Number of chunks to read: {}", num_chunks);

    // 4) We'll collect partial vectors from each chunk in parallel, then combine.
    // (0..num_chunks) -> ParIter
    let edges: Vec<(u32, u32)> = (0..num_chunks).into_par_iter()
        .map(|chunk_idx| {
            // Each parallel job will read [start_offset..end_offset)
            let start_offset = chunk_idx * chunk_size;
            let mut end_offset = start_offset + chunk_size;
            if end_offset > file_size {
                end_offset = file_size;
            }

            // We'll parse from start_offset up to end_offset, ignoring any
            // partial edge at the tail if it doesn't align. 
            // Because chunk_size is multiple of 8, only the last chunk
            // might have partial leftover if the file_size isn't multiple of 8.
            let length_to_read = end_offset - start_offset;
            println!("        Chunk {}: offset=[{}..{}], length={}",
                chunk_idx, start_offset, end_offset, length_to_read);

            // Open the file for this chunk, seek to start_offset
            let file = File::open(&path).expect("Unable to open file in parallel load");
            let mut reader = BufReader::new(file);
            reader.seek(io::SeekFrom::Start(start_offset))
                  .expect("Failed to seek in parallel load");

            // Read the entire chunk into memory
            let mut buf = vec![0u8; length_to_read as usize];
            let mut total_read = 0usize;
            while total_read < buf.len() {
                let n = reader.read(&mut buf[total_read..])
                              .expect("Failed to read chunk");
                if n == 0 {
                    break; // EOF
                }
                total_read += n;
            }

            // Now parse edges from buf. Only parse multiples of 8 fully contained.
            let valid_bytes = (total_read / 8) * 8; 
            let mut local_edges = Vec::new();

            let mut i = 0;
            let mut edge_count_local = 0usize;
            while i + 7 < valid_bytes {
                let from = u32::from_le_bytes(buf[i..i+4].try_into().unwrap());
                let to   = u32::from_le_bytes(buf[i+4..i+8].try_into().unwrap());
                i += 8;

                edge_count_local += 1;
                // Filter by [start_node..end_node]
                if (start_node..=end_node).contains(&(from as usize))
                   && (start_node..=end_node).contains(&(to as usize)) {
                    local_edges.push((from, to));
                }
            }

            println!("        Chunk {} done: parsed {} edges, kept {}",
                chunk_idx, edge_count_local, local_edges.len());
            local_edges
        })
        .reduce(
            || Vec::new(), // identity
            |mut acc, mut part| {
                acc.append(&mut part); // merges partial vectors
                acc
            }
        );

    let total_kept = edges.len();
    println!("    Parallel parse complete. Total edges in range: {}", total_kept);

    let func_duration = func_start.elapsed();
    println!("    === load_adjacency_matrix END (total time: {:.4?}) ===", func_duration);

    Ok(edges)
}

/// Fast Laplacian construction directly from GAM file, in parallel.
/// Each chunk accumulates a partial (dim x dim) Laplacian + degrees,
/// then we merge them in the reduce step.
pub fn fast_laplacian_from_gam<P: AsRef<Path>>(
    path: P,
    start_node: usize,
    end_node: usize,
) -> io::Result<Array2<f64>> {
    println!("    === fast_laplacian_from_gam BEGIN (PARALLEL) ===");
    let func_start = Instant::now();

    let dim = end_node - start_node + 1;
    println!("    Allocating Laplacian: {} x {}", dim, dim);

    // 1) Get file size
    let file_size = std::fs::metadata(&path)?.len();
    println!("    File size: {} bytes", file_size);

    // 2) chunk_size multiple of 8
    let mut chunk_size = 1_073_741_824u64; // 1 GB
    if chunk_size % 8 != 0 {
        chunk_size -= chunk_size % 8; 
    }
    // 3) number of chunks
    let num_chunks = if file_size == 0 {
        0
    } else {
        (file_size + chunk_size - 1) / chunk_size
    };
    println!("    Using chunk_size = {} bytes, total chunks = {}", chunk_size, num_chunks);

    // We'll create a parallel iterator over chunk indices
    // Each chunk returns a partial Laplacian (dim x dim) + partial degrees,
    // which we reduce into a final sum.
    let (laplacian, degrees) = (0..num_chunks).into_par_iter()
        .map(|chunk_idx| {
            // Allocate partial structures for this chunk
            let mut part_lap = Array2::<f64>::zeros((dim, dim));
            let mut part_deg = Array1::<f64>::zeros(dim);

            let start_offset = chunk_idx * chunk_size;
            let mut end_offset = start_offset + chunk_size;
            if end_offset > file_size {
                end_offset = file_size;
            }
            let length_to_read = end_offset - start_offset;
            println!("        Chunk {}: offset=[{}..{}], length={}",
                     chunk_idx, start_offset, end_offset, length_to_read);

            // Open + seek
            let file = File::open(&path).expect("Unable to open file in parallel Laplacian");
            let mut reader = BufReader::new(file);
            reader.seek(io::SeekFrom::Start(start_offset))
                  .expect("Failed to seek in parallel Laplacian");

            // Read chunk
            let mut buf = vec![0u8; length_to_read as usize];
            let mut total_read = 0usize;
            while total_read < buf.len() {
                let n = reader.read(&mut buf[total_read..])
                              .expect("Failed to read chunk");
                if n == 0 {
                    break;
                }
                total_read += n;
            }

            // parse fully aligned edges
            let valid_bytes = (total_read / 8) * 8; 
            let mut edge_count_local = 0usize;

            let mut i = 0;
            while i + 7 < valid_bytes {
                let from = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as usize;
                let to   = u32::from_le_bytes(buf[i+4..i+8].try_into().unwrap()) as usize;
                i += 8;
                edge_count_local += 1;

                // If both from, to in [start_node..end_node], accumulate
                if (start_node..=end_node).contains(&from)
                   && (start_node..=end_node).contains(&to) {
                    let r = from - start_node;
                    let c = to   - start_node;
                    // undirected edge => symmetrical updates
                    part_lap[[r, c]] -= 1.0;
                    part_lap[[c, r]] -= 1.0;
                    part_deg[r] += 1.0;
                    part_deg[c] += 1.0;
                }
            }

            println!("        Chunk {} done: parsed {} edges", chunk_idx, edge_count_local);
            (part_lap, part_deg)
        })
        .reduce(
            // Identity
            || (
                Array2::<f64>::zeros((dim, dim)),
                Array1::<f64>::zeros(dim),
            ),
            // Combine partials
            |(mut lap_a, mut deg_a), (lap_b, deg_b)| {
                lap_a = lap_a + &lap_b; // elementwise add
                deg_a = deg_a + &deg_b;
                (lap_a, deg_a)
            }
        );

    // Now fill diagonal with degrees
    println!("    Summation done, filling diagonal...");
    for i in 0..dim {
        laplacian[[i, i]] = degrees[i];
    }

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
