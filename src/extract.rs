// src/extract.rs

//! Module for extracting adjacency submatrix from edge list, eigendecomposition, and performing analysis.

use lapack_sys::dsbevd_;
use ndarray::prelude::*;
use std::ffi::c_char;
use std::os::raw::c_int;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use csv::WriterBuilder;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use std::cmp::min;

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

    // Initialize progress tracking
    let total_nodes = end_node - start_node + 1;

    // Compute Laplacian and eigendecomposition
    println!("🔬 Computing Laplacian matrix and eigendecomposition...");

    let adj_matrix =
        adjacency_matrix_to_ndarray(&adjacency_matrix.lock().unwrap(), start_node, end_node);

    // Compute degree matrix
    let degrees = adj_matrix.sum_axis(Axis(1));
    let degree_matrix = Array2::<f64>::from_diag(&degrees);

    // Compute Laplacian matrix: L = D - A
    let laplacian = &degree_matrix - &adj_matrix;

    // Progress print for Laplacian matrix processing
    for i in 0..total_nodes {
        if i % (total_nodes / 10).max(1) == 0 {
            let elapsed = start_time.elapsed();
            let progress = (i as f64 / total_nodes as f64) * 100.0;
            let estimated_remaining = elapsed / (i as u32 + 1) * (total_nodes as u32 - i as u32);
            println!(
                "Laplacian Matrix: Processed {} out of {} nodes ({:.2}%) | Estimated remaining: {:.2?}",
                i, total_nodes, progress, estimated_remaining
            );
        }
    }

    // Save Laplacian matrix to CSV
    let laplacian_csv_path = output_path.as_ref().with_extension("laplacian.csv");
    save_array_to_csv(&laplacian, &laplacian_csv_path)?;
    println!(
        "✅ Laplacian matrix saved to {}",
        laplacian_csv_path.display()
    );

    // Compute eigenvalues and eigenvectors using LAPACK's dsbevd
    println!("🔬 Performing eigendecomposition using LAPACK's dsbevd...");
    let (eigvals, eigvecs) = compute_eigenvalues_and_vectors(&laplacian)?;

    // Progress print for eigenvector matrix computation
    let total_eigenvectors = eigvecs.ncols();
    for i in 0..total_eigenvectors {
        if i % (total_eigenvectors / 10).max(1) == 0 {
            let elapsed = start_time.elapsed();
            let progress = (i as f64 / total_eigenvectors as f64) * 100.0;
            let estimated_remaining =
                elapsed / (i as u32 + 1) * (total_eigenvectors as u32 - i as u32);
            println!(
                "Eigenvector Matrix: Processed {} out of {} vectors ({:.2}%) | Estimated remaining: {:.2?}",
                i, total_eigenvectors, progress, estimated_remaining
            );
        }
    }

    // Save eigenvectors to CSV
    let eigen_csv_path = output_path.as_ref().with_extension("eigenvectors.csv");
    save_array_to_csv(&eigvecs, &eigen_csv_path)?;
    println!("✅ Eigenvectors saved to {}", eigen_csv_path.display());

    // Save eigenvalues to CSV
    let eigenvalues_csv_path = output_path.as_ref().with_extension("eigenvalues.csv");
    save_vector_to_csv(&eigvals, &eigenvalues_csv_path)?;
    println!("✅ Eigenvalues saved to {}", eigenvalues_csv_path.display());

    // Print heatmaps
    println!("🎨 Printing heatmaps:");
    println!("Laplacian Matrix:");
    print_heatmap(&laplacian.view());

    println!("Eigenvectors:");
    let eigenvecs_subset = eigvecs.slice(s![.., 0..min(5000, eigvecs.ncols())]); // Display at max first 5000
    print_heatmap_ndarray(&eigenvecs_subset.to_owned());

    println!("Eigenvalues:");
    print_eigenvalues_heatmap(&eigvals);

    let duration = start_time.elapsed();
    println!("⏰ Completed in {:.2?} seconds.", duration);

    Ok(())
}

/// Converts a 2D matrix to a banded matrix representation required for dsbevd
fn to_banded_format(matrix: &Array2<f64>, kd: usize) -> Array2<f64> {
    let (n, _) = matrix.dim();
    let mut banded = Array2::<f64>::zeros((kd + 1, n));

    for j in 0..n {
        for i in 0..=kd {
            if j >= i {
                banded[(kd - i, j)] = matrix[(j - i, j)];
            }
        }
    }

    banded
}

/// Computes eigenvalues and eigenvectors using LAPACK's dsbevd
fn compute_eigenvalues_and_vectors(
    laplacian: &Array2<f64>,
) -> io::Result<(Array1<f64>, Array2<f64>)> {
    let n = laplacian.nrows() as c_int;
    let kd = 1; // Number of subdiagonals/superdiagonals

    // Convert to the banded format expected by dsbevd
    let mut banded_matrix = to_banded_format(laplacian, kd as usize).into_raw_vec_and_offset().0;

    // Prepare arrays for eigenvalues and eigenvectors
    let mut eigvals = vec![0.0_f64; n as usize];
    let mut eigvecs = vec![0.0_f64; (n * n) as usize]; // Flat array for eigenvectors
    let mut work = vec![0.0_f64; (8 * n) as usize];  // Workspace
    let work_len = (work.len()) as c_int;
    let mut iwork = vec![0_i32; (5 * n) as usize];   // Integer workspace
    let iwork_len = (iwork.len()) as c_int;
    let mut info: c_int = 0;

    let jobz = b'V' as c_char; // Compute eigenvalues and eigenvectors
    let uplo = b'L' as c_char; // Lower triangle

    unsafe {
        dsbevd_(
            &jobz,
            &uplo,
            &n,
            &kd,
            banded_matrix.as_mut_ptr(),
            &(kd + 1) as *const c_int,
            eigvals.as_mut_ptr(),
            eigvecs.as_mut_ptr(),
            &n,
            work.as_mut_ptr(),
            &work_len,
            iwork.as_mut_ptr(),
            &iwork_len,
            &mut info,
        );
    }

    if info != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("LAPACK dsbevd failed with error code {}", info),
        ));
    }

    // Convert outputs to ndarray types
    let eigvals_nd = Array1::from(eigvals);
    let eigvecs_nd = Array2::from_shape_vec((n as usize, n as usize), eigvecs)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    Ok((eigvals_nd, eigvecs_nd))
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

/// Converts the adjacency matrix edge list to ndarray::Array2<f64>
pub fn adjacency_matrix_to_ndarray(
    edges: &[(u32, u32)],
    start_node: usize,
    end_node: usize,
) -> Array2<f64> {
    let mut adj_array =
        Array2::<f64>::zeros((end_node - start_node + 1, end_node - start_node + 1));
    for &(a, b) in edges {
        let local_a = a as usize - start_node;
        let local_b = b as usize - start_node;
        adj_array[(local_a, local_b)] = 1.0;
        adj_array[(local_b, local_a)] = 1.0;
    }
    adj_array
}

/// Saves a 2D ndarray::Array2<f64> to a CSV file
pub fn save_array_to_csv<P: AsRef<Path>>(matrix: &Array2<f64>, csv_path: P) -> io::Result<()> {
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;
    for row in matrix.rows() {
        wtr.serialize(row.to_vec())?;
    }
    wtr.flush()?;
    Ok(())
}

/// Saves a 1D ndarray::Array1<f64> to a CSV file
pub fn save_vector_to_csv<P: AsRef<Path>>(vector: &Array1<f64>, csv_path: P) -> io::Result<()> {
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;
    let row = vector.iter().cloned().collect::<Vec<f64>>();
    wtr.serialize(row)?;
    wtr.flush()?;
    Ok(())
}

/// Prints a heatmap of a 2D ndarray::ArrayView2<f64> to the terminal
fn print_heatmap(matrix: &ArrayView2<f64>) {
    let num_rows = matrix.nrows();
    let num_cols = matrix.ncols();

    let non_zero_values: Vec<f64> = matrix.iter().cloned().filter(|&x| x != 0.0).collect();

    let max_value = non_zero_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_value = non_zero_values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut stdout = stdout.lock();

    for i in 0..num_rows {
        for j in 0..num_cols {
            let value = matrix[[i, j]];
            let intensity = if (max_value - min_value) != 0.0 {
                ((value - min_value) / (max_value - min_value)).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let color = if value == 0.0 {
                Color::Black
            } else {
                Color::Rgb(
                    (intensity * 255.0) as u8,
                    0,
                    ((1.0 - intensity) * 255.0) as u8,
                )
            };

            let mut color_spec = ColorSpec::new();
            color_spec.set_fg(Some(color));
            let _ = stdout.set_color(&color_spec);
            let _ = write!(stdout, "██");
        }
        let _ = stdout.reset();
        let _ = writeln!(stdout);
    }
}

/// Prints a heatmap of a nalgebra::Array2<f64> to the terminal with Z-normalization
fn print_heatmap_ndarray(matrix: &Array2<f64>) {
    let num_rows = matrix.nrows();
    let num_cols = matrix.ncols();

    // Collect all values for calculating mean and standard deviation
    let all_values: Vec<f64> = matrix.iter().cloned().collect();

    // Calculate mean and standard deviation for Z-normalization
    let mean = all_values.iter().sum::<f64>() / all_values.len() as f64;
    let stddev = (all_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / all_values.len() as f64)
        .sqrt();

    // Collect final normalized values before printing
    let mut normalized_values: Vec<Vec<f64>> = Vec::new();

    for i in 0..num_rows {
        let mut row_values: Vec<f64> = Vec::new();
        for j in 0..num_cols {
            let value = matrix[(i, j)];

            // Zero is always black
            let final_value = if value == 0.0 {
                0.0 // Zero remains black
            } else {
                // Z-normalize other values
                let z_value = (value - mean) / stddev;

                let adjusted_intensity = 0.5 * z_value + 0.5; // Shift range to [0, 1] for contrast
                adjusted_intensity.clamp(0.0, 1.0) // Clamp intensity to [0, 1]
            };

            row_values.push(final_value);
        }
        normalized_values.push(row_values);
    }

    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut stdout = stdout.lock();

    for i in 0..num_rows {
        for j in 0..num_cols {
            let final_value = normalized_values[i][j];

            let color = if final_value == 0.0 {
                Color::Black
            } else {
                Color::Rgb(
                    (final_value * 255.0) as u8,
                    0,
                    ((1.0 - final_value) * 255.0) as u8,
                )
            };

            let mut color_spec = ColorSpec::new();
            color_spec.set_fg(Some(color));
            let _ = stdout.set_color(&color_spec);
            let _ = write!(stdout, "██");
        }
        let _ = stdout.reset();
        let _ = writeln!(stdout);
    }
}

/// Prints a heatmap of eigenvalues
fn print_eigenvalues_heatmap(vector: &Array1<f64>) {
    let num_eigvals = vector.len();

    let max_value = vector.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_value = vector.iter().cloned().fold(f64::INFINITY, f64::min);

    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut stdout = stdout.lock();

    for i in 0..num_eigvals {
        let value = vector[i];
        let intensity = if (max_value - min_value) != 0.0 {
            ((value - min_value) / (max_value - min_value)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let color = if value == 0.0 {
            Color::Black
        } else {
            Color::Rgb(
                (intensity * 255.0) as u8,
                0,
                ((1.0 - intensity) * 255.0) as u8,
            )
        };

        let mut color_spec = ColorSpec::new();
        color_spec.set_fg(Some(color));
        let _ = stdout.set_color(&color_spec);
        let _ = write!(stdout, "██");
    }
    let _ = stdout.reset();
    let _ = writeln!(stdout);
}
