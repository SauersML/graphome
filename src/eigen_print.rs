// src/eigen.rs

//! Module for eigendecomposition and analyses that depend on eigenvectors or eigenvalues.

use ndarray::prelude::*;
use std::io::{self, Write};
use std::path::Path;
use csv::WriterBuilder;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use nalgebra::{DVector, DMatrix, SymmetricEigen};

/// Computes the eigendecomposition of the Laplacian matrix using nalgebra's SymmetricEigen.
pub fn call_eigendecomp(laplacian: &Array2<f64>) -> io::Result<(Array1<f64>, Array2<f64>)> {
    // Use nalgebra's SymmetricEigen for the matrix
    println!("Using nalgebra's SymmetricEigen for the matrix.");
    let (eigvals, eigvecs) = compute_eigenvalues_and_vectors_sym(laplacian)?;

    // Convert nalgebra's DVector and DMatrix to ndarray's Array1 and Array2
    let eigvals_nd = Array1::from(eigvals.iter().cloned().collect::<Vec<f64>>());
    let eigvecs_nd = Array2::from_shape_vec(
        (eigvecs.nrows(), eigvecs.ncols()),
        eigvecs.iter().cloned().collect(),
    ).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    Ok((eigvals_nd, eigvecs_nd))
}

/// Computes eigenvalues and eigenvectors for a given Laplacian matrix with SymmetricEigen
pub fn compute_eigenvalues_and_vectors_sym(
    laplacian: &Array2<f64>,
) -> io::Result<(DVector<f64>, DMatrix<f64>)> {
    // Convert ndarray::Array2<f64> to nalgebra::DMatrix<f64>
    let nalgebra_laplacian = ndarray_to_nalgebra_matrix(laplacian)?;

    // Compute eigendecomposition using nalgebra's SymmetricEigen
    let symmetric_eigen = SymmetricEigen::new(nalgebra_laplacian);

    let eigvals = symmetric_eigen.eigenvalues;
    let eigvecs = symmetric_eigen.eigenvectors;

    Ok((eigvals, eigvecs))
}

/// Converts an ndarray::Array2<f64> to nalgebra::DMatrix<f64>
pub fn ndarray_to_nalgebra_matrix(matrix: &Array2<f64>) -> io::Result<DMatrix<f64>> {
    let (rows, cols) = matrix.dim();
    let mut nalgebra_matrix = DMatrix::<f64>::zeros(rows, cols);

    for ((i, j), value) in matrix.indexed_iter() {
        nalgebra_matrix[(i, j)] = *value;
    }

    Ok(nalgebra_matrix)
}

/// Saves a nalgebra::DMatrix<f64> to a CSV file
pub fn save_nalgebra_matrix_to_csv<P: AsRef<Path>>(
    matrix: &DMatrix<f64>,
    csv_path: P,
) -> io::Result<()> {
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;
    for i in 0..matrix.nrows() {
        let row = matrix.row(i).iter().cloned().collect::<Vec<f64>>();
        wtr.serialize(row)?;
    }
    wtr.flush()?;
    Ok(())
}

/// Saves a nalgebra::DVector<f64> to a CSV file
pub fn save_nalgebra_vector_to_csv<P: AsRef<Path>>(
    vector: &DVector<f64>,
    csv_path: P,
) -> io::Result<()> {
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;
    let row = vector.iter().cloned().collect::<Vec<f64>>();
    wtr.serialize(row)?;
    wtr.flush()?;
    Ok(())
}

/// Converts the adjacency matrix edge list to an `ndarray::Array2<f64>`.
/// 
/// This function takes a list of edges and creates an adjacency matrix for the nodes
/// in the specified range from `start_node` to `end_node`, inclusive. Each entry in
/// the returned matrix indicates whether an edge is present between the corresponding nodes.
/// 
/// Any edges that refer to nodes outside the given range are ignored.
/// 
/// # Arguments
/// 
/// * `edges` - A slice of `(u32, u32)` tuples, each representing an edge between two nodes.
/// * `start_node` - The first node index of the subgraph of interest.
/// * `end_node` - The last node index of the subgraph of interest (inclusive).
/// 
/// # Returns
/// 
/// An `Array2<f64>` that is a square adjacency matrix of size `(end_node - start_node + 1) x (end_node - start_node + 1)`.
/// 
/// # Example
/// 
/// ```
/// // Suppose we have edges (1,2), (2,3), and we want the adjacency matrix for nodes [1,3].
/// let edges = vec![(1,2),(2,3)];
/// let adj = adjacency_matrix_to_ndarray(&edges, 1, 3);
/// // The resulting matrix would be:
/// // [[0, 1, 0],
/// //  [1, 0, 1],
/// //  [0, 1, 0]]
/// ```
pub fn adjacency_matrix_to_ndarray(
    edges: &[(u32, u32)],
    start_node: usize,
    end_node: usize,
) -> Array2<f64> {
    let size = end_node - start_node + 1;
    let mut adj_array = Array2::<f64>::zeros((size, size));

    for &(a, b) in edges {
        // Check if both nodes are within the specified range
        if (a as usize) >= start_node && (b as usize) >= start_node {
            let local_a = (a as usize) - start_node;
            let local_b = (b as usize) - start_node;

            // Only set entries if indices are within the bounds of the array
            if local_a < size && local_b < size {
                adj_array[(local_a, local_b)] = 1.0;
                adj_array[(local_b, local_a)] = 1.0;
            }
        }
    }

    adj_array
}


/// Computes the Normalized Global Eigen-Complexity (NGEC) based on eigenvalues.
/// Ignores eigenvalues that are negative within a small epsilon due to floating-point precision.
///
/// # Arguments
///
/// * `eigenvalues` - A reference to an Array1<f64> containing the eigenvalues.
///
/// # Returns
///
/// * `Ok(f64)` - The computed NGEC value.
/// * `Err(io::Error)` - If the computation fails due to invalid input.
pub fn compute_ngec(eigenvalues: &Array1<f64>) -> io::Result<f64> {
    let epsilon = 1e-9; // Small epsilon to account for floating-point precision
    let m = eigenvalues.len();
    if m == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Eigenvalues array is empty.",
        ));
    }

    // Check for eigenvalues significantly below zero (beyond precision tolerance)
    if eigenvalues.iter().any(|&x| x < -epsilon) {
        let negative_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .filter(|&&x| x < -epsilon)
            .cloned()
            .take(5)
            .collect();
    
        println!("❗ Significant negative eigenvalues found: {:?}", negative_eigenvalues);
    
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Eigenvalues contain significant negative values."
            ),
        ));
    }

    // Calculate the sum of eigenvalues, ignoring small negative values due to precision
    let sum_eigen = eigenvalues
        .iter()
        .filter(|&&x| x >= -epsilon)  // Only include eigenvalues >= -epsilon
        .sum::<f64>();
    
    if sum_eigen <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Sum of eigenvalues must be positive.",
        ));
    }

    // Normalize the eigenvalues, ignoring small negative values
    let normalized_eigen = eigenvalues.mapv(|x| if x >= -epsilon { x / sum_eigen } else { 0.0 });

    // Compute the entropy
    // Handle cases where normalized eigenvalues are zero
    let entropy: f64 = normalized_eigen
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x * x.ln())
        .sum::<f64>();

    // Calculate log(m)
    let log_m = (m as f64).ln();
    if log_m == 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Logarithm of the number of eigenvalues resulted in zero.",
        ));
    }

    // Compute NGEC
    let ngec = -entropy / log_m;

    Ok(ngec)
}

/// Prints a heatmap of a 2D ndarray::ArrayView2<f64> to the terminal
pub fn print_heatmap(matrix: &ArrayView2<f64>) {
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
pub fn print_heatmap_ndarray(matrix: &Array2<f64>) {
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
pub fn print_eigenvalues_heatmap(vector: &Array1<f64>) {
    let num_eigvals = vector.len();

    let max_value = vector
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_value = vector
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

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
