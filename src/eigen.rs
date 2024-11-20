// src/eigen.rs

//! Module for eigendecomposition and analyses that depend on eigenvectors or eigenvalues.

// Import necessary crates
use ndarray::prelude::*;
use std::io::{self, Write};
use std::path::Path;
use csv::WriterBuilder;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use nalgebra::{DVector, DMatrix};
use nalgebra::SymmetricEigen;
use sprs::{CsMat, TriMatI};

/// Struct representing the Laplacian operator for efficient matrix-vector multiplication.
#[derive(Clone)]
pub struct LaplacianOperator {
    laplacian: CsMat<f64>,
}

impl SymmetricOperator<f64> for LaplacianOperator {
    fn dim(&self) -> usize {
        self.laplacian.rows()
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.dim());
        assert_eq!(y.len(), self.dim());

        // Compute y = A * x efficiently using sparse matrix-vector multiplication
        self.laplacian.mul_mat_vec(x, y);
    }
}

/// Performs eigendecomposition using the Lanczos algorithm optimized for sparse banded matrices.
pub fn call_eigendecomp(
    edges: &[(usize, usize)],
    num_nodes: usize,
    k: usize,
) -> io::Result<(Vec<f64>, Vec<Vec<f64>>)> {
    // Build the Laplacian matrix as CsMat<f64>
    let laplacian_csmat = build_laplacian_csmat(edges, num_nodes);

    // Create LaplacianOperator
    let laplacian_operator = LaplacianOperator {
        laplacian: laplacian_csmat,
    };

    // Use SymmetricLanczos to compute k eigenvalues and eigenvectors
    let mut lanczos = SymmetricLanczos::new(laplacian_operator, k);
    lanczos.max_iterations = 1000;
    lanczos.tolerance = 1e-10;

    let result = lanczos.compute().map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Lanczos algorithm failed: {}", e),
        )
    })?;

    // Get eigenvalues and eigenvectors
    let eigenvalues = result.eigenvalues;
    let eigenvectors = result.eigenvectors;

    // Convert eigenvectors to Vec<Vec<f64>>
    let eigenvectors_vec = eigenvectors
        .column_iter()
        .map(|col| col.iter().cloned().collect())
        .collect();

    Ok((eigenvalues.to_vec(), eigenvectors_vec))
}

/// Builds the Laplacian matrix as a sparse CsMat<f64> from the edge list.
pub fn build_laplacian_csmat(edges: &[(usize, usize)], num_nodes: usize) -> CsMat<f64> {
    // Build the Laplacian matrix in triplet format
    let mut triplet =
        TriMatI::<f64, usize>::with_capacity((num_nodes, num_nodes), edges.len() * 2 + num_nodes);

    // Compute degrees and add off-diagonal entries
    let mut degrees = vec![0.0; num_nodes];
    for &(i, j) in edges {
        degrees[i] += 1.0;
        degrees[j] += 1.0;

        // Add off-diagonal entries
        triplet.add_triplet(i, j, -1.0);
        triplet.add_triplet(j, i, -1.0);
    }

    // Add diagonal entries
    for i in 0..num_nodes {
        triplet.add_triplet(i, i, degrees[i]);
    }

    // Convert triplet to CSR format
    triplet.to_csr()
}

/// Computes the Normalized Global Eigen-Complexity (NGEC) based on eigenvalues.
pub fn compute_ngec(eigenvalues: &[f64]) -> io::Result<f64> {
    let epsilon = 1e-9; // Small epsilon to account for floating-point precision
    let m = eigenvalues.len();
    if m == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Eigenvalues array is empty.",
        ));
    }

    // Check for significant negative eigenvalues
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
            "Eigenvalues contain significant negative values.",
        ));
    }

    // Calculate the sum of eigenvalues, ignoring small negative values due to precision
    let sum_eigen = eigenvalues
        .iter()
        .filter(|&&x| x >= -epsilon)
        .sum::<f64>();

    if sum_eigen <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Sum of eigenvalues must be positive.",
        ));
    }

    // Normalize the eigenvalues, ignoring small negative values
    let normalized_eigen: Vec<f64> = eigenvalues
        .iter()
        .map(|&x| if x >= -epsilon { x / sum_eigen } else { 0.0 })
        .collect();

    // Compute the entropy
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

/// Converts the adjacency matrix edge list to ndarray::Array2<f64>
pub fn adjacency_matrix_to_ndarray(
    edges: &[(usize, usize)],
    num_nodes: usize,
) -> Array2<f64> {
    let mut adj_array = Array2::<f64>::zeros((num_nodes, num_nodes));
    for &(a, b) in edges {
        adj_array[(a, b)] = 1.0;
        adj_array[(b, a)] = 1.0;
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

/// Saves a 1D Vec<f64> to a CSV file
pub fn save_vector_to_csv<P: AsRef<Path>>(vector: &[f64], csv_path: P) -> io::Result<()> {
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;
    wtr.serialize(vector)?;
    wtr.flush()?;
    Ok(())
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
pub fn print_eigenvalues_heatmap(vector: &[f64]) {
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
