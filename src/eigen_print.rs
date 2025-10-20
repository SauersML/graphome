// src/eigen_print.rs

//! Module for eigendecomposition and analyses that depend on eigenvectors or eigenvalues.

use csv::WriterBuilder;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, MatRef, Side};
use std::io::{self, Write};
use std::path::Path;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

/// Computes the eigendecomposition of the Laplacian matrix using faer's self-adjoint solver.
pub fn call_eigendecomp(laplacian: MatRef<'_, f64>) -> io::Result<(Vec<f64>, Mat<f64>)> {
    println!("Using faer's self-adjoint eigensolver for the matrix.");
    compute_eigenvalues_and_vectors_sym(laplacian)
}

/// Computes eigenvalues and eigenvectors for a given Laplacian matrix with faer.
pub fn compute_eigenvalues_and_vectors_sym(
    laplacian: MatRef<'_, f64>,
) -> io::Result<(Vec<f64>, Mat<f64>)> {
    if laplacian.nrows() != laplacian.ncols() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Laplacian matrix must be square.",
        ));
    }

    let decomposition = SelfAdjointEigen::new(laplacian, Side::Lower)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, format!("{:?}", err)))?;

    let eigenvalues = decomposition
        .S()
        .column_vector()
        .iter()
        .copied()
        .collect::<Vec<f64>>();

    let eigenvectors = decomposition.U().to_owned();

    Ok((eigenvalues, eigenvectors))
}

/// Saves a dense faer matrix to a CSV file.
pub fn save_matrix_to_csv<P: AsRef<Path>>(matrix: MatRef<'_, f64>, csv_path: P) -> io::Result<()> {
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;

    for i in 0..matrix.nrows() {
        let row: Vec<f64> = (0..matrix.ncols()).map(|j| matrix[(i, j)]).collect();
        wtr.serialize(row)?;
    }

    wtr.flush()?;
    Ok(())
}

/// Saves an eigenvalue vector to a CSV file.
pub fn save_vector_to_csv<P: AsRef<Path>>(vector: &[f64], csv_path: P) -> io::Result<()> {
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;
    wtr.serialize(vector.iter().copied().collect::<Vec<f64>>())?;
    wtr.flush()?;
    Ok(())
}

/// Converts the adjacency matrix edge list to a `faer::Mat<f64>`.
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
/// A `Mat<f64>` that is a square adjacency matrix of size `(end_node - start_node + 1) x (end_node - start_node + 1)`.
pub fn adjacency_matrix_to_dense(
    edges: &[(u32, u32)],
    start_node: usize,
    end_node: usize,
) -> Mat<f64> {
    if end_node < start_node {
        return Mat::<f64>::zeros(0, 0);
    }

    let size = end_node - start_node + 1;
    let mut adj_matrix = Mat::<f64>::zeros(size, size);

    for &(a, b) in edges {
        if (a as usize) >= start_node
            && (a as usize) <= end_node
            && (b as usize) >= start_node
            && (b as usize) <= end_node
        {
            let local_a = (a as usize) - start_node;
            let local_b = (b as usize) - start_node;

            *adj_matrix.get_mut(local_a, local_b) = 1.0;
            *adj_matrix.get_mut(local_b, local_a) = 1.0;
        }
    }

    adj_matrix
}

/// Computes the Normalized Global Eigen-Complexity (NGEC) based on eigenvalues.
/// Ignores eigenvalues that are negative within a small epsilon due to floating-point precision.
///
/// # Arguments
///
/// * `eigenvalues` - A slice containing the eigenvalues.
///
/// # Returns
///
/// * `Ok(f64)` - The computed NGEC value.
/// * `Err(io::Error)` - If the computation fails due to invalid input.
pub fn compute_ngec(eigenvalues: &[f64]) -> io::Result<f64> {
    let epsilon = 1e-9; // Small epsilon to account for floating-point precision
    let m = eigenvalues.len();
    if m == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Eigenvalues array is empty.",
        ));
    }

    if eigenvalues.iter().any(|&x| x < -epsilon) {
        let negative_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .copied()
            .filter(|&x| x < -epsilon)
            .take(5)
            .collect();

        println!(
            "❗ Significant negative eigenvalues found: {:?}",
            negative_eigenvalues
        );

        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Eigenvalues contain significant negative values.",
        ));
    }

    let mut sum_eigen = 0.0;
    for &value in eigenvalues {
        if value >= -epsilon {
            sum_eigen += value;
        }
    }

    if sum_eigen <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Sum of eigenvalues must be positive.",
        ));
    }

    let mut entropy = 0.0;
    for &value in eigenvalues {
        if value >= -epsilon {
            let normalized = value / sum_eigen;
            if normalized > 0.0 {
                entropy += normalized * normalized.ln();
            }
        }
    }

    let log_m = (m as f64).ln();
    if log_m == 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Logarithm of the number of eigenvalues resulted in zero.",
        ));
    }

    Ok(-entropy / log_m)
}

/// Prints a heatmap of a dense matrix to the terminal.
pub fn print_heatmap(matrix: MatRef<'_, f64>) {
    let num_rows = matrix.nrows();
    let num_cols = matrix.ncols();

    let mut non_zero_values = Vec::new();
    for i in 0..num_rows {
        for j in 0..num_cols {
            let value = matrix[(i, j)];
            if value != 0.0 {
                non_zero_values.push(value);
            }
        }
    }

    if non_zero_values.is_empty() {
        println!("(heatmap omitted: matrix is all zeros)");
        return;
    }

    let max_value = non_zero_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_value = non_zero_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut stdout = stdout.lock();

    for i in 0..num_rows {
        for j in 0..num_cols {
            let value = matrix[(i, j)];
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

/// Prints a heatmap of a dense matrix to the terminal with Z-normalization.
pub fn print_heatmap_normalized(matrix: MatRef<'_, f64>) {
    let num_rows = matrix.nrows();
    let num_cols = matrix.ncols();

    if num_rows == 0 || num_cols == 0 {
        println!("(heatmap omitted: empty matrix)");
        return;
    }

    let mut all_values = Vec::with_capacity(num_rows * num_cols);
    for i in 0..num_rows {
        for j in 0..num_cols {
            all_values.push(matrix[(i, j)]);
        }
    }

    let mean = all_values.iter().sum::<f64>() / all_values.len() as f64;
    let variance =
        all_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all_values.len() as f64;
    let stddev = variance.sqrt();

    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut stdout = stdout.lock();

    for i in 0..num_rows {
        for j in 0..num_cols {
            let value = matrix[(i, j)];
            let final_value = if value == 0.0 {
                0.0
            } else if stddev > 0.0 {
                let z_value = (value - mean) / stddev;
                (0.5 * z_value + 0.5).clamp(0.0, 1.0)
            } else {
                0.5
            };

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

/// Prints a heatmap of eigenvalues.
pub fn print_eigenvalues_heatmap(vector: &[f64]) {
    if vector.is_empty() {
        println!("(heatmap omitted: eigenvalue vector is empty)");
        return;
    }

    let max_value = vector.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_value = vector.iter().copied().fold(f64::INFINITY, f64::min);

    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut stdout = stdout.lock();

    for &value in vector {
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
