// src/eigen_print.rs

//! Module for eigendecomposition and analyses that depend on eigenvectors or eigenvalues.

use csv::WriterBuilder;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, MatRef, Side};
use ndarray::{Array2, ArrayView2};
use std::io::{self, Write};
use std::path::Path;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

/// Supported input types for [`call_eigendecomp`].
pub enum LaplacianInput<'a> {
    /// A view into a Faer matrix. This is the fast path that avoids allocating.
    Borrowed(MatRef<'a, f64>),
    /// An owned Faer matrix produced from another representation.
    Owned(Mat<f64>),
}

/// Trait that converts various Laplacian representations into a form usable by faer.
pub trait IntoLaplacian<'a> {
    fn into_laplacian(self) -> LaplacianInput<'a>;
}

impl<'a> IntoLaplacian<'a> for MatRef<'a, f64> {
    fn into_laplacian(self) -> LaplacianInput<'a> {
        LaplacianInput::Borrowed(self)
    }
}

impl<'a> IntoLaplacian<'a> for &'a Mat<f64> {
    fn into_laplacian(self) -> LaplacianInput<'a> {
        LaplacianInput::Borrowed(self.as_ref())
    }
}

impl<'a> IntoLaplacian<'a> for &'a Array2<f64> {
    fn into_laplacian(self) -> LaplacianInput<'a> {
        self.view().into_laplacian()
    }
}

impl<'a> IntoLaplacian<'a> for ArrayView2<'a, f64> {
    fn into_laplacian(self) -> LaplacianInput<'a> {
        ndarray_to_laplacian_input(self)
    }
}

impl<'a> IntoLaplacian<'a> for Array2<f64> {
    fn into_laplacian(self) -> LaplacianInput<'a> {
        LaplacianInput::Owned(copy_array_into_mat(self.view()))
    }
}

fn ndarray_to_laplacian_input<'a>(array: ArrayView2<'a, f64>) -> LaplacianInput<'a> {
    LaplacianInput::Owned(copy_array_into_mat(array))
}

fn copy_array_into_mat(array: ArrayView2<'_, f64>) -> Mat<f64> {
    let (nrows, ncols) = array.dim();
    let mut mat = Mat::<f64>::zeros(nrows, ncols);

    if nrows == 0 || ncols == 0 {
        return mat;
    }

    if let Some(slice) = array.as_slice_memory_order() {
        let view = MatRef::from_row_major_slice(slice, nrows, ncols);
        mat.as_mut().copy_from(view);
        return mat;
    }

    for (i, row) in array.outer_iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            *mat.get_mut(i, j) = *value;
        }
    }

    mat
}

/// Computes the eigendecomposition of the Laplacian matrix using faer's self-adjoint solver.
///
/// This helper now accepts either a Faer matrix view (`MatRef`) or an ndarray 2D array. When an
/// ndarray is provided, the data is copied into a temporary Faer matrix before invoking the
/// solver. This maintains backwards compatibility for existing callers while allowing the tests
/// (and any downstream users) to supply Laplacians that are manipulated as ndarray matrices.
pub fn call_eigendecomp<'a, T>(laplacian: T) -> io::Result<(Vec<f64>, Mat<f64>)>
where
    T: IntoLaplacian<'a>,
{
    println!("Using faer's self-adjoint eigensolver for the matrix.");

    match laplacian.into_laplacian() {
        LaplacianInput::Borrowed(view) => compute_eigenvalues_and_vectors_sym(view),
        LaplacianInput::Owned(mat) => compute_eigenvalues_and_vectors_sym(mat.as_ref()),
    }
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
        .map_err(|err| io::Error::other(format!("{:?}", err)))?;

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
    wtr.serialize(vector.to_vec())?;
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

/// Converts an adjacency edge list into an `ndarray::Array2` adjacency matrix.
///
/// The semantics match [`adjacency_matrix_to_dense`]: edges are treated as undirected, so both
/// `(a, b)` and `(b, a)` entries are set to 1.0 whenever an edge falls within the requested node
/// range.
pub fn adjacency_matrix_to_ndarray(
    edges: &[(u32, u32)],
    start_node: usize,
    end_node: usize,
) -> Array2<f64> {
    if end_node < start_node {
        return Array2::zeros((0, 0));
    }

    let size = end_node - start_node + 1;
    let mut adj_matrix = Array2::<f64>::zeros((size, size));

    for &(a, b) in edges {
        if (a as usize) >= start_node
            && (a as usize) <= end_node
            && (b as usize) >= start_node
            && (b as usize) <= end_node
        {
            let local_a = (a as usize) - start_node;
            let local_b = (b as usize) - start_node;

            adj_matrix[(local_a, local_b)] = 1.0;
            adj_matrix[(local_b, local_a)] = 1.0;
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
