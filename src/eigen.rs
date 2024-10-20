// src/eigen.rs

//! Module for eigendecomposition.

// Try: export RUSTFLAGS="-llapack -lopenblas"
// export RUSTFLAGS="-L/usr/lib/x86_64-linux-gnu -llapack -lopenblas"

use lapack_sys::dsbevd_;
use ndarray::prelude::*;
use std::ffi::c_char;
use std::os::raw::c_int;
use std::io::{self, Write};
use std::path::Path;
use csv::WriterBuilder;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use nalgebra::{DVector, DMatrix, SymmetricEigen};
use std::cmp::min;

// determine which matrix algorithm to use =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
/// Computes the eigendecomposition of the Laplacian matrix, choosing between LAPACK's dsbevd and SymmetricEigen based on the matrix's bandedness.
pub fn call_eigendecomp(laplacian: &Array2<f64>) -> io::Result<(Array1<f64>, Array2<f64>)> {
    // Compute the maximum bandedness (kd) of the matrix
    let kd = max_band(laplacian);
    let n = laplacian.nrows() as i32;

    // Decide which eigendecomposition method to use based on kd
    if kd < (n as f64 / 3.0).ceil() as i32 {
        // Use LAPACK's dsbevd for banded matrices
        println!("Using LAPACK's dsbevd for banded matrices (kd = {}, n = {})", kd, n);
        compute_eigenvalues_and_vectors_sym_band(laplacian, kd)
    } else {
        // Use nalgebra's SymmetricEigen for non-banded or less banded matrices
        println!("Using nalgebra's SymmetricEigen for non-banded matrices (kd = {}), n = {}", kd, n);
        let (eigvals, eigvecs) = compute_eigenvalues_and_vectors_sym(laplacian)?;

        // Convert nalgebra's DVector and DMatrix to ndarray's Array1 and Array2
        let eigvals_nd = Array1::from(eigvals.iter().cloned().collect::<Vec<f64>>());
        let eigvecs_nd = Array2::from_shape_vec(
            (eigvecs.nrows(), eigvecs.ncols()),
            eigvecs.iter().cloned().collect(),
        ).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        Ok((eigvals_nd, eigvecs_nd))
    }
}


/// Computes the maximum bandedness (`kd`) of a symmetric matrix.
/// The bandedness is determined by finding the farthest diagonal from the main diagonal
/// that contains a non-zero element. `kd` is set to the distance of this diagonal plus one.
pub fn max_band(laplacian: &Array2<f64>) -> i32 {
    let n = laplacian.nrows() as i32;

    // Iterate from the outermost upper diagonal towards the main diagonal
    for k in (1..n).rev() {
        let mut has_non_zero = false;
        for i in 0..(n - k) {
            if laplacian[[i as usize, (i + k) as usize]] != 0.0 {
                has_non_zero = true;
                break;
            }
        }
        if has_non_zero {
            return k + 1; // Add one for good measure
        }
    }

    // If all upper diagonals are zero, set kd to 1 (only main diagonal is non-zero)
    1
}

// dsbevd eigendecomposition section =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

/// Converts a 2D matrix to a banded matrix representation required for dsbevd
pub fn to_banded_format(matrix: &Array2<f64>, kd: i32) -> Array2<f64> {
    let (n, _) = matrix.dim();
    let mut banded = Array2::<f64>::zeros(((kd + 1) as usize, n));

    for j in 0..n {
        for i in j..min(n, (j as i32 + kd + 1) as usize) {
            let row = (i - j) as usize;
            let col = j as usize;
            banded[[row, col]] = matrix[[i, j]];
        }
    }

    banded
}

/// Computes eigenvalues and eigenvectors for a symmetric band matrix using LAPACK's dsbevd
pub fn compute_eigenvalues_and_vectors_sym_band(
    laplacian: &Array2<f64>,
    kd: i32,
) -> io::Result<(Array1<f64>, Array2<f64>)> {
    let n = laplacian.nrows() as c_int;

    // Convert to the banded format expected by dsbevd
    let mut banded_matrix = to_banded_format(laplacian, kd)
        .reversed_axes() // Convert to column-major order
        .into_raw_vec_and_offset();

    // Initialize workspace query parameters
    let jobz = b'V' as c_char; // Compute eigenvalues and eigenvectors
    let uplo = b'L' as c_char; // Lower triangle

    // Workspace query: set LWORK = -1 and LIWORK = -1
    let mut work_query = vec![0.0_f64];
    let mut iwork_query = vec![0_i32];
    let lwork_query = -1;
    let liwork_query = -1;
    let mut info: c_int = 0;

    let mut eigvals_dummy = vec![0.0_f64; 1];
    let mut eigvecs_dummy = vec![0.0_f64; 1];

    // TODO: Since this can cause a segfault, the program should catch this and fallback to the safe method
    unsafe {
        dsbevd_(
            &jobz,
            &uplo,
            &n,
            &kd,
            banded_matrix.0.as_mut_ptr(),
            &(kd + 1),
            eigvals_dummy.as_mut_ptr(),
            eigvecs_dummy.as_mut_ptr(),
            &n,
            work_query.as_mut_ptr(),
            &lwork_query,
            iwork_query.as_mut_ptr(),
            &liwork_query,
            &mut info,
        );
    }

    if info != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("LAPACK dsbevd (workspace query) failed with error code {}", info),
        ));
    }

    // Extract optimal LWORK and LIWORK from the workspace query
    let optimal_lwork = work_query[0] as usize;
    let optimal_liwork = iwork_query[0] as usize;

    // Allocate WORK and IWORK with optimal sizes
    let mut work = vec![0.0_f64; optimal_lwork];
    let mut iwork = vec![0_i32; optimal_liwork];
    let mut eigvals = vec![0.0_f64; n as usize];
    let mut eigvecs = vec![0.0_f64; (n * n) as usize]; // Flat array for eigenvectors

    // Perform the actual eigendecomposition with optimal WORK and IWORK
    unsafe {
        dsbevd_(
            &jobz,
            &uplo,
            &n,
            &kd,
            banded_matrix.0.as_mut_ptr(),
            &(kd + 1),
            eigvals.as_mut_ptr(),
            eigvecs.as_mut_ptr(),
            &n,
            work.as_mut_ptr(),
            &(optimal_lwork as c_int),
            iwork.as_mut_ptr(),
            &(optimal_liwork as c_int),
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

/// Converts the adjacency matrix edge list to ndarray::Array2<f64>
pub fn adjacency_matrix_to_ndarray(
    edges: &[(u32, u32)],
    start_node: usize,
    end_node: usize,
) -> Array2<f64> {
    let size = end_node - start_node + 1;
    let mut adj_array = Array2::<f64>::zeros((size, size));
    for &(a, b) in edges {
        let local_a = a as usize - start_node;
        let local_b = b as usize - start_node;
        if local_a < size && local_b < size {
            adj_array[(local_a, local_b)] = 1.0;
            adj_array[(local_b, local_a)] = 1.0;
        }
    }
    adj_array
}

/// Saves a 2D ndarray::Array2<f64> to a CSV file
pub fn save_array_to_csv_dsbevd<P: AsRef<Path>>(matrix: &Array2<f64>, csv_path: P) -> io::Result<()> {
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
pub fn save_vector_to_csv_dsbevd<P: AsRef<Path>>(vector: &Array1<f64>, csv_path: P) -> io::Result<()> {
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path(csv_path)?;
    let row = vector.iter().cloned().collect::<Vec<f64>>();
    wtr.serialize(row)?;
    wtr.flush()?;
    Ok(())
}

// SymmetricEigen eigendecomposition section =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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


// Compute Normalized Global Eigen-Complexity (NGEC) =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

/// Computes the Normalized Global Eigen-Complexity (NGEC) based on eigenvalues.
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
    let m = eigenvalues.len();
    if m == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Eigenvalues array is empty.",
        ));
    }

    // Check for negative eigenvalues
    if eigenvalues.iter().any(|&x| x < 0.0) {
        let negative_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .filter(|&&x| x < 0.0)
            .cloned()
            .take(5)
            .collect();
    
        println!("❗ Negative eigenvalues found: {:?}", negative_eigenvalues);
    
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Eigenvalues contain negative values."
            ),
        ));
    }

    // Calculate the sum of eigenvalues
    let sum_eigen = eigenvalues.sum();
    if sum_eigen <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Sum of eigenvalues must be positive.",
        ));
    }

    // Normalize the eigenvalues
    let normalized_eigen = eigenvalues.mapv(|x| x / sum_eigen);

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


// Load and output section =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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
