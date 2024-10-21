// tests/test_eigen.rs

//! Unit tests for the eigen module.

use ndarray::array;
use graphome::eigen::{
    call_eigendecomp,
    compute_eigenvalues_and_vectors_sym,
    compute_eigenvalues_and_vectors_sym_band,
    compute_ngec,
    max_band,
    save_array_to_csv_dsbevd,
    to_banded_format,
};
use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use nalgebra::DMatrix;

const TOLERANCE: f64 = 1e-6;

/// Test the `to_banded_format` function with a known symmetric matrix and `kd = 1`.
#[test]
fn test_to_banded_format_kd_1() {
    // Define a known symmetric matrix
    let matrix = array![
        [1.0, 2.0, 0.0],
        [2.0, 3.0, 4.0],
        [0.0, 4.0, 5.0]
    ];
    let kd = 1;

    // Define the expected banded format for kd = 1 based on the `to_banded_format` implementation
    // According to the function, the main diagonal is in the last row,
    // and upper diagonals are above.
    let expected_banded = array![
        [0.0, 2.0, 4.0], // Upper diagonal (k = 1)
        [1.0, 3.0, 5.0]  // Main diagonal
    ];

    // Convert the matrix to banded format
    let banded = to_banded_format(&matrix, kd);

    // Assert that the banded matrix matches the expected format
    assert_eq!(
        banded,
        expected_banded,
        "Banded matrix with kd = 1 does not match the expected output."
    );
}

/// Test the `to_banded_format` function with a known symmetric matrix and `kd = 2`.
#[test]
fn test_to_banded_format_kd_2() {
    // Define a known symmetric matrix
    let matrix = array![
        [4.0, 1.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ];
    let kd = 2;

    // Define the expected banded format for kd = 2 based on the `to_banded_format` implementation
    let expected_banded = array![
        [0.0, 0.0, 0.0, 0.0], // k = 2 (No elements beyond the second upper diagonal)
        [0.0, 1.0, 1.0, 1.0], // k = 1
        [4.0, 3.0, 2.0, 1.0]  // Main diagonal
    ];

    // Convert the matrix to banded format
    let banded = to_banded_format(&matrix, kd);

    // Assert that the banded matrix matches the expected format
    assert_eq!(
        banded,
        expected_banded,
        "Banded matrix with kd = 2 does not match the expected output."
    );
}

/// Test the `compute_eigenvalues_and_vectors_sym_band` function with a known symmetric banded matrix.
#[test]
fn test_compute_eigenvalues_and_vectors_sym_band_known_matrix() {
    // Define a known symmetric banded Laplacian matrix
    let laplacian = array![
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0]
    ];
    let kd = max_band(&laplacian);

    // Perform eigendecomposition using LAPACK's dsbevd
    let (eigvals_lapack, eigvecs_lapack) =
        compute_eigenvalues_and_vectors_sym_band(&laplacian, kd)
            .expect("Eigendecomposition with dsbevd failed");

    // Perform eigendecomposition using SymmetricEigen for comparison
    let (eigvals_sym, eigvecs_sym) =
        compute_eigenvalues_and_vectors_sym(&laplacian)
            .expect("Eigendecomposition with SymmetricEigen failed");

    // Convert eigenvalues to vectors for sorting
    let mut eigvals_lapack_sorted = eigvals_lapack.to_vec();
    let mut eigvals_sym_sorted = eigvals_sym.as_slice().to_vec();

    // Sort the eigenvalues for consistent comparison
    eigvals_lapack_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigvals_sym_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare eigenvalues within the specified tolerance
    for (computed, reference) in eigvals_lapack_sorted.iter().zip(eigvals_sym_sorted.iter()) {
        assert!(
            (*computed - *reference).abs() <= TOLERANCE,
            "Eigenvalues mismatch: computed = {}, reference = {}, difference = {}",
            computed,
            reference,
            (*computed - reference).abs()
        );
    }

    // Compare eigenvectors within the specified tolerance
    // Handle potential sign differences
    for col in 0..eigvecs_lapack.ncols() {
        for row in 0..eigvecs_lapack.nrows() {
            let v1 = eigvecs_lapack[(row, col)];
            let v2 = eigvecs_sym[(row, col)];
            let diff = (v1 - v2).abs();
            let diff_neg = (v1 + v2).abs();
            assert!(
                diff <= TOLERANCE || diff_neg <= TOLERANCE,
                "Eigenvector elements mismatch at ({}, {}): v1 = {}, v2 = {}, |v1 - v2| = {}, |v1 + v2| = {}",
                row,
                col,
                v1,
                v2,
                diff,
                diff_neg
            );
        }
    }
}

/// Test the `compute_eigenvalues_and_vectors_sym` function with a known symmetric matrix.
#[test]
fn test_compute_eigenvalues_and_vectors_sym_known_matrix() {
    // Define a known symmetric Laplacian matrix
    let laplacian = array![
        [3.0, -1.0, -1.0],
        [-1.0, 3.0, -1.0],
        [-1.0, -1.0, 3.0]
    ];

    // Perform eigendecomposition using SymmetricEigen
    let (eigvals_sym, eigvecs_sym) =
        compute_eigenvalues_and_vectors_sym(&laplacian)
            .expect("Eigendecomposition with SymmetricEigen failed");

    // Perform eigendecomposition using LAPACK's dsbevd for comparison
    let kd = max_band(&laplacian);
    let (eigvals_lapack, eigvecs_lapack) =
        compute_eigenvalues_and_vectors_sym_band(&laplacian, kd)
            .expect("Eigendecomposition with dsbevd failed");

    // Convert eigenvalues to vectors for sorting
    let mut eigvals_sym_sorted = eigvals_sym.as_slice().to_vec();
    let mut eigvals_lapack_sorted = eigvals_lapack.to_vec();

    // Sort the eigenvalues for consistent comparison
    eigvals_sym_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigvals_lapack_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare eigenvalues within the specified tolerance
    for (computed, reference) in eigvals_sym_sorted.iter().zip(eigvals_lapack_sorted.iter()) {
        assert!(
            (*computed - *reference).abs() <= TOLERANCE,
            "Eigenvalues mismatch: computed = {}, reference = {}, difference = {}",
            computed,
            reference,
            (*computed - reference).abs()
        );
    }

    // Compare eigenvectors within the specified tolerance
    // Handle potential sign differences
    for col in 0..eigvecs_sym.ncols() {
        for row in 0..eigvecs_sym.nrows() {
            let v1 = eigvecs_sym[(row, col)];
            let v2 = eigvecs_lapack[(row, col)];
            let diff = (v1 - v2).abs();
            let diff_neg = (v1 + v2).abs();
            assert!(
                diff <= TOLERANCE || diff_neg <= TOLERANCE,
                "Eigenvector elements mismatch at ({}, {}): v1 = {}, v2 = {}, |v1 - v2| = {}, |v1 + v2| = {}",
                row,
                col,
                v1,
                v2,
                diff,
                diff_neg
            );
        }
    }
}

/// Test the `compute_ngec` function with a set of positive eigenvalues.
#[test]
fn test_compute_ngec_with_positive_eigenvalues() {
    // Define a set of positive eigenvalues
    let eigenvalues = array![1.0, 2.0, 3.0];

    // Compute NGEC
    let ngec = compute_ngec(&eigenvalues)
        .expect("Failed to compute NGEC with positive eigenvalues");

    // Assert that NGEC is within the expected range (0, 1)
    assert!(
        ngec > 0.0 && ngec < 1.0,
        "NGEC value out of expected range: {}",
        ngec
    );
}

/// Test the `compute_ngec` function with a set of eigenvalues containing significant negative values.
#[test]
fn test_compute_ngec_with_negative_eigenvalues() {
    // Define a set of eigenvalues with a significant negative value
    let eigenvalues = array![1.0, -2.0, 3.0];

    // Compute NGEC and expect an error
    let result = compute_ngec(&eigenvalues);

    // Assert that an error is returned
    assert!(
        result.is_err(),
        "Expected an error due to negative eigenvalues, but computation succeeded."
    );

    if let Err(e) = result {
        assert!(
            e.to_string().contains("significant negative values"),
            "Unexpected error message: {}",
            e
        );
    }
}

/// Test the `compute_ngec` function with an empty eigenvalues array.
#[test]
fn test_compute_ngec_with_empty_eigenvalues() {
    // Define an empty eigenvalues array
    let eigenvalues = array![];

    // Compute NGEC and expect an error
    let result = compute_ngec(&eigenvalues);

    // Assert that an error is returned
    assert!(
        result.is_err(),
        "Expected an error due to empty eigenvalues array, but computation succeeded."
    );

    if let Err(e) = result {
        assert!(
            e.to_string().contains("Eigenvalues array is empty"),
            "Unexpected error message: {}",
            e
        );
    }
}

/// Test the `save_array_to_csv_dsbevd` function by saving and reading back a known array.
#[test]
fn test_save_array_to_csv_dsbevd() {
    // Create a small array to save
    let array = array![[1.0, 2.0], [3.0, 4.0]];
    let output_path = Path::new("test_output_dsbevd.csv");

    // Save the array to a CSV file
    save_array_to_csv_dsbevd(&array, &output_path)
        .expect("Failed to save array to CSV using dsbevd");

    // Read back the file contents
    let mut file = File::open(&output_path)
        .expect("Failed to open CSV file saved by dsbevd");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read CSV file saved by dsbevd");

    // Define the expected CSV content
    let expected_contents = "1.0,2.0\n3.0,4.0";

    assert_eq!(
        contents.trim(),
        expected_contents,
        "CSV contents do not match expected output."
    );

    // Clean up the test output file
    fs::remove_file(output_path)
        .expect("Failed to delete test output CSV file saved by dsbevd");
}

/// Test the `call_eigendecomp` function with a small symmetric matrix.
#[test]
fn test_call_eigendecomp_with_small_matrix() {
    // Define a small symmetric Laplacian matrix
    let laplacian = array![
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0]
    ];

    // Perform eigendecomposition using `call_eigendecomp`
    let (eigvals, eigvecs) =
        call_eigendecomp(&laplacian).expect("Eigendecomposition using call_eigendecomp failed");

    // Assert that the eigenvalues array has the correct length
    assert_eq!(eigvals.len(), 3, "Eigenvalues array length mismatch.");

    // Assert that the eigenvectors matrix has the correct dimensions
    assert_eq!(
        eigvecs.dim(),
        (3, 3),
        "Eigenvectors matrix dimensions mismatch."
    );

    // Assert that all eigenvalues are non-negative
    for &val in eigvals.iter() {
        assert!(
            val >= 0.0,
            "Eigenvalue is negative: {}",
            val
        );
    }
}

/// Test the `compute_eigenvalues_and_vectors_sym_band` function with a diagonal matrix and `kd = 0`.
#[test]
fn test_compute_eigenvalues_and_vectors_sym_band_diagonal_matrix() {
    // Define a diagonal matrix
    let matrix = array![
        [5.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    let kd = 0;

    // Perform eigendecomposition using LAPACK's dsbevd
    let (eigvals_lapack, eigvecs_lapack) =
        compute_eigenvalues_and_vectors_sym_band(&matrix, kd)
            .expect("Eigendecomposition with dsbevd failed for diagonal matrix");

    // Perform eigendecomposition using SymmetricEigen for comparison
    let (eigvals_sym, eigvecs_sym) =
        compute_eigenvalues_and_vectors_sym(&matrix)
            .expect("Eigendecomposition with SymmetricEigen failed for diagonal matrix");

    // Convert eigenvalues to vectors for sorting
    let mut eigvals_lapack_sorted = eigvals_lapack.to_vec();
    let mut eigvals_sym_sorted = eigvals_sym.as_slice().to_vec();

    // Sort the eigenvalues for consistent comparison
    eigvals_lapack_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigvals_sym_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare eigenvalues within the specified tolerance
    for (computed, reference) in eigvals_lapack_sorted.iter().zip(eigvals_sym_sorted.iter()) {
        assert!(
            (*computed - *reference).abs() <= TOLERANCE,
            "Eigenvalues mismatch: computed = {}, reference = {}, difference = {}",
            computed,
            reference,
            (*computed - reference).abs()
        );
    }

    // Compare eigenvectors within the specified tolerance
    // For diagonal matrices, eigenvectors should be the standard basis vectors
    for col in 0..eigvecs_lapack.ncols() {
        for row in 0..eigvecs_lapack.nrows() {
            let v1 = eigvecs_lapack[(row, col)];
            let v2 = eigvecs_sym[(row, col)];
            let diff = (v1 - v2).abs();
            let diff_neg = (v1 + v2).abs();
            assert!(
                diff <= TOLERANCE || diff_neg <= TOLERANCE,
                "Eigenvector elements mismatch at ({}, {}): v1 = {}, v2 = {}, |v1 - v2| = {}, |v1 + v2| = {}",
                row,
                col,
                v1,
                v2,
                diff,
                diff_neg
            );
        }
    }
}

/// Test the `compute_eigenvalues_and_vectors_sym_band` function with a larger symmetric banded matrix.
#[test]
fn test_compute_eigenvalues_and_vectors_sym_band_larger_matrix() {
    // Define a larger symmetric banded Laplacian matrix
    let laplacian = array![
        [3.0, -1.0, -1.0, 0.0],
        [-1.0, 3.0, -1.0, -1.0],
        [-1.0, -1.0, 3.0, -1.0],
        [0.0, -1.0, -1.0, 3.0]
    ];
    let kd = max_band(&laplacian);

    // Perform eigendecomposition using LAPACK's dsbevd
    let (eigvals_lapack, eigvecs_lapack) =
        compute_eigenvalues_and_vectors_sym_band(&laplacian, kd)
            .expect("Eigendecomposition with dsbevd failed for larger matrix");

    // Perform eigendecomposition using SymmetricEigen for comparison
    let (eigvals_sym, eigvecs_sym) =
        compute_eigenvalues_and_vectors_sym(&laplacian)
            .expect("Eigendecomposition with SymmetricEigen failed for larger matrix");

    // Convert eigenvalues to vectors for sorting
    let mut eigvals_lapack_sorted = eigvals_lapack.to_vec();
    let mut eigvals_sym_sorted = eigvals_sym.as_slice().to_vec();

    // Sort the eigenvalues for consistent comparison
    eigvals_lapack_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigvals_sym_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare eigenvalues within the specified tolerance
    for (computed, reference) in eigvals_lapack_sorted.iter().zip(eigvals_sym_sorted.iter()) {
        assert!(
            (*computed - *reference).abs() <= TOLERANCE,
            "Eigenvalues mismatch: computed = {}, reference = {}, difference = {}",
            computed,
            reference,
            (*computed - reference).abs()
        );
    }

    // Compare eigenvectors within the specified tolerance
    // Handle potential sign differences
    for col in 0..eigvecs_lapack.ncols() {
        for row in 0..eigvecs_lapack.nrows() {
            let v1 = eigvecs_lapack[(row, col)];
            let v2 = eigvecs_sym[(row, col)];
            let diff = (v1 - v2).abs();
            let diff_neg = (v1 + v2).abs();
            assert!(
                diff <= TOLERANCE || diff_neg <= TOLERANCE,
                "Eigenvector elements mismatch at ({}, {}): v1 = {}, v2 = {}, |v1 - v2| = {}, |v1 + v2| = {}",
                row,
                col,
                v1,
                v2,
                diff,
                diff_neg
            );
        }
    }
}

/// Test that all eigenvalues computed by LAPACK are non-negative.
#[test]
fn test_non_negative_eigenvalues_lapack() {
    // Define a small symmetric Laplacian matrix
    let laplacian = array![
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0]
    ];
    let kd = max_band(&laplacian);

    // Perform eigendecomposition using LAPACK's dsbevd
    let (eigvals_lapack, _) =
        compute_eigenvalues_and_vectors_sym_band(&laplacian, kd)
            .expect("Eigendecomposition with dsbevd failed");

    for &val in eigvals_lapack.iter() {
        assert!(
            val >= 0.0,
            "LAPACK eigenvalue is negative: {}",
            val
        );
    }
}

/// Test that all eigenvalues computed by SymmetricEigen are non-negative.
#[test]
fn test_non_negative_eigenvalues_symmetric() {
    // Define a small symmetric Laplacian matrix
    let laplacian = array![
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0]
    ];

    // Perform eigendecomposition using SymmetricEigen
    let (eigvals_sym, _) =
        compute_eigenvalues_and_vectors_sym(&laplacian)
            .expect("Eigendecomposition with SymmetricEigen failed");

    for &val in eigvals_sym.iter() {
        assert!(
            val >= 0.0,
            "SymmetricEigen eigenvalue is negative: {}",
            val
        );
    }
}
