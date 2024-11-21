// tests/test_eigen.rs

//! Unit tests for the eigen module.

use ndarray::array;
use graphome::eigen::{
    call_eigendecomp,
    compute_eigenvalues_and_vectors_sym,
    compute_ngec,
    ndarray_to_nalgebra_matrix, // Add this import
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

fn to_banded_format(matrix: &Array2<f64>, kd: usize) -> Array2<f64> {
    let n = matrix.nrows();
    let mut banded_matrix = Array2::zeros((kd + 1, n));

    for i in 0..n {
        for j in 0..n {
            let k = (j - i).abs();
            if k <= kd {
                banded_matrix[[kd - k, i]] = matrix[[i, j]];
            }
        }
    }
    banded_matrix
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

fn max_band(matrix: &Array2<f64>) -> usize {
    let n = matrix.nrows();
    let mut kd = 0;
    for i in 0..n {
        for j in 0..n {
            if matrix[[i, j]] != 0.0 {
                kd = std::cmp::max(kd, (i - j).abs());
            }
        }
    }
    kd as usize
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
