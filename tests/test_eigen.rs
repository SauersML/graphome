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

/// Test the `compute_eigenvalues_and_vectors_sym` function with a known symmetric matrix.
#[test]
fn test_compute_eigenvalues_and_vectors_sym_known_matrix() {
    // Define a known symmetric matrix
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

    // Sort eigenvalues for consistent comparison
    let mut eigvals_sym_sorted = eigvals_sym.to_owned();
    let mut eigvals_lapack_sorted = eigvals_lapack.to_owned();
    eigvals_sym_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigvals_lapack_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare eigenvalues within the specified tolerance
    for (computed, reference) in eigvals_sym_sorted.iter().zip(eigvals_lapack_sorted.iter()) {
        assert!(
            (*computed - *reference).abs() <= TOLERANCE,
            "Eigenvalues mismatch: computed = {}, reference = {}, difference = {}",
            computed,
            reference,
            (*computed - *reference).abs()
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

    // Sort eigenvalues for consistent comparison
    let mut eigvals_lapack_sorted = eigvals_lapack.to_owned();
    let mut eigvals_sym_sorted = eigvals_sym.to_owned();
    eigvals_lapack_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigvals_sym_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare eigenvalues within the specified tolerance
    for (computed, reference) in eigvals_lapack_sorted.iter().zip(eigvals_sym_sorted.iter()) {
        assert!(
            (*computed - *reference).abs() <= TOLERANCE,
            "Eigenvalues mismatch: computed = {}, reference = {}, difference = {}",
            computed,
            reference,
            (*computed - *reference).abs()
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
        [3.0, -1.0,  0.0,  0.0],
        [-1.0, 3.0, -1.0,  0.0],
        [0.0, -1.0, 3.0, -1.0],
        [0.0,  0.0, -1.0, 3.0]
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

    // Sort eigenvalues for consistent comparison
    let mut eigvals_lapack_sorted = eigvals_lapack.to_owned();
    let mut eigvals_sym_sorted = eigvals_sym.to_owned();
    eigvals_lapack_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigvals_sym_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare eigenvalues within the specified tolerance
    for (computed, reference) in eigvals_lapack_sorted.iter().zip(eigvals_sym_sorted.iter()) {
        assert!(
            (*computed - *reference).abs() <= TOLERANCE,
            "Eigenvalues mismatch: computed = {}, reference = {}, difference = {}",
            computed,
            reference,
            (*computed - *reference).abs()
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
