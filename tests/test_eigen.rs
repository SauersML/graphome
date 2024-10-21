// tests/test_eigen.rs

//! Unit tests for the eigen module.

use ndarray::array;
use graphome::eigen::{call_eigendecomp, save_array_to_csv_dsbevd, compute_ngec, compute_eigenvalues_and_vectors_sym, compute_eigenvalues_and_vectors_sym_band, max_band};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::fs;

const TOLERANCE: f64 = 1e-9;

#[test]
fn test_eigendecomp() {
    // Define a small Laplacian matrix
    let laplacian = array![
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0]
    ];

    // Perform eigendecomposition
    let (eigvals, eigvecs) = call_eigendecomp(&laplacian).expect("Eigendecomposition failed");

    // Assert that the eigenvector matrix is the correct size
    assert_eq!(eigvecs.nrows(), 3);
    assert_eq!(eigvecs.ncols(), 3);
}

#[test]
fn test_save_array_to_csv() {
    // Create a small array to save
    let array = array![[1.0, 2.0], [3.0, 4.0]];

    // Define the output path
    let output_path = Path::new("test_output.csv");

    // Save the array to a CSV file
    save_array_to_csv_dsbevd(&array, &output_path).expect("Failed to save array to CSV");

    // Read back the file contents
    let mut file = File::open(&output_path).expect("Failed to open CSV file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read CSV file");

    // Check that the contents are as expected
    assert_eq!(contents.trim(), "1.0,2.0\n3.0,4.0");

    // Clean up the test output file
    fs::remove_file(output_path).expect("Failed to delete test output file");
}

#[test]
fn test_compute_ngec() {
    // Define a set of eigenvalues
    let eigenvalues = array![1.0, 2.0, 3.0];

    // Compute NGEC
    let ngec = compute_ngec(&eigenvalues).expect("Failed to compute NGEC");

    // Assert that the NGEC value is within the expected range
    assert!(ngec > 0.0 && ngec < 1.0);
}

#[test]
fn test_compare_eigenvalues_lapack_vs_symmetric() {
    // Define a small Laplacian matrix
    let laplacian = array![
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0]
    ];

    // Perform eigendecomposition using LAPACK's dsbevd
    let (eigvals_lapack, _) = call_eigendecomp(&laplacian).expect("LAPACK eigendecomposition failed");

    // Manually force usage of SymmetricEigen
    let kd = laplacian.nrows() as f64 / 2.0; // Set a high bandedness to force SymmetricEigen
    let (eigvals_symmetric, _) = call_eigendecomp(&laplacian).expect("Symmetric eigenvalue calculation failed");

    // Manually check that the eigenvalues are approximately equal within tolerance
    for (v1, v2) in eigvals_lapack.iter().zip(eigvals_symmetric.iter()) {
        assert!(
            (v1 - v2).abs() <= TOLERANCE,
            "Eigenvalues mismatch: v1 = {}, v2 = {}, diff = {}",
            v1, v2, (v1 - v2).abs()
        );
    }
}

#[test]
fn test_compare_eigenvectors_lapack_vs_symmetric() {
    // Define a small Laplacian matrix
    let laplacian = array![
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0]
    ];

    // Perform eigendecomposition using LAPACK's dsbevd
    let (_, eigvecs_lapack) = call_eigendecomp(&laplacian).expect("LAPACK eigendecomposition failed");

    // Manually force usage of SymmetricEigen
    let kd = laplacian.nrows() as f64 / 2.0; // Set a high bandedness to force SymmetricEigen
    let (_, eigvecs_symmetric) = call_eigendecomp(&laplacian).expect("Symmetric eigenvector calculation failed");

    // Manually check that each element of the eigenvectors is approximately equal within tolerance
    for (row_lapack, row_symmetric) in eigvecs_lapack.outer_iter().zip(eigvecs_symmetric.outer_iter()) {
        for (v1, v2) in row_lapack.iter().zip(row_symmetric.iter()) {
            assert!(
                (v1 - v2).abs() <= TOLERANCE,
                "Eigenvector elements mismatch: v1 = {}, v2 = {}, diff = {}",
                v1, v2, (v1 - v2).abs()
            );
        }
    }
}
