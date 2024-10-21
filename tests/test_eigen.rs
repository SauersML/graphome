// tests/test_eigen.rs

//! Unit tests for the eigen module.

use ndarray::array;
use graphome::eigen::{call_eigendecomp, save_array_to_csv_dsbevd, compute_ngec};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::fs;

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
