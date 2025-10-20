//! Unit tests for the eigen module.

use graphome::eigen_print::{
    adjacency_matrix_to_ndarray, compute_eigenvalues_and_vectors_sym, compute_ngec,
    save_vector_to_csv,
};
use ndarray::{array, Array1};

use std::io::ErrorKind;

/// Test that all eigenvalues computed by faer are non-negative.
#[test]
fn test_non_negative_eigenvalues_symmetric() {
    // Define a small symmetric Laplacian matrix
    let laplacian = array![[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];

    // Perform eigendecomposition using faer
    let (eigvals_sym, _) = compute_eigenvalues_and_vectors_sym(&laplacian)
        .expect("Eigendecomposition with faer failed");

    for &val in eigvals_sym.iter() {
        assert!(val >= 0.0, "faer eigenvalue is negative: {}", val);
    }
}

// Assuming this test uses some functions from src/eigen.rs,  I'll add some example tests for other likely functions:

#[test]
fn test_adjacency_matrix_to_ndarray() {
    let edges = vec![(0, 1), (1, 2), (0, 2)];
    let start_node = 0;
    let end_node = 2;
    let expected = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0],];

    let adj_matrix = adjacency_matrix_to_ndarray(&edges, start_node, end_node);

    assert_eq!(adj_matrix, expected);
}

#[test]
fn test_compute_ngec_empty_eigenvalues() {
    let eigenvalues = ndarray::Array1::<f64>::zeros(0);
    let result = compute_ngec(&eigenvalues);
    assert!(matches!(result, Err(e) if e.kind() == ErrorKind::InvalidInput));
}

#[test]
fn test_compute_ngec_with_negative_eigenvalue() {
    let eigenvalues = ndarray::array![-2.0, 1.0, 0.5]; // Negative eigenvalue present
    let epsilon = 1e-9;
    if eigenvalues.iter().any(|&x| x < -epsilon) {
        let result = compute_ngec(&eigenvalues);
        assert!(matches!(result, Err(e) if e.kind() == ErrorKind::InvalidInput));
    }
}

#[test]
fn test_save_and_load_vector_csv() {
    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let tmp_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    let csv_path = tmp_file.path();

    save_vector_to_csv(&vector, csv_path).expect("Failed to save vector to CSV");

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(csv_path)
        .expect("Failed to create CSV reader");

    let loaded_row: Vec<f64> = rdr
        .deserialize()
        .next()
        .expect("Failed to read row")
        .expect("Failed to deserialize");

    let loaded_vector = Array1::from_vec(loaded_row);

    assert_eq!(
        vector.to_vec(),
        loaded_vector.to_vec(),
        "Loaded vector does not match the original vector."
    );
}
