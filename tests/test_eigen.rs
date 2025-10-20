use graphome::eigen_print::{
    adjacency_matrix_to_ndarray, call_eigendecomp, compute_eigenvalues_and_vectors_sym,
    compute_ngec, save_array1_to_csv,
};
use ndarray::{array, Array1};
use std::io::ErrorKind;
use tempfile::NamedTempFile;

#[test]
fn test_non_negative_eigenvalues_symmetric() {
    let laplacian = array![[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0],];

    let (eigvals, _) =
        compute_eigenvalues_and_vectors_sym(&laplacian).expect("faer eigendecomposition failed");

    for &val in eigvals.iter() {
        assert!(val >= 0.0, "faer eigenvalue is negative: {}", val);
    }
}

#[test]
fn test_call_eigendecomp_matches_direct_call() {
    let laplacian = array![[3.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 3.0],];

    let (direct_vals, direct_vecs) =
        compute_eigenvalues_and_vectors_sym(&laplacian).expect("direct eigendecomposition failed");
    let (call_vals, call_vecs) =
        call_eigendecomp(&laplacian).expect("wrapper eigendecomposition failed");

    assert_eq!(direct_vals.len(), call_vals.len());
    assert_eq!(direct_vecs.shape(), call_vecs.shape());

    for (a, b) in direct_vals.iter().zip(call_vals.iter()) {
        assert!((a - b).abs() < 1e-9);
    }

    for ((i, j), a) in direct_vecs.indexed_iter() {
        let b = call_vecs[(i, j)];
        assert!((a - b).abs() < 1e-9);
    }
}

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
    let eigenvalues = Array1::<f64>::zeros(0);
    let result = compute_ngec(&eigenvalues);
    assert!(matches!(result, Err(e) if e.kind() == ErrorKind::InvalidInput));
}

#[test]
fn test_compute_ngec_with_negative_eigenvalue() {
    let eigenvalues = array![-2.0, 1.0, 0.5];
    let result = compute_ngec(&eigenvalues);
    assert!(matches!(result, Err(e) if e.kind() == ErrorKind::InvalidInput));
}

#[test]
fn test_save_and_load_vector_csv() {
    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let tmp_file = NamedTempFile::new().expect("Failed to create temp file");
    let csv_path = tmp_file.path();

    save_array1_to_csv(&vector, csv_path).expect("Failed to save vector to CSV");

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
        vector, loaded_vector,
        "Loaded vector does not match original vector"
    );
}
