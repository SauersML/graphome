//! Unit tests for the eigen module.

use ndarray::{array, Array1, Array2, ArrayView2};
use graphome::eigen_print::{
    call_eigendecomp,
    compute_eigenvalues_and_vectors_sym,
    compute_ngec,
    ndarray_to_nalgebra_matrix,
    adjacency_matrix_to_ndarray,
    print_heatmap,
    save_nalgebra_vector_to_csv,
};

use std::io::ErrorKind;
use tempfile::NamedTempFile;

const TOLERANCE: f64 = 1e-6;


fn to_banded_format(matrix: &Array2<f64>, kd: usize) -> Array2<f64> {
    let n = matrix.nrows();
    let mut banded_matrix = Array2::zeros((kd + 1, n));

    for i in 0..n {
        for j in 0..n {
            let k = (j as isize - i as isize).abs() as usize; // Explicit cast to isize for subtraction
            if k <= kd {
                banded_matrix[[kd - k, i]] = matrix[[i, j]];
            }
        }
    }
    banded_matrix
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

// Assuming this test uses some functions from src/eigen.rs,  I'll add some example tests for other likely functions:

#[test]
fn test_ndarray_to_nalgebra_matrix() {
    let ndarray_matrix = array![[1.0, 2.0], [3.0, 4.0]];
    let nalgebra_matrix = ndarray_to_nalgebra_matrix(&ndarray_matrix).unwrap();

    assert_eq!(nalgebra_matrix.nrows(), 2);
    assert_eq!(nalgebra_matrix.ncols(), 2);
    assert_eq!(nalgebra_matrix[(0, 0)], 1.0);
    assert_eq!(nalgebra_matrix[(0, 1)], 2.0);
    assert_eq!(nalgebra_matrix[(1, 0)], 3.0);
    assert_eq!(nalgebra_matrix[(1, 1)], 4.0);
}


#[test]
fn test_adjacency_matrix_to_ndarray() {
    let edges = vec![(0, 1), (1, 2), (0, 2)];
    let start_node = 0;
    let end_node = 2;
    let expected = array![
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ];

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
    let vector = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let tmp_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    let csv_path = tmp_file.path();

    save_nalgebra_vector_to_csv(&vector, csv_path).expect("Failed to save vector to CSV");

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(csv_path)
        .expect("Failed to create CSV reader");

    let loaded_row: Vec<f64> = rdr
        .deserialize()
        .next()
        .expect("Failed to read row")
        .expect("Failed to deserialize");

    let loaded_vector = nalgebra::DVector::from_vec(loaded_row);

    assert_eq!(
        vector, loaded_vector,
        "Loaded vector does not match the original vector."
    );
}
