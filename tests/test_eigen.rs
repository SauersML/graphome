//! Unit tests for the eigen module.

use ndarray::{array, Array1, Array2, ArrayView2};
use graphome::eigen::{
    call_eigendecomp,
    compute_eigenvalues_and_vectors_sym,
    compute_ngec,
    ndarray_to_nalgebra_matrix,
    adjacency_matrix_to_ndarray,
    print_heatmap,
    save_nalgebra_vector_to_csv,
};

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    /// Test the `to_banded_format` function...
    #[test]
    fn test_to_banded_format_kd_1() {
        // Define a known symmetric matrix
        let matrix = array![
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 4.0],
            [0.0, 4.0, 5.0]
        ];
        let kd = 1;
    
        // Define the expected banded format for kd = 1 
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
                let k = (j as isize - i as isize).abs() as usize; // Explicit cast to isize for subtraction
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
    
        let expected_banded = array![
            [0.0, 0.0, 0.0, 0.0], // k = 2
            [0.0, 1.0, 1.0, 1.0], // k = 1
            [4.0, 3.0, 2.0, 1.0]  // Main diagonal
        ];
    
        let banded = to_banded_format(&matrix, kd);
    
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
                    kd = std::cmp::max(kd, (i as isize - j as isize).abs() as usize); // Cast to isize
                }
            }
        }
        kd
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
    
        let adj_matrix = graphome::eigen::adjacency_matrix_to_ndarray(&edges, start_node, end_node);
    
        assert_eq!(adj_matrix, expected);
    
    }
}
