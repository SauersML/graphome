// tests/test_dsbevd.rs

#[cfg(test)]
mod tests {
    use graphome::dsbevd::SymmetricBandedMatrix;
    use rand_distr::{Normal, Distribution};
    use nalgebra::{DMatrix, SymmetricEigen};
    use std::time::Instant;
    use std::f64;

    // Helper function to create a known matrix with exact eigenvalues
    fn create_known_matrix(n: usize, kd: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut ab = vec![vec![0.0; n]; kd + 1];
        // Create diagonal matrix with known eigenvalues 1,2,3,...,n
        for i in 0..n {
            ab[kd][i] = (i + 1) as f64;
        }
        let eigenvalues = (1..=n).map(|x| x as f64).collect();
        (ab, eigenvalues)
    }

    // Helper to check if matrix is properly banded
    fn verify_band_structure(ab: &[Vec<f64>], n: usize, kd: usize) {
        for i in 0..=kd {
            for j in 0..n {
                if j + i >= n {
                    assert!(ab[i][j].abs() < 1e-10,
                        "Matrix element outside band should be zero: ({},{})", i, j);
                }
            }
        }
    }

    #[test]
    fn test_matrix_construction() {
        let n = 5;
        let kd = 2;
        let ab = vec![vec![0.0; n]; kd + 1];

        // Test basic construction
        let matrix = SymmetricBandedMatrix::new(n, kd, ab.clone());

        // Test invalid constructions
        let result = std::panic::catch_unwind(|| {
            SymmetricBandedMatrix::new(n, kd, vec![vec![0.0; n]; kd]);  // Too few rows
        });
        assert!(result.is_err(), "Should panic with too few rows");
    }

    #[test]
    fn test_diagonal_matrix() {
        let n = 5;
        let kd = 0;  // Diagonal matrix
        let mut ab = vec![vec![0.0; n]; 1];
        // Set diagonal elements to 1,2,3,4,5
        for i in 0..n {
            ab[0][i] = (i + 1) as f64;
        }

        let matrix = SymmetricBandedMatrix::new(n, kd, ab);
        let results = matrix.dsbevd();

        // For diagonal matrix, eigenvalues should exactly equal diagonal elements
        for i in 0..n {
            assert!((results.eigenvalues[i] - (i + 1) as f64).abs() < 1e-10,
                "Diagonal eigenvalue incorrect at position {}", i);
        }
    }
}
