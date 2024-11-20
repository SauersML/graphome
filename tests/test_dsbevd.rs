// tests/test_dsbevd.rs

#[cfg(test)]
mod tests {
    use graphome::dsbevd::SymmetricBandedMatrix;
    use rand_distr::{Normal, Distribution};
    use nalgebra::{DMatrix, SymmetricEigen};
    use std::time::Instant;
    use std::f64;

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
