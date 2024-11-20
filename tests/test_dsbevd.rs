#[cfg(test)]
mod tests {
    use graphome::dsbevd::SymmetricBandedMatrix;
    use rand_distr::{Normal, Distribution};
    use nalgebra::{DMatrix, SymmetricEigen};
    use std::time::Instant;
    use std::f64;
    
    // Helper function to convert banded storage to dense matrix
    fn banded_to_dense(n: usize, kd: usize, ab: &Vec<Vec<f64>>) -> DMatrix<f64> {
        let mut dense = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in (i.saturating_sub(kd))..=std::cmp::min(i + kd, n - 1) {
                if i <= j {
                    dense[(i, j)] = ab[j - i][i];
                } else {
                    dense[(i, j)] = ab[i - j][j];
                }
            }
        }
        // Make symmetric
        for i in 0..n {
            for j in 0..i {
                dense[(j, i)] = dense[(i, j)];
            }
        }
        dense
    }

    // Helper function to generate random banded matrix
    fn generate_random_banded(n: usize, kd: usize) -> Vec<Vec<f64>> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();
        let mut ab = vec![vec![0.0; n]; kd + 1];
        
        // Fill with random values
        for i in 0..=kd {
            for j in 0..n {
                if j + i < n {
                    ab[i][j] = normal.sample(&mut rng);
                }
            }
        }
        
        // Make diagonally dominant for stability
        for j in 0..n {
            let diag_abs: f64 = ab[0][j].abs();
            let sum: f64 = (1..=kd)
                .filter(|&i| j + i < n)
                .map(|i| ab[i][j].abs())
                .sum();
            ab[0][j] = diag_abs + sum + 1.0;
        }
        
        ab
    }

    #[test]
    fn test_matrix_construction() {
        let n = 5;
        let kd = 2;
        let ab = vec![vec![0.0; n]; kd + 1];
        
        // Test basic construction
        let _matrix = SymmetricBandedMatrix::new(n, kd, ab.clone());
        
        // Test invalid constructions
        let result = std::panic::catch_unwind(|| {
            SymmetricBandedMatrix::new(n, kd, vec![vec![0.0; n]; kd]);  // Too few rows
        });
        assert!(result.is_err(), "Should panic with too few rows");
        
        let result = std::panic::catch_unwind(|| {
            SymmetricBandedMatrix::new(n, kd, vec![vec![0.0; n-1]; kd+1]);  // Wrong column count
        });
        assert!(result.is_err(), "Should panic with wrong column count");
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
        
        // For diagonal matrix, eigenvalues should exactly equal diagonal elements in ascending order
        for i in 0..n {
            let diff = (results.eigenvalues[i] - (i + 1) as f64).abs();
            assert!(diff < 1e-10,
                "Diagonal eigenvalue incorrect at position {}", i);
            
            // Check eigenvector is unit vector
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (results.eigenvectors[i][j] - expected).abs();
                assert!(diff < 1e-10,
                    "Diagonal eigenvector incorrect at position ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_tridiagonal_toeplitz() {
        let n = 10;
        let kd = 1;
        let mut ab = vec![vec![0.0; n]; kd + 1];
        
        // Create tridiagonal Toeplitz matrix with 2 on diagonal and -1 on sub/super-diagonals
        for x in ab[0].iter_mut() {
            *x = 2.0;
        }
        for x in ab[1].iter_mut().take(n-1) {
            *x = -1.0;
        }
        
        let matrix = SymmetricBandedMatrix::new(n, kd, ab);
        let results = matrix.dsbevd();
        
        // Known eigenvalues for this matrix are: 2 - 2cos(πj/(n+1)) for j=1,...,n
        let mut expected: Vec<f64> = (1..=n)
            .map(|j| 2.0 - 2.0 * (std::f64::consts::PI * j as f64 / (n as f64 + 1.0)).cos())
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        for (i, (&computed, &expected)) in results.eigenvalues.iter()
            .zip(expected.iter()).enumerate() {
            let diff = (computed - expected).abs();
            assert!(diff < 1e-10,
                "Eigenvalue mismatch at position {}: computed={}, expected={}", 
                i, computed, expected);
        }
    }

    #[test]
    fn test_comparison_with_nalgebra() {
        let sizes = vec![10, 20, 50];
        let bandwidths = vec![1, 3, 5];
        
        for &n in &sizes {
            for &kd in &bandwidths {
                let ab = generate_random_banded(n, kd);
                let matrix = SymmetricBandedMatrix::new(n, kd, ab.clone());
                
                // Our implementation
                let start = Instant::now();
                let our_result = matrix.dsbevd();
                let our_time = start.elapsed();
                
                // nalgebra implementation
                let dense = banded_to_dense(n, kd, &ab);
                let start = Instant::now();
                let nalgebra_result = SymmetricEigen::new(dense);
                let nalgebra_time = start.elapsed();
                
                // Compare eigenvalues (they should be in ascending order)
                for i in 0..n {
                    let diff = (our_result.eigenvalues[i] - nalgebra_result.eigenvalues[i]).abs();
                    assert!(diff < 1e-8,
                        "Eigenvalue mismatch at position {}", i);
                }
                
                println!("Size={}, Bandwidth={}: Our time={:?}, Nalgebra time={:?}",
                    n, kd, our_time, nalgebra_time);
            }
        }
    }

    #[test]
    fn test_orthogonality() {
        let n = 20;
        let kd = 3;
        let ab = generate_random_banded(n, kd);
        let matrix = SymmetricBandedMatrix::new(n, kd, ab);
        let results = matrix.dsbevd();
        
        // Check orthogonality of eigenvectors
        for i in 0..n {
            for j in 0..n {
                let dot_product: f64 = (0..n)
                    .map(|k| results.eigenvectors[i][k] * results.eigenvectors[j][k])
                    .sum();
                
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (dot_product - expected).abs();
                assert!(diff < 1e-8,
                    "Eigenvectors {} and {} not orthonormal", i, j);
            }
        }
    }

    #[test]
    fn test_eigendecomposition_reconstruction() {
        let n = 15;
        let kd = 2;
        let ab = generate_random_banded(n, kd);
        let matrix = SymmetricBandedMatrix::new(n, kd, ab.clone());
        let results = matrix.dsbevd();
        
        // Convert to dense format for comparison
        let original = banded_to_dense(n, kd, &ab);
        
        // Reconstruct A = QΛQ^T
        let mut reconstructed = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    reconstructed[(i, j)] += results.eigenvectors[k][i] 
                        * results.eigenvalues[k] 
                        * results.eigenvectors[k][j];
                }
            }
        }
        
        // Compare original and reconstructed matrices
        for i in 0..n {
            for j in 0..n {
                let diff = (original[(i, j)] - reconstructed[(i, j)]).abs();
                assert!(diff < 1e-8,
                    "Matrix reconstruction failed at position ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        // 1x1 matrix
        let ab = vec![vec![2.0]];
        let matrix = SymmetricBandedMatrix::new(1, 0, ab);
        let results = matrix.dsbevd();
        assert_eq!(results.eigenvalues.len(), 1);
        assert!((results.eigenvalues[0] - 2.0).abs() < 1e-10);
        
        // Zero matrix
        let n = 3;
        let kd = 1;
        let ab = vec![vec![0.0; n]; kd + 1];
        let matrix = SymmetricBandedMatrix::new(n, kd, ab);
        let results = matrix.dsbevd();
        for &x in &results.eigenvalues {
            assert!(x.abs() < 1e-10);
        }
        
        // Matrix with maximum bandwidth
        let n = 4;
        let kd = n - 1;
        let ab = generate_random_banded(n, kd);
        let matrix = SymmetricBandedMatrix::new(n, kd, ab);
        let results = matrix.dsbevd();
        assert_eq!(results.eigenvalues.len(), n);
    }

    #[test]
    fn test_performance_scaling() {
        let sizes = vec![100, 200, 400];
        let kd = 5;  // Fixed bandwidth
        
        for &n in &sizes {
            let ab = generate_random_banded(n, kd);
            let matrix = SymmetricBandedMatrix::new(n, kd, ab);
            
            let start = Instant::now();
            let _results = matrix.dsbevd();
            let duration = start.elapsed();
            
            println!("Size {} took {:?}", n, duration);
        }
    }
}
