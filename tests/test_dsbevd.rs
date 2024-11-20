// tests/test_dsbevd.rs

#[cfg(test)]
mod tests {
    use crate::dsbevd::SymmetricBandedMatrix;
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

    #[test]
    fn test_tridiagonal_matrix() {
        let n = 5;
        let kd = 1;
        let mut ab = vec![vec![0.0; n]; kd + 1];

        // Create tridiagonal matrix with 2 on diagonal and -1 on off-diagonals
        for i in 0..n {
            ab[1][i] = 2.0;  // diagonal
            if i < n - 1 {
                ab[0][i] = -1.0;  // off-diagonal
            }
        }

        let matrix = SymmetricBandedMatrix::new(n, kd, ab);
        let results = matrix.dsbevd();

        // Known eigenvalues for this specific tridiagonal matrix
        let expected: Vec<f64> = (1..=n).map(|i| {
            let x = (i as f64 * std::f64::consts::PI) / (n as f64 + 1.0);
            2.0 - 2.0 * x.cos()
        }).collect();

        for (i, (&computed, &expected)) in
            results.eigenvalues.iter().zip(expected.iter()).enumerate() {
            assert!((computed - expected).abs() < 1e-6,
                "Tridiagonal eigenvalue mismatch at {}: {} vs {}", i, computed, expected);
        }
    }

    #[test]
    fn test_householder_reflector() {
        let n = 5;
        let kd = 2;
        let mut ab = vec![vec![0.0; n]; kd + 1];

        // Test with simple known case
        ab[2][0] = 4.0;
        ab[1][0] = 2.0;
        ab[0][0] = 1.0;

        let matrix = SymmetricBandedMatrix::new(n, kd, ab);
        let results = matrix.dsbevd();

        // Since the matrix is symmetric positive definite, eigenvalues should be positive
        for val in &results.eigenvalues {
            assert!(*val > 0.0, "Eigenvalue is not positive: {}", val);
        }
    }

    #[test]
    fn test_numerical_stability() {
        let n = 3;
        let kd = 1;
        let mut ab = vec![vec![0.0; n]; kd + 1];

        // Test with very small values
        ab[1][0] = 1e-15;
        ab[1][1] = 1e-15;
        ab[1][2] = 1e-15;
        ab[0][0] = 1e-15;
        ab[0][1] = 1e-15;

        let matrix = SymmetricBandedMatrix::new(n, kd, ab);
        let results = matrix.dsbevd();

        // Results should be well-scaled
        for val in &results.eigenvalues {
            assert!(!val.is_nan() && !val.is_infinite());
        }
    }

    #[test]
    fn test_degenerate_eigenvalues() {
        let n = 5;
        let kd = 2;
        let mut ab = vec![vec![0.0; n]; kd + 1];

        // Create matrix with known repeated eigenvalues
        for i in 0..n {
            ab[kd][i] = 1.0;  // All diagonal elements = 1
        }
        for i in 0..n - 1 {
            ab[kd - 1][i] = 0.0;
        }
        for i in 0..n - 2 {
            ab[kd - 2][i] = 0.0;
        }

        let matrix = SymmetricBandedMatrix::new(n, kd, ab);
        let results = matrix.dsbevd();

        // All eigenvalues should be very close to 1.0
        for val in results.eigenvalues.iter() {
            assert!((val - 1.0).abs() < 1e-6,
                "Degenerate eigenvalue not correctly computed: {}", val);
        }

        // Eigenvectors should still be orthogonal
        for i in 0..n {
            for j in i + 1..n {
                let dot: f64 = results.eigenvectors[i].iter()
                    .zip(&results.eigenvectors[j])
                    .map(|(&x, &y)| x * y)
                    .sum();
                assert!(dot.abs() < 1e-6,
                    "Eigenvectors not orthogonal for degenerate eigenvalues: {}", dot);
            }
        }
    }

    #[test]
    fn test_random_matrices_comprehensive() {
        let sizes = vec![3, 5, 10, 20];
        let bandwidths = vec![0, 1, 2, 4];

        for &n in sizes.iter() {
            for &kd in bandwidths.iter() {
                if kd >= n { continue; }

                let mut rng = rand::thread_rng();
                let normal = Normal::new(0.0, 1.0).unwrap();
                let mut ab = vec![vec![0.0; n]; kd + 1];

                // Generate random symmetric banded matrix
                for i in 0..=kd {
                    for j in 0..n {
                        if j + i < n {
                            let val: f64 = normal.sample(&mut rng);
                            ab[i][j] = val;
                        }
                    }
                }

                let sb_matrix = SymmetricBandedMatrix::new(n, kd, ab.clone());
                let results = sb_matrix.dsbevd();

                // Convert to full matrix for nalgebra comparison
                let mut full_matrix = DMatrix::<f64>::zeros(n, n);
                for j in 0..n {
                    for i in 0..=kd {
                        if j + i < n {
                            let val = ab[i][j];
                            full_matrix[(j + i, j)] = val;
                            full_matrix[(j, j + i)] = val;
                        }
                    }
                }

                let sym_eigen = SymmetricEigen::new(full_matrix.clone());

                // Compare eigenvalues with detailed diagnostics
                for (i, (lambda_dsbevd, lambda_nalgebra)) in
                    results.eigenvalues.iter().zip(sym_eigen.eigenvalues.iter()).enumerate() {
                    assert!((lambda_dsbevd - lambda_nalgebra).abs() < 1e-6,
                        "Eigenvalue mismatch at position {} for n={}, kd={}: {} vs {}",
                        i, n, kd, lambda_dsbevd, lambda_nalgebra);
                }

                // Verify eigenvector properties
                for i in 0..n {
                    // Verify Av = λv
                    let lambda = results.eigenvalues[i];
                    let v = &results.eigenvectors[i];

                    // Compute Av
                    let mut av = vec![0.0; n];
                    for j in 0..n {
                        for k in 0..n {
                            if (j as i32 - k as i32).abs() <= kd as i32 {
                                av[j] += full_matrix[(j, k)] * v[k];
                            }
                        }
                    }

                    // Check Av = λv
                    for j in 0..n {
                        assert!((av[j] - lambda * v[j]).abs() < 1e-6,
                            "Eigenvector equation failed for n={}, kd={}, vec={}, pos={}",
                            n, kd, i, j);
                    }
                }
            }
        }
    }

    #[test]
    fn test_dsbevd_small_matrix() {
        let n = 5;
        let kd = 2; // Bandwidth

        // Generate a random symmetric banded matrix
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut ab = vec![vec![0.0; n]; kd + 1];
        for i in 0..=kd {
            for j in 0..n {
                if j + i < n {
                    let val: f64 = normal.sample(&mut rng);
                    ab[i][j] = val;
                }
            }
        }

        let sb_matrix = SymmetricBandedMatrix::new(n, kd, ab.clone());
        let start = Instant::now();
        let results = sb_matrix.dsbevd();
        let duration = start.elapsed();
        println!("dsbevd took {:?}", duration);

        // Now construct the full matrix and compute eigenvalues using nalgebra
        let mut full_matrix = DMatrix::<f64>::zeros(n, n);
        for j in 0..n {
            for i in 0..=kd {
                if j + i < n {
                    let val = ab[i][j];
                    full_matrix[(j + i, j)] = val;
                    full_matrix[(j, j + i)] = val;
                }
            }
        }

        let sym_eigen = SymmetricEigen::new(full_matrix.clone());

        // Compare eigenvalues
        for (lambda_dsbevd, lambda_nalgebra) in results.eigenvalues.iter().zip(sym_eigen.eigenvalues.iter()) {
            assert!((lambda_dsbevd - lambda_nalgebra).abs() < 1e-6,
                "Eigenvalue mismatch: {} vs {}", lambda_dsbevd, lambda_nalgebra);
        }

        // Compare eigenvectors
        // Note: Eigenvectors may differ in sign or order
        // So we need to check that the eigenvectors are equivalent up to sign

        for i in 0..n {
            let v1 = &results.eigenvectors[i];
            let v2 = sym_eigen.eigenvectors.column(i);

            // Determine the sign that makes the vectors as close as possible
            let dot_product: f64 = v1.iter()
                .zip(v2.iter())
                .map(|(a, b)| a * b)
                .sum();
            let sign = if dot_product >= 0.0 { 1.0 } else { -1.0 };

            for j in 0..n {
                assert!((v1[j] - sign * v2[j]).abs() < 1e-6,
                    "Eigenvector mismatch at index {}, component {}: {} vs {}",
                    i, j, v1[j], sign * v2[j]);
            }
        }
    }

    #[test]
    fn test_dsbevd_large_matrix_performance() {
        let n = 500;
        let kd = 5; // Bandwidth

        // Generate a random symmetric banded matrix
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut ab = vec![vec![0.0; n]; kd + 1];
        for i in 0..=kd {
            for j in 0..n {
                if j + i < n {
                    let val: f64 = normal.sample(&mut rng);
                    ab[i][j] = val;
                }
            }
        }

        let sb_matrix = SymmetricBandedMatrix::new(n, kd, ab.clone());

        // Time dsbevd
        let start = Instant::now();
        let results = sb_matrix.dsbevd();
        let duration = start.elapsed();
        println!("dsbevd on large matrix took {:?}", duration);

        // Check that eigenvalues are sorted
        for i in 1..n {
            assert!(results.eigenvalues[i - 1] <= results.eigenvalues[i],
                "Eigenvalues not sorted at index {}: {} > {}",
                i - 1, results.eigenvalues[i - 1], results.eigenvalues[i]);
        }

        // Check orthogonality of eigenvectors
        for i in 0..n {
            for j in i + 1..n {
                let dot_product: f64 = results.eigenvectors[i]
                    .iter()
                    .zip(&results.eigenvectors[j])
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(dot_product.abs() < 1e-6,
                    "Eigenvectors not orthogonal: dot product between vector {} and {} is {}",
                    i, j, dot_product);
            }
        }

        // Check that eigenvectors are normalized
        for i in 0..n {
            let norm: f64 = results.eigenvectors[i]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-6,
                "Eigenvector not normalized: vector {} has norm {}", i, norm);
        }
    }
}
