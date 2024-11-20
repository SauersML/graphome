// tests/test_dsbevd.rs

#[cfg(test)]
mod tests {
    use super::super::*;
    use rand::prelude::*;
    use rand_distr::{Normal, Distribution};
    use nalgebra::{DMatrix, SymmetricEigen};
    use std::time::Instant;

    #[test]
    fn test_dsbevd_small_matrix() {
        let n = 5;
        let kd = 2; // Bandwidth
        let uplo = 'L';

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
            assert!((lambda_dsbevd - lambda_nalgebra).abs() < 1e-6);
        }

        // Compare eigenvectors
        // Note: Eigenvectors may differ in sign or order
        // So we need to check that the eigenvectors are equivalent up to sign

        for i in 0..n {
            let v1 = &results.eigenvectors[i];
            let v2 = sym_eigen.eigenvectors.column(i);

            // Determine if the vectors are proportional
            let mut dot_product = 0.0;
            for j in 0..n {
                dot_product += v1[j] * v2[j];
            }
            let sign = if dot_product >= 0.0 { 1.0 } else { -1.0 };

            for j in 0..n {
                assert!((v1[j] - sign * v2[j]).abs() < 1e-6);
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
            assert!(results.eigenvalues[i - 1] <= results.eigenvalues[i]);
        }

        // Check orthogonality of eigenvectors
        for i in 0..n {
            for j in i + 1..n {
                let dot_product: f64 = results.eigenvectors[i]
                    .iter()
                    .zip(&results.eigenvectors[j])
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(dot_product.abs() < 1e-6);
            }
        }

        // Check that eigenvectors are normalized
        for i in 0..n {
            let norm: f64 = results.eigenvectors[i]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }
}
