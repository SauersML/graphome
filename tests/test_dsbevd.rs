// tests/test_dsbevd.rs

#[cfg(test)]
mod tests {
    use graphome::dsbevd::SymmetricBandedMatrix;
    use rand::prelude::*;
    use nalgebra::{DMatrix, SymmetricEigen};

    #[test]
    fn test_dsbevd() {
        let n = 5;
        let kd = 1; // For simplicity, use kd = 1
        let ldab = kd + 1;
        let uplo = 'L';

        // Generate a random symmetric banded matrix
        let mut rng = rand::thread_rng();
        let mut ab = vec![0.0; ldab * n];
        for j in 0..n {
            for i in 0..(kd + 1) {
                if j + i < n {
                    let val = rng.gen::<f64>();
                    ab[i + j * ldab] = val;
                    // Since the matrix is symmetric, set the corresponding element
                    if i > 0 {
                        let row = j + i;
                        let col = j;
                        let idx = (row - col) + col * ldab;
                        ab[idx] = val;
                    }
                }
            }
        }

        let sb_matrix = SymmetricBandedMatrix::new(n, kd, ab.clone(), ldab, uplo);
        let results = sb_matrix.dsbevd();

        // Now construct the full matrix and compute eigenvalues using nalgebra
        let mut full_matrix = DMatrix::<f64>::zeros(n, n);
        for j in 0..n {
            for i in 0..(kd + 1) {
                if j + i < n {
                    let val = ab[i + j * ldab];
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

            // Check if v1 and v2 are proportional
            let mut factor = v1[0] / v2[0];
            if factor == 0.0 {
                factor = 1.0;
            }

            for j in 0..n {
                assert!((v1[j] - factor * v2[j]).abs() < 1e-6);
            }
        }
    }
}
