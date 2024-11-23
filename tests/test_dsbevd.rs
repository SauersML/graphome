use graphome::dsbevd::SymmetricBandedMatrix;
use nalgebra::{DMatrix, SymmetricEigen};
use rand_distr::{Distribution, Normal};
use std::f64;
use std::f64::EPSILON;
use std::time::Instant;

use graphome::dsbevd::dcopy;
use graphome::dsbevd::dgemm;
use graphome::dsbevd::dgemv;
use graphome::dsbevd::dlacpy;
use graphome::dsbevd::dlaed0;
use graphome::dsbevd::dlaeda;
use graphome::dsbevd::dlaed6;
use graphome::dsbevd::dlaed4;
use graphome::dsbevd::dlaed5;
use graphome::dsbevd::dlaev2;
use graphome::dsbevd::dlamc3;
use graphome::dsbevd::dlamch;
use graphome::dsbevd::dlamrg;
use graphome::dsbevd::dlanst as dlanst_function;
use graphome::dsbevd::dlapy2;
use graphome::dsbevd::dlar2v;
use graphome::dsbevd::dlargv;
use graphome::dsbevd::dlartg;
use graphome::dsbevd::dlartv;
use graphome::dsbevd::dlascl;
use graphome::dsbevd::dlaset;
use graphome::dsbevd::dlassq;
use graphome::dsbevd::dnrm2;
use graphome::dsbevd::drot;
use graphome::dsbevd::dsbtrd_wrapper;
use graphome::dsbevd::dscal;
use graphome::dsbevd::dstedc;
use graphome::dsbevd::dsteqr;
use graphome::dsbevd::dswap;
use graphome::dsbevd::get_mut_bands;
use graphome::dsbevd::idamax;
use graphome::dsbevd::ilaenv;
use graphome::dsbevd::EigenResults;
use graphome::dsbevd::Error;

// Helper function to convert banded storage to dense matrix
fn banded_to_dense(n: usize, kd: usize, ab: &Vec<Vec<f64>>) -> DMatrix<f64> {
    let mut dense = DMatrix::<f64>::zeros(n, n);
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
        let diag_abs = f64::abs(ab[0][j]);
        let sum: f64 = (1..=kd)
            .filter(|&i| j + i < n)
            .map(|i| f64::abs(ab[i][j]))
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
        SymmetricBandedMatrix::new(n, kd, vec![vec![0.0; n]; kd]); // Too few rows
    });
    assert!(result.is_err(), "Should panic with too few rows");

    let result = std::panic::catch_unwind(|| {
        SymmetricBandedMatrix::new(n, kd, vec![vec![0.0; n - 1]; kd + 1]); // Wrong column count
    });
    assert!(result.is_err(), "Should panic with wrong column count");
}

#[test]
fn test_diagonal_matrix() {
    let n = 5;
    let kd = 0; // Diagonal matrix
    let mut ab = vec![vec![0.0; n]; 1];

    // Set diagonal elements to 1,2,3,4,5
    for i in 0..n {
        ab[0][i] = (i + 1) as f64;
    }

    let matrix = SymmetricBandedMatrix::new(n, kd, ab);
    let results = matrix.dsbevd().unwrap();

    // For diagonal matrix, eigenvalues should exactly equal diagonal elements in ascending order
    for i in 0..n {
        let diff = f64::abs(results.eigenvalues[i] - (i + 1) as f64);
        assert!(
            diff < 1e-10,
            "Diagonal eigenvalue incorrect at position {}",
            i
        );

        // Check eigenvector is unit vector
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = f64::abs(results.eigenvectors[i][j] - expected);
            assert!(
                diff < 1e-10,
                "Diagonal eigenvector incorrect at position ({},{})",
                i,
                j
            );
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
    for x in ab[1].iter_mut().take(n - 1) {
        *x = -1.0;
    }

    let matrix = SymmetricBandedMatrix::new(n, kd, ab);
    let results = matrix.dsbevd().unwrap();

    // Known eigenvalues for this matrix are: 2 - 2cos(πj/(n+1)) for j=1,...,n
    let mut expected: Vec<f64> = (1..=n)
        .map(|j| 2.0 - 2.0 * (std::f64::consts::PI * j as f64 / (n as f64 + 1.0)).cos())
        .collect();
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (i, (&computed, &expected)) in results.eigenvalues.iter().zip(expected.iter()).enumerate() {
        let diff = f64::abs(computed - expected);
        assert!(
            diff < 1e-10,
            "Eigenvalue mismatch at position {}: computed={}, expected={}",
            i,
            computed,
            expected
        );
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
            let our_result = matrix.dsbevd().unwrap();
            let our_time = start.elapsed();

            // nalgebra implementation
            let dense = banded_to_dense(n, kd, &ab);
            let start = Instant::now();
            let nalgebra_result = SymmetricEigen::new(dense);
            let nalgebra_time = start.elapsed();

            // Compare eigenvalues (they should be in ascending order)
            for i in 0..n {
                let diff = f64::abs(our_result.eigenvalues[i] - nalgebra_result.eigenvalues[i]);
                assert!(diff < 1e-8, "Eigenvalue mismatch at position {}", i);
            }

            println!(
                "Size={}, Bandwidth={}: Our time={:?}, Nalgebra time={:?}",
                n, kd, our_time, nalgebra_time
            );
        }
    }
}

#[test]
fn test_orthogonality() {
    let n = 20;
    let kd = 3;
    let ab = generate_random_banded(n, kd);
    let matrix = SymmetricBandedMatrix::new(n, kd, ab);
    let results = matrix.dsbevd().unwrap();

    // Check orthogonality of eigenvectors
    for i in 0..n {
        for j in 0..n {
            let dot_product: f64 = (0..n)
                .map(|k| results.eigenvectors[i][k] * results.eigenvectors[j][k])
                .sum();

            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = f64::abs(dot_product - expected);
            assert!(diff < 1e-8, "Eigenvectors {} and {} not orthonormal", i, j);
        }
    }
}

#[test]
fn test_eigendecomposition_reconstruction() {
    let n = 15;
    let kd = 2;
    let ab = generate_random_banded(n, kd);
    let matrix = SymmetricBandedMatrix::new(n, kd, ab.clone());
    let results = matrix.dsbevd().unwrap();

    // Convert to dense format for comparison
    let original = banded_to_dense(n, kd, &ab);

    // Reconstruct A = QΛQ^T
    let mut reconstructed = DMatrix::<f64>::zeros(n, n);
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
            let diff = f64::abs(original[(i, j)] - reconstructed[(i, j)]);
            assert!(
                diff < 1e-8,
                "Matrix reconstruction failed at position ({},{})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_edge_cases() {
    // 1x1 matrix
    let ab = vec![vec![2.0]];
    let matrix = SymmetricBandedMatrix::new(1, 0, ab);
    let results = matrix.dsbevd().unwrap();
    assert_eq!(results.eigenvalues.len(), 1);
    assert!(f64::abs(results.eigenvalues[0] - 2.0) < 1e-10);

    // Zero matrix
    let n = 3;
    let kd = 1;
    let ab = vec![vec![0.0; n]; kd + 1];
    let matrix = SymmetricBandedMatrix::new(n, kd, ab);
    let results = matrix.dsbevd().unwrap();
    for &x in &results.eigenvalues {
        assert!(f64::abs(x) < 1e-10);
    }

    // Matrix with maximum bandwidth
    let n = 4;
    let kd = n - 1;
    let ab = generate_random_banded(n, kd);
    let matrix = SymmetricBandedMatrix::new(n, kd, ab);
    let results = matrix.dsbevd().unwrap();
    assert_eq!(results.eigenvalues.len(), n);
}

#[test]
fn test_performance_scaling() {
    let sizes = vec![100, 200, 400];
    let kd = 5; // Fixed bandwidth

    for &n in &sizes {
        let ab = generate_random_banded(n, kd);
        let matrix = SymmetricBandedMatrix::new(n, kd, ab);

        let start = Instant::now();
        let _results = matrix.dsbevd().unwrap();
        let duration = start.elapsed();

        println!("Size {} took {:?}", n, duration);
    }
}

// Helper function to compare two floating-point numbers
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 100.0 * EPSILON
}

#[test]
fn test_symmetric_banded_matrix_dsbevd() {
    let n = 4;
    let kd = 1;
    let ab = vec![vec![4.0, 1.0, 1.0, 1.0], vec![1.0, 4.0, 1.0, 1.0]];
    let matrix = SymmetricBandedMatrix::new(n, kd, ab);
    let result = matrix.dsbevd();
    assert!(result.is_ok());
    let eigen_results = result.unwrap();
    assert_eq!(eigen_results.eigenvalues.len(), n);
    assert_eq!(eigen_results.eigenvectors.len(), n);
    for vec in eigen_results.eigenvectors.iter() {
        assert_eq!(vec.len(), n);
    }
    // Check that the eigenvalues are in ascending order
    for i in 0..n - 1 {
        assert!(eigen_results.eigenvalues[i] <= eigen_results.eigenvalues[i + 1]);
    }
}

#[test]
fn test_dstedc() {
    let mut d = vec![4.0, 1.0, 3.0];
    let mut e = vec![0.5, 0.5];
    let n = d.len();
    let mut z = vec![vec![0.0; n]; n];
    for i in 0..n {
        z[i][i] = 1.0;
    }
    let result = dstedc(&mut d, &mut e, &mut z);
    assert!(result.is_ok());
    assert_eq!(d.len(), n);
    assert_eq!(e.len(), n - 1);

    // Test for index out of bounds
    let mut e_incorrect = vec![0.5]; // One element less
    let result = dstedc(&mut d, &mut e_incorrect, &mut z);
    assert!(result.is_err());
}

#[test]
fn test_dlaed4() {
    let d1 = vec![1.0, 2.0];
    let d2 = vec![3.0, 4.0];
    let z = vec![0.5, 0.5, 0.5, 0.5];
    let rho = 0.1;
    let mut d = vec![0.0; 4];
    let mut z_out = vec![vec![0.0; 4]; 4];
    let info = dlaed4(&d1, &d2, &z, rho, &mut d, &mut z_out);
    assert_eq!(result.is_ok())
    assert_eq!(d.len(), 4);
    assert_eq!(z_out.len(), 4);
}

#[test]
fn test_dgemm() {
    let q = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let z = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    let result = dgemm(&q, &z);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].len(), 2);
    assert_eq!(result[1].len(), 2);

    // Expected result manually computed
    let expected = vec![
        vec![1.0 * 5.0 + 2.0 * 7.0, 1.0 * 6.0 + 2.0 * 8.0],
        vec![3.0 * 5.0 + 4.0 * 7.0, 3.0 * 6.0 + 4.0 * 8.0],
    ];
    assert_eq!(result, expected);

    // Test for index out of bounds
    let q_incorrect = vec![vec![1.0], vec![3.0]]; // One column less
    let result = std::panic::catch_unwind(|| dgemm(&q_incorrect, &z));
    assert!(result.is_err());
}

#[test]
fn test_dlamc3() {
    let a = 1.0;
    let b = 2.0;
    let result = dlamc3(a, b);
    assert_eq!(result, a + b);
}

#[test]
fn test_dlaev2() {
    let a = 1.0;
    let b = 0.5;
    let c = 3.0;
    let (rt1, rt2, cs1, sn1, _) = dlaev2(a, b, c);
    assert!(rt1 >= rt2);

    // Check that the eigenvalues satisfy the characteristic equation
    let eigenvalues = vec![rt1, rt2];
    for lambda in eigenvalues {
        let det = (a - lambda) * (c - lambda) - b * b;
        assert!(approx_eq(det, 0.0));
    }
}

#[test]
fn test_dlapy2() {
    let x = 3.0;
    let y = 4.0;
    let result = dlapy2(x, y);
    assert_eq!(result, 5.0);
}

#[test]
fn test_dnrm2() {
    let x = vec![3.0, 4.0];
    let result = dnrm2(2, &x, 1);
    assert_eq!(result, 5.0);

    // Test with zero vector
    let x_zero = vec![0.0, 0.0];
    let result = dnrm2(2, &x_zero, 1);
    assert_eq!(result, 0.0);
}

#[test]
fn test_dscal() {
    let mut x = vec![1.0, 2.0, 3.0];
    dscal(3, 2.0, &mut x, 1);
    assert_eq!(x, vec![2.0, 4.0, 6.0]);

    // Test with zero scaling
    dscal(3, 0.0, &mut x, 1);
    assert_eq!(x, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_idamax() {
    let x = vec![-1.0, 3.0, -2.0];
    let index = idamax(3, &x, 1);
    assert_eq!(index, 1); // Index of element with max absolute value

    // Test with empty vector
    let x_empty: Vec<f64> = vec![];
    let index = idamax(0, &x_empty, 1);
    assert_eq!(index, 0);
}

#[test]
fn test_dswap() {
    let mut dx = vec![1.0, 2.0, 3.0];
    let mut dy = vec![4.0, 5.0, 6.0];
    dswap(3, &mut dx, 1, &mut dy, 1);
    assert_eq!(dx, vec![4.0, 5.0, 6.0]);
    assert_eq!(dy, vec![1.0, 2.0, 3.0]);

    // Test with zero increments (should do nothing)
    let mut dx = vec![1.0, 2.0, 3.0];
    let mut dy = vec![4.0, 5.0, 6.0];
    dswap(3, &mut dx, 0, &mut dy, 0);
    assert_eq!(dx, vec![1.0, 2.0, 3.0]);
    assert_eq!(dy, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_dcopy() {
    let n = 3;
    let dx = vec![1.0, 2.0, 3.0];
    let mut dy = vec![0.0; n];
    dcopy(n, &dx, 1, &mut dy, 1);
    assert_eq!(dy, dx);

    // Test with negative increment (should copy in reverse)
    let mut dy = vec![0.0; n];
    dcopy(n, &dx, 1, &mut dy, -1);
    assert_eq!(dy, vec![3.0, 2.0, 1.0]);
}

#[test]
fn test_dlassq() {
    let x = vec![1.0, 2.0, 3.0];
    let mut scale = 0.0;
    let mut sumsq = 1.0;
    dlassq(3, &x, 1, &mut scale, &mut sumsq);
    let norm = scale * sumsq.sqrt();
    assert!(approx_eq(norm, dnrm2(3, &x, 1)));

    // Test with zero vector
    let x_zero = vec![0.0, 0.0, 0.0];
    let mut scale = 0.0;
    let mut sumsq = 1.0;
    dlassq(3, &x_zero, 1, &mut scale, &mut sumsq);
    assert_eq!(scale, 0.0);
    assert_eq!(sumsq, 1.0);
}

#[test]
fn test_ilaenv() {
    let result = ilaenv(9, "", "", 0, 0, 0, 0);
    assert_eq!(result, 25);

    // Test with different ispec (should return 1)
    let result = ilaenv(1, "", "", 0, 0, 0, 0);
    assert_eq!(result, 1);
}

#[test]
fn test_dlaed0() {
    let n = 5;
    let mut d = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let mut e = vec![1.0; n - 1];
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        q[i][i] = 1.0;
    }
    let mut qstore = vec![vec![0.0; n]; n * n];
    let mut qptr = vec![0; n + 2];
    let mut prmptr = vec![0; n + 2];
    let mut perm = vec![0; n * n];
    let mut givptr = vec![0usize; n + 2];
    let mut givcol = vec![vec![0; n]; 2];
    let mut givnum = vec![vec![0.0; n]; 2];
    let mut work = vec![0.0; 4 * n + n * n];
    let mut iwork = vec![0; 3 + 5 * n];

    let result = dlaed0(
        2,
        n,
        n,
        0,
        0,
        0,
        &mut d,
        &mut e,
        &mut q,
        n,
        &mut qstore,
        &mut qptr,
        &mut prmptr,
        &mut perm,
        &mut givptr,
        &mut givcol,
        &mut givnum,
        &mut work,
        &mut iwork,
    );
    assert!(result.is_ok());

    // Eigenvalues should be sorted in ascending order
    for i in 0..n - 1 {
        assert!(d[i] <= d[i + 1]);
    }
}

#[test]
fn test_dlamch() {
    let eps = dlamch('E');
    assert_eq!(eps, f64::EPSILON / 2.0);

    let base = dlamch('B');
    assert_eq!(base, 2.0);

    // Test with invalid cmach (should return zero)
    let invalid = dlamch('X');
    assert_eq!(invalid, 0.0);
}

#[test]
fn test_dlacpy() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let lda = 2;
    let mut b = vec![vec![0.0; 2]; 2];
    let ldb = 2;
    dlacpy('A', 2, 2, &a, lda, &mut b, ldb);
    assert_eq!(a, b);

    // Test copying only upper triangle
    let mut b = vec![vec![0.0; 2]; 2];
    dlacpy('U', 2, 2, &a, lda, &mut b, ldb);
    assert_eq!(b, vec![vec![1.0, 2.0], vec![0.0, 4.0],]);
}

#[test]
fn test_dlamrg() {
    let a = vec![1.0, 3.0, 2.0, 4.0];
    let n1 = 2;
    let n2 = 2;
    let dtrd1 = 1;
    let dtrd2 = 1;
    let mut index = vec![0; n1 + n2];
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    assert_eq!(index, vec![0, 2, 1, 3]);

    // Test with decreasing order
    let dtrd1 = -1;
    let dtrd2 = -1;
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    assert_eq!(index, vec![1, 3, 0, 2]);
}

#[test]
fn test_dlaed5() {
    let i = 1;
    let d = vec![1.0, 2.0];
    let z = vec![0.5, 0.5];
    let rho = 0.1;
    let mut delta = vec![0.0, 0.0];
    let mut dlam = 0.0;
    let result = dlaed5(i, &d, &z, &mut delta, rho, &mut dlam);
    assert!(result.is_ok());
    assert!(dlam >= d[0] && dlam <= d[1]);
}

#[test]
fn test_dlargv() {
    // Test normal case
    let mut x = vec![3.0, 4.0, 5.0];
    let mut y = vec![4.0, 3.0, 12.0];
    let mut c = vec![0.0; 3];
    dlargv(3, &mut x, 1, &mut y, 1, &mut c, 1);

    // Verify results
    for i in 0..3 {
        assert!((c[i] * c[i] + y[i] * y[i] - 1.0).abs() < 1e-10);
    }

    // Test zero cases
    let mut x_zero = vec![0.0, 0.0];
    let mut y_zero = vec![1.0, 1.0];
    let mut c_zero = vec![0.0; 2];
    dlargv(2, &mut x_zero, 1, &mut y_zero, 1, &mut c_zero, 1);
    assert_eq!(c_zero, vec![0.0, 0.0]);
    assert_eq!(y_zero, vec![1.0, 1.0]);
}

#[test]
fn test_dlartv() {
    let mut x = vec![1.0, 2.0, 3.0];
    let mut y = vec![4.0, 5.0, 6.0];
    let c = vec![0.8, 0.6, 0.7];
    let s = vec![0.6, 0.8, 0.7];

    // Make copies for verification
    let x_orig = x.clone();
    let y_orig = y.clone();

    dlartv(3, &mut x, 1, &mut y, 1, &c, &s, 1);

    // Verify results - check if rotation preserves lengths
    for i in 0..3 {
        let orig_length = (x_orig[i] * x_orig[i] + y_orig[i] * y_orig[i]).sqrt();
        let new_length = (x[i] * x[i] + y[i] * y[i]).sqrt();
        assert!((orig_length - new_length).abs() < 1e-10);
    }

    // Test with empty arrays
    let mut x_empty: Vec<f64> = vec![];
    let mut y_empty: Vec<f64> = vec![];
    let c_empty: Vec<f64> = vec![];
    let s_empty: Vec<f64> = vec![];
    dlartv(0, &mut x_empty, 1, &mut y_empty, 1, &c_empty, &s_empty, 1);
}

#[test]
fn test_dlamc3_overflow() {
    // Test potential overflow cases
    let max = f64::MAX;
    let result = dlamc3(max, max);
    assert!(result.is_infinite());

    // Test underflow cases
    let min = f64::MIN_POSITIVE;
    let result = dlamc3(min, -min);
    assert_eq!(result, 0.0);

    // Test NaN handling
    let nan = f64::NAN;
    let result = dlamc3(nan, 1.0);
    assert!(result.is_nan());
}

#[test]
fn test_dgemv() {
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let x = vec![1.0, 1.0, 1.0];
    let mut y = vec![0.0; 3];

    // Test normal multiplication
    dgemv(false, 3, 3, 1.0, &a, &x, 0.0, &mut y);
    assert_eq!(y, vec![6.0, 15.0, 24.0]);

    // Test transpose multiplication
    let mut y = vec![0.0; 3];
    dgemv(true, 3, 3, 1.0, &a, &x, 0.0, &mut y);
    assert_eq!(y, vec![12.0, 15.0, 18.0]);

    // Test with zero alpha
    let mut y = vec![1.0; 3];
    dgemv(false, 3, 3, 0.0, &a, &x, 1.0, &mut y);
    assert_eq!(y, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_dlaset() {
    // Test full matrix fill
    let mut a = vec![vec![1.0; 3]; 3];
    dlaset('A', 3, 3, 2.0, 3.0, &mut a);
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(a[i][j], if i == j { 3.0 } else { 2.0 });
        }
    }

    // Test upper triangle
    let mut a = vec![vec![1.0; 3]; 3];
    dlaset('U', 3, 3, 2.0, 3.0, &mut a);
    for i in 0..3 {
        for j in 0..3 {
            if i <= j {
                assert_eq!(a[i][j], if i == j { 3.0 } else { 2.0 });
            } else {
                assert_eq!(a[i][j], 1.0);
            }
        }
    }

    // Test empty matrix
    let mut a: Vec<Vec<f64>> = vec![vec![]];
    dlaset('A', 0, 0, 2.0, 3.0, &mut a);
}

#[test]
fn test_dlascl() {
    let mut a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    // Test normal scaling
    let result = dlascl(&mut a, 2.0, 1.0);
    assert!(result.is_ok());
    assert_eq!(a, vec![vec![0.5, 1.0], vec![1.5, 2.0]]);

    // Test scaling by zero (should return error)
    let result = dlascl(&mut a, 0.0, 1.0);
    assert!(result.is_err());

    // Test overflow handling
    let result = dlascl(&mut a, f64::MIN_POSITIVE, f64::MAX);
    assert!(result.is_ok());
}


#[test]
fn test_dlartg() {
    // Test normal case
    let (c, s) = dlartg(3.0, 4.0);
    assert!((c * c + s * s - 1.0).abs() < 1e-10);
    assert!((3.0 * c + 4.0 * s).abs() > (3.0 * s - 4.0 * c).abs());

    // Test zero cases
    let (c, s) = dlartg(0.0, 0.0);
    assert_eq!(c, 1.0);
    assert_eq!(s, 0.0);

    // Test f = 0
    let (c, s) = dlartg(0.0, 1.0);
    assert_eq!(c, 0.0);
    assert_eq!(s, 1.0);
}


#[test]
fn test_dlaed6() {
    let mut tau = 0.0;
    let mut info = 0;
    let kniter = 2;
    let orgati = true;
    let rho = 1.0;
    let mut d = vec![1.0, 2.0, 3.0];
    let mut z = vec![0.5, 0.5, 0.5];
    let finit = 1.0;

    dlaed6(
        kniter, orgati, rho, &mut d, &mut z, finit, &mut tau, &mut info,
    );

    // Check that info is valid
    assert!(info == 0 || info == 1);

    // Test with zero rho
    dlaed6(
        kniter, orgati, 0.0, &mut d, &mut z, finit, &mut tau, &mut info,
    );
    assert!(info == 0 || info == 1);
}

#[test]
fn test_dlaeda() {
    let n = 4;
    let tlvls = 2;
    let curlvl = 1;
    let curpbm = 0;
    let prmptr = vec![0, 2, 4];
    let perm = vec![1, 2, 3, 4];
    let givptr = vec![1, 2];
    let givcol = vec![vec![1, 2], vec![2, 3]];
    let givnum = vec![vec![0.8, 0.6], vec![0.6, 0.8]];
    let q = vec![1.0, 0.0, 0.0, 1.0];
    let qptr = vec![0, 2, 4];
    let mut z = vec![0.0; n];
    let mut ztemp = vec![0.0; n];

    let result = dlaeda(
        n, tlvls, curlvl, curpbm, &prmptr, &perm, &givptr, &givcol, &givnum, &q, &qptr, &mut z,
        &mut ztemp,
    );

    assert!(result.is_ok());

    // Test empty case
    let result = dlaeda(
        0, tlvls, curlvl, curpbm, &prmptr, &perm, &givptr, &givcol, &givnum, &q, &qptr, &mut z,
        &mut ztemp,
    );
    assert!(result.is_ok());
}
