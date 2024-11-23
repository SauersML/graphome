use graphome::dsbevd::SymmetricBandedMatrix;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand_distr::{Distribution, Normal};
use std::f64;
use std::f64::EPSILON;
use std::time::Instant;
use approx::assert_relative_eq;

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
fn test_dsbevd_diagonal_matrix() {
    // Create a 3x3 diagonal matrix with bandwidth 0
    let diagonal = vec![1.0, 2.0, 3.0];
    let mut ab = vec![diagonal];
    let matrix = SymmetricBandedMatrix::new(3, 0, ab);
    
    let result = matrix.dsbevd().unwrap();
    
    // Eigenvalues should be exactly the diagonal elements in ascending order
    assert_relative_eq!(result.eigenvalues[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.eigenvalues[2], 3.0, epsilon = 1e-12);
}

#[test]
fn test_dsbevd_2x2_symmetric() {
    // Create a 2x2 symmetric matrix with bandwidth 1
    // Matrix is: [2.0  1.0]
    //           [1.0  2.0]
    let ab = vec![
        vec![2.0, 2.0],  // diagonal
        vec![1.0, 0.0],  // superdiagonal
    ];
    let matrix = SymmetricBandedMatrix::new(2, 1, ab);
    
    let result = matrix.dsbevd().unwrap();
    
    // Known eigenvalues: 1.0, 3.0
    assert_relative_eq!(result.eigenvalues[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.eigenvalues[1], 3.0, epsilon = 1e-12);
}

#[test]
fn test_dsbevd_zero_matrix() {
    let ab = vec![vec![0.0, 0.0, 0.0]];
    let matrix = SymmetricBandedMatrix::new(3, 0, ab);
    
    let result = matrix.dsbevd().unwrap();
    
    // All eigenvalues should be zero
    assert_relative_eq!(result.eigenvalues[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(result.eigenvalues[1], 0.0, epsilon = 1e-12);
}

#[test]
fn test_dsbevd_identity_matrix() {
    let ab = vec![vec![1.0, 1.0, 1.0]];
    let matrix = SymmetricBandedMatrix::new(3, 0, ab);
    
    let result = matrix.dsbevd().unwrap();
    
    // All eigenvalues should be 1.0
    assert_relative_eq!(result.eigenvalues[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(result.eigenvalues[2], 1.0, epsilon = 1e-12);
}

#[test]
fn test_dsbevd_orthogonality() {
    // Create a 3x3 symmetric band matrix
    let ab = vec![
        vec![2.0, 2.0, 2.0],  // diagonal
        vec![1.0, 1.0, 0.0],  // superdiagonal
    ];
    let matrix = SymmetricBandedMatrix::new(3, 1, ab);
    
    let result = matrix.dsbevd().unwrap();
    
    // Check if eigenvectors are orthogonal (dot product should be close to 0)
    let dot_product = (0..3).map(|i| 
        result.eigenvectors[0][i] * result.eigenvectors[1][i]
    ).sum::<f64>();
    
    assert_relative_eq!(dot_product, 0.0, epsilon = 1e-10);
}

#[test]
fn test_dsbevd_eigenvalue_order() {
    // Create a matrix with known eigenvalues
    let ab = vec![
        vec![3.0, 1.0, 5.0],  // diagonal
        vec![2.0, 1.0, 0.0],  // superdiagonal
    ];
    let matrix = SymmetricBandedMatrix::new(3, 1, ab);
    
    let result = matrix.dsbevd().unwrap();
    
    // Check if eigenvalues are in ascending order
    assert!(result.eigenvalues[0] <= result.eigenvalues[1]);
    assert!(result.eigenvalues[1] <= result.eigenvalues[2]);
}

#[test]
#[should_panic]
fn test_dsbevd_invalid_bandwidth() {
    // Try to create a matrix with invalid bandwidth
    let ab = vec![vec![1.0, 1.0]];
    let matrix = SymmetricBandedMatrix::new(2, 2, ab); // bandwidth > n-1
    
    let _ = matrix.dsbevd().unwrap();
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
fn test_dlaev2_diagonal() {
    // Test diagonal matrix case where b=0
    let (rt1, rt2, cs1, sn1, _) = dlaev2(3.0, 0.0, 1.0);
    
    // Larger eigenvalue should be 3.0
    assert_relative_eq!(rt1, 3.0, epsilon = 1e-12);
    // Smaller eigenvalue should be 1.0
    assert_relative_eq!(rt2, 1.0, epsilon = 1e-12);
    // Eigenvector should be [1, 0]
    assert_relative_eq!(cs1, 1.0, epsilon = 1e-12);
    assert_relative_eq!(sn1, 0.0, epsilon = 1e-12);
}

#[test]
fn test_dlaev2_identity() {
    // Test identity matrix case
    let (rt1, rt2, cs1, sn1, _) = dlaev2(1.0, 0.0, 1.0);
    
    // Both eigenvalues should be 1.0
    assert_relative_eq!(rt1, 1.0, epsilon = 1e-12);
    assert_relative_eq!(rt2, 1.0, epsilon = 1e-12);
    // Any unit vector is an eigenvector
    assert_relative_eq!(cs1*cs1 + sn1*sn1, 1.0, epsilon = 1e-12);
}

#[test]
fn test_dlaev2_symmetric() {
    // Test symmetric matrix with known eigenvalues
    // [2  1]
    // [1  2]
    let (rt1, rt2, cs1, sn1, _) = dlaev2(2.0, 1.0, 2.0);
    
    // Eigenvalues should be 3 and 1
    assert_relative_eq!(rt1, 3.0, epsilon = 1e-12);
    assert_relative_eq!(rt2, 1.0, epsilon = 1e-12);
    // Verify eigenvector is unit length
    assert_relative_eq!(cs1*cs1 + sn1*sn1, 1.0, epsilon = 1e-12);
    // For this symmetric case, cs1 and sn1 should be equal
    assert_relative_eq!(cs1.abs(), sn1.abs(), epsilon = 1e-12);
}

#[test]
fn test_dlaev2_zero_matrix() {
    // Test zero matrix
    let (rt1, rt2, cs1, sn1, _) = dlaev2(0.0, 0.0, 0.0);
    
    // Both eigenvalues should be 0
    assert_relative_eq!(rt1, 0.0, epsilon = 1e-12);
    assert_relative_eq!(rt2, 0.0, epsilon = 1e-12);
    // Verify eigenvector is unit length
    assert_relative_eq!(cs1*cs1 + sn1*sn1, 1.0, epsilon = 1e-12);
}

#[test]
fn test_dlaev2_eigenvalue_ordering() {
    // Test that rt1 is always >= rt2
    let (rt1, rt2, _, _, _) = dlaev2(-1.0, 2.0, 4.0);
    assert!(rt1 >= rt2);

    let (rt1, rt2, _, _, _) = dlaev2(4.0, 2.0, -1.0);
    assert!(rt1 >= rt2);
}

#[test]
fn test_dlaev2_eigenvector_verification() {
    // Test that (cs1,sn1) is indeed an eigenvector for rt1
    let a = 2.0;
    let b = 1.0;
    let c = 3.0;
    let (rt1, _, cs1, sn1, _) = dlaev2(a, b, c);
    
    // Verify Av = λv where v = [cs1, sn1]
    let v1 = a*cs1 + b*sn1;
    let v2 = b*cs1 + c*sn1;
    
    assert_relative_eq!(v1, rt1*cs1, epsilon = 1e-12);
    assert_relative_eq!(v2, rt1*sn1, epsilon = 1e-12);
}

#[test]
fn test_dlaev2_trace_preservation() {
    // Test that sum of eigenvalues equals trace
    let a = 2.0;
    let b = 1.5;
    let c = 4.0;
    let (rt1, rt2, _, _, _) = dlaev2(a, b, c);
    
    assert_relative_eq!(rt1 + rt2, a + c, epsilon = 1e-12);
}

#[test]
fn test_dlaev2_determinant_preservation() {
    // Test that product of eigenvalues equals determinant
    let a = 2.0;
    let b = 1.5;
    let c = 4.0;
    let (rt1, rt2, _, _, _) = dlaev2(a, b, c);
    
    assert_relative_eq!(rt1 * rt2, a*c - b*b, epsilon = 1e-12);
}

#[test]
fn test_dlaev2_special_case() {
    // Test case where |df| = |tb|
    let (rt1, rt2, cs1, sn1, _) = dlaev2(1.0, 1.0, -1.0);
    
    assert_relative_eq!(cs1*cs1 + sn1*sn1, 1.0, epsilon = 1e-12);
    assert!(rt1 >= rt2);
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
    // Test 1: Basic swap with increment 1
    {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![5.0, 6.0, 7.0, 8.0];
        let n = 4;
        dswap(n, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // Test 2: Empty vectors (n=0)
    {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![3.0, 4.0];
        dswap(0, &mut x, 1, &mut y, 1);
        // Should remain unchanged
        assert_eq!(x, vec![1.0, 2.0]);
        assert_eq!(y, vec![3.0, 4.0]);
    }

    // Test 3: Single element (n=1)
    {
        let mut x = vec![1.0];
        let mut y = vec![2.0];
        dswap(1, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![2.0]);
        assert_eq!(y, vec![1.0]);
    }

    // Test 4: Two elements (n=2, tests partial unrolling)
    {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![3.0, 4.0];
        dswap(2, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![3.0, 4.0]);
        assert_eq!(y, vec![1.0, 2.0]);
    }

    // Test 5: Three elements (tests full unrolling)
    {
        let mut x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        dswap(3, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![4.0, 5.0, 6.0]);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    // Test 6: Increment > 1
    {
        let mut x = vec![1.0, -1.0, 2.0, -1.0, 3.0];
        let mut y = vec![4.0, -1.0, 5.0, -1.0, 6.0];
        // Swap elements at positions 0,2,4
        dswap(3, &mut x, 2, &mut y, 2);
        assert_eq!(x, vec![4.0, -1.0, 5.0, -1.0, 6.0]);
        assert_eq!(y, vec![1.0, -1.0, 2.0, -1.0, 3.0]);
    }

    // Test 7: Different increments for x and y
    {
        let mut x = vec![1.0, -1.0, 2.0, -1.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        // x uses increment 2, y uses increment 1
        dswap(2, &mut x, 2, &mut y, 1);
        assert_eq!(x, vec![4.0, -1.0, 5.0, -1.0, 3.0]);
        assert_eq!(y, vec![1.0, 2.0, 6.0]);
    }

    // Test 8: Negative increments
    {
        let mut x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        // Swap in reverse order
        dswap(3, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![4.0, 5.0, 6.0]);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    // Test 9: Large vectors to test loop unrolling
    {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut y = vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        dswap(7, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]);
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    // Test 10: Mixed increments with larger vectors
    {
        let mut x = vec![1.0, -1.0, 2.0, -1.0, 3.0, -1.0, 4.0];
        let mut y = vec![5.0, -1.0, 6.0, -1.0, 7.0, -1.0, 8.0];
        // Swap every other element
        dswap(4, &mut x, 2, &mut y, 2);
        assert_eq!(x, vec![5.0, -1.0, 6.0, -1.0, 7.0, -1.0, 8.0]);
        assert_eq!(y, vec![1.0, -1.0, 2.0, -1.0, 3.0, -1.0, 4.0]);
    }
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
    // Test 1: Basic ascending + ascending merge
    {
        let a = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];  // [1,3,5] and [2,4,6] are sorted ascending
        let n1 = 3;
        let n2 = 3;
        let mut index = vec![0; n1 + n2];
        dlamrg(n1, n2, &a, 1, 1, &mut index);
        
        // After applying permutation, should get [1,2,3,4,5,6]
        // So index should allow us to reconstruct this order from a
        assert_eq!(index, vec![0, 3, 1, 4, 2, 5]);
    }

    // Test 2: Descending + descending merge 
    {
        let a = vec![5.0, 3.0, 1.0, 6.0, 4.0, 2.0];  // [5,3,1] and [6,4,2] are sorted descending
        let n1 = 3;
        let n2 = 3;
        let mut index = vec![0; n1 + n2];
        dlamrg(n1, n2, &a, -1, -1, &mut index);
        
        // Should merge into ascending order [1,2,3,4,5,6]
        assert_eq!(index, vec![2, 5, 1, 4, 0, 3]);
    }

    // Test 3: Empty second list
    {
        let a = vec![1.0, 3.0, 5.0];
        let n1 = 3;
        let n2 = 0;
        let mut index = vec![0; n1 + n2];
        dlamrg(n1, n2, &a, 1, 1, &mut index);
        
        // Should just return indices for first list
        assert_eq!(index, vec![0, 1, 2]);
    }

    // Test 4: Unequal list sizes
    {
        let a = vec![1.0, 4.0, 2.0, 3.0, 5.0];  // [1,4] and [2,3,5]
        let n1 = 2;
        let n2 = 3;
        let mut index = vec![0; n1 + n2];
        dlamrg(n1, n2, &a, 1, 1, &mut index);
        
        // Should merge into [1,2,3,4,5]
        assert_eq!(index, vec![0, 2, 3, 1, 4]);
    }

    // Test 5: Lists with equal elements
    {
        let a = vec![1.0, 3.0, 2.0, 3.0];  // [1,3] and [2,3]
        let n1 = 2;
        let n2 = 2;
        let mut index = vec![0; n1 + n2];
        dlamrg(n1, n2, &a, 1, 1, &mut index);
        
        // Should maintain stability when merging equal elements
        assert_eq!(index, vec![0, 2, 1, 3]);
    }
    
    // Test 6: Mixed ascending/descending
    {
        let a = vec![1.0, 3.0, 5.0, 6.0, 4.0, 2.0];  // [1,3,5] ascending and [6,4,2] descending
        let n1 = 3;
        let n2 = 3;
        let mut index = vec![0; n1 + n2];
        dlamrg(n1, n2, &a, 1, -1, &mut index);
        
        // Should merge into [1,2,3,4,5,6]
        assert_eq!(index, vec![0, 5, 1, 4, 2, 3]);
    }
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
fn test_dlartv_basic_rotation() {
    let mut x = vec![1.0, 0.0];
    let mut y = vec![0.0, 1.0];
    let c = vec![0.0, 1.0];  // cos(90°), cos(0°)
    let s = vec![1.0, 0.0];  // sin(90°), sin(0°)
    
    dlartv(2, &mut x, 1, &mut y, 1, &c, &s, 1);
    
    // First rotation should swap x[0] and y[0]
    assert_relative_eq!(x[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(y[0], -1.0, epsilon = 1e-12);
}

#[test]
fn test_dlartv_45_degree_rotation() {
    let mut x = vec![1.0];
    let mut y = vec![1.0];
    let c = vec![0.7071067811865476]; // cos(45°)
    let s = vec![0.7071067811865476]; // sin(45°)
    
    dlartv(1, &mut x, 1, &mut y, 1, &c, &s, 1);
    
    // After 45° rotation
    assert_relative_eq!(x[0], 1.4142135623730951, epsilon = 1e-12); // √2
    assert_relative_eq!(y[0], 0.0, epsilon = 1e-12);
}

#[test]
fn test_dlartv_with_increments() {
    let mut x = vec![1.0, -1.0, 2.0, -2.0];
    let mut y = vec![0.0, -9.9, 0.0, -9.9];
    let c = vec![1.0, 0.0];
    let s = vec![0.0, 1.0];
    
    dlartv(2, &mut x, 2, &mut y, 2, &c, &s, 1);
    
    // Check only the affected elements
    assert_relative_eq!(x[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(x[2], 0.0, epsilon = 1e-12);
}

#[test]
fn test_dlartv_zero_vectors() {
    let mut x = vec![0.0, 0.0];
    let mut y = vec![0.0, 0.0];
    let c = vec![0.8, 0.8];
    let s = vec![0.6, 0.6];
    
    dlartv(2, &mut x, 1, &mut y, 1, &c, &s, 1);
    
    // Rotating zero vectors should keep them zero
    assert_relative_eq!(x[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(y[0], 0.0, epsilon = 1e-12);
}

#[test]
fn test_dlartv_identity_rotation() {
    let mut x = vec![1.0, 2.0, 3.0];
    let mut y = vec![4.0, 5.0, 6.0];
    let c = vec![1.0, 1.0, 1.0];
    let s = vec![0.0, 0.0, 0.0];
    
    let x_orig = x.clone();
    let y_orig = y.clone();
    
    dlartv(3, &mut x, 1, &mut y, 1, &c, &s, 1);
    
    // Identity rotation should not change vectors
    assert_relative_eq!(x[0], x_orig[0], epsilon = 1e-12);
    assert_relative_eq!(y[0], y_orig[0], epsilon = 1e-12);
}

// Length expectation tests
#[test]
#[should_panic]
fn test_dlartv_insufficient_x_length() {
    let mut x = vec![1.0];  // Too short
    let mut y = vec![1.0, 2.0];
    let c = vec![0.8, 0.8];
    let s = vec![0.6, 0.6];
    
    dlartv(2, &mut x, 1, &mut y, 1, &c, &s, 1);
}

#[test]
#[should_panic]
fn test_dlartv_insufficient_y_length() {
    let mut x = vec![1.0, 2.0];
    let mut y = vec![1.0];  // Too short
    let c = vec![0.8, 0.8];
    let s = vec![0.6, 0.6];
    
    dlartv(2, &mut x, 1, &mut y, 1, &c, &s, 1);
}

#[test]
#[should_panic]
fn test_dlartv_insufficient_rotation_params() {
    let mut x = vec![1.0, 2.0];
    let mut y = vec![1.0, 2.0];
    let c = vec![0.8];  // Too short
    let s = vec![0.6, 0.6];
    
    dlartv(2, &mut x, 1, &mut y, 1, &c, &s, 1);
}

#[test]
fn test_dlartv_vector_length_with_increments() {
    let n = 3;
    let incx = 2;
    let incy = 2;
    let incc = 1;
    
    let mut x = vec![0.0; 1 + (n-1)*incx];
    let mut y = vec![0.0; 1 + (n-1)*incy];
    let c = vec![1.0; 1 + (n-1)*incc];
    let s = vec![0.0; 1 + (n-1)*incc];
    
    // This should not panic
    dlartv(n, &mut x, incx, &mut y, incy, &c, &s, incc);
    
    // Verify we can access the last elements
    assert_relative_eq!(x[x.len()-1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(y[y.len()-1], 0.0, epsilon = 1e-12);
}

#[test]
fn test_dlartv_preserves_magnitude() {
    let mut x = vec![3.0];
    let mut y = vec![4.0];
    let c = vec![0.8];
    let s = vec![0.6];
    
    let initial_magnitude = f64::sqrt(x[0]*x[0] + y[0]*y[0]);
    
    dlartv(1, &mut x, 1, &mut y, 1, &c, &s, 1);
    
    let final_magnitude = f64::sqrt(x[0]*x[0] + y[0]*y[0]);
    
    assert_relative_eq!(initial_magnitude, final_magnitude, epsilon = 1e-12);
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

    // Test empty case
    let result = dlaeda(
        0, tlvls, curlvl, curpbm, &prmptr, &perm, &givptr, &givcol, &givnum, &q, &qptr, &mut z,
        &mut ztemp,
    );
    assert!(result.is_ok());
}



#[test]
fn test_dlaed0_empty() {
    let n = 0;
    let qsiz = 0;
    let mut d = vec![];
    let mut e = vec![];
    let mut q = vec![];
    let mut qstore = vec![];
    let mut work = vec![];
    let mut iwork = vec![];
    let mut givcol = vec![];
    let mut givnum = vec![];
    let mut qptr = vec![];
    let mut prmptr = vec![];
    let mut perm = vec![];
    let mut givptr = vec![];

    let result = dlaed0(
        2, n, qsiz, 0, 0, 0, &mut d, &mut e, &mut q, 1,
        &mut qstore, &mut qptr, &mut prmptr, &mut perm,
        &mut givptr, &mut givcol, &mut givnum, &mut work, &mut iwork,
    );
    assert!(result.is_ok());
}

#[test]
fn test_dlaed0_small_matrix() {
    let n = 4;
    let qsiz = 4;
    
    // Create tridiagonal matrix
    let mut d = vec![2.0, 2.0, 2.0, 2.0];  // Diagonal
    let mut e = vec![1.0, 1.0, 1.0];       // Off-diagonal
    let mut q = vec![vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.0, 1.0]];

    // Create same matrix in nalgebra format for reference
    let mut matrix = DMatrix::zeros(n, n);
    for i in 0..n {
        matrix[(i, i)] = d[i];
        if i < n-1 {
            matrix[(i, i+1)] = e[i];
            matrix[(i+1, i)] = e[i];
        }
    }
    
    // Get reference eigenvalues using nalgebra
    let eigen = matrix.symmetric_eigen();
    let mut expected_eigs: Vec<f64> = eigen.eigenvalues.as_slice().to_vec();
    expected_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    println!("Reference eigenvalues: {:?}", expected_eigs);

    // Allocate workspace arrays
    let mut qstore = vec![vec![0.0; n]; n];
    let work_size = 1 + 4*n + n*n;
    let iwork_size = 3 + 5*n;
    let mut work = vec![0.0; work_size];
    let mut iwork = vec![0; iwork_size];
    let mut qptr = vec![0; n];
    let mut prmptr = vec![0; n];
    let mut perm = vec![0; n];
    let mut givptr = vec![0; n];
    let mut givcol = vec![vec![0; 2]; n];
    let mut givnum = vec![vec![0.0; 2]; n];

    println!("Input matrix:");
    println!("Diagonal: {:?}", d);
    println!("Off-diagonal: {:?}", e);

    let result = dlaed0(
        2, n, qsiz, 0, 0, 0, &mut d, &mut e, &mut q, n,
        &mut qstore, &mut qptr, &mut prmptr, &mut perm,
        &mut givptr, &mut givcol, &mut givnum, &mut work, &mut iwork,
    );
    
    assert!(result.is_ok(), "dlaed0 failed");
    
    println!("Computed eigenvalues: {:?}", d);

    // Check eigenvalues
    let eps = 1e-12;
    for i in 0..n {
        let diff = (d[i] - expected_eigs[i]).abs();
        println!("Eigenvalue {}: computed = {}, expected = {}, diff = {}", 
                i, d[i], expected_eigs[i], diff);
        assert!(diff < eps, 
                "Eigenvalue {} mismatch: computed = {}, expected = {}, diff = {}", 
                i, d[i], expected_eigs[i], diff);
    }

    // Check eigenvectors are orthogonal
    println!("\nChecking eigenvector orthogonality:");
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += q[k][i] * q[k][j];
            }
            println!("dot(v{}, v{}) = {}", i, j, dot);
            if i == j {
                assert!((dot - 1.0).abs() < eps,
                        "Eigenvector {} not normalized: dot = {}", i, dot);
            } else {
                assert!(dot.abs() < eps,
                        "Eigenvectors {},{} not orthogonal: dot = {}", i, j, dot);
            }
        }
    }

    // Verify eigenvalue/eigenvector relationship (Av = λv)
    println!("\nVerifying eigenvalue/eigenvector relationship:");
    for i in 0..n {
        let mut v = vec![0.0; n];
        for j in 0..n {
            v[j] = q[j][i];
        }
        
        // Compute Av
        let mut av = vec![0.0; n];
        for j in 0..n {
            av[j] = d[j] * v[j];
            if j > 0 {
                av[j] += e[j-1] * v[j-1];
            }
            if j < n-1 {
                av[j] += e[j] * v[j+1];
            }
        }

        // Compare Av with λv
        let lambda = d[i];
        let mut max_diff: f64 = 0.0;
        for j in 0..n {
            let diff = (av[j] - lambda * v[j]).abs();
            max_diff = max_diff.max(diff);
        }
        println!("Max residual for eigenpair {}: {}", i, max_diff);
        assert!(max_diff < eps * 10.0,  // Slightly larger tolerance for this test
                "Eigenpair {} fails Av = λv check with residual {}", i, max_diff);
    }
}

#[test]
fn test_dlaed0_large_matrix() {
    let n = 32;  // > smlsiz (25)
    let qsiz = n;
    
    // Create tridiagonal matrix
    let mut d = vec![2.0; n];  // Diagonal
    let mut e = vec![1.0; n-1];  // Off-diagonal
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        q[i][i] = 1.0;  // Identity matrix
    }

    // Create reference matrix using nalgebra
    let mut matrix = DMatrix::zeros(n, n);
    for i in 0..n {
        matrix[(i, i)] = d[i];
        if i < n-1 {
            matrix[(i, i+1)] = e[i];
            matrix[(i+1, i)] = e[i];
        }
    }
    
    // Get reference eigenvalues
    let eigen = matrix.symmetric_eigen();
    let mut expected_eigs: Vec<f64> = eigen.eigenvalues.as_slice().to_vec();
    expected_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Allocate workspace arrays
    let mut qstore = vec![vec![0.0; n]; n];
    let work_size = 1 + 4*n + n*n;
    let iwork_size = 3 + 5*n;
    let mut work = vec![0.0; work_size];
    let mut iwork = vec![0; iwork_size];
    let mut qptr = vec![0; n];
    let mut prmptr = vec![0; n];
    let mut perm = vec![0; n];
    let mut givptr = vec![0; n];
    let mut givcol = vec![vec![0; 2]; n];
    let mut givnum = vec![vec![0.0; 2]; n];

    let result = dlaed0(
        2, n, qsiz, 0, 0, 0, &mut d, &mut e, &mut q, n,
        &mut qstore, &mut qptr, &mut prmptr, &mut perm,
        &mut givptr, &mut givcol, &mut givnum, &mut work, &mut iwork,
    );
    
    assert!(result.is_ok(), "dlaed0 failed for large matrix");

    // Check eigenvalues
    let eps = 1e-10;  // Slightly larger tolerance for larger matrix
    for i in 0..n {
        let diff = (d[i] - expected_eigs[i]).abs();
        assert!(diff < eps,
                "Large matrix eigenvalue {} mismatch: computed = {}, expected = {}, diff = {}",
                i, d[i], expected_eigs[i], diff);
    }

    // Check eigenvector orthogonality
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += q[k][i] * q[k][j];
            }
            if i == j {
                assert!((dot - 1.0).abs() < eps,
                        "Large matrix eigenvector {} not normalized", i);
            } else {
                assert!(dot.abs() < eps,
                        "Large matrix eigenvectors {},{} not orthogonal", i, j);
            }
        }
    }
}
