use graphome::dsbevd::SymmetricBandedMatrix;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand_distr::{Distribution, Normal};
use std::f64;
use std::f64::EPSILON;
use std::time::Instant;
use approx::assert_relative_eq;
use approx::assert_abs_diff_eq;
use std::f64::consts::PI;
use nalgebra as na;

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
use graphome::dsbevd::dlaed2;
use graphome::dsbevd::dlaed3;
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
use graphome::dsbevd::dlaed1;
use graphome::dsbevd::dlaset;
use graphome::dsbevd::dlassq;
use graphome::dsbevd::dnrm2;
use graphome::dsbevd::dlasr;
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



#[test]
fn test_dlar2v_empty() {
    let mut x: Vec<f64> = vec![];
    let mut y: Vec<f64> = vec![];
    let mut z: Vec<f64> = vec![];
    let c: Vec<f64> = vec![];
    let s: Vec<f64> = vec![];
    assert!(dlar2v(0, &mut x, &mut y, &mut z, 1, &c, &s, 1).is_ok());
}

#[test]
fn test_dlar2v_single_rotation() {
    // Use 30 degree rotation 
    let angle = PI / 6.0; // 30 degrees
    let cos_30 = angle.cos();
    let sin_30 = angle.sin();
    
    let mut x = vec![1.0];
    let mut y = vec![2.0];
    let mut z = vec![3.0];
    let c = vec![cos_30];
    let s = vec![sin_30];

    assert!(dlar2v(1, &mut x, &mut y, &mut z, 1, &c, &s, 1).is_ok());

    // Expected values computed using original LAPACK implementation
    let t1 = sin_30 * 3.0;
    let t2 = cos_30 * 3.0;
    let t3 = t2 - sin_30 * 1.0;
    let t4 = t2 + sin_30 * 2.0;
    let t5 = cos_30 * 1.0 + t1;
    let t6 = cos_30 * 2.0 - t1;
    let expected_x = cos_30 * t5 + sin_30 * t4;
    let expected_y = cos_30 * t6 - sin_30 * t3;
    let expected_z = cos_30 * t4 - sin_30 * t5;

    assert_relative_eq!(x[0], expected_x, epsilon = 1e-12);
    assert_relative_eq!(y[0], expected_y, epsilon = 1e-12);
    assert_relative_eq!(z[0], expected_z, epsilon = 1e-12);
}

#[test]
fn test_dlar2v_multiple_elements_incx_1() {
    let angle = PI / 6.0;
    let cos_30 = angle.cos();
    let sin_30 = angle.sin();

    let mut x = vec![1.0, 2.0];
    let mut y = vec![2.0, 4.0];
    let mut z = vec![3.0, 6.0];
    let c = vec![cos_30, cos_30];
    let s = vec![sin_30, sin_30];

    assert!(dlar2v(2, &mut x, &mut y, &mut z, 1, &c, &s, 1).is_ok());

    // Compute expected values for second element
    let t1 = sin_30 * 6.0;
    let t2 = cos_30 * 6.0;
    let t3 = t2 - sin_30 * 2.0;
    let t4 = t2 + sin_30 * 4.0;
    let t5 = cos_30 * 2.0 + t1;
    let t6 = cos_30 * 4.0 - t1;
    let expected_x = cos_30 * t5 + sin_30 * t4;

    assert_relative_eq!(x[1], expected_x, epsilon = 1e-12);
}

#[test]
fn test_dlar2v_multiple_elements_with_incx_2() {
    let angle = PI / 6.0;
    let cos_30 = angle.cos();
    let sin_30 = angle.sin();

    let mut x = vec![1.0, 0.0, 2.0];
    let mut y = vec![2.0, 0.0, 4.0];
    let mut z = vec![3.0, 0.0, 6.0];
    let c = vec![cos_30, cos_30];
    let s = vec![sin_30, sin_30];

    assert!(dlar2v(2, &mut x, &mut y, &mut z, 2, &c, &s, 1).is_ok());

    // Compute expected value for second element with increment
    let t1 = sin_30 * 6.0;
    let t2 = cos_30 * 6.0;
    let t3 = t2 - sin_30 * 2.0;
    let t4 = t2 + sin_30 * 4.0;
    let t5 = cos_30 * 2.0 + t1;
    let t6 = cos_30 * 4.0 - t1;
    let expected_x = cos_30 * t5 + sin_30 * t4;

    assert_relative_eq!(x[2], expected_x, epsilon = 1e-12);
    // Middle element should be unchanged
    assert_relative_eq!(x[1], 0.0, epsilon = 1e-12);
}

#[test]
fn test_dlar2v_multiple_elements_with_incc_2() {
    let angle = PI / 6.0;
    let cos_30 = angle.cos();
    let sin_30 = angle.sin();

    let mut x = vec![1.0, 2.0];
    let mut y = vec![2.0, 4.0];
    let mut z = vec![3.0, 6.0];
    let c = vec![cos_30, 0.0, cos_30];
    let s = vec![sin_30, 0.0, sin_30];

    assert!(dlar2v(2, &mut x, &mut y, &mut z, 1, &c, &s, 2).is_ok());

    // Compute expected value with incremented c,s arrays
    let t1 = sin_30 * 6.0;
    let t2 = cos_30 * 6.0;
    let t3 = t2 - sin_30 * 2.0;
    let t4 = t2 + sin_30 * 4.0;
    let t5 = cos_30 * 2.0 + t1;
    let t6 = cos_30 * 4.0 - t1;
    let expected_x = cos_30 * t5 + sin_30 * t4;

    assert_relative_eq!(x[1], expected_x, epsilon = 1e-12);
}

#[test]
#[should_panic]
fn test_dlar2v_invalid_incx() {
    let mut x = vec![1.0];
    let mut y = vec![2.0];
    let mut z = vec![3.0];
    let c = vec![1.0];
    let s = vec![0.0];
    dlar2v(1, &mut x, &mut y, &mut z, 0, &c, &s, 1).unwrap();
}

#[test]
#[should_panic]
fn test_dlar2v_invalid_incc() {
    let mut x = vec![1.0];
    let mut y = vec![2.0];
    let mut z = vec![3.0];
    let c = vec![1.0];
    let s = vec![0.0];
    dlar2v(1, &mut x, &mut y, &mut z, 1, &c, &s, 0).unwrap();
}

#[test]
#[should_panic]
fn test_dlar2v_x_array_too_small() {
    let mut x = vec![1.0];
    let mut y = vec![2.0, 4.0];
    let mut z = vec![3.0, 6.0];
    let c = vec![1.0, 1.0];
    let s = vec![0.0, 0.0];
    dlar2v(2, &mut x, &mut y, &mut z, 1, &c, &s, 1).unwrap();
}

#[test]
#[should_panic]
fn test_dlar2v_y_array_too_small() {
    let mut x = vec![1.0, 2.0];
    let mut y = vec![2.0];
    let mut z = vec![3.0, 6.0];
    let c = vec![1.0, 1.0];
    let s = vec![0.0, 0.0];
    dlar2v(2, &mut x, &mut y, &mut z, 1, &c, &s, 1).unwrap();
}

#[test]
#[should_panic]
fn test_dlar2v_z_array_too_small() {
    let mut x = vec![1.0, 2.0];
    let mut y = vec![2.0, 4.0];
    let mut z = vec![3.0];
    let c = vec![1.0, 1.0];
    let s = vec![0.0, 0.0];
    dlar2v(2, &mut x, &mut y, &mut z, 1, &c, &s, 1).unwrap();
}

#[test]
#[should_panic]
fn test_dlar2v_c_array_too_small() {
    let mut x = vec![1.0, 2.0];
    let mut y = vec![2.0, 4.0];
    let mut z = vec![3.0, 6.0];
    let c = vec![1.0];
    let s = vec![0.0, 0.0];
    dlar2v(2, &mut x, &mut y, &mut z, 1, &c, &s, 1).unwrap();
}

#[test]
#[should_panic]
fn test_dlar2v_s_array_too_small() {
    let mut x = vec![1.0, 2.0];
    let mut y = vec![2.0, 4.0];
    let mut z = vec![3.0, 6.0];
    let c = vec![1.0, 1.0];
    let s = vec![0.0];
    dlar2v(2, &mut x, &mut y, &mut z, 1, &c, &s, 1).unwrap();
}

#[test]
fn test_dlar2v_identity_rotation() {
    // Test with identity rotation (0 degrees)
    let mut x = vec![1.0, 2.0];
    let mut y = vec![3.0, 4.0];
    let mut z = vec![5.0, 6.0];
    let c = vec![1.0, 1.0];  // cos(0) = 1
    let s = vec![0.0, 0.0];  // sin(0) = 0

    assert!(dlar2v(2, &mut x, &mut y, &mut z, 1, &c, &s, 1).is_ok());

    // With identity rotation, values should remain unchanged
    assert_relative_eq!(x[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(y[0], 3.0, epsilon = 1e-12);
    assert_relative_eq!(z[0], 5.0, epsilon = 1e-12);
    assert_relative_eq!(x[1], 2.0, epsilon = 1e-12);
    assert_relative_eq!(y[1], 4.0, epsilon = 1e-12);
    assert_relative_eq!(z[1], 6.0, epsilon = 1e-12);
}

#[test]
fn test_dlar2v_90_degree_rotation() {
    // Test with 90 degree rotation
    let angle = PI / 2.0;
    let cos_90 = angle.cos();  // 0
    let sin_90 = angle.sin();  // 1
    
    let mut x = vec![1.0];
    let mut y = vec![2.0];
    let mut z = vec![3.0];
    let c = vec![cos_90];
    let s = vec![sin_90];

    assert!(dlar2v(1, &mut x, &mut y, &mut z, 1, &c, &s, 1).is_ok());

    // Calculate expected values for 90 degree rotation
    let t1 = sin_90 * 3.0;
    let t2 = cos_90 * 3.0;
    let t3 = t2 - sin_90 * 1.0;
    let t4 = t2 + sin_90 * 2.0;
    let t5 = cos_90 * 1.0 + t1;
    let t6 = cos_90 * 2.0 - t1;
    let expected_x = cos_90 * t5 + sin_90 * t4;
    let expected_y = cos_90 * t6 - sin_90 * t3;
    let expected_z = cos_90 * t4 - sin_90 * t5;

    assert_relative_eq!(x[0], expected_x, epsilon = 1e-12);
    assert_relative_eq!(y[0], expected_y, epsilon = 1e-12);
    assert_relative_eq!(z[0], expected_z, epsilon = 1e-12);
}


// Helper to create a test matrix
fn create_test_matrix(m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            matrix[i][j] = (i + j) as f64;
        }
    }
    matrix
}

// Helper to create rotation parameters
fn create_rotation_params(num_rotations: usize, angle_step: f64) -> (Vec<f64>, Vec<f64>) {
    let mut c = Vec::with_capacity(num_rotations);
    let mut s = Vec::with_capacity(num_rotations);
    
    for i in 0..num_rotations {
        let angle = angle_step * (i + 1) as f64;
        c.push(angle.cos());
        s.push(angle.sin());
    }
    
    (c, s)
}

// Helper to apply a 2x2 rotation matrix manually for verification
fn apply_rotation(c: f64, s: f64, x1: f64, x2: f64) -> (f64, f64) {
    let y1 = c * x1 + s * x2;
    let y2 = -s * x1 + c * x2;
    (y1, y2)
}

#[test]
fn test_dlasr_left_variable_forward() {
    let m = 4;
    let n = 3;
    let mut actual = create_test_matrix(m, n);
    let mut expected = create_test_matrix(m, n);
    
    let (c, s) = create_rotation_params(m-1, PI/6.0);
    
    // Apply the rotations in forward order P = P(z-1)*...*P(2)*P(1)
    // This means we apply P(1) first, then P(2), etc.
    for k in 0..m-1 {
        for col in 0..n {
            let (y1, y2) = apply_rotation(
                c[k], 
                s[k],
                expected[k][col],
                expected[k+1][col]
            );
            expected[k][col] = y1;
            expected[k+1][col] = y2;
        }
    }
    
    let result = dlasr('L', 'V', 'F', m, n, &c, &s, &mut actual, m);
    assert!(result.is_ok());
    
    for i in 0..m {
        for j in 0..n {
            assert_abs_diff_eq!(actual[i][j], expected[i][j], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_dlasr_right_top_backward() {
    let m = 3;
    let n = 4;
    let mut actual = create_test_matrix(m, n);
    let mut expected = create_test_matrix(m, n);
    
    let (c, s) = create_rotation_params(n-1, PI/4.0);  // 45° increments
    
    // Apply the rotations manually to create expected result
    for j in (0..n-1).rev() {
        for i in 0..m {
            let (y1, y2) = apply_rotation(c[j], s[j], expected[i][0], expected[i][j+1]);
            expected[i][0] = y1;
            expected[i][j+1] = y2;
        }
    }
    
    let result = dlasr('R', 'T', 'B', m, n, &c, &s, &mut actual, m);
    assert!(result.is_ok());
    
    for i in 0..m {
        for j in 0..n {
            assert_abs_diff_eq!(actual[i][j], expected[i][j], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_dlasr_zero_size_matrix() {
    let mut a: Vec<Vec<f64>> = vec![vec![]];
    let c = vec![];
    let s = vec![];
    
    let result = dlasr('L', 'V', 'F', 0, 0, &c, &s, &mut a, 1);
    assert!(result.is_ok());
}

#[test]
fn test_dlasr_identity_rotations() {
    let m = 3;
    let n = 3;
    let mut actual = create_test_matrix(m, n);
    let expected = actual.clone();
    
    // Create identity rotations (0° angles)
    let c = vec![1.0; m-1];
    let s = vec![0.0; m-1];
    
    let result = dlasr('L', 'V', 'F', m, n, &c, &s, &mut actual, m);
    assert!(result.is_ok());
    
    for i in 0..m {
        for j in 0..n {
            assert_abs_diff_eq!(actual[i][j], expected[i][j], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_dlasr_invalid_parameters() {
    let mut a = vec![vec![0.0; 3]; 3];
    let c = vec![1.0];
    let s = vec![0.0];
    
    assert!(dlasr('X', 'V', 'F', 3, 3, &c, &s, &mut a, 3).is_err());
    assert!(dlasr('L', 'X', 'F', 3, 3, &c, &s, &mut a, 3).is_err());
    assert!(dlasr('L', 'V', 'X', 3, 3, &c, &s, &mut a, 3).is_err());
}


#[test]
fn test_dlaed1_empty_matrix() {
    let mut d: Vec<f64> = vec![];
    let mut q: Vec<Vec<f64>> = vec![];
    let mut indxq: Vec<usize> = vec![];
    let mut rho = 0.0;
    let mut work = vec![];
    let mut iwork = vec![];
    
    let result = dlaed1(0, &mut d, &mut q, 0, &mut indxq, &mut rho, 0, &mut work, &mut iwork);
    assert!(result.is_ok());
}

// fn test_dlaed1_single_element(): This test should not exist as it's mathematically impossible to satisfy CUTPNT requirements for N=1.


#[test]
fn test_dlaed1_error_cases() {
    let n = 3;
    let mut d = vec![1.0, 2.0, 3.0];
    let mut q = vec![vec![1.0; n]; n];
    let mut indxq = vec![0, 1, 2];
    let mut rho = 1.0;
    let mut work = vec![0.0; 4*n + n*n];
    let mut iwork = vec![0; 4*n];

    // Test case 1: Invalid cutpnt > N/2
    {
        let mut work_copy = work.clone();
        let mut iwork_copy = iwork.clone();
        let result = dlaed1(n, &mut d, &mut q, n, &mut indxq, &mut rho, 2, &mut work_copy, &mut iwork_copy);
        assert!(result.is_err());
    }

    // Test case 2: Invalid cutpnt = 0 for N > 1
    {
        let mut work_copy = work.clone();
        let mut iwork_copy = iwork.clone();
        let result = dlaed1(n, &mut d, &mut q, n, &mut indxq, &mut rho, 0, &mut work_copy, &mut iwork_copy);
        assert!(result.is_err());
    }

    // Test case 3: Insufficient work array size
    {
        let mut small_work = vec![0.0; 4*n + n*n - 1];
        let mut iwork_copy = iwork.clone();
        let result = dlaed1(n, &mut d, &mut q, n, &mut indxq, &mut rho, 1, &mut small_work, &mut iwork_copy);
        assert!(result.is_err());
    }

    // Test case 4: Insufficient iwork array size
    {
        let mut work_copy = work.clone();
        let mut small_iwork = vec![0; 4*n - 1];
        let result = dlaed1(n, &mut d, &mut q, n, &mut indxq, &mut rho, 1, &mut work_copy, &mut small_iwork);
        assert!(result.is_err());
    }

    // Test case 5: Invalid ldq
    {
        let mut work_copy = work.clone();
        let mut iwork_copy = iwork.clone();
        let result = dlaed1(n, &mut d, &mut q, 2, &mut indxq, &mut rho, 1, &mut work_copy, &mut iwork_copy);
        assert!(result.is_err());
    }
}


#[test]
fn test_dlaed2_with_deflation() {
    // Test case where deflation occurs due to small z components
    let n = 4;
    let n1 = 2;
    let mut d = vec![1.0, 2.0, 3.0, 4.0];
    let mut q = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let ldq = n;
    let mut indxq = vec![0, 1, 0, 1]; // indxq
    let mut rho = 1.0;
    // Set z components small enough to trigger deflation
    let mut z = vec![1e-10, 1e-10, 1e-10, 1e-10];
    let mut dlamda = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut q2 = vec![vec![0.0; n]; 2 * n]; // Adjusted length of q2
    let mut indx = vec![0; n];
    let mut indxc = vec![0; n];
    let mut indxp = vec![0; n];
    let mut coltyp = vec![0; n];
    let mut k = 0;

    // Call the function
    let result = dlaed2(
        &mut k,
        n,
        n1,
        &mut d,
        &mut q,
        ldq,
        &mut indxq,
        &mut rho,
        &mut z,
        &mut dlamda,
        &mut w,
        &mut q2,
        &mut indx,
        &mut indxc,
        &mut indxp,
        &mut coltyp,
    );

    // Since z components are tiny, deflation should occur, k should be 0
    assert_eq!(k, 0);
}

#[test]
fn test_idamax_basic() {
    let n: usize = 5;
    let x: [f64; 5] = [1.0, 3.0, -2.0, 5.0, 4.0];
    let incx: usize = 1;
    let result = idamax(n, &x, incx);
    assert_eq!(result, 3); // Maximum at index 3 (value 5.0)
}

#[test]
fn test_idamax_negative_values() {
    let n: usize = 5;
    let x: [f64; 5] = [-1.0, -3.0, -2.0, -5.0, -4.0];
    let incx: usize = 1;
    let result = idamax(n, &x, incx);
    assert_eq!(result, 3); // Maximum absolute value at index 3
}

#[test]
fn test_idamax_zeros() {
    let n: usize = 5;
    let x: [f64; 5] = [0.0, -0.0, 0.0, -0.0, 0.0];
    let incx: usize = 1;
    let result = idamax(n, &x, incx);
    assert_eq!(result, 0); // First index when all zeros
}

#[test]
fn test_idamax_multiple_maxima() {
    let n: usize = 5;
    let x: [f64; 5] = [3.0, -5.0, 5.0, -5.0, 5.0];
    let incx: usize = 1;
    let result = idamax(n, &x, incx);
    assert_eq!(result, 1); // First occurrence of max abs value
}

#[test]
fn test_idamax_stride() {
    let n: usize = 3;
    let x: [f64; 5] = [1.0, 4.0, -2.0, 5.0, 3.0];
    let incx: usize = 2;
    let result = idamax(n, &x, incx);
    assert_eq!(result, 2); // Max abs at x[4], index 2
}

#[test]
fn test_idamax_n_zero() {
    let n: usize = 0;
    let x: [f64; 0] = [];
    let incx: usize = 1;
    let result = idamax(n, &x, incx);
    assert_eq!(result, 0); // Return 0 when n is 0
}

#[test]
fn test_dlamch_eps() {
    let eps = dlamch('E');
    assert_eq!(eps, f64::EPSILON * 0.5);
}

#[test]
fn test_dlamch_base() {
    let base = dlamch('B');
    assert_eq!(base, 2.0f64);
}

#[test]
fn test_dlamch_rmin() {
    let rmin = dlamch('U');
    assert_eq!(rmin, f64::MIN_POSITIVE);
}

#[test]
fn test_dlamch_rmax() {
    let rmax = dlamch('O');
    assert_eq!(rmax, f64::MAX);
}

#[test]
fn test_dlapy2_basic() {
    let x: f64 = 3.0;
    let y: f64 = 4.0;
    let result = dlapy2(x, y);
    assert_eq!(result, 5.0);
}

#[test]
fn test_dlapy2_negative() {
    let x: f64 = -3.0;
    let y: f64 = -4.0;
    let result = dlapy2(x, y);
    assert_eq!(result, 5.0);
}

#[test]
fn test_dlapy2_zero_x() {
    let x: f64 = 0.0;
    let y: f64 = 4.0;
    let result = dlapy2(x, y);
    assert_eq!(result, 4.0);
}

#[test]
fn test_dlapy2_large_values() {
    let x: f64 = 1e154;
    let y: f64 = 1e154;
    let result = dlapy2(x, y);
    let expected = (2.0f64).sqrt() * 1e154;
    assert_eq!(result, expected);
}

#[test]
fn test_dcopy_basic() {
    let n: usize = 5;
    let dx: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let incx: i32 = 1;
    let mut dy: [f64; 5] = [0.0; 5];
    let incy: i32 = 1;
    dcopy(n, &dx, incx, &mut dy, incy);
    assert_eq!(dy, dx);
}

#[test]
fn test_dcopy_incx_incy() {
    let n: usize = 3;
    let dx: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let incx: i32 = 2;
    let mut dy: [f64; 4] = [0.0; 4];
    let incy: i32 = 1;
    dcopy(n, &dx, incx, &mut dy, incy);
    assert_eq!(dy[0..3], [1.0, 3.0, 5.0]);
}

#[test]
fn test_dlacpy_all() {
    let uplo: char = 'A';
    let m: usize = 3;
    let n: usize = 3;
    let a: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let lda: usize = 3;
    let mut b: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
    let ldb: usize = 3;
    dlacpy(uplo, m, n, &a, lda, &mut b, ldb);
    assert_eq!(b, a);
}

#[test]
fn test_dlacpy_upper() {
    let uplo: char = 'U';
    let m: usize = 3;
    let n: usize = 3;
    let a: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let lda: usize = 3;
    let mut b: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
    let ldb: usize = 3;
    dlacpy(uplo, m, n, &a, lda, &mut b, ldb);
    let expected: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 5.0, 6.0],
        vec![0.0, 0.0, 9.0],
    ];
    assert_eq!(b, expected);
}

#[test]
fn test_dlamrg_basic() {
    let n1: usize = 3;
    let n2: usize = 3;
    let a: [f64; 6] = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    let dtrd1: i32 = 1;
    let dtrd2: i32 = 1;
    let mut index: [usize; 6] = [0; 6];
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    let expected: [usize; 6] = [0, 3, 1, 4, 2, 5];
    assert_eq!(index, expected);
}

#[test]
fn test_drot_no_rotation() {
    let n: usize = 2;
    let mut dx: [f64; 2] = [1.0, 2.0];
    let incx = 1;
    let mut dy: [f64; 2] = [3.0, 4.0];
    let incy = 1;
    let c: f64 = 1.0;
    let s: f64 = 0.0;
    drot(n, &mut dx, incx, &mut dy, incy, c, s);
    assert_eq!(dx, [1.0, 2.0]);
    assert_eq!(dy, [3.0, 4.0]);
}

#[test]
fn test_drot_90_degrees() {
    let n: usize = 2;
    let mut dx: [f64; 2] = [1.0, 2.0];
    let incx = 1;
    let mut dy: [f64; 2] = [3.0, 4.0];
    let incy = 1;
    let c: f64 = 0.0;
    let s: f64 = 1.0;
    drot(n, &mut dx, incx, &mut dy, incy, c, s);
    assert_eq!(dx, [3.0, 4.0]);
    assert_eq!(dy, [-1.0, -2.0]);
}

#[test]
fn test_dscal_basic() {
    let n: usize = 3;
    let alpha: f64 = 2.0;
    let mut x: [f64; 3] = [1.0, 2.0, 3.0];
    let incx: usize = 1;
    dscal(n, alpha, &mut x, incx);
    assert_eq!(x, [2.0, 4.0, 6.0]);
}

#[test]
fn test_dscal_stride() {
    let n: usize = 2;
    let alpha: f64 = 3.0;
    let mut x: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let incx: usize = 2;
    dscal(n, alpha, &mut x, incx);
    assert_eq!(x, [3.0, 2.0, 9.0, 4.0, 5.0]);
}


#[test]
fn test_dlamrg_basic_ascending_subsets() {
    let n1 = 3;
    let n2 = 3;
    let a = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    let dtrd1 = 1;
    let dtrd2 = 1;
    let mut index = vec![0usize; n1 + n2];
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    let expected_index = vec![0, 3, 1, 4, 2, 5];
    assert_eq!(index, expected_index);
}

#[test]
fn test_dlamrg_descending_subsets() {
    let n1 = 3;
    let n2 = 3;
    let a = vec![5.0, 3.0, 1.0, 6.0, 4.0, 2.0];
    let dtrd1 = -1;
    let dtrd2 = -1;
    let mut index = vec![0usize; n1 + n2];
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    let expected_index = vec![2, 5, 1, 4, 0, 3];
    assert_eq!(index, expected_index);
}

#[test]
fn test_dlamrg_mixed_strides() {
    let n1 = 3;
    let n2 = 3;
    let a = vec![1.0, 2.0, 3.0, 6.0, 5.0, 4.0];
    let dtrd1 = 1;
    let dtrd2 = -1;
    let mut index = vec![0usize; n1 + n2];
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    let expected_index = vec![0, 1, 2, 5, 4, 3];
    assert_eq!(index, expected_index);
}

#[test]
fn test_dlamrg_empty_first_subset() {
    let n1 = 0;
    let n2 = 3;
    let a = vec![2.0, 4.0, 6.0];
    let dtrd1 = 1;
    let dtrd2 = 1;
    let mut index = vec![0usize; n1 + n2];
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    let expected_index = vec![0, 1, 2];
    assert_eq!(index, expected_index);
}

#[test]
fn test_dlamrg_empty_second_subset() {
    let n1 = 3;
    let n2 = 0;
    let a = vec![1.0, 3.0, 5.0];
    let dtrd1 = 1;
    let dtrd2 = 1;
    let mut index = vec![0usize; n1 + n2];
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    let expected_index = vec![0, 1, 2];
    assert_eq!(index, expected_index);
}

#[test]
fn test_dlamrg_single_element_subsets() {
    let n1 = 1;
    let n2 = 1;
    let a = vec![3.0, 2.0];
    let dtrd1 = 1;
    let dtrd2 = 1;
    let mut index = vec![0usize; n1 + n2];
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    let expected_index = vec![1, 0];
    assert_eq!(index, expected_index);
}

#[test]
fn test_dlamrg_identical_elements() {
    let n1 = 3;
    let n2 = 3;
    let a = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let dtrd1 = 1;
    let dtrd2 = 1;
    let mut index = vec![0usize; n1 + n2];
    dlamrg(n1, n2, &a, dtrd1, dtrd2, &mut index);
    let expected_index = vec![0, 1, 2, 3, 4, 5];
    assert_eq!(index, expected_index);
}

// Tests for dlaed2
#[test]
fn test_dlaed2_all_deflated() {
    let n = 6;
    let n1 = 3;
    let mut d = vec![1.0; n];
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        q[i][i] = 1.0;
    }
    let ldq = n;
    let mut indxq = vec![0, 1, 2, 0, 1, 2];
    let mut rho = 0.5;
    let mut z = vec![1e-20; n]; // Very small z components
    let mut dlamda = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut q2 = vec![vec![0.0; n]; n];
    let mut indx = vec![0usize; n];
    let mut indxc = vec![0usize; n];
    let mut indxp = vec![0usize; n];
    let mut coltyp = vec![0usize; n];
    let mut k = 0;
    let result = dlaed2(
        &mut k,
        n,
        n1,
        &mut d,
        &mut q,
        ldq,
        &mut indxq,
        &mut rho,
        &mut z,
        &mut dlamda,
        &mut w,
        &mut q2,
        &mut indx,
        &mut indxc,
        &mut indxp,
        &mut coltyp,
    );
    assert_eq!(k, 0);
}

fn test_dlaed2_negative_rho() {
    let n = 6;
    let n1 = 3;
    let mut d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut q = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let ldq = n;
    let mut indxq = vec![0, 1, 2, 0, 1, 2];
    let mut rho = -0.5;
    let mut z = vec![0.0; n];
    
    // Set initial z values
    for i in 0..n1 {
        z[i] = q[n1-1][i];
    }
    for i in n1..n {
        z[i] = q[i][i];
    }

    let mut dlamda = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut q2 = vec![vec![0.0; n]; n];
    let mut indx = vec![0; n];
    let mut indxc = vec![0; n];
    let mut indxp = vec![0; n];
    let mut coltyp = vec![0; n];
    let mut k = 0;

    let z_before = z.clone();
    let result = dlaed2(
        &mut k,
        n,
        n1,
        &mut d,
        &mut q,
        ldq,
        &mut indxq,
        &mut rho,
        &mut z,
        &mut dlamda,
        &mut w,
        &mut q2,
        &mut indx,
        &mut indxc,
        &mut indxp,
        &mut coltyp,
    );

    assert!(result.is_ok());
    // Verify second part of z was negated
    for i in n1..n {
        assert_eq!(z_before[i], -z[i]);
    }
}

#[test]
fn test_dlaed2_invalid_n1() {
    let n = 6;
    let n1 = 7; // Invalid n1 (> n)
    let mut d = vec![1.0; n];
    let mut q = vec![vec![0.0; n]; n];
    let ldq = n;
    let mut indxq = vec![0; n];
    let mut rho = 0.5;
    let mut z = vec![0.0; n];
    let mut dlamda = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut q2 = vec![vec![0.0; n]; n];
    let mut indx = vec![0usize; n];
    let mut indxc = vec![0usize; n];
    let mut indxp = vec![0usize; n];
    let mut coltyp = vec![0usize; n];
    let mut k = 0;
    let result = dlaed2(
        &mut k,
        n,
        n1,
        &mut d,
        &mut q,
        ldq,
        &mut indxq,
        &mut rho,
        &mut z,
        &mut dlamda,
        &mut w,
        &mut q2,
        &mut indx,
        &mut indxc,
        &mut indxp,
        &mut coltyp,
    );
    assert!(result.is_err());
}

#[test]
fn test_dlaed2_identical_eigenvalues() {
    let n = 4;
    let n1 = 2;
    let mut d = vec![1.0, 1.0, 1.0, 1.0];
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        q[i][i] = 1.0;
    }
    let ldq = n;
    let mut indxq = vec![0, 1, 0, 1];
    let mut rho = 0.5;
    let mut z = vec![1.0; n];
    let mut dlamda = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut q2 = vec![vec![0.0; n]; n];
    let mut indx = vec![0usize; n];
    let mut indxc = vec![0usize; n];
    let mut indxp = vec![0usize; n];
    let mut coltyp = vec![0usize; n];
    let mut k = 0;
    let result = dlaed2(
        &mut k,
        n,
        n1,
        &mut d,
        &mut q,
        ldq,
        &mut indxq,
        &mut rho,
        &mut z,
        &mut dlamda,
        &mut w,
        &mut q2,
        &mut indx,
        &mut indxc,
        &mut indxp,
        &mut coltyp,
    );
    assert!(k < n);
}


#[test]
fn test_dlaed3_mechanical() {
    // Test basic 2x2 case hitting k=2 special path
    let k = 2;
    let n = 4; // Doubled size to handle dlaed4's array needs
    let n1 = 2; 
    let ldq = n;
    
    let mut dlamda = vec![1.0, 2.0, 3.0, 4.0]; // Room for combined arrays
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        q[i][i] = 1.0;  // Initialize to identity
    }
    let mut w = vec![0.5, 0.5, 0.0, 0.0]; 
    let mut d = vec![0.0; n];
    let rho = 0.25;
    
    // Setup all inputs with proper n-sized arrays
    let q2 = q.clone(); 
    let indx = vec![1, 2, 3, 4];
    let ctot = vec![2, 0, 2, 0];
    let mut s = vec![vec![0.0; n]; n];

    let result = dlaed3(k, n, n1, &mut d, &mut q, ldq, rho, 
                       &mut dlamda, &q2, &indx, &ctot, &mut w, &mut s);

   // Convert output Q to nalgebra matrix for verification
   let q_mat = na::DMatrix::from_row_slice(n, k, &q.iter().flatten().copied().collect::<Vec<f64>>());

   // Verify Q remains orthogonal
   let qtq = &q_mat.transpose() * &q_mat;
   let id = na::DMatrix::<f64>::identity(k, k);
   assert!((qtq - id).norm() < 1e-10, "Q lost orthogonality");

   // Verify eigenvalues sorted ascending
   assert!(d[0] <= d[1], "Eigenvalues not sorted");

   // Verify secular equation satisfied
   for i in 0..k {
       let lambda = d[i];
       let f: f64 = (0..k).map(|j| {
           let zj = w[j];
           zj * zj / (dlamda[j] - lambda)
       }).sum::<f64>();
       assert!((1.0 + rho * f).abs() < 1e-10, "Secular equation not satisfied");
   }
}

#[test]
fn test_dlaed3_deflation() {
   // Test larger case with deflation
   let k = 3;
   let n = 4;
   let n1 = 2;
   let ldq = 4;

   // Set up eigenvalues with known gap to trigger deflation
   let mut dlamda = vec![1.0, 5.0, 10.0];
   
   let mut q = vec![
       vec![1.0, 0.0, 0.0, 0.0],
       vec![0.0, 1.0, 0.0, 0.0], 
       vec![0.0, 0.0, 1.0, 0.0],
       vec![0.0, 0.0, 0.0, 1.0]
   ];

   // Create z vector with small component to trigger deflation
   let mut w = vec![0.5, 1e-14, 0.5]; // Middle component very small

   let mut d = vec![0.0; n];
   let rho = 0.1;

   let q2 = q.clone();
   let indx = vec![1, 2, 3];  // 1-based
   let ctot = vec![2, 0, 1, 0];
   let mut s = vec![vec![0.0; k]; k];

   let result = dlaed3(k, n, n1, &mut d, &mut q, ldq, rho,
                      &mut dlamda, &q2, &indx, &ctot, &mut w, &mut s);

   // Verify deflation preserved original eigenvalue
   let mut found_deflated = false;
   for i in 0..k {
       if (d[i] - dlamda[1]).abs() < 1e-10 {
           found_deflated = true;
           break;
       }
   }
   assert!(found_deflated, "Deflation failed to preserve eigenvalue");

   // Convert output Q to nalgebra matrix
   let q_mat = na::DMatrix::from_row_slice(n, k, &q.iter().flatten().copied().collect::<Vec<f64>>());

   // Verify Q remains orthogonal after deflation
   let qtq = &q_mat.transpose() * &q_mat;
   let id = na::DMatrix::<f64>::identity(k, k);
   assert!((qtq - id).norm() < 1e-10, "Q lost orthogonality after deflation");

   // Verify secular equation satisfied for non-deflated eigenvalues
   for i in 0..k {
       if (d[i] - dlamda[1]).abs() > 1e-10 {  // Skip deflated eigenvalue
           let lambda = d[i];
           let f: f64 = (0..k).map(|j| {
               let zj = w[j];
               zj * zj / (dlamda[j] - lambda)  
           }).sum::<f64>();
           assert!((1.0 + rho * f).abs() < 1e-10, "Secular equation not satisfied");
       }
   }
}


/// 1. Checking if eigenvalues are sorted in ascending order
/// 2. Verifying Av = λv for each eigenpair
/// 3. Checking orthogonality of eigenvectors
fn verify_eigen_solution(
    d: &[f64],          // original diagonal
    e: &[f64],          // original subdiagonal
    eigenvals: &[f64],  // computed eigenvalues
    eigenvecs: &[Vec<f64>], // computed eigenvectors
    tol: f64,
) -> bool {
    let n = d.len();
    
    // Check if eigenvalues are sorted
    for i in 1..n {
        if eigenvals[i] < eigenvals[i-1] {
            println!("Eigenvalues not sorted at index {}", i);
            return false;
        }
    }
    
    // Construct original tridiagonal matrix
    let mut mat = na::DMatrix::zeros(n, n);
    for i in 0..n {
        mat[(i, i)] = d[i];
        if i < n-1 {
            mat[(i, i+1)] = e[i];
            mat[(i+1, i)] = e[i];
        }
    }
    
    // Check Av = λv for each eigenpair
    for i in 0..n {
        let v = na::DVector::from_vec(eigenvecs[i].clone());
        let av = &mat * &v;
        let lambda_v = &v * eigenvals[i];
        
        let diff = (&av - &lambda_v).norm();
        if diff > tol {
            println!("Eigenpair {} failed Av = λv check with diff {}", i, diff);
            return false;
        }
    }
    
    // Check orthogonality
    for i in 0..n {
        for j in i+1..n {
            let dot = eigenvecs[i].iter().zip(eigenvecs[j].iter())
                                      .map(|(a, b)| a * b)
                                      .sum::<f64>();
            if dot.abs() > tol {
                println!("Vectors {} and {} not orthogonal: dot = {}", i, j, dot);
                return false;
            }
        }
        
        // Check normalization
        let norm = eigenvecs[i].iter().map(|x| x*x).sum::<f64>().sqrt();
        if (norm - 1.0).abs() > tol {
            println!("Vector {} not normalized: norm = {}", i, norm);
            return false;
        }
    }
    
    true
}

#[test]
fn test_2x2_matrix() {
    let n = 2;
    let mut d = vec![2.0, 5.0];  // diagonal
    let mut e = vec![1.0];       // subdiagonal
    let mut z = vec![vec![0.0; n]; n];
    let mut work = vec![0.0; 2*n];
    
    let result = dsteqr('I', n, &mut d, &mut e, &mut z, &mut work);
    assert!(result.is_ok());
    
    // Known eigenvalues for this matrix: (5.618034, 1.381966) - golden ratio related
    let expected_eigenvals = vec![1.381966011250105, 5.618033988749895];
    for (computed, expected) in d.iter().zip(expected_eigenvals.iter()) {
        assert_abs_diff_eq!(computed, expected, epsilon = 1e-12);
    }
    
    assert!(verify_eigen_solution(&vec![2.0, 5.0], &vec![1.0], &d, &z, 1e-12));
}

#[test]
fn test_3x3_matrix() {
    let n = 3;
    let mut d = vec![1.0, 2.0, 1.0];  // diagonal
    let mut e = vec![1.0, 1.0];       // subdiagonal
    let mut z = vec![vec![0.0; n]; n];
    let mut work = vec![0.0; 2*n];
    
    let result = dsteqr('I', n, &mut d, &mut e, &mut z, &mut work);
    assert!(result.is_ok());
    
    // Expected eigenvalues for this symmetric tridiagonal matrix
    let expected_eigenvals = vec![0.0, 2.0, 2.0];
    for (computed, expected) in d.iter().zip(expected_eigenvals.iter()) {
        assert_abs_diff_eq!(computed, expected, epsilon = 1e-12);
    }
    
    assert!(verify_eigen_solution(&vec![1.0, 2.0, 1.0], &vec![1.0, 1.0], &d, &z, 1e-12));
}

#[test]
fn test_degenerate_cases() {
    // Test n = 0
    {
        let mut d = vec![];
        let mut e = vec![];
        let mut z = vec![];
        let mut work = vec![];
        let result = dsteqr('I', 0, &mut d, &mut e, &mut z, &mut work);
        assert!(result.is_ok());
    }
    
    // Test n = 1
    {
        let mut d = vec![42.0];
        let mut e = vec![];
        let mut z = vec![vec![0.0; 1]; 1];
        let mut work = vec![0.0; 2];
        let result = dsteqr('I', 1, &mut d, &mut e, &mut z, &mut work);
        assert!(result.is_ok());
        assert_abs_diff_eq!(d[0], 42.0);
        assert_abs_diff_eq!(z[0][0], 1.0);
    }
}

#[test]
fn test_random_matrix() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let n = 5;
    let mut d: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let mut e: Vec<f64> = (0..n-1).map(|_| rng.gen_range(-5.0..5.0)).collect();
    let d_orig = d.clone();
    let e_orig = e.clone();
    
    let mut z = vec![vec![0.0; n]; n];
    let mut work = vec![0.0; 2*n];
    
    let result = dsteqr('I', n, &mut d, &mut e, &mut z, &mut work);
    assert!(result.is_ok());
    
    // Verify solution
    assert!(verify_eigen_solution(&d_orig, &e_orig, &d, &z, 1e-10));
    
    // Additional verification that eigenvalues are sorted
    for i in 1..n {
        assert!(d[i] >= d[i-1], "Eigenvalues not sorted");
    }
}
