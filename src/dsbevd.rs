// src/dsbevd.rs

use rayon::prelude::*;
use rand_distr::Distribution;
use std::cmp::min;
use nalgebra::linalg::SymmetricTridiagonal;

/// Represents a real symmetric banded matrix.
#[derive(Clone)]
pub struct SymmetricBandedMatrix {
    n: usize,           // Order of the matrix
    kd: usize,          // Number of subdiagonals (assuming 'L' storage)
    ab: Vec<Vec<f64>>,  // The lower triangle of the symmetric band matrix A, stored in banded format
}

pub struct EigenResults {
    pub eigenvalues: Vec<f64>,       // Eigenvalues in ascending order
    pub eigenvectors: Vec<Vec<f64>>, // Eigenvectors corresponding to the eigenvalues
}

impl SymmetricBandedMatrix {
    /// Creates a new symmetric banded matrix.
    pub fn new(n: usize, kd: usize, ab: Vec<Vec<f64>>) -> Self {
        assert!(ab.len() == kd + 1, "Incorrect number of rows in 'ab' matrix");
        assert!(ab[0].len() == n, "Incorrect number of columns in 'ab' matrix");
        SymmetricBandedMatrix { n, kd, ab }
    }

    /// Computes all eigenvalues and eigenvectors of the symmetric banded matrix.
    pub fn dsbevd(&self) -> EigenResults {
        let safmin = f64::MIN_POSITIVE;
        let eps = f64::EPSILON;
        let smlnum = safmin / eps;
        let bignum = 1.0 / smlnum;
        let rmin = smlnum.sqrt();
        let rmax = bignum.sqrt();
    
        let anrm = self.matrix_norm();
        let mut scale = 1.0;
        let mut iscale = 0;
        
        if anrm > 0.0 && anrm < rmin {
            iscale = 1;
            scale = rmin / anrm;
        } else if anrm > rmax {
            iscale = 2;
            scale = rmax / anrm;
        }
    
        let working_matrix = if scale != 1.0 {
            self.scaled_copy(scale)
        } else {
            self.clone()
        };
    
        let (mut d, mut e, mut q) = working_matrix.reduce_to_tridiagonal();
    
        // Use reliable tridiagonal solver
        let (eigenvals, eigenvecs) = {
            let mut tri = SymmetricTridiagonal::new(d.clone(), e[..n-1].clone());
            let decomp = tri.eigendecomposition(true).unwrap();
            (decomp.eigenvalues, decomp.eigenvectors)
        };
    
        // Transform eigenvectors back
        let eigenvectors = multiply_q(&q, &eigenvecs);
    
        // Rescale eigenvalues
        let mut eigenvalues = eigenvals;
        if scale != 1.0 {
            for eigenval in &mut eigenvalues {
                *eigenval /= scale;
            }
        }
    
        EigenResults {
            eigenvalues,
            eigenvectors,
        }
    }

    /// Reduces the symmetric banded matrix to tridiagonal form.
    /// Returns the diagonal elements `d`, off-diagonal elements `e`, and the accumulated orthogonal matrix `q`.
    fn reduce_to_tridiagonal(&self) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let n = self.n;
        let kd = self.kd;
        let mut ab = self.ab.clone();
        
        // Output arrays
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n-1];
        let mut q = vec![vec![0.0; n]; n];
        
        // Initialize Q to identity
        for i in 0..n {
            q[i][i] = 1.0;
        }
    
        if kd == 0 {
            // Diagonal case
            for i in 0..n {
                d[i] = ab[0][i];
            }
            return (d, e, q);
        }
    
        // Use DSBTRD algorithm structure
        let mut work = vec![0.0; n];
        
        for i in 0..n-1 {
            // Generate Householder to annihilate A(i+2:min(i+kd+1,n),i)
            let nrt = (n-i-1).min(kd);
            if nrt > 0 {
                let mut temp = Vec::with_capacity(nrt);
                for j in 0..nrt {
                    temp.push(ab[j+1][i]);
                }
                let (v, tau) = householder_reflector(&temp);
                
                // Apply transformation
                let mut sum = 0.0;
                for j in 0..nrt {
                    sum += v[j] * ab[j+1][i+j+1];
                }
                sum *= tau;
                
                for j in 0..nrt {
                    ab[j+1][i+j+1] -= sum * v[j];
                }
                
                // Accumulate transformation in q
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..nrt {
                        sum += q[j][i+k+1] * v[k];
                    }
                    sum *= tau;
                    for k in 0..nrt {
                        q[j][i+k+1] -= sum * v[k];
                    }
                }
            }
        }
        
        // Copy diagonal and subdiagonal
        for i in 0..n {
            d[i] = ab[0][i];
            if i < n-1 {
                e[i] = ab[1][i];
            }
        }
        
        (d, e, q)
    }
    
    fn matrix_norm(&self) -> f64 {
        let mut value = 0.0;
        if self.kd == 0 {
            // Diagonal case
            for &val in &self.ab[0] {
                value = value.max(val.abs());
            }
            return value;
        }
    
        // Band matrix case
        for j in 0..self.n {
            let mut sum = 0.0;
            for i in 0..=self.kd {
                if i + j < self.n {
                    sum += self.ab[i][j].abs();
                }
            }
            value = value.max(sum);
        }
        value
    }
    

    
    fn scaled_copy(&self, scale: f64) -> Self {
        let mut scaled = (*self).clone();
        for row in &mut scaled.ab {
            for val in row {
                *val *= scale;
            }
        }
        scaled
    }
}

/// Computes the Householder reflector for a vector x.
/// Returns the Householder vector v and the scalar tau.
/// The reflector is of the form H = I - tau * v * v^T
fn householder_reflector(x: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    if n == 0 { return (vec![], 0.0); }
    
    let safmin = f64::MIN_POSITIVE;
    let eps = f64::EPSILON;
    let safemin = safmin.max(eps * x[0].abs());
    
    let mut v = x.to_vec();
    if n == 1 { return (v, 0.0); }

    let mut scale = 0.0;
    let mut ssq = 0.0;
    
    // Two-pass scale computation for numerical stability
    for &xi in x.iter().skip(1) {
        scale = scale.max(xi.abs());
    }
    
    if scale == 0.0 {
        v[0] = x[0];
        for i in 1..n {
            v[i] = 0.0;
        }
        return (v, 0.0);
    }

    for &xi in x.iter().skip(1) {
        let temp = xi / scale;
        ssq += temp * temp;
    }
    
    let mut xnorm = scale * ssq.sqrt();
    let alpha = x[0];
    
    if xnorm == 0.0 {
        return (v, 0.0);
    }

    let mut beta = -alpha.signum() * (alpha.abs().hypot(xnorm));
    
    if beta.abs() < safemin {
        beta = -safemin.copysign(alpha);
    }

    let tau = (beta - alpha) / beta;
    let scale = 1.0 / (alpha - beta);
    
    for i in 1..n {
        v[i] *= scale;
    }
    v[0] = beta;
    
    (v, tau)
}


/// Computes the eigenvalues and eigenvectors of a symmetric tridiagonal matrix using the divide and conquer algorithm.
/// `d`: diagonal elements
/// `e`: off-diagonal elements
/// Returns the eigenvalues and the eigenvectors.
fn tridiagonal_eigen_dc(d: &[f64], e: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = d.len();
    let mut diag = d.to_vec();
    let mut off_diag = e.to_vec();

    // Initialize eigenvector matrix as identity
    let mut z = vec![vec![0.0; n]; n];
    for i in 0..n {
        z[i][i] = 1.0;
    }

    // Call the divide and conquer algorithm
    divide_and_conquer(&mut diag, &mut off_diag, &mut z);

    // The eigenvalues are in diag
    // The eigenvectors are in z

    // Sort the eigenvalues and eigenvectors
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| diag[i].partial_cmp(&diag[j]).unwrap());

    let eigenvalues = idx.iter().map(|&i| diag[i]).collect::<Vec<f64>>();
    let eigenvectors = idx
        .iter()
        .map(|&i| z.iter().map(|row| row[i]).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();

    (eigenvalues, eigenvectors)
}

/// Divide and conquer algorithm for symmetric tridiagonal eigenproblem.
/// Modifies `d`, `e`, and `z` in place.
fn divide_and_conquer(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>]) {
    let n = d.len();
    
    // Base cases
    const SMLSIZ: usize = 25;
    
    if n == 0 {
        return;
    }
    
    if n == 1 {
        z[0][0] = 1.0;
        return; 
    }
    
    if n <= SMLSIZ {
        tridiagonal_qr(d, e, z);
        return;
    }

    // Check for deflation
    let eps = f64::EPSILON;
    let mut deflated = false;
    for i in 0..n-1 {
        let tiny = eps * (d[i].abs().sqrt() * d[i+1].abs().sqrt());
        if e[i].abs() <= tiny {
            e[i] = 0.0;
            deflated = true;
        }
    }
    
    if deflated {
        // Find blocks and solve each separately
        let mut start = 0;
        for i in 0..n-1 {
            if e[i] == 0.0 {
                let block_size = i - start + 1;
                if block_size > 1 {
                    let d_block = &mut d[start..=i];
                    let e_block = &mut e[start..i];
                    let z_block = &mut z[start..=i];
                    divide_and_conquer(d_block, e_block, z_block);
                }
                start = i + 1;
            }
        }
        return;
    }

    // Divide step
    let m = n / 2;

    if m == 0 || m == n {
        return;
    }
    
    // Save and zero out off-diagonal element
    let e_m_minus_1 = e[m - 1];
    e[m - 1] = 0.0;
    
    // Recursively solve subproblems
    let (d_left, d_right) = d.split_at_mut(m);
    let (e_left, e_right) = e.split_at_mut(m - 1);
    let (z_left, z_right) = z.split_at_mut(m);
    
    divide_and_conquer(d_left, e_left, z_left);
    divide_and_conquer(d_right, e_right, z_right);
    
    // Form rank-one modification
    let mut z_vec = vec![0.0; n];
    for i in 0..m {
        z_vec[i] = z_left[i][m-1];
    }
    for i in 0..n-m {
        z_vec[m + i] = z_right[i][0];
    }
    
    // Solve secular equation
    let mut new_d = vec![0.0; n];
    let mut new_z = vec![vec![0.0; n]; n];
    
    solve_secular_equation(
        d_left,
        d_right, 
        &z_vec,
        e_m_minus_1,
        &mut new_d,
        &mut new_z
    );
    
    // Copy results back
    d.copy_from_slice(&new_d);
    
    // Update eigenvectors
    let mut temp_z = vec![vec![0.0; n]; n];
    
    // Left portion
    for i in 0..m {
        for j in 0..n {
            temp_z[i][j] = 0.0;
            for l in 0..m {
                temp_z[i][j] += z_left[i][l] * new_z[l][j];
            }
        }
    }
    
    // Right portion  
    for i in 0..n-m {
        for j in 0..n {
            temp_z[m+i][j] = 0.0;
            for l in 0..n-m {
                temp_z[m+i][j] += z_right[i][l] * new_z[m+l][j];
            }
        }
    }
    
    // Copy results back
    for i in 0..n {
        for j in 0..n {
            z[i][j] = temp_z[i][j];
        }
    }
}

/// Solves the secular equation in the divide and conquer algorithm.
/// `d1`, `d2`: Eigenvalues from the left and right subproblems.
/// `z`: Combined z vector from left and right subproblems.
/// `rho`: The rank-one update scalar.
/// `d`: Output eigenvalues.
/// `z_out`: Output eigenvectors.
fn solve_secular_equation(
    d1: &[f64], 
    d2: &[f64],
    z: &[f64],
    rho: f64,
    d: &mut [f64],
    z_out: &mut [Vec<f64>]
) {
    let n1 = d1.len();
    let n2 = d2.len();
    let n = n1 + n2;
    
    // Machine constants for numerical stability
    let eps = f64::EPSILON;
    let safmin = f64::MIN_POSITIVE;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt();

    let mut dlamda = vec![0.0; n];
    let mut delta = vec![0.0; n]; 
    
    // Copy eigenvalues and compute gaps
    for i in 0..n1 {
        dlamda[i] = d1[i];
    }
    for i in 0..n2 {
        dlamda[n1 + i] = d2[i];
    }
    
    // Sort eigenvalues and compute gaps
    dlamda.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..n-1 {
        delta[i] = dlamda[i+1] - dlamda[i];
    }
    delta[n-1] = rmax;

    // Normalize z vector
    let mut z_norm = 0.0;
    for i in 0..n {
        z_norm += z[i] * z[i];
    }
    z_norm = z_norm.sqrt();
    
    let mut z_scaled = vec![0.0; n];
    for i in 0..n {
        z_scaled[i] = z[i] / z_norm;
    }

    // Main secular equation solving loop
    for i in 0..n {
        let mut left = if i == 0 { 
            dlamda[0] - (dlamda[1] - dlamda[0]).abs() 
        } else { 
            dlamda[i-1] 
        };
        let mut right = if i == n-1 { 
            dlamda[n-1] + (dlamda[n-1] - dlamda[n-2]).abs()
        } else { 
            dlamda[i+1] 
        };
        
        let mut mid = dlamda[i];
        let mut converged = false;
        
        // Newton iterations
        for iter in 0..50 {
            let mut mid = (left + right) / 2.0;
            let mut sum = 0.0;
            let mut deriv = 0.0;
            
            for j in 0..n {
                if j != i {
                    let temp = z_scaled[j] / (dlamda[j] - mid);
                    sum += z_scaled[j] * temp;
                    deriv += temp * temp;
                }
            }
            
            let tol = eps * mid.abs();
            if sum.abs() <= tol {
                d[i] = mid;
                break;
            } else if iter >= 50 {
                // If Newton fails, do additional bisection steps
                let mut l = left;
                let mut r = right;
                for _ in 0..20 {
                    let m = (l + r) / 2.0;
                    let mut sum = 0.0;
                    for j in 0..n {
                        if j != i {
                            sum += z_scaled[j] * z_scaled[j] / (dlamda[j] - m);
                        }
                    }
                    if sum > 0.0 {
                        r = m;
                    } else {
                        l = m;
                    }
                }
                d[i] = (l + r) / 2.0;
                break;
            }
            
            let update = sum / deriv;
            if sum > 0.0 {
                right = mid;
            } else {
                left = mid;
            }
            
            // Newton step
            let new_mid = mid - update;
            if new_mid >= left && new_mid <= right {
                if (new_mid - mid).abs() < eps * mid.abs() {
                    d[i] = new_mid;
                    break;
                }
                d[i] = new_mid;
                mid = new_mid;
            } else {
                d[i] = mid;
                break;
            }
        }
        
        // Compute eigenvector components
        for j in 0..n {
            if j != i {
                z_out[j][i] = z_scaled[j] / (dlamda[j] - d[i]);
            } else {
                z_out[j][i] = 1.0;
            }
        }
        
        // Normalize eigenvector
        let mut norm = 0.0;
        for j in 0..n {
            norm += z_out[j][i] * z_out[j][i]; 
        }
        norm = norm.sqrt();
        for j in 0..n {
            z_out[j][i] /= norm;
        }
    }
}

/// Tridiagonal QR algorithm for small matrices.
/// `d`: Diagonal elements
/// `e`: Off-diagonal elements
/// `z`: Eigenvectors (output)
fn tridiagonal_qr(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>]) {
    let n = d.len();
    let mut tri = SymmetricTridiagonal::new(d.to_vec(), e[..n-1].to_vec());
    let decomp = tri.eigendecomposition(true).unwrap();
    
    // Copy eigenvalues back
    d.copy_from_slice(&decomp.eigenvalues);
    
    // Copy eigenvectors
    for i in 0..n {
        for j in 0..n {
            z[i][j] = decomp.eigenvectors[(i,j)];
        }
    }
}

/// Computes the Givens rotation coefficients c and s such that
/// [c -s; s c]^T * [a; b] = [r; 0]
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    let eps = f64::EPSILON;
    let safmin = f64::MIN_POSITIVE;
    
    if b == 0.0 {
        (1.0, 0.0)
    } else if a == 0.0 {
        (0.0, 1.0)
    } else {
        let abs_a = a.abs();
        let abs_b = b.abs();
        if abs_b >= abs_a {
            let t = a / b;
            let t2 = t * t;
            let u = (1.0 + t2).sqrt();
            let s = if b > 0.0 { 1.0 / u } else { -1.0 / u };
            let c = t * s;
            (c, s)
        } else {
            let t = b / a;
            let t2 = t * t;
            let u = (1.0 + t2).sqrt();
            let c = if a > 0.0 { 1.0 / u } else { -1.0 / u };
            let s = t * c;
            (c, s)
        }
    }
}

/// Multiplies q and z matrices to get the eigenvectors of the original matrix.
fn multiply_q(q: &[Vec<f64>], z: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = q.len();
    let mut result = vec![vec![0.0; n]; n];

    // Use parallelism for large matrices
    result
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..n {
                row[j] = (0..n).map(|k| q[i][k] * z[k][j]).sum();
            }
        });

    result
}
