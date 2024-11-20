// src/dsbevd.rs

use rayon::prelude::*;
use rand_distr::Distribution;
use std::cmp::min;

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
        // Get machine constants
        let safmin = f64::MIN_POSITIVE;
        let eps = f64::EPSILON;
        let rmin = safmin.sqrt();
        let rmax = (1.0/safmin).sqrt();

        // Matrix norm and scaling
        let anrm = self.matrix_norm();
        let scale = if anrm > 0.0 && anrm < rmin {
            rmin / anrm
        } else if anrm > rmax {
            rmax / anrm
        } else {
            1.0
        };

        // Create scaled copy
        let working_matrix = if scale != 1.0 {
            self.scaled_copy(scale)
        } else {
            self.clone()
        };

        // Step 1: Reduce to tridiagonal form
        let (d, e, q) = working_matrix.reduce_to_tridiagonal();

        // Step 2: Compute eigenvalues and eigenvectors of the tridiagonal matrix using divide and conquer
        let (mut eigenvalues, eigenvectors_tridiag) = tridiagonal_eigen_dc(&d, &e);

        // Step 3: Transform eigenvectors back to those of the original matrix
        let eigenvectors = multiply_q(&q, &eigenvectors_tridiag);

        // Rescale eigenvalues
        if scale != 1.0 {
            for eigenval in eigenvalues.iter_mut() {
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

        // Initialize q as the identity matrix
        let mut q = vec![vec![0.0; n]; n];
        for i in 0..n {
            q[i][i] = 1.0;
        }

        // Diagonal and off-diagonal elements
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n - 1];

        // Process lower triangle (assuming 'L' storage)
        for i in 0..n {
            let m = min(kd, n - i - 1);
            if m > 0 {
                // Form the vector x consisting of the diagonal and subdiagonals in column i
                let mut x = Vec::with_capacity(m + 1);
                for j in 0..=m {
                    x.push(ab[j][i]);
                }

                // Compute Householder reflector
                let (v, tau) = householder_reflector(&x);

                // Apply reflector to A
                for j in i..n {
                    let mut sum = 0.0;
                    for k in 0..=m {
                        if i + k < n && j < n {
                            sum += ab[k][i + k] * v[k];
                        }
                    }
                    sum *= tau;
                    for k in 0..=m {
                        if i + k < n && j < n {
                            ab[k][i + k] -= sum * v[k];
                        }
                    }
                }

                // Apply reflector to Q
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..=m {
                        if i + k < n {
                            sum += q[j][i + k] * v[k];
                        }
                    }
                    sum *= tau;
                    for k in 0..=m {
                        if i + k < n {
                            q[j][i + k] -= sum * v[k];
                        }
                    }
                }

                // Store the subdiagonal element
                if i + 1 < n {
                    e[i] = ab[1][i];
                }
                // Update the diagonal element
                d[i] = ab[0][i];
                // Zero out the elements below the subdiagonal
                for k in 1..=m {
                    ab[k][i] = 0.0;
                }
            } else {
                // No reflector needed; just copy the diagonal and subdiagonal elements
                d[i] = ab[0][i];
                if i + 1 < n {
                    e[i] = ab[1][i];
                }
            }
        }

        (d, e, q)
    }
    fn matrix_norm(&self) -> f64 {
        let mut max: f64 = 0.0;
        for row in &self.ab {
            for &val in row {
                max = max.max(val.abs());
            }
        }
        max
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
    let mut v = x.to_vec();
    
    // Machine constants
    let safmin = f64::MIN_POSITIVE;
    let eps = f64::EPSILON;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    
    let alpha = x[0];
    let mut sigma = 0.0;
    for xi in x.iter().skip(1) {
        sigma += xi * xi;
    }
    
    let mut tau = 0.0;
    let mut beta = 0.0;
    
    if sigma == 0.0 {
        if alpha < 0.0 {
            tau = 2.0;
            v[0] = -alpha;
        }
    } else {
        let mu = (alpha * alpha + sigma).sqrt();
        if alpha <= 0.0 {
            v[0] = alpha - mu;
        } else {
            v[0] = -sigma / (alpha + mu);
        }
        
        tau = 2.0 * v[0] * v[0] / (sigma + v[0] * v[0]);
        let norm = (sigma + v[0] * v[0]).sqrt();
        if norm < smlnum {
            beta = 0.0;
        } else {
            beta = 2.0 / (norm * norm);
        }

        // Scale if needed
        if beta > bignum {
            let scale = (bignum / beta).sqrt();
            for vi in v.iter_mut() {
                *vi *= scale;
            }
            beta *= scale * scale;
        }
        
        for vi in v.iter_mut() {
            *vi *= beta;
        }
    }
    
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
        let mut left = dlamda[i];
        let mut right = if i < n-1 { 
            dlamda[i] + delta[i]
        } else {
            dlamda[i] * (1.0 + 4.0 * eps)
        };
        
        // Binary search refinement 
        for _ in 0..50 {  // Max iterations
            let mid = (left + right) / 2.0;
            let mut sum = 0.0;
            let mut deriv = 0.0;
            
            for j in 0..n {
                if j != i {
                    let temp = z_scaled[j] / (dlamda[j] - mid);
                    sum += z_scaled[j] * temp;
                    deriv += temp * temp;
                }
            }
            
            if sum.abs() <= eps {
                d[i] = mid;
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

    // Initialize eigenvector matrix as identity
    for i in 0..n {
        for j in 0..n {
            z[i][j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    // Implement the QR algorithm for tridiagonal matrices
    const MAX_ITER: usize = 100;
    for m in (0..n).rev() {
        let mut iter = 0;
        loop {
            let mut done = true;
            for i in 0..m {
                let scale = d[i].abs() + d[i + 1].abs();
                let eps = f64::EPSILON;
                if scale == 0.0 {
                    if e[i].abs() > eps {
                        done = false;
                    }
                } else if e[i].abs() > eps * scale {
                    done = false;
                }
            }
            if done {
                break;
            }
            if iter >= MAX_ITER {
                panic!("QR algorithm failed to converge");
            }
            iter += 1;

            // Perform implicit QR step
            let dm = d[m];
            let em1 = e[m-1];
            let dm1 = d[m-1];
            let diff = (dm1 - dm) / 2.0;
            let sign = if diff >= 0.0 { 1.0 } else { -1.0 };
            let shift = dm - em1 * em1 / 
                (diff + sign * (diff * diff + em1 * em1).sqrt());
            let mut x = d[0] - shift;
            let mut zeta = e[0];

            for k in 0..m {
                let (c, s) = givens_rotation(x, zeta);
                let temp = c * d[k] - s * e[k];
                e[k] = s * d[k + 1];
                d[k + 1] = c * d[k + 1];
                d[k] = temp;

                // Apply rotation to z
                for i in 0..n {
                    let temp = c * z[i][k] - s * z[i][k + 1];
                    z[i][k + 1] = s * z[i][k] + c * z[i][k + 1];
                    z[i][k] = temp;
                }

                if k < m - 1 {
                    x = e[k];
                    zeta = s * e[k + 1];
                    e[k + 1] = c * e[k + 1];
                }
            }
            d[m] += shift;
        }
    }
}

/// Computes the Givens rotation coefficients c and s such that
/// [c -s; s c]^T * [a; b] = [r; 0]
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    let eps = f64::EPSILON;
    if b.abs() < eps * a.abs() {
        (1.0, 0.0)
    } else {
        let scale = a.abs().max(b.abs());
        let r;
        let c;
        let s;
        if scale == 0.0 {
            c = 1.0;
            s = 0.0;
        } else {
            let scaled_a = a / scale;
            let scaled_b = b / scale;
            r = scale * (scaled_a * scaled_a + scaled_b * scaled_b).sqrt();
            c = a / r;
            s = b / r;
        }
        (c, s)
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
