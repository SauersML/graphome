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
        let (eigenvals, eigenvecs) = tridiagonal_eigen_dc(&d, &e);
    
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
        let mut value: f64 = 0.0;
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

    let mut scale: f64 = 0.0;
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
    let safmin: f64 = f64::MIN_POSITIVE;
    let n = d1.len() + d2.len();
    let eps = f64::EPSILON;
    
    // Sort d values and permute z 
    let mut perm: Vec<usize> = (0..n).collect();
    let mut dlamda = vec![0.0; n];
    for i in 0..d1.len() {
        dlamda[i] = d1[i];
    }
    for i in 0..d2.len() {
        dlamda[d1.len() + i] = d2[i];
    }
    perm.sort_by(|&i, &j| dlamda[i].partial_cmp(&dlamda[j]).unwrap());

    // Apply permutation and compute gaps
    let mut z_perm = vec![0.0; n];
    let mut dlamda_sorted = vec![0.0; n];
    let mut delta = vec![0.0; n];
    
    for i in 0..n {
        z_perm[i] = z[perm[i]];
        dlamda_sorted[i] = dlamda[perm[i]];
        if i > 0 {
            delta[i-1] = dlamda_sorted[i] - dlamda_sorted[i-1];
        }
    }
    
    // Normalize z vector to avoid overflow
    let zmax = z_perm.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    let z_scale = if zmax > safmin { zmax } else { 1.0 };
    
    for j in 0..n {
        let mut lambda = dlamda_sorted[j];
        
        // Initial bounds based on gaps
        let mut left = if j == 0 {
            lambda - delta[0].abs()
        } else {
            lambda - delta[j-1].abs() * 0.5
        };
        
        let mut right = if j == n-1 {
            lambda + delta[n-2].abs()
        } else {
            lambda + delta[j].abs() * 0.5
        };

        // Fast Newton iterations with bisection fallback
        for iter in 0..8 {  // DLAED4 uses max 8 iterations for most cases
            let mut f = rho * (z_perm[j] / z_scale).powi(2);
            let mut df = 0.0;
            
            for i in 0..n {
                if i != j {
                    let del = lambda - dlamda_sorted[i];
                    if del.abs() < eps * lambda.abs() {
                        continue;
                    }
                    let temp = z_perm[i] / (z_scale * del);
                    f += z_perm[i] * temp;
                    df += temp * temp;
                }
            }
            
            // Newton update with bounds
            if df == 0.0 {
                break;
            }
            let delta = f / df;
            let new_lambda = lambda - delta;
            
            if new_lambda <= left || new_lambda >= right {
                // Bisect if Newton step outside bounds
                lambda = (left + right) * 0.5;
                if f > 0.0 {
                    right = lambda;
                } else {
                    left = lambda;
                }
            } else {
                lambda = new_lambda;
                if f > 0.0 {
                    right = lambda;
                } else {
                    left = lambda;
                }
                if delta.abs() <= eps * lambda.abs() {
                    break;
                }
            }
        }
        
        d[j] = lambda;
        
        // Compute eigenvector with scaled computation
        let mut norm = 0.0;
        for i in 0..n {
            if i != j {
                let temp = z_perm[i] / (dlamda_sorted[i] - lambda);
                z_out[i][j] = temp;
                norm += temp * temp;
            } else {
                z_out[i][j] = 1.0;
                norm += 1.0;
            }
        }
        
        // Normalize
        norm = norm.sqrt();
        for i in 0..n {
            z_out[i][j] /= norm;
        }
    }
}

/// Tridiagonal QR algorithm for small matrices.
/// `d`: Diagonal elements
/// `e`: Off-diagonal elements
/// `z`: Eigenvectors (output)
fn tridiagonal_qr(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>]) {
    let n = d.len();
    let eps = f64::EPSILON;
    let eps2 = eps * eps;
    let safmin = f64::MIN_POSITIVE;
    let safmax = 1.0 / safmin;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = safmin.sqrt() / eps2;

    // Initialize z to identity if needed
    for i in 0..n {
        for j in 0..n {
            z[i][j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    let mut nmaxit = n * 30;  // Same as DSTEQR's maxit=30
    let mut jtot = 0;

    'outer: while jtot < nmaxit {
        // Look for small off-diagonal elements
        for m in 1..n {
            let test = e[m-1].abs();
            if test <= (d[m-1].abs().sqrt() * d[m].abs().sqrt()) * eps {
                e[m-1] = 0.0;
                continue 'outer;
            }
        }

        // Choose shift
        let m = n - 1;
        if m == 0 {
            break;
        }

        let g = (d[m] - d[m-1]) / (2.0 * e[m-1]);
        let r = g.hypot(1.0);
        let mut g = d[m] - d[m-1] + e[m-1]/(g + r.copysign(g));

        let mut s = 1.0;
        let mut c = 1.0;
        let mut p = 0.0;

        for i in (0..m).rev() {
            let f = s * e[i];
            let b = c * e[i];
            let (cs, sn) = givens_rotation(g, f);
            c = cs;
            s = sn;

            if i < m - 1 {
                e[i+1] = r;
            }
            g = d[i+1] - p;
            let r = (d[i] - g) * s + 2.0 * c * b;
            p = s * r;
            d[i+1] = g + p;
            g = c * r - b;

            // Accumulate transformation for eigenvectors
            for k in 0..n {
                let temp = z[k][i+1];
                z[k][i+1] = s * z[k][i] + c * temp;
                z[k][i] = c * z[k][i] - s * temp;
            }
        }

        d[0] -= p;
        e[m-1] = g;
        jtot += 1;
    }

    // Sort eigenvalues and eigenvectors
    for i in 0..n-1 {
        let mut k = i;
        let mut p = d[i];
        for j in i+1..n {
            if d[j] < p {
                k = j;
                p = d[j];
            }
        }
        if k != i {
            d.swap(k, i);
            for j in 0..n {
                z[j].swap(k, i);
            }
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
