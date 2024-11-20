// src/dsbevd.rs

use rayon::prelude::*;
use rand_distr::{Normal, Distribution};
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
        const SAFMIN: f64 = 1e-300;
        const EPS: f64 = 1e-14;
        let rmin = SAFMIN.sqrt();
        let rmax = (1.0/SAFMIN).sqrt();

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
    let alpha = x[0];
    let sigma = x[1..].iter().map(|&xi| xi * xi).sum::<f64>();

    let mut tau = 0.0;
    if sigma == 0.0 {
        tau = 0.0;
    } else {
        let mut beta = 0.0;
        if sigma == 0.0 {
            beta = 0.0;
        } else {
            let s = (alpha * alpha + sigma).abs();
            if s <= 0.0 {
                beta = 0.0; 
            } else {
                let sqrt_s = s.sqrt();
                beta = if alpha >= 0.0 { -sqrt_s } else { sqrt_s };
            }
        }
        v[0] = alpha - beta;
        for i in 1..n {
            v[i] = x[i];
        }
        let v_norm = (beta * v[0]).abs().sqrt();
        for vi in v.iter_mut() {
            *vi /= v_norm;
        }
        tau = (beta - alpha) / beta;
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

    if n == 1 {
        return;
    }

    // Threshold for small subproblems
    let smlsiz = 25;

    // If the problem size is small, use QR algorithm
    if n <= smlsiz {
        tridiagonal_qr(d, e, z);
        return;
    }

    // Divide step
    let m = n / 2;

    let mut e_m_minus_1 = e[m - 1];
    e[m - 1] = 0.0;

    // Recurse on left and right subproblems
    let (d_left, d_right) = d.split_at_mut(m);
    let (e_left, e_right) = e.split_at_mut(m - 1);
    let (z_left, z_right) = z.split_at_mut(m);

    divide_and_conquer(d_left, e_left, z_left);
    divide_and_conquer(d_right, e_right, z_right);

    // Conquer step
    // Form rank-one modification
    let rho = e_m_minus_1;

    let k = m;

    // Build the secular equation components
    let n1 = m;
    let n2 = n - m;

    let d1 = d_left;
    let d2 = d_right;

    let z1 = z_left;
    let z2 = z_right;

    let mut z_vec = vec![0.0; n];
    let mut d_vec = vec![0.0; n];

    // Build z vector
    for i in 0..n1 {
        z_vec[i] = z1[i][n1 - 1];
    }
    for i in 0..n2 {
        z_vec[n1 + i] = z2[i][0];
    }

    // Compute updated eigenvalues and eigenvectors
    let mut new_d = vec![0.0; n];
    let mut new_z = vec![vec![0.0; n]; n];

    // Solve secular equation
    solve_secular_equation(
        &d1,
        &d2,
        &z_vec,
        rho,
        &mut new_d,
        &mut new_z,
    );

    // Copy new_d to d
    d.copy_from_slice(&new_d);

    // Create temporary matrix for results
    let mut temp_z = vec![vec![0.0; n]; n];

    // Left eigenvectors
    for i in 0..n1 {
        for j in 0..n {
            temp_z[i][j] = 0.0;
            for l in 0..n1 {
                temp_z[i][j] += z1[i][l] * new_z[l][j];
            }
        }
    }

    // Right eigenvectors
    for i in 0..n2 {
        for j in 0..n {
            temp_z[n1 + i][j] = 0.0;
            for l in 0..n2 {
                temp_z[n1 + i][j] += z2[i][l] * new_z[n1 + l][j];
            }
        }
    }

    // Copy results back to z
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
    z_out: &mut [Vec<f64>],
) {
    let n1 = d1.len();
    let n2 = d2.len();
    let n = n1 + n2;

    // Initialize variables
    let mut dlamda = vec![0.0; n];
    let mut q2 = vec![vec![0.0; n2]; n];

    // Copy d1 and d2 into dlamda
    for i in 0..n1 {
        dlamda[i] = d1[i];
    }
    for i in 0..n2 {
        dlamda[n1 + i] = d2[i];
    }

    // Compute the eigenvalues and eigenvectors
    // For each eigenvalue, solve the secular equation
    for i in 0..n {
        let lambda = dlamda[i];

        // Compute the secular equation denominator
        let mut denom = rho;
        for j in 0..n {
            if j != i {
                denom += z[j] * z[j] / (dlamda[j] - lambda);
            }
        }

        // Compute the eigenvalue
        d[i] = lambda + rho * z[i] * z[i] / denom;

        // Compute the eigenvector components
        for j in 0..n {
            if j != i {
                z_out[j][i] = z[j] / (dlamda[j] - d[i]);
            } else {
                z_out[j][i] = 1.0;
            }
        }

        // Normalize the eigenvector
        let norm = z_out.iter().map(|row| row[i] * row[i]).sum::<f64>().sqrt();
        for row in z_out.iter_mut() {
            row[i] /= norm;
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
                if scale == 0.0 {
                    if e[i].abs() > 0.0 {
                        done = false;
                    }
                } else if e[i].abs() > 1e-14 * scale {
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
    if b == 0.0 {
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
