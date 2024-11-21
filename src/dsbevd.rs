// src/dsbevd.rs

// We ONLY care about large matrices, and we ALWAYS want both eigenvectors and eigenvalues

use rayon::prelude::*;
use std::cmp::{max, min};

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
        let (eigenvals, eigenvecs) = dstedc(&d, &e);
    
        // Transform eigenvectors back
        let eigenvectors = dgemm(&q, &eigenvecs);
    
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

    fn reduce_to_tridiagonal(&self) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let n = self.n;
        let kd = self.kd;
        let mut ab = self.ab.clone();
        
        // Output arrays
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n.saturating_sub(1)];
        let mut q = vec![vec![0.0; n]; n];
        
        // Initialize Q to identity
        for i in 0..n {
            q[i][i] = 1.0;
        }
        
        if kd == 0 {
            for i in 0..n {
                d[i] = ab[0][i];
            }
            return (d, e, q);
        }
        
        let kd1 = kd + 1;
        let kdm1 = kd - 1;
        let inca = kd1;
        
        let mut nr = 0;
        let mut j1 = kd;
        let mut j2 = 1;
    
        // Declare the work array
        let mut work = vec![0.0; n];
    
        // Main reduction loop matching LAPACK's structure
        for i in 0..(n - 2) {
            for k in (2..=kd1).rev() {
                j1 += kd;
                j2 += kd;
    
                if nr > 0 {
                    // Work arrays for current k iteration
                    let mut x_temp = vec![0.0; nr];
                    let mut y_temp = vec![0.0; nr];
    
                    // Generate plane rotations
                    for idx in 0..nr {
                        let j = j1 - kd - 1 + idx * kd;
                        if j < ab[kd].len() {
                            x_temp[idx] = ab[kd][j];
                            if kd >= 1 {
                                y_temp[idx] = ab[kd - 1][j];
                            }
                        }
                    }
    
                    dlargv(
                        nr,
                        &mut x_temp,
                        1,
                        &mut y_temp,
                        1,
                        &mut work[..nr],
                        1,
                    );
    
                    // Apply rotations based on number of diagonals
                    if nr > 2 * kd - 1 {
                        for l in 1..=kdm1 {
                            let mut v1 = vec![];
                            let mut v2 = vec![];
    
                            for idx in 0..nr {
                                let j = j1 - kd + l + idx * kd;
                                if j < ab[0].len() && j + 1 < ab[0].len() && (kd - l) < ab.len() && (kd - l + 1) < ab.len() {
                                    v1.push(ab[kd - l][j]);
                                    v2.push(ab[kd - l + 1][j]);
                                }
                            }
    
                            if !v1.is_empty() {
                                let len = v1.len();
                                dlartv(
                                    len,
                                    &mut v1,
                                    inca,
                                    &mut v2,
                                    inca,
                                    &work[..len],
                                    &y_temp[..len],
                                    1,
                                );
                                
                                for (idx, (val1, val2)) in v1.iter().zip(v2.iter()).enumerate() {
                                    let j = j1 - kd + l + idx * kd;
                                    if j < ab[0].len() && (kd - l) < ab.len() && (kd - l + 1) < ab.len() {
                                        ab[kd - l][j] = *val1;
                                        ab[kd - l + 1][j] = *val2;
                                    }
                                }
                            }
                        }
                    } else {
                        let jend = j1 + kd1 * (nr - 1);
                        for jinc in (j1..=jend).step_by(kd1) {
                            if jinc >= kd {
                                let mut row1 = vec![];
                                let mut row2 = vec![];
    
                                for idx in 0..kdm1 {
                                    if jinc - kd + idx < ab[0].len() && (kd) < ab.len() && (kd1) < ab.len() {
                                        row1.push(ab[kd][jinc - kd + idx]);
                                        row2.push(ab[kd1][jinc - kd + idx]);
                                    }
                                }
    
                                if !row1.is_empty() {
                                    let jidx = (jinc - j1) / kd1;
                                    if jidx < work.len() && jidx < y_temp.len() {
                                        drot(
                                            &mut row1,
                                            &mut row2,
                                            work[jidx],
                                            y_temp[jidx],
                                        );
    
                                        for (idx, (val1, val2)) in row1.iter().zip(row2.iter()).enumerate() {
                                            if jinc - kd + idx < ab[0].len() && (kd) < ab.len() && (kd1) < ab.len() {
                                                ab[kd][jinc - kd + idx] = *val1;
                                                ab[kd1][jinc - kd + idx] = *val2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
    
                    // Update Q matrix safely
                    for j in j1..=j2 {
                        if j + kd < n {
                            let mut block = Vec::with_capacity(3);
                            for jj in j..=j + 2 {
                                if jj < ab[0].len() {
                                    block.push(ab[kd][jj]);
                                }
                            }
    
                            if block.len() >= 3 {
                                let mut x = vec![block[0]];
                                let mut y = vec![block[1]];
                                let mut z = vec![block[2]];
    
                                let j_idx = (j - j1) / kd1;
                                if j_idx + 1 <= work.len() && j_idx + 1 <= y_temp.len() {
                                    dlar2v(
                                        1,
                                        &mut x,
                                        &mut y,
                                        &mut z,
                                        1,
                                        &work[j_idx..j_idx + 1],
                                        &y_temp[j_idx..j_idx + 1],
                                        1,
                                    );
    
                                    ab[kd][j] = x[0];
                                    ab[kd][j + 1] = y[0];
                                    ab[kd][j + 2] = z[0];
                                }
                            }
                        }
    
                        // Update Q matrix
                        if j < n - 1 {
                            let j_idx = (j - j1) / kd1;
                            if j_idx < work.len() && j_idx < y_temp.len() {
                                for k in 0..n {
                                    let temp = work[j_idx] * q[k][j] + y_temp[j_idx] * q[k][j + 1];
                                    q[k][j + 1] = -y_temp[j_idx] * q[k][j] + work[j_idx] * q[k][j + 1];
                                    q[k][j] = temp;
                                }
                            }
                        }
                    }
                }
    
                // Handle inner elements of band for current k
                if k > 2 && k <= n - i {
                    // Safe indexing
                    if kd - 2 < ab.len() && i < ab[kd - 2].len() {
                        let f = ab[kd - 2][i];
                        let g = ab[kd - 1][i];
                        let (cs, sn) = givens_rotation(f, g);
                        ab[kd - 2][i] = cs * f + sn * g;
    
                        // Apply from the left
                        let start = i + 1;
                        let end = (i + k - 1).min(n - 1);
                        if start <= end {
                            for j in start..=end {
                                if kd - 2 < ab.len() && kd - 1 < ab.len() && j < ab[kd - 2].len() && j < ab[kd - 1].len() {
                                    let temp = cs * ab[kd - 2][j] + sn * ab[kd - 1][j];
                                    ab[kd - 1][j] = -sn * ab[kd - 2][j] + cs * ab[kd - 1][j];
                                    ab[kd - 2][j] = temp;
                                }
                            }
                        }
                    }
                    nr += 1;
                    j1 = j1.saturating_sub(kd + 1);
                }
    
                // Adjust bounds
                if j2 + kd > n {
                    nr = nr.saturating_sub(1);
                    j2 = j2.saturating_sub(kd + 1);
                }
            }
        }
    
        // Copy final results
        for i in 0..n {
            d[i] = ab[0][i];
            if i < n - 1 && i + 1 < ab[1].len() {
                e[i] = ab[1][i + 1];
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

/// Generates a vector of plane rotations for 2-by-2 matrices.
/// 
/// For i = 1,2,...,n:
///    [  c(i)  s(i) ] [ x(i) ] = [ a(i) ]
///    [ -s(i)  c(i) ] [ y(i) ] = [  0  ]
/// 
/// Returns updated x values (now containing a(i)) and c,s rotation values
fn dlargv(
    n: usize,
    x: &mut [f64],
    incx: usize,
    y: &mut [f64], 
    incy: usize,
    c: &mut [f64],
    incc: usize,
) {
    debug_assert!(incx > 0, "incx must be positive");
    debug_assert!(incy > 0, "incy must be positive"); 
    debug_assert!(incc > 0, "incc must be positive");
    debug_assert!(x.len() >= 1 + (n-1)*incx, "x array too small");
    debug_assert!(y.len() >= 1 + (n-1)*incy, "y array too small");
    debug_assert!(c.len() >= 1 + (n-1)*incc, "c array too small");

    let mut ix = 0;
    let mut iy = 0;
    let mut ic = 0;

    for _ in 0..n {
        let f = x[ix];
        let g = y[iy];

        if g == 0.0 {
            // Special case: g = 0 implies rotation by 0 degrees
            c[ic] = 1.0;
            y[iy] = 0.0;
        } else if f == 0.0 {
            // Special case: f = 0 implies rotation by 90 degrees
            c[ic] = 0.0;
            y[iy] = 1.0;
            x[ix] = g;
        } else if f.abs() > g.abs() {
            // Case |f| > |g|: use f to calculate stable rotation
            let t = g / f;
            let tt = (1.0 + t * t).sqrt();
            c[ic] = 1.0 / tt;
            y[iy] = t * c[ic];
            x[ix] = f * tt;
        } else {
            // Case |g| >= |f|: use g to calculate stable rotation  
            let t = f / g;
            let tt = (1.0 + t * t).sqrt();
            y[iy] = 1.0 / tt;
            c[ic] = t * y[iy];
            x[ix] = g * tt;
        }

        ix += incx;
        iy += incy; 
        ic += incc;
    }
}

fn dlartv(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize, c: &[f64], s: &[f64], incc: usize) {
    let mut ix = 0;
    let mut iy = 0;
    let mut ic = 0;

    for _ in 0..n {
        let xi = x[ix];
        let yi = y[iy];
        let ci = c[ic];
        let si = s[ic];

        x[ix] = ci * xi + si * yi;
        y[iy] = ci * yi - si * xi;

        ix += incx;
        iy += incy;
        ic += incc;
    }
}

/// Apply plane rotation
fn drot(dx: &mut [f64], dy: &mut [f64], c: f64, s: f64) {
    assert_eq!(dx.len(), dy.len(), "Vector lengths must match");
    for i in 0..dx.len() {
        let temp = c * dx[i] + s * dy[i];
        dy[i] = -s * dx[i] + c * dy[i];
        dx[i] = temp;
    }
}


fn dlar2v(
    n: usize,
    x: &mut [f64],
    y: &mut [f64],
    z: &mut [f64],
    incx: usize,
    c: &[f64],
    s: &[f64],
    incc: usize,
) {
    debug_assert!(incx > 0, "incx must be positive");
    debug_assert!(incc > 0, "incc must be positive");
    debug_assert!(x.len() >= 1 + (n-1)*incx, "x array too small");
    debug_assert!(y.len() >= 1 + (n-1)*incx, "y array too small");
    debug_assert!(z.len() >= 1 + (n-1)*incx, "z array too small");
    debug_assert!(c.len() >= 1 + (n-1)*incc, "c array too small");
    debug_assert!(s.len() >= 1 + (n-1)*incc, "s array too small");

    let mut ix = 0;
    let mut ic = 0;

    for _ in 0..n {
        let xi = x[ix];
        let yi = y[ix];
        let zi = z[ix];
        let ci = c[ic];
        let si = s[ic];

        let t1 = si * zi;
        let t2 = ci * zi;
        let t3 = t2 - si * xi;
        let t4 = t2 + si * yi;
        let t5 = ci * xi + t1;
        let t6 = ci * yi - t1;

        x[ix] = ci * t5 + si * t4;
        y[ix] = ci * t6 - si * t3;
        z[ix] = ci * t4 - si * t5;

        ix += incx;
        ic += incc;
    }
}

/// Computes the eigenvalues and eigenvectors of a symmetric tridiagonal matrix using the divide and conquer algorithm.
/// `d`: diagonal elements
/// `e`: off-diagonal elements
/// Returns the eigenvalues and the eigenvectors.
fn dstedc(d: &[f64], e: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
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
/// Modifies `d`, `e`, and `z` in place to compute eigenvalues and eigenvectors.
/// This function corresponds to LAPACK's DSTEDC subroutine.
fn divide_and_conquer(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>]) {
    let n = d.len();
    if n == 1 {
        // Trivial case: the eigenvalue is d[0], and the eigenvector is [1]
        z[0][0] = 1.0;
        return;
    }

    // Parameters
    let smlsiz = 25; // Minimum size to use divide and conquer. We don't care about small input matrices but we need this for a base case.
    let eps = f64::EPSILON;

    // We always compute eigenvectors

    if n <= smlsiz {
        // Use QR algorithm for small matrices
        dsteqr(d, e, z);
        return;
    }

    // Scale the matrix if necessary
    let orgnrm = d.iter().map(|&x| x.abs()).chain(e.iter().map(|&x| x.abs())).fold(0.0_f64, f64::max);
    if orgnrm == 0.0 {
        // The matrix is zero; all eigenvalues are zero
        return;
    }

    // Compute splitting points
    let mut submat_start = 0;
    while submat_start < n {
        // Find the end of the current submatrix
        let mut submat_end = submat_start;
        while submat_end < n - 1 {
            let tiny = eps * (d[submat_end].abs().sqrt() * d[submat_end + 1].abs().sqrt());
            if e[submat_end].abs() <= tiny {
                e[submat_end] = 0.0; // Deflate
                break;
            }
            submat_end += 1;
        }

        let m = submat_end - submat_start + 1;

        if m <= smlsiz {
            // Use QR algorithm for small submatrices
            let mut d_sub = d[submat_start..=submat_end].to_vec();
            let mut e_sub = e[submat_start..submat_end].to_vec();
            let mut z_sub = vec![vec![0.0; m]; m];
            for i in 0..m {
                z_sub[i][i] = 1.0;
            }
            dsteqr(&mut d_sub, &mut e_sub, &mut z_sub);

            // Copy back results
            for i in 0..m {
                d[submat_start + i] = d_sub[i];
                for j in 0..m {
                    z[submat_start + i][submat_start + j] = z_sub[i][j];
                }
            }
        } else {
            // Recursive divide and conquer
            // Divide the matrix into two submatrices
            let mid = submat_start + m / 2 - 1;
            let rho = e[mid];
            e[mid] = 0.0; // Split the matrix

            // Left subproblem
            let left_size = mid - submat_start + 1;
            let mut d_left = d[submat_start..=mid].to_vec();
            let mut e_left = e[submat_start..mid].to_vec();
            let mut z_left = vec![vec![0.0; left_size]; left_size];
            for i in 0..left_size {
                z_left[i][i] = 1.0;
            }
            divide_and_conquer(&mut d_left, &mut e_left, &mut z_left);

            // Right subproblem
            let right_size = submat_end - mid;
            let mut d_right = d[(mid + 1)..=submat_end].to_vec();
            let mut e_right = e[(mid + 1)..submat_end].to_vec();
            let mut z_right = vec![vec![0.0; right_size]; right_size];
            for i in 0..right_size {
                z_right[i][i] = 1.0;
            }
            divide_and_conquer(&mut d_right, &mut e_right, &mut z_right);

            // Merge the two subproblems
            let mut d_merged = vec![0.0; m];
            let mut z_merged = vec![vec![0.0; m]; m];

            // Copy eigenvalues
            for i in 0..left_size {
                d_merged[i] = d_left[i];
            }
            for i in 0..right_size {
                d_merged[left_size + i] = d_right[i];
            }

            // Form the z vector for the rank-one update
            let mut z_vector = vec![0.0; m];
            for i in 0..left_size {
                z_vector[i] = z_left[i][left_size - 1];
            }
            for i in 0..right_size {
                z_vector[left_size + i] = z_right[i][0];
            }

            // Initialize z_out for dlaed4
            let mut z_out = vec![vec![0.0; m]; m];

            // Solve the secular equation
            let info = dlaed4(
                &d_left,
                &d_right,
                &z_vector,
                rho,
                &mut d_merged,
                &mut z_out,
            );
            if info != 0 {
                panic!("Error in dlaed4: info = {}", info);
            }

            // Copy eigenvalues back
            for i in 0..m {
                d[submat_start + i] = d_merged[i];
            }

            // Compute the updated eigenvectors
            // Multiply the eigenvectors of the left and right subproblems with z_out
            for i in 0..m {
                for j in 0..m {
                    let mut sum = 0.0;
                    if i < left_size {
                        for k in 0..left_size {
                            sum += z_left[i][k] * z_out[k][j];
                        }
                    } else {
                        for k in 0..right_size {
                            sum += z_right[i - left_size][k] * z_out[left_size + k][j];
                        }
                    }
                    z_merged[i][j] = sum;
                }
            }

            // Copy back eigenvectors
            for i in 0..m {
                for j in 0..m {
                    z[submat_start + i][submat_start + j] = z_merged[i][j];
                }
            }
        }

        submat_start = submat_end + 1;
    }

    // Sort the eigenvalues and eigenvectors
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap());

    let sorted_d = idx.iter().map(|&i| d[i]).collect::<Vec<f64>>();
    let sorted_z = idx
        .iter()
        .map(|&i| z.iter().map(|row| row[i]).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();

    // Copy back sorted eigenvalues and eigenvectors
    d.copy_from_slice(&sorted_d);
    for i in 0..n {
        for j in 0..n {
            z[i][j] = sorted_z[j][i];
        }
    }
}

/// Solves the secular equation in the divide and conquer algorithm.
/// `d1`, `d2`: Eigenvalues from the left and right subproblems.
/// `z`: Combined z vector from left and right subproblems.
/// `rho`: The rank-one update scalar.
/// `d`: Output eigenvalues.
/// `z_out`: Output eigenvectors.
fn dlaed4(
    d1: &[f64],
    d2: &[f64],
    z: &[f64],
    rho: f64,
    d: &mut [f64],
    z_out: &mut [Vec<f64>],
) -> i32 {
    let safmin: f64 = f64::MIN_POSITIVE;
    let n = d1.len() + d2.len();
    let eps = f64::EPSILON;

    // Combine d1 and d2 into dlamda
    let mut dlamda = Vec::with_capacity(n);
    dlamda.extend_from_slice(d1);
    dlamda.extend_from_slice(d2);

    // Combine z1 and z2 (assumed to be stored in z) into z_perm
    let mut z_perm = z.to_vec();

    // Sort dlamda and permute z accordingly
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| dlamda[i].partial_cmp(&dlamda[j]).unwrap());

    let mut dlamda_sorted = vec![0.0; n];
    let mut z_sorted = vec![0.0; n];
    for (new_i, &old_i) in idx.iter().enumerate() {
        dlamda_sorted[new_i] = dlamda[old_i];
        z_sorted[new_i] = z_perm[old_i];
    }

    // Compute gaps
    let mut delta = vec![0.0; n - 1];
    for i in 0..(n - 1) {
        delta[i] = dlamda_sorted[i + 1] - dlamda_sorted[i];
    }

    // Normalize z vector to avoid overflow
    let zmax = z_sorted.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    let z_scale = if zmax > safmin { zmax } else { 1.0 };

    // Initialize eigenvalues and eigenvectors
    for j in 0..n {
        let mut lambda = dlamda_sorted[j];

        // Initial bounds based on gaps
        let mut left = if j == 0 {
            lambda - delta.get(0).unwrap_or(&0.0).abs()
        } else {
            lambda - delta[j - 1].abs() * 0.5
        };

        let mut right = if j == n - 1 {
            lambda + delta.get(n - 2).unwrap_or(&0.0).abs()
        } else {
            lambda + delta[j].abs() * 0.5
        };

        // Fast Newton iterations with bisection fallback
        for _iter in 0..8 {
            let mut f = rho * (z_sorted[j] / z_scale).powi(2);
            let mut df = 0.0;

            for i in 0..n {
                if i != j {
                    let del = dlamda_sorted[i] - lambda;
                    if del.abs() < eps * lambda.abs() {
                        continue;
                    }
                    let temp = z_sorted[i] / (z_scale * (dlamda_sorted[i] - lambda));
                    f += temp * z_sorted[i] / z_scale;
                    df += temp * temp;
                }
            }

            // Newton update with bounds
            if df == 0.0 {
                break;
            }
            let delta_lambda = f / df;
            let new_lambda = lambda - delta_lambda;

            if new_lambda <= left || new_lambda >= right {
                // Bisect if Newton step outside bounds
                lambda = (left + right) * 0.5;
                if f > 0.0 {
                    left = lambda;
                } else {
                    right = lambda;
                }
            } else {
                lambda = new_lambda;
                if delta_lambda.abs() <= eps * lambda.abs() {
                    break;
                }
                if f > 0.0 {
                    left = lambda;
                } else {
                    right = lambda;
                }
            }
        }

        d[j] = lambda;

        // Compute eigenvector with scaled computation
        let mut norm = 0.0;
        for i in 0..n {
            let denom = dlamda_sorted[i] - lambda;
            let temp = if denom.abs() < eps {
                0.0
            } else {
                z_sorted[i] / denom
            };
            z_out[i][j] = temp;
            norm += temp * temp;
        }
        z_out[j][j] = 1.0;
        norm += 1.0;

        // Normalize
        norm = norm.sqrt();
        for i in 0..n {
            z_out[i][j] /= norm;
        }
    }

    0 // Return 0 to indicate success
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
fn dgemm(q: &[Vec<f64>], z: &[Vec<f64>]) -> Vec<Vec<f64>> {
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


/// Force a and b to be stored prior to addition
fn dlamc3(a: f64, b: f64) -> f64 {
    a + b
}

/// Compute the norm of a symmetric tridiagonal matrix
fn dlanst(norm_type: char, n: usize, d: &[f64], e: &[f64]) -> f64 {
    match norm_type {
        'M' | 'm' => {
            // Maximum absolute value
            let mut result = 0.0_f64;
            for val in d.iter() {
                result = result.max(val.abs());
            }
            for val in e.iter() {
                result = result.max(val.abs());
            }
            result
        },
        'O' | 'o' | '1' | 'I' | 'i' => {
            // One norm and Infinity norm are same for symmetric matrix
            if n == 0 {
                return 0.0_f64;
            }
            if n == 1 {
                return d[0].abs();
            }
            
            let mut work = vec![0.0_f64; n];
            // First row
            work[0] = d[0].abs() + e[0].abs();
            // Middle rows
            for i in 1..n-1 {
                work[i] = e[i-1].abs() + d[i].abs() + e[i].abs();
            }
            // Last row
            work[n-1] = e[n-2].abs() + d[n-1].abs();
            
            let mut max_val = 0.0_f64;
            for val in work.iter() {
                max_val = max_val.max(*val);
            }
            max_val
        },
        'F' | 'f' | 'E' | 'e' => {
            // Frobenius norm
            let mut scale = 0.0_f64;
            let mut sumsq = 1.0_f64;
            
            // Add diagonal elements
            dlassq(n, d, 1, &mut scale, &mut sumsq);
            
            // Add off-diagonal elements
            if n > 1 {
                dlassq(n-1, e, 1, &mut scale, &mut sumsq);
                sumsq *= 2.0_f64;
            }
            
            scale * sumsq.sqrt()
        },
        _ => panic!("Invalid norm type for dlanst")
    }
}

/// Compute parameters for a 2x2 eigenvalue problem
fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64, f64) {
    if b == 0.0 && c == 0.0 {
        // Matrix is diagonal
        if a >= c {
            (a, c, 1.0_f64, 0.0_f64, 0.0_f64)
        } else {
            (c, a, 0.0_f64, 1.0_f64, 0.0_f64)
        }
    } else {
        let sm = a + c;
        let df = a - c;
        let adf = df.abs();
        let tb = b + b;
        let ab = tb.abs();
        
        let (acmx, acmn) = if a.abs() > c.abs() {
            (a, c)
        } else {
            (c, a)
        };
        
        let (rt1, rt2, cs1, sn1) = if adf > ab {
            let rt = adf * (1.0_f64 + (ab/adf).powi(2)).sqrt();
            if df >= 0.0 {
                let rt1_val = sm + rt;
                let rt2_val = (acmx / rt1_val) * acmn - (b / rt1_val) * b;
                let cs = df + rt;
                let sn = tb;
                let norm = (cs * cs + sn * sn).sqrt();
                (rt1_val, rt2_val, cs/norm, sn/norm)
            } else {
                let rt1_val = (acmx / (sm - rt)) * acmn - (b / (sm - rt)) * b;
                let rt2_val = sm - rt;
                let cs = tb;
                let sn = df + rt;
                let norm = (cs * cs + sn * sn).sqrt();
                (rt1_val, rt2_val, cs/norm, sn/norm)
            }
        } else if ab == 0.0 {
            (sm, 0.0_f64, 1.0_f64, 0.0_f64)
        } else {
            let rt = ab * (1.0_f64 + (adf/ab).powi(2)).sqrt();
            if sm >= 0.0 {
                let rt1_val = 0.5_f64 * (sm + rt);
                let rt2_val = (acmx/rt1_val) * acmn - (b/rt1_val) * b;
                let sn = if tb >= 0.0 {
                    if ab >= 0.0 { ab } else { -ab }
                } else {
                    if ab >= 0.0 { -ab } else { ab }
                };
                let norm = (1.0_f64 + sn * sn).sqrt();
                (rt1_val, rt2_val, 1.0_f64/norm, sn/norm)
            } else {
                let rt2_val = 0.5_f64 * (sm - rt);
                let rt1_val = (acmx/rt2_val) * acmn - (b/rt2_val) * b;
                let sn = if tb >= 0.0 {
                    if ab >= 0.0 { ab } else { -ab }
                } else {
                    if ab >= 0.0 { -ab } else { ab }
                };
                let norm = (1.0_f64 + sn * sn).sqrt();
                (rt1_val, rt2_val, 1.0_f64/norm, sn/norm)
            }
        };
        
        (rt1, rt2, cs1, sn1, 0.0_f64)
    }
}

/// Safe computation of sqrt(x*x + y*y)
fn dlapy2(x: f64, y: f64) -> f64 {
    let x_abs = x.abs();
    let y_abs = y.abs();
    
    if x_abs > y_abs {
        let temp = y_abs / x_abs;
        x_abs * (1.0_f64 + temp * temp).sqrt()
    } else if y_abs > x_abs {
        let temp = x_abs / y_abs;
        y_abs * (1.0_f64 + temp * temp).sqrt()
    } else {
        x_abs * (2.0_f64).sqrt()
    }
}

/// Initialize a matrix with diagonal and off-diagonal values
fn dlaset(uplo: char, m: usize, n: usize, alpha: f64, beta: f64, a: &mut [Vec<f64>]) {
    match uplo {
        'U' | 'u' => {
            // Upper triangle
            for j in 0..n {
                for i in 0..j.min(m) {
                    a[i][j] = alpha;
                }
                if j < m {
                    a[j][j] = beta;
                }
            }
        },
        'L' | 'l' => {
            // Lower triangle
            for j in 0..n {
                for i in j+1..m {
                    a[i][j] = alpha;
                }
                if j < m {
                    a[j][j] = beta;
                }
            }
        },
        _ => {
            // Full matrix
            for i in 0..m {
                for j in 0..n {
                    a[i][j] = if i == j { beta } else { alpha };
                }
            }
        }
    }
}

// Machine parameters
const SAFE_MIN: f64 = f64::MIN_POSITIVE;
const EPSILON: f64 = f64::EPSILON;
const BASE: f64 = 2.0_f64;
const PRECISION: f64 = f64::EPSILON;
const RMAX: f64 = f64::MAX;
const RMIN: f64 = f64::MIN_POSITIVE;

/// Matrix-vector multiplication
/// Computes y := alpha*A*x + beta*y or y := alpha*A^T*x + beta*y
pub fn dgemv(trans: bool, m: usize, n: usize, alpha: f64, a: &[Vec<f64>], 
             x: &[f64], beta: f64, y: &mut [f64]) {
    if !trans {
        // y := alpha*A*x + beta*y
        for i in 0..m {
            let mut temp = 0.0;
            for j in 0..n {
                temp += alpha * a[i][j] * x[j];
            }
            y[i] = temp + beta * y[i];
        }
    } else {
        // y := alpha*A^T*x + beta*y
        for i in 0..n {
            y[i] *= beta;
        }
        for i in 0..m {
            let temp = alpha * x[i];
            for j in 0..n {
                y[j] += temp * a[i][j];
            }
        }
    }
}

/// Euclidean norm of a vector
pub fn dnrm2(n: usize, x: &[f64], incx: usize) -> f64 {
    if n < 1 || incx < 1 {
        return 0.0;
    }
    if n == 1 {
        return x[0].abs();
    }

    let mut scale = 0.0;
    let mut ssq = 1.0;
    
    // Computing the scaled sum of squares
    for i in (0..n*incx).step_by(incx) {
        if x[i] != 0.0 {
            let absxi = x[i].abs();
            if scale < absxi {
                let temp = scale / absxi;
                ssq = 1.0 + ssq * temp * temp;
                scale = absxi;
            } else {
                let temp = absxi / scale;
                ssq += temp * temp;
            }
        }
    }
    scale * ssq.sqrt()
}

/// Scale a vector by a constant
pub fn dscal(n: usize, alpha: f64, x: &mut [f64], incx: usize) {
    if n < 1 || incx < 1 {
        return;
    }
    for i in (0..n*incx).step_by(incx) {
        x[i] *= alpha;
    }
}

/// Find index of element with maximum absolute value
pub fn idamax(n: usize, x: &[f64], incx: usize) -> usize {
    if n < 1 || incx < 1 {
        return 0;
    }
    if n == 1 {
        return 0;
    }

    let mut max_idx = 0;
    let mut max_val = x[0].abs();

    for i in (incx..n*incx).step_by(incx) {
        let abs_val = x[i].abs();
        if abs_val > max_val {
            max_idx = i/incx;
            max_val = abs_val;
        }
    }
    max_idx
}

// Helper function to pass workspace arrays and handle error conditions safely
pub fn dsbtrd_wrapper(uplo: char, n: usize, kd: usize, ab: &mut [Vec<f64>]) 
    -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) 
{
    assert!(n > 0, "Matrix dimension must be positive");
    assert!(kd >= 0 && kd < n, "Band width must be valid");
    assert!(ab.len() >= kd + 1, "Invalid array dimension for ab");
    assert!(ab[0].len() >= n, "Invalid array dimension for ab");

    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n-1];
    let mut q = vec![vec![0.0; n]; n];

    dsbtrd(uplo, n, kd, ab, &mut d, &mut e, &mut q);

    (d, e, q)
}

// Helper function for safe mutable band access
fn get_mut_bands(ab: &mut [Vec<f64>], k1: usize, k2: usize, start: usize, len: usize) 
   -> (&mut [f64], &mut [f64]) 
{
   assert!(k1 != k2, "Cannot borrow same band twice");
   let (min_k, max_k) = if k1 < k2 { (k1, k2) } else { (k2, k1) };
   let (lower, upper) = ab.split_at_mut(max_k);
   if k1 < k2 {
       (&mut lower[k1][start..start+len], &mut upper[0][start..start+len])
   } else {
       (&mut upper[0][start..start+len], &mut lower[k2][start..start+len])
   }
}

fn dlartg(f: f64, g: f64) -> (f64, f64) {
   if g == 0.0 {
       (1.0, 0.0)
   } else if f == 0.0 {
       (0.0, 1.0)
   } else {
       let abs_f = f.abs();
       let abs_g = g.abs();
       let scale = abs_f.max(abs_g);
       let fs = f / scale;
       let gs = g / scale;
       let norm = (fs * fs + gs * gs).sqrt();
       let c = fs / norm;
       let s = gs / norm;
       (c, s)
   }
}

fn dsbtrd(uplo: char, n: usize, kd: usize, ab: &mut [Vec<f64>], d: &mut [f64], 
         e: &mut [f64], q: &mut [Vec<f64>]) {
   let kd1 = kd + 1;
   let kdm1 = kd - 1;
   let mut nr: usize = 0;
   let mut j1: usize = kd1;
   let mut j2: usize = 1;
   let mut work = vec![0.0; n];
   let mut rotations = Vec::new();

   // Initialize Q to identity
   dlaset('F', n, n, 0.0, 1.0, q);

   if uplo == 'U' {
       for j in 0..n-2 {
           for k in (0..kd-1).rev() {
               j1 = if j1 > kd1 { j1 - kd1 } else { 0 };
               j2 = if j2 > kd1 { j2 - kd1 } else { 0 };

               if nr > 0 {
                   let j_start = j1.saturating_sub(1);
                   let len = nr.min(ab[0].len() - j_start);

                   // Store rotations
                   rotations.clear();
                   for idx in 0..nr {
                       let j = j1 - kd - 1 + idx * kd;
                       if j < ab[kd].len() {
                           rotations.push((ab[kd][j], ab[kd-1][j]));
                       }
                   }

                   // Apply stored rotations
                   for (idx, &(x, y)) in rotations.iter().enumerate() {
                       let (cs, sn) = dlartg(x, y);
                       let start = j_start + idx;
                       if start + 1 < ab[0].len() {
                           let (band1, band2) = get_mut_bands(ab, k, k+1, start, 1);
                           band1[0] = cs * x + sn * y;
                           band2[0] = -sn * x + cs * y;
                       }
                   }

                   for l in 0..k {
                       let mut v1 = Vec::new();
                       let mut v2 = Vec::new();
                       for idx in 0..len {
                           let j = j_start + idx;
                           if j < ab[0].len() {
                               v1.push(ab[kd-l][j]);
                               v2.push(ab[kd-l+1][j]);
                           }
                       }
                       
                       for i in 0..v1.len() {
                           let (cs, sn) = dlartg(v1[i], v2[i]);
                           let j = j_start + i;
                           if j < ab[0].len() {
                               ab[kd-l][j] = cs * v1[i] + sn * v2[i];
                               ab[kd-l+1][j] = -sn * v1[i] + cs * v2[i];
                           }
                       }
                   }
               }

               if k < kdm1 {
                   let x = ab[kd-k][j+k];
                   let y = ab[kd-k-1][j+k+1];
                   let (cs, sn) = dlartg(x, y);
                   
                   // Apply rotation
                   let len = k.min(n - (j+k) - 1);
                   let mut temp_row1 = Vec::new();
                   let mut temp_row2 = Vec::new();
                   
                   for i in 0..len {
                       temp_row1.push(ab[kd-k][j+k+i]);
                       temp_row2.push(ab[kd-k-1][j+k+1+i]);
                   }
                   
                   for i in 0..len {
                       ab[kd-k][j+k+i] = cs * temp_row1[i] + sn * temp_row2[i];
                       ab[kd-k-1][j+k+1+i] = -sn * temp_row1[i] + cs * temp_row2[i];
                   }
                   
                   ab[kd-k-1][j+k+1] = 0.0;
               }

               nr += 1;
           }

           if j < n-1 {
               d[j] = ab[kd][j];
               e[j] = ab[kdm1][j+1];
           }
       }
       d[n-1] = ab[kd][n-1];
   } else {
       for j in 0..n-2 {
           for k in (0..kd-1).rev() {
               j1 = if j1 > kd1 { j1 - kd1 } else { 0 };
               j2 = if j2 > kd1 { j2 - kd1 } else { 0 };

               if nr > 0 {
                   let j_start = j1;
                   let len = nr.min(ab[0].len() - j_start);
                   
                   let mut temp_bands = Vec::new();
                   for l in 0..=k {
                       let mut band = Vec::new();
                       for idx in 0..len {
                           band.push(ab[l][j_start+idx]);
                       }
                       temp_bands.push(band);
                   }

                   for l in 0..k {
                       for i in 0..len {
                           let (cs, sn) = dlartg(temp_bands[l][i], temp_bands[l+1][i]);
                           if j_start + i < ab[0].len() {
                               ab[l][j_start+i] = cs * temp_bands[l][i] + sn * temp_bands[l+1][i];
                               ab[l+1][j_start+i] = -sn * temp_bands[l][i] + cs * temp_bands[l+1][i];
                           }
                       }
                   }
               }

               if k < kdm1 {
                   let (cs, sn) = dlartg(ab[k+1][j], ab[k+2][j]);
                   
                   let len = k.min(n - j - 1);
                   let mut temp1 = Vec::new();
                   let mut temp2 = Vec::new();
                   
                   for i in 0..len {
                       temp1.push(ab[k+2][j+i]);
                       temp2.push(ab[k+1][j+1+i]);
                   }
                   
                   for i in 0..len {
                       ab[k+2][j+i] = cs * temp1[i] + sn * temp2[i];
                       ab[k+1][j+1+i] = -sn * temp1[i] + cs * temp2[i];
                   }
                   
                   ab[k+2][j] = 0.0;
               }

               nr += 1;
           }

           if j < n-1 {
               d[j] = ab[0][j];
               e[j] = ab[1][j];
           }
       }
       d[n-1] = ab[0][n-1];
   }
}

fn dsteqr(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>]) {
   let n = d.len();
   let mut e_ext = vec![0.0; n];
   e_ext[..n-1].copy_from_slice(e);

   for l in 0..n {
       let mut iter = 0;
       loop {
           let mut m = l;
           while m < n - 1 {
               let dd = d[m].abs() + d[m + 1].abs();
               if e_ext[m].abs() <= f64::EPSILON * dd {
                   break;
               }
               m += 1;
           }

           if m == l {
               break;
           }

           iter += 1;
           if iter > 1000 {
               panic!("Too many iterations in dsteqr");
           }

           let delta = (d[m - 1] - d[m]).abs() / 2.0;
           let mu = d[m] - (e_ext[m - 1].powi(2)) / (delta + (delta.powi(2) + e_ext[m - 1].powi(2)).sqrt());

           for i in l..n {
               d[i] -= mu;
           }

           for i in l..m {
               let (cs, sn) = dlartg(d[i], e_ext[i]);
               
               let temp = cs * d[i] - sn * e_ext[i];
               e_ext[i] = sn * d[i] + cs * e_ext[i];
               d[i] = temp;

               if i + 1 < n {
                   for k in 0..n {
                       let temp = cs * z[k][i] - sn * z[k][i + 1];
                       z[k][i + 1] = sn * z[k][i] + cs * z[k][i + 1];
                       z[k][i] = temp;
                   }
               }
           }
       }
   }

   let mut idx: Vec<usize> = (0..n).collect();
   idx.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap());

   let sorted_d = idx.iter().map(|&i| d[i]).collect::<Vec<f64>>();
   let sorted_z = idx.iter()
       .map(|&i| z.iter().map(|row| row[i]).collect::<Vec<f64>>())
       .collect::<Vec<Vec<f64>>>();

   d.copy_from_slice(&sorted_d);
   for i in 0..n {
       for j in 0..n {
           z[i][j] = sorted_z[j][i];
       }
   }
}


/// Scales a matrix by cto/cfrom without over/underflow.
/// Translated from LAPACK's DLASCL for the general matrix case (type 'G').
fn dlascl(a: &mut [Vec<f64>], cfrom: f64, cto: f64) -> Result<(), &'static str> {
    if cfrom == 0.0 {
        return Err("CFROM cannot be zero");
    }
    
    let smlnum = f64::MIN_POSITIVE;
    let bignum = 1.0 / smlnum;

    let mut cfromc = cfrom;
    let mut ctoc = cto;
    let mut done = false;
    let mut mul: f64;

    while !done {
        let cfrom1 = cfromc * smlnum;
        let cto1 = ctoc / bignum;
        if cfrom1 == cfromc {
            // cfromc is an inf. Multiply by something to get a finite cto
            mul = ctoc / cfromc;
            done = true;
        } else if cto1 == ctoc {
            // ctoc is either 0 or an inf.
            mul = ctoc;
            done = true;
        } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
            mul = smlnum;
            done = false;
            cfromc = cfrom1;
        } else if ctoc.abs() > cfromc.abs() {
            mul = bignum;
            done = false;
            ctoc = cto1;
        } else {
            mul = ctoc / cfromc;
            done = true;
        }

        // Scale the matrix
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                a[i][j] *= mul;
            }
        }
    }
    Ok(())
}

/// Swaps two vectors.
/// Translated from BLAS's DSWAP
fn dswap(n: usize, dx: &mut [f64], incx: usize, dy: &mut [f64], incy: usize) {
    if n == 0 {
        return;
    }
    if incx == 1 && incy == 1 {
        // Code for increments equal to 1
        let m = n % 3;
        if m != 0 {
            for i in 0..m {
                let temp = dx[i];
                dx[i] = dy[i];
                dy[i] = temp;
            }
            if n < 3 {
                return;
            }
        }
        for i in (m..n).step_by(3) {
            let temp = dx[i];
            dx[i] = dy[i];
            dy[i] = temp;
            let temp = dx[i + 1];
            dx[i + 1] = dy[i + 1];
            dy[i + 1] = temp;
            let temp = dx[i + 2];
            dx[i + 2] = dy[i + 2];
            dy[i + 2] = temp;
        }
    } else {
        // Code for unequal increments or increments not equal to 1
        let mut ix = if incx > 0 { 0 } else { (n - 1) * incx };
        let mut iy = if incy > 0 { 0 } else { (n - 1) * incy };
        for _ in 0..n {
            let temp = dx[ix];
            dx[ix] = dy[iy];
            dy[iy] = temp;
            ix += incx;
            iy += incy;
        }
    }
}


/// Updates a sum of squares represented in scaled form.
/// This function computes values `scale` and `sumsq` such that
/// `(scale**2) * sumsq = x(1)**2 + x(2)**2 + ... + x(n)**2 + (scale_in**2)*sumsq_in`,
/// where x is the slice of vector elements.
///
/// It's designed to prevent overflow or underflow during the accumulation of the sum of squares.
///
/// # Arguments
///
/// * `n` - The number of elements to be used from the vector x. n ≥ 0.
/// * `x` - The vector for which a scaled sum of squares is computed.
///         x[i] = x(1 + i*incx), for i = 0 to n-1.
/// * `incx` - The increment between successive values of the vector x. incx > 0.
/// * `scale` - On entry, the value `scale_in` in the equation above. On exit, `scale_out`.
/// * `sumsq` - On entry, the value `sumsq_in` in the equation above. On exit, `sumsq_out`.
///
/// # Note
///
/// This function corresponds to the LAPACK routine DLASSQ.
///
/// # Example
///
/// ```
/// let mut x = vec![2.0, 3.0];
/// let mut scale = 1.0;
/// let mut sumsq = 1.0;
/// dlassq(x.len(), &x, 1, &mut scale, &mut sumsq);
/// // Now, scale and sumsq contain the updated values.
/// ```
fn dlassq(n: usize, x: &[f64], incx: usize, scale: &mut f64, sumsq: &mut f64) {
    if n == 0 {
        return;
    }

    let mut ix = 0;
    for _ in 0..n {
        let absxi = x[ix].abs();
        if absxi > 0.0 {
            if *scale < absxi {
                let temp = *scale / absxi;
                *sumsq = 1.0 + *sumsq * temp * temp;
                *scale = absxi;
            } else {
                let temp = absxi / *scale;
                *sumsq += temp * temp;
            }
        }
        ix += incx;
    }
}

/// Computes all eigenvalues and corresponding eigenvectors of an unreduced
/// symmetric tridiagonal matrix using the divide and conquer method.
/// This is a recursive function corresponding to LAPACK's DLAED0 subroutine.
fn dlaed0(
    icompq: i32,
    n: usize,
    qsiz: usize,
    tlvls: usize,
    curlvl: usize,
    curpbm: usize,
    d: &mut [f64],
    e: &mut [f64],
    q: &mut [Vec<f64>],
    qstore: &mut [Vec<f64>],
    qptr: &mut [usize],
    prmptr: &mut [usize],
    perm: &mut [usize],
    givptr: &mut [usize],
    givcol: &mut [Vec<usize>],
    givnum: &mut [Vec<f64>],
    work: &mut [f64],
    iwork: &mut [usize],
) -> Result<(), &'static str> {
    // Test the input parameters.
    if icompq < 0 || icompq > 2 {
        return Err("Invalid value for ICOMPQ");
    }
    if icompq == 1 && qsiz < n {
        return Err("QSIZ must be at least N when ICOMPQ == 1");
    }
    if n < 0 {
        return Err("N must be non-negative");
    }
    if q.len() < n || q[0].len() < n {
        return Err("Dimension of Q is too small");
    }
    if qstore.len() < n || qstore[0].len() < n {
        return Err("Dimension of QSTORE is too small");
    }

    // Quick return if possible
    if n == 0 {
        return Ok(());
    }

    let smlsiz = ilaenv(9, "DLAED0", " ", 0, 0, 0, 0) as usize;

    if n <= 1 {
        if icompq == 1 {
            q[0][0] = 1.0;
        }
        return Ok(());
    }

    // Determine the size and placement of the submatrices, and save in IWORK.
    let mut subpbs = 1;
    iwork[0] = n;
    let mut tlvls_local = 0;
    while iwork[subpbs - 1] > smlsiz {
        for j in (0..subpbs).rev() {
            let i1 = iwork[j] / 2;
            let i2 = iwork[j] - i1;
            iwork[2 * j] = i1;
            iwork[2 * j + 1] = i2;
        }
        tlvls_local += 1;
        subpbs *= 2;
    }

    let total_problems = subpbs;
    for i in 1..subpbs {
        iwork[i] += iwork[i - 1];
    }

    // Divide the matrix into submatrices of size at most smlsiz+1
    let mut submat: usize = 0;
    let mut info = 0;

    // Recursively solve each submatrix eigenproblem
    for i in 0..total_problems {
        let smm1: usize;
        if i == 0 {
            submat = 0;
            smm1 = 0;
        } else {
            submat = iwork[i - 1];
            smm1 = submat - 1;
        }
        let matsiz = iwork[i] - submat;
        if curlvl == tlvls {
            // Solve the submatrix eigenproblem
            // Base case: use DSTEQR or similar
            let mut d_sub = d[submat..(submat + matsiz)].to_vec();
            let mut e_sub = e[submat..(submat + matsiz - 1)].to_vec();
            let mut q_sub = vec![vec![0.0; matsiz]; matsiz];
            for j in 0..matsiz {
                q_sub[j][j] = 1.0;
            }
            let result = dsteqr(icompq, matsiz, &mut d_sub, &mut e_sub, &mut q_sub);

            if let Err(_err) = result {
                info = submat * (n + 1) + submat + matsiz - 1;
                return Err("Error in DSTEQR");
            }

            // Copy back results
            d[submat..(submat + matsiz)].copy_from_slice(&d_sub);
            for j in 0..matsiz {
                for k in 0..matsiz {
                    q[submat + j][submat + k] = q_sub[j][k];
                }
            }
        } else {
            // Recursive call for larger submatrices
            // Prepare parameters for the recursive call
            let new_curlvl = curlvl + 1;
            let new_tlvls = tlvls;
            let new_curpbm = 2 * curpbm - 1 + i;

            // Call dlaed0 recursively
            dlaed0(
                icompq,
                matsiz,
                qsiz,
                new_tlvls,
                new_curlvl,
                new_curpbm,
                &mut d[submat..(submat + matsiz)],
                &mut e[submat..(submat + matsiz - 1)],
                q,
                qstore,
                qptr,
                prmptr,
                perm,
                givptr,
                givcol,
                givnum,
                work,
                iwork,
            )?;
        }
    }

    // Merge back the subproblems
    for lvl in (0..tlvls).rev() {
        let num_merge = 1 << lvl;
        for idx in 0..num_merge {
            let k = idx * (n / num_merge);
            let k1 = k;
            let k2 = k + (n / num_merge) - 1;

            // Merge the two subproblems from k1 to k2
            let n1 = (k2 - k1 + 1) / 2;
            let n2 = k2 - k1 + 1 - n1;
            let mut idxq = vec![0_usize; n1 + n2];

            // Merge the eigenvalues
            dlamrg(
                n1,
                n2,
                &d[k1..=k2],
                1,
                1,
                &mut idxq,
            );

            // Apply permutations to eigenvalues and eigenvectors
            let mut d_temp = vec![0.0; k2 - k1 + 1];
            let mut q_temp = vec![vec![0.0; k2 - k1 + 1]; n];
            for i in 0..(k2 - k1 + 1) {
                d_temp[i] = d[k1 + idxq[i]];
                for j in 0..n {
                    q_temp[j][i] = q[j][k1 + idxq[i]];
                }
            }
            d[k1..=k2].copy_from_slice(&d_temp);
            for i in 0..n {
                q[i][k1..=k2].copy_from_slice(&q_temp[i]);
            }
        }
    }

    Ok(())
}

/// Merges two sorted lists of numbers into a single sorted list with a permutation index.
/// This function corresponds to LAPACK's DLAMRG subroutine.
fn dlamrg(
    n1: usize,
    n2: usize,
    a: &[f64],
    dtrd1: i32,
    dtrd2: i32,
    index: &mut [usize],
) {
    let mut ind1: isize = if dtrd1 > 0 { 0 } else { n1 as isize - 1 };
    let mut ind2: isize = if dtrd2 > 0 { n1 as isize } else { (n1 + n2) as isize - 1 };
    let mut i: usize = 0;
    let mut n1sv = n1 as isize;
    let mut n2sv = n2 as isize;

    while n1sv > 0 && n2sv > 0 {
        let a_ind1 = a[ind1 as usize];
        let a_ind2 = a[ind2 as usize];
        if a_ind1 <= a_ind2 {
            index[i] = ind1 as usize;
            i += 1;
            ind1 += dtrd1 as isize;
            n1sv -= 1;
        } else {
            index[i] = ind2 as usize;
            i += 1;
            ind2 += dtrd2 as isize;
            n2sv -= 1;
        }
    }

    if n1sv == 0 {
        for _ in 0..n2sv {
            index[i] = ind2 as usize;
            i += 1;
            ind2 += dtrd2 as isize;
        }
    } else {
        for _ in 0..n1sv {
            index[i] = ind1 as usize;
            i += 1;
            ind1 += dtrd1 as isize;
        }
    }
}

/// Query function for machine-dependent parameters
fn ilaenv(ispec: i32, name: &str, opts: &str, n1: i32, n2: i32, n3: i32, n4: i32) -> i32 {
    // - For ispec = 9 (used in DLAED0), return the block size for the D&C algorithm.
    // In LAPACK, ILAENV(9, ...) returns the value of SMLSIZ, is always 25.
    if ispec == 9 {
        25
    } else {
        1
    }
}


/*
Not yet implemented functions:

- DLAMCH
  - Description: Determines double-precision real machine parameters, such as the machine precision (EPS), the safe minimum (SAFMIN), the base of the machine, the maximum and minimum exponents, etc. This function is essential for setting up constants used in scaling and convergence criteria in numerical algorithms.
  - When it's called: It's called during `dsbevd` to get machine constants like EPS (precision), SAFMIN (safe minimum), SMLNUM, BIGNUM, RMIN, RMAX. These constants are crucial for scaling decisions, convergence checks, and to avoid overflow or underflow during computations.

- DLAED0
  - Description: Computes all eigenvalues and corresponding eigenvectors of an unreduced symmetric tridiagonal matrix using the divide and conquer method. It serves as a high-level driver routine that orchestrates the divide and conquer process.
  - When it's called: Within `dstedc` when dealing with larger matrices (size greater than a threshold `SMLSIZ`), `dlaed0` is called to compute the eigenvalues and eigenvectors using the divide and conquer strategy.
  - Perhaps partially in divide_and_conquer function

- DCOPY
  - Description: Copies the contents of one vector to another, i.e., performs the operation `y := x`. It handles copying vectors with specific increments between elements.
  - When it's called: Used throughout the algorithms to copy vectors or portions of vectors. In routines like `dlaed0`, `dlaed1`, and `dlaed2`, it is essential for duplicating data during the divide and conquer steps.

- DLACPY
  - Description: Copies all or part of a two-dimensional matrix `A` to another matrix `B`. It can copy the entire matrix or just the upper or lower triangular part, depending on the specified options.
  - When it's called: Used to copy matrices or submatrices, particularly when manipulating eigenvector matrices. In `dstedc` and `dlaed0`, `dlacpy` is called to manage workspace matrices during computations.

- DLAED1
  - Description: Computes the updated eigensystem of a diagonal matrix after modification by a rank-one symmetric matrix. It is specifically used when the original matrix is tridiagonal and involves deflation techniques.
  - When it's called: Within `dlaed0` during the divide step of the divide and conquer algorithm, when handling merged eigenvalues and eigenvectors.

- DLAED2
  - Description: Merges eigenvalues and deflates the secular equation, reducing the problem size when possible. It handles cases with multiple eigenvalues or negligible entries in the updating vector.
  - When it's called: Called by `dlaed1` to perform deflation during the divide and conquer process, optimizing the computation by exploiting the structure of the problem.

- DLAED5
  - Description: Computes the i-th eigenvalue and eigenvector of a symmetric rank-one modification of a 2-by-2 diagonal matrix. It handles the special case where the problem size is two.
  - When it's called: Within `dlaed4` when the problem has been reduced to size two, allowing for a direct computation of eigenvalues and eigenvectors in the divide and conquer algorithm.

- DLAED6
  - Description: Computes one Newton step in the solution of the secular equation, specializing in refining a single eigenvalue. It is used for efficiently finding roots of the secular equation.
  - When it's called: Invoked by `dlaed4` during the iterative solution of the secular equation, particularly when dealing with closely spaced eigenvalues that require careful handling to ensure convergence.

- DLAED7
  - Description: Computes the updated eigensystem of a diagonal matrix after modification by a rank-one symmetric matrix, used when the original matrix is dense. It specifically handles larger subproblems in the divide and conquer algorithm.
  - When it's called: Within `dstedc`, `dlaed7` is called during recursive steps of the divide and conquer process to update eigenvalues and eigenvectors after merging subproblems.

- DLAED8
  - Description: Merges eigenvalues and deflates the secular equation for dense matrices. It works alongside `dlaed7` to handle eigenvalue deflation and convergence during the divide and conquer algorithm.
  - When it's called: Called by `dlaed7` to perform the merging and deflation steps necessary for efficiently computing the eigenvalues and eigenvectors of the updated matrix.

- DLAED9
  - Description: Finds the roots of the secular equation and updates the eigenvectors. This function computes updated eigenvalues and eigenvectors for a rank-one modification of a diagonal matrix.
  - When it's called: Invoked by `dlaed7` during the last stages of the divide and conquer algorithm to compute the final eigenvectors that will be combined to form the solution to the original problem.

- DLAEDA
  - Description: Computes the Z vector determining the rank-one modification of the diagonal matrix. It effectively prepares the data needed for the rank-one update in the divide step of the algorithm.
  - When it's called: Used within `dlaed0` during the divide and conquer process to set up vectors necessary for merging eigenvalues and eigenvectors from subproblems.

- DLASR
  - Description: Applies a sequence of plane rotations to a general rectangular matrix. This function can apply rotations from the left or the right, in forward or backward order, and with different pivot strategies.
  - When it's called: Used in `dsteqr` to apply accumulated rotations to eigenvectors during the QR iteration process for computing eigenvalues and eigenvectors of a tridiagonal matrix.
*/
