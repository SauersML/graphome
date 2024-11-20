// src/dsbevd.rs

use rayon::prelude::*;

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
            for k in (2..=kd + 1).rev() {
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
                            y_temp[idx] = ab[kd - 1][j];
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
                        for l in 1..kd {
                            let mut v1 = vec![];
                            let mut v2 = vec![];
    
                            for idx in 0..nr {
                                let j = j1 - kd + l + idx * kd;
                                if j < ab[0].len() && j + 1 < ab[0].len() {
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
                                    if j < ab[0].len() {
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
                                    if jinc - kd + idx < ab[0].len() {
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
                                            if jinc - kd + idx < ab[0].len() {
                                                ab[kd][jinc - kd + idx] = *val1;
                                                ab[kd1][jinc - kd + idx] = *val2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
    
                    // Block diagonal updates moved inside k-loop
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
                                let (left, right) = q.split_at_mut(j + 1);
                                drot(
                                    &mut left[j],
                                    &mut right[0],
                                    work[j_idx],
                                    y_temp[j_idx],
                                );
                            }
                        }
                    }
                }
    
                // Handle inner elements of band for current k
                if k > 2 && k <= n - i {
                    let f = ab[k - 2][i];
                    let g = ab[k - 1][i];
                    let (cs, sn) = givens_rotation(f, g);
                    ab[k - 2][i] = cs * f + sn * g;
    
                    // Apply from the left
                    let start = i + 1;
                    let end = (i + k - 1).min(n - 1);
                    if start <= end {
                        for j in start..=end {
                            let temp = cs * ab[k - 2][j] + sn * ab[k - 1][j];
                            ab[k - 1][j] = -sn * ab[k - 2][j] + cs * ab[k - 1][j];
                            ab[k - 2][j] = temp;
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
            if i < n - 1 {
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

fn drot(dx: &mut [f64], dy: &mut [f64], c: f64, s: f64) {
    let n = dx.len();
    for i in 0..n {
        let temp = c * dx[i] + s * dy[i];
        dy[i] = c * dy[i] - s * dx[i];
        dx[i] = temp;
    }
}

fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    if g == 0.0 {
        let cs = f.signum();
        let sn = 0.0;
        let r = f.abs();
        (cs, sn, r)
    } else if f == 0.0 {
        let cs = 0.0;
        let sn = g.signum();
        let r = g.abs();
        (cs, sn, r)
    } else {
        let scale = f.abs().max(g.abs());
        let fs = f / scale;
        let gs = g / scale;
        let r = scale * (fs * fs + gs * gs).sqrt();
        let cs = f / r;
        let sn = g / r;
        (cs, sn, r)
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
    
    let xnorm = scale * ssq.sqrt();
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
    let smlsiz = 25; // Minimum size to use divide and conquer
    let eps = f64::EPSILON;

    // We always compute eigenvectors

    if n <= smlsiz {
        // Use QR algorithm for small matrices
        tridiagonal_qr(d, e, z);
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
            tridiagonal_qr(&mut d_sub, &mut e_sub, &mut z_sub);

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

            // Initialize z_out for solve_secular_equation
            let mut z_out = vec![vec![0.0; m]; m];

            // Solve the secular equation
            let info = solve_secular_equation(
                &d_left,
                &d_right,
                &z_vector,
                rho,
                &mut d_merged,
                &mut z_out,
            );
            if info != 0 {
                panic!("Error in solve_secular_equation: info = {}", info);
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
fn solve_secular_equation(
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

/// Tridiagonal QR algorithm for small matrices.
/// `d`: Diagonal elements
/// `e`: Off-diagonal elements
/// `z`: Eigenvectors (output)
fn tridiagonal_qr(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>]) {
    let n = d.len();
    let mut e_ext = vec![0.0; n];
    e_ext[..(n - 1)].copy_from_slice(e);

    // Initialize z to identity matrix
    for i in 0..n {
        for j in 0..n {
            z[i][j] = if i == j { 1.0 } else { 0.0 };
        }
    }

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
            if iter > 30 {
                panic!("Too many iterations in tridiagonal_qr");
            }

            // Compute shift
            let g = d[l];
            let p = (d[l + 1] - g) / (2.0 * e_ext[l]);
            let r = p.hypot(1.0);
            let mut t = if p >= 0.0 {
                g - e_ext[l] / (p + r)
            } else {
                g - e_ext[l] / (p - r)
            };

            for i in l..n {
                d[i] -= t;
            }

            let mut s = 0.0;
            let mut c = 1.0;

            for i in l..m {
                let f = s * e_ext[i];
                let b = c * e_ext[i];
                let (r, cs, sn) = plane_rotation(d[i] - t, f);
                e_ext[i] = r;
                s = sn;
                c = cs;
                let temp = c * d[i] - s * e_ext[i + 1];
                e_ext[i + 1] = s * d[i] + c * e_ext[i + 1];
                d[i] = temp;

                // Apply rotation to eigenvectors
                for k in 0..n {
                    let temp = c * z[k][i] - s * z[k][i + 1];
                    z[k][i + 1] = s * z[k][i] + c * z[k][i + 1];
                    z[k][i] = temp;
                }
            }

            let temp = c * d[m] - s * e_ext[m];
            e_ext[m] = s * d[m] + c * e_ext[m];
            d[m] = temp;
        }
    }

    // Sort eigenvalues and eigenvectors
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

/// Compute plane rotation parameters
fn plane_rotation(f: f64, g: f64) -> (f64, f64, f64) {
    if g == 0.0 {
        (f.abs(), f.signum(), 0.0)
    } else if f == 0.0 {
        (g.abs(), 0.0, g.signum())
    } else {
        let r = f.hypot(g);
        let cs = f / r;
        let sn = g / r;
        (r, cs, sn)
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
