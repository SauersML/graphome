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
    pub fn dsbevd(&self) -> Result<EigenResults, i32> {
        let safmin = f64::MIN_POSITIVE;
        let eps = f64::EPSILON;
        let smlnum = safmin / eps;
        let bignum = 1.0 / smlnum;
        let rmin = smlnum.sqrt();
        let rmax = bignum.sqrt();
    
        let anrm = self.dlanst();
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
            self.dlascl(scale)
        } else {
            self.clone()
        };
    
        let (mut d, mut e, mut q) = working_matrix.dsbtrd();
    
        // Use reliable tridiagonal solver
        let mut d = d.to_vec();
        let mut e = e.to_vec();
        let mut z = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            z[i][i] = 1.0;
        }
        
        // Call dstedc with mutable references
        dstedc(&mut d, &mut e, &mut z)?;
        
        // Convert results to required format
        let eigenvalues = d.clone();
        let eigenvectors = z.clone(); // Clone z here
        
        // Rescale eigenvalues
        let mut eigenvalues = eigenvalues; // Use the cloned eigenvalues
        if scale != 1.0 {
            for eigenval in &mut eigenvalues {
                *eigenval /= scale;
            }
        }
    
        Ok(EigenResults {
            eigenvalues,
            eigenvectors,
        })
    }

    fn dsbtrd(&self) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
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
                                        let n_rot = row1.len(); // Should we assume row1 and row2 have the same length?
                                        drot(n_rot, &mut row1, 1, &mut row2, 1, work[jidx], y_temp[jidx]);
    
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
                        let (cs, sn) = dlartg(f, g);
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
    
        

    fn dlanst(&self) -> f64 {
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
    

    
    fn dlascl(&self, scale: f64) -> Self {
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

/// Performs a Givens rotation. (LAPACK's DROT)
fn drot(n: usize, dx: &mut [f64], incx: i32, dy: &mut [f64], incy: i32, c: f64, s: f64) {
    if n == 0 {
        return;
    }

    let mut ix = if incx > 0 { 0 } else { (1-n as i32) * incx };
    let mut iy = if incy > 0 { 0 } else { (1-n as i32) * incy };

    for _ in 0..n {
        let temp = c * dx[ix as usize] + s * dy[iy as usize];
        dy[iy as usize] = c * dy[iy as usize] - s * dx[ix as usize];
        dx[ix as usize] = temp;

        ix += incx;
        iy += incy;
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



/// Computes all eigenvalues and eigenvectors of a symmetric tridiagonal matrix using divide and conquer.
/// 
/// # Arguments
/// * `d` - On entry, the diagonal elements. On exit, the eigenvalues in ascending order.
/// * `e` - On entry, the subdiagonal elements. On exit, destroyed.
/// * `z` - On entry (when computing eigenvectors), the identity matrix.
///        On exit, the orthonormal eigenvectors of the tridiagonal matrix.
/// 
/// # Returns
/// * `Ok(())` on successful computation
/// * `Err(i)` if the algorithm failed to compute eigenvalue i
pub fn dstedc(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>]) -> Result<(), i32> {
    let n = d.len();
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        z[0][0] = 1.0;
        return Ok(());
    }

    // Parameters
    let smlsiz = 25; // Minimum size for divide-conquer. Use STEQR for smaller.
    let eps = f64::EPSILON;

    if n <= smlsiz {
        // Use QR algorithm for small matrices
        let mut work = vec![0.0; 2*n-2];
        return dsteqr('I', n, d, e, z, &mut work).map_err(|_| 1);
    }

    // Scale the matrix if necessary
    let orgnrm = d.iter().map(|&x| x.abs())
                  .chain(e.iter().map(|&x| x.abs()))
                  .fold(0.0_f64, f64::max);
    if orgnrm == 0.0 {
        // Zero matrix, eigenvalues all zero, z stays identity
        return Ok(());
    }

    let mut info = 0;
    let mut start = 0;

    // Main divide and conquer loop
    while start < n {
        // Find the end of the current submatrix (look for small subdiagonal)
        let mut finish = start;
        while finish < n - 1 {
            let tiny = eps * (d[finish].abs().sqrt() * d[finish + 1].abs().sqrt());
            if e[finish].abs() <= tiny {
                e[finish] = 0.0; // Deflate
                break;
            }
            finish += 1;
        }

        // Process submatrix from start to finish
        let m = finish - start + 1;

        if m == 1 {
            // 1x1 block, already diagonal
            start = finish + 1;
            continue;
        }

        if m <= smlsiz {
            // Small subproblem - use STEQR
            let mut d_sub = d[start..=finish].to_vec();
            let mut e_sub = e[start..finish].to_vec();
            let mut z_sub = vec![vec![0.0; m]; m];
            for i in 0..m {
                z_sub[i][i] = 1.0;
            }
            let mut work = vec![0.0; 2*m-2];
            if let Err(err) = dsteqr('I', m, &mut d_sub, &mut e_sub, &mut z_sub, &mut work) {
                info = (start + 1) * (n + 1) + finish;
                break;
            }

            // Copy results back
            for i in 0..m {
                d[start + i] = d_sub[i];
                for j in 0..m {
                    z[start + i][start + j] = z_sub[i][j];
                }
            }
        } else {
            // Large subproblem - use divide and conquer
            let mid = start + m/2 - 1;
            let rho = e[mid];
            e[mid] = 0.0; // Split matrix

            // Solve left subproblem
            let left_size = mid - start + 1;
            let mut d_left = d[start..=mid].to_vec();
            let mut e_left = e[start..mid].to_vec();
            let mut z_left = vec![vec![0.0; left_size]; left_size];
            for i in 0..left_size {
                z_left[i][i] = 1.0;
            }
            dstedc(&mut d_left, &mut e_left, &mut z_left)?;

            // Solve right subproblem
            let right_size = finish - mid;
            let mut d_right = d[mid+1..=finish].to_vec();
            let mut e_right = e[mid+1..finish].to_vec();
            let mut z_right = vec![vec![0.0; right_size]; right_size];
            for i in 0..right_size {
                z_right[i][i] = 1.0;
            }
            dstedc(&mut d_right, &mut e_right, &mut z_right)?;

            // Merge solutions
            let mut d_merged = vec![0.0; m];
            let mut z_merged = vec![vec![0.0; m]; m];

            // Form rank-one modification
            let mut z_vec = vec![0.0; m];
            for i in 0..left_size {
                z_vec[i] = z_left[i][left_size - 1];
            }
            for i in 0..right_size {
                z_vec[left_size + i] = z_right[i][0];
            }

            // Solve secular equation
            let mut z_out = vec![vec![0.0; m]; m];
            let err = dlaed4(&d_left, &d_right, &z_vec, rho, &mut d_merged, &mut z_out);
            
            if err != 0 {
                info = (start + err as usize) * (n + 1);
                break;
            }

            // Update eigenvectors
            let mut z_out = vec![vec![0.0; m]; m];
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

            // Copy back results
            for i in 0..m {
                d[start + i] = d_merged[i];
                for j in 0..m {
                    z[start + i][start + j] = z_merged[i][j];
                }
            }
        }

        start = finish + 1;
    }

    if info != 0 {
        return Err(info.try_into().unwrap());
    }

    // Final eigenvalue sort using selection sort to minimize eigenvector swaps
    for ii in 1..n {
        let i = ii - 1;
        let mut k = i;
        let p = d[i];
        
        for j in ii..n {
            if d[j] < p {
                k = j;
            }
        }
        
        if k != i {
            // Swap eigenvalues and eigenvectors
            d.swap(k, i);
            for row in z.iter_mut() {
                row.swap(k, i);
            }
        }
    }

    Ok(())
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


fn dsteqr(
    compz: char,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    z: &mut [Vec<f64>],
    work: &mut [f64],
) -> Result<(), &'static str> {
    // Constants
    const MAXIT: usize = 30;
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const THREE: f64 = 3.0;

    // Check input parameters
    let icompz = match compz {
        'N' | 'n' => 0,
        'V' | 'v' => 1,
        'I' | 'i' => 2,
        _ => return Err("Invalid value for compz in dsteqr"),
    };

    if n == 0 {
        return Ok(());
    }

    if icompz == 1 && (z.len() < n || z[0].len() < n) {
        return Err("Dimension of z is too small");
    }

    if icompz == 2 {
        // Initialize z to identity matrix
        for i in 0..n {
            for j in 0..n {
                z[i][j] = if i == j { ONE } else { ZERO };
            }
        }
    }

    let mut info = 0;
    let mut nm1 = n.saturating_sub(1);

    if n == 1 {
        return Ok(());
    }

    // Machine constants
    let eps = f64::EPSILON;
    let eps2 = eps * eps;
    let safmin = f64::MIN_POSITIVE;
    let safmax = ONE / safmin;
    let ssfmax = (safmax.sqrt()) / THREE;
    let ssfmin = (safmin.sqrt()) / eps2;

    // Initialize work arrays
    let mut work_c = vec![0.0; n];
    let mut work_s = vec![0.0; n];

    // Main loop variables
    let mut l1 = 0;
    let mut jtot = 0;
    let nmaxit = n * MAXIT;

    while l1 < n {
        let mut m = l1;
        // Find small subdiagonal element
        while m < nm1 {
            let tst = e[m].abs().powi(2);
            if tst <= (eps2 * d[m].abs() * d[m + 1].abs() + safmin) {
                break;
            }
            m += 1;
        }

        let mut l = l1;
        let lend = m;
        if m < nm1 {
            e[m] = ZERO;
        }
        l1 = m + 1;

        if l > lend {
            // QR iteration
            while l > lend {
                if jtot >= nmaxit {
                    info += l;
                    break;
                }
                jtot += 1;

                // Form shift
                let mut g = (d[l - 1] - d[l]) / (TWO * e[l - 1]);
                let mut r = dlapy2(g, ONE);
                g = d[l] - d[l - 1] + e[l - 1] / (g + r.copysign(g));
                let mut s = ONE;
                let mut c = ONE;
                let mut p = ZERO;

                for i in (lend..l).rev() {
                    let f = s * e[i];
                    let b = c * e[i];
                    let (c_temp, s_temp) = dlartg(g, f);
                    if i < l - 1 {
                        e[i + 1] = r;
                    }
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s_temp + TWO * c_temp * b;
                    p = s_temp * r;
                    d[i + 1] = g + p;
                    g = c_temp * r - b;

                    // Save rotations
                    if icompz > 0 {
                        work_c[i] = c_temp;
                        work_s[i] = s_temp;
                    }
                }
                d[l] = d[l] - p;
                e[l - 1] = g;

                // Apply rotations
                if icompz > 0 {
                    let mm = l - lend + 1;
                    for j in 0..z.len() {
                        let mut k = l - mm + 1;
                        while k <= l {
                            let temp = work_c[k - 1] * z[j][k - 1] - work_s[k - 1] * z[j][k];
                            z[j][k] = work_s[k - 1] * z[j][k - 1] + work_c[k - 1] * z[j][k];
                            z[j][k - 1] = temp;
                            k += 1;
                        }
                    }
                }

                if e[l - 1].abs() <= eps2 * (d[l - 1].abs() + d[l].abs()) {
                    e[l - 1] = ZERO;
                }

                l -= 1;
            }
        } else if l < lend {
            // QL iteration
            while l < lend {
                if jtot >= nmaxit {
                    info += l;
                    break;
                }
                jtot += 1;

                // Form shift
                let mut g = (d[l + 1] - d[l]) / (TWO * e[l]);
                let mut r = dlapy2(g, ONE);
                g = d[l] - d[l + 1] + e[l] / (g + r.copysign(g));
                let mut s = ONE;
                let mut c = ONE;
                let mut p = ZERO;

                for i in l..lend {
                    let f = s * e[i];
                    let b = c * e[i];
                    let (c_temp, s_temp) = dlartg(g, f);
                    if i > l {
                        e[i - 1] = r;
                    }
                    g = d[i] - p;
                    r = (d[i + 1] - g) * s_temp + TWO * c_temp * b;
                    p = s_temp * r;
                    d[i] = g + p;
                    g = c_temp * r - b;

                    // Save rotations
                    if icompz > 0 {
                        work_c[i] = c_temp;
                        work_s[i] = s_temp;
                    }
                }
                d[lend] = d[lend] - p;
                e[lend - 1] = g;

                // Apply rotations
                if icompz > 0 {
                    let mm = lend - l;
                    for j in 0..z.len() {
                        let mut k = l;
                        while k < lend {
                            let temp = work_c[k] * z[j][k] - work_s[k] * z[j][k + 1];
                            z[j][k + 1] = work_s[k] * z[j][k] + work_c[k] * z[j][k + 1];
                            z[j][k] = temp;
                            k += 1;
                        }
                    }
                }

                if e[l].abs() <= eps2 * (d[l].abs() + d[l + 1].abs()) {
                    e[l] = ZERO;
                }

                l += 1;
            }
        } else {
            // Diagonal element
            if d[l].is_nan() {
                return Err("NaN encountered");
            }
            l += 1;
        }
    }

    // Order eigenvalues and eigenvectors
    if icompz == 0 {
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_d = idx.iter().map(|&i| d[i]).collect::<Vec<f64>>();
        let sorted_z = idx
            .iter()
            .map(|&i| z.iter().map(|row| row[i]).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();

        d.copy_from_slice(&sorted_d);
        for i in 0..n {
            for j in 0..n {
                z[i][j] = sorted_z[j][i];
            }
        }
    }

    if info > 0 {
        Err("Failed to compute all eigenvalues")
    } else {
        Ok(())
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


/// Computes all eigenvalues and corresponding eigenvectors of an unreduced
/// symmetric tridiagonal matrix using the divide and conquer method.
/// This function corresponds to LAPACK's DLAED0 subroutine.
///
/// Parameters:
/// - `icompq`: Specifies whether to compute eigenvectors. NOT NEEDED. We always compute BOTH eigenvectors and eigenvalues.
///     - 0: Compute eigenvalues only.
///     - 1: Compute eigenvectors of the original symmetric matrix.
///     - 2: Compute eigenvectors of the tridiagonal matrix.
/// - `n`: The order of the matrix.
/// - `qsiz`: The dimension of the orthogonal matrix used to reduce
///           the full matrix to tridiagonal form.
/// - `tlvls`: The total number of merging levels in the overall divide and conquer tree.
/// - `curlvl`: The current level in the overall merge routine.
/// - `curpbm`: The current problem in the current level.
/// - `d`: On entry, the diagonal elements of the tridiagonal matrix.
///        On exit, contains the eigenvalues.
/// - `e`: On entry, the off-diagonal elements of the tridiagonal matrix.
///        On exit, it is destroyed.
/// - `q`: On entry, if `icompq = 1`, the orthogonal matrix used to reduce the original matrix
///        to tridiagonal form. If `icompq = 2`, it is initialized to the identity matrix.
///        On exit, if `icompq > 0`, contains the eigenvectors.
/// - `qstore`: Workspace array to store intermediate eigenvectors.
/// - `qptr`, `prmptr`, `perm`, `givptr`, `givcol`, `givnum`: Workspace arrays for DLAED0.
/// - `work`: Workspace array.
/// - `iwork`: Integer workspace array.
///
/// Returns:
/// - `Result<(), &'static str>` indicating success or an error message.
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
        return Err("Invalid value for icompq");
    }
    if icompq == 1 && qsiz < n {
        return Err("QSIZ must be at least N when icompq == 1");
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

    if n <= smlsiz {
        // Use dsteqr to compute the eigenvalues and eigenvectors.
        let compz = match icompq {
            0 => 'N',
            1 => 'V',
            2 => 'I',
            _ => return Err("Invalid value for icompq"),
        };
        // Initialize q if necessary
        if compz == 'I' {
            for i in 0..n {
                for j in 0..n {
                    q[i][j] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }
        // Prepare work array
        let mut work_dsteqr = vec![0.0; max(1, 2 * n - 2)];
        // Call dsteqr
        let result = dsteqr(compz, n, d, e, q, &mut work_dsteqr);
        if let Err(err) = result {
            return Err(err);
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
    let mut submat: usize;

    // Recursively solve each submatrix eigenproblem
    for i in 0..total_problems {
        let matsiz = if i == 0 {
            iwork[0]
        } else {
            iwork[i] - iwork[i - 1]
        };
        submat = if i == 0 {
            0
        } else {
            iwork[i - 1]
        };

        if curlvl == tlvls {
            // Use dsteqr to compute the eigenvalues and eigenvectors
            let mut d_sub = d[submat..submat + matsiz].to_vec();
            let mut e_sub = e[submat..submat + matsiz - 1].to_vec();

            // Initialize q_sub
            let mut q_sub = vec![vec![0.0; matsiz]; matsiz];
            for j in 0..matsiz {
                q_sub[j][j] = 1.0;
            }

            let compz = match icompq {
                0 => 'N',
                1 => 'V',
                2 => 'I',
                _ => return Err("Invalid value for icompq"),
            };
            let mut work_dsteqr = vec![0.0; max(1, 2 * matsiz - 2)];
            let result = dsteqr(compz, matsiz, &mut d_sub, &mut e_sub, &mut q_sub, &mut work_dsteqr);

            if let Err(_err) = result {
                let info = submat * (n + 1) + submat + matsiz - 1;
                return Err("Error in dsteqr");
            }

            // Copy back results
            d[submat..submat + matsiz].copy_from_slice(&d_sub);
            for i in 0..matsiz {
                for j in 0..matsiz {
                    q[submat + i][submat + j] = q_sub[i][j];
                }
            }
        } else {
            // Recursion to lower levels
            let new_curlvl = curlvl + 1;
            let new_tlvls = tlvls;
            let new_curpbm = 2 * curpbm - 1 + i;

            // Recursive call
            dlaed0(
                icompq,
                matsiz,
                qsiz,
                new_tlvls,
                new_curlvl,
                new_curpbm,
                &mut d[submat..submat + matsiz],
                &mut e[submat..submat + matsiz - 1],
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

    // WE MUST MERGE BACK THE SUBPROBLEMS.

    Ok(())
}


/// Determines double-precision real machine parameters.
/// This function corresponds to LAPACK's DLAMCH subroutine.
pub fn dlamch(cmach: char) -> f64 {
    let one: f64 = 1.0;
    let zero: f64 = 0.0;

    // Assume rounding, not chopping.
    let rnd: f64 = one;

    let eps: f64 = if one == rnd {
        f64::EPSILON * 0.5
    } else {
        f64::EPSILON
    };

    match cmach {
        'E' | 'e' => eps,
        'S' | 's' => {
            let mut sfmin: f64 = f64::MIN_POSITIVE; // tiny(zero)
            let small: f64 = 1.0 / f64::MAX; // 1 / huge(zero)
            if small >= sfmin {
                // Use SMALL plus a bit, to avoid the possibility of rounding causing overflow when computing 1/sfmin.
                sfmin = small * (one + eps);
            }
            sfmin
        }
        'B' | 'b' => 2.0f64, // radix(zero). We assume base 2.
        'P' | 'p' => eps * 2.0f64, // eps * radix(zero)
        'N' | 'n' => f64::MANTISSA_DIGITS as f64, // digits(zero)
        'R' | 'r' => rnd,
        'M' | 'm' => f64::MIN_EXP as f64, // minexponent(zero)
        'U' | 'u' => f64::MIN_POSITIVE, // tiny(zero)
        'L' | 'l' => f64::MAX_EXP as f64, // maxexponent(zero)
        'O' | 'o' => f64::MAX, // huge(zero)
        _ => zero,
    }
}



/// Copies a vector, x, to a vector, y.
///
/// This function corresponds to the BLAS level 1 routine DCOPY.  It uses unrolled loops
/// for the common case where increment values are equal to one.
///
/// # Arguments
/// * `n` - The number of vector elements to be copied. `n ≥ 0`.
/// * `dx` - The source vector.
/// * `incx` - The increment between elements of `dx`. `incx ≠ 0`.
/// * `dy` - The destination vector.
/// * `incy` - The increment between elements of `dy`. `incy ≠ 0`.
pub fn dcopy(n: usize, dx: &[f64], incx: i32, dy: &mut [f64], incy: i32) {
    if n == 0 {
        return;
    }

    if incx == 1 && incy == 1 {
        // Optimized case for increments equal to 1, using unrolled loops
        dy[..n].copy_from_slice(&dx[..n]);
    } else {
        // General case for unequal or non-unit increments
        let mut ix = if incx > 0 { 0 } else { (1 - (n as i32)) * incx } as isize;
        let mut iy = if incy > 0 { 0 } else { (1 - (n as i32)) * incy } as isize;

        for _ in 0..n {
            dy[iy as usize] = dx[ix as usize];
            ix += incx as isize;
            iy += incy as isize;
        }
    }
}


/// Copies all or part of a 2D matrix A to another matrix B.
///
/// This function corresponds to LAPACK's DLACPY subroutine.
///
/// # Arguments
///
/// * `uplo` - Specifies the part of the matrix A to be copied to B:
///     - 'U': Upper triangular part
///     - 'L': Lower triangular part
///     - Other: All of the matrix A
/// * `m` - The number of rows of the matrix A.  m >= 0.
/// * `n` - The number of columns of the matrix A.  n >= 0.
/// * `a` - The m-by-n matrix A.
/// * `lda` - The leading dimension of the array A. lda >= max(1,m).
/// * `b` - On exit, B = A in the locations specified by uplo.
/// * `ldb` - The leading dimension of the array B. ldb >= max(1,m).
pub fn dlacpy(uplo: char, m: usize, n: usize, a: &[Vec<f64>], lda: usize, b: &mut [Vec<f64>], ldb: usize) {
    if m == 0 || n == 0 {
        return; // Quick return if possible
    }

    match uplo {
        'U' | 'u' => {
            // Copy the upper triangular part of A to B
            for j in 0..n {
                for i in 0..min(j + 1, m) {
                    b[i][j] = a[i][j];
                }
            }
        }
        'L' | 'l' => {
            // Copy the lower triangular part of A to B
            for j in 0..n {
                for i in j..m {
                    b[i][j] = a[i][j];
                }
            }
        }
        _ => {
            // Copy the entire matrix A to B
            for j in 0..n {
                for i in 0..m {
                    b[i][j] = a[i][j];
                }
            }
        }
    }
}


/// Merges eigenvalues and deflates the secular equation.
/// This function corresponds to LAPACK's DLAED2 subroutine.
pub fn dlaed2(
    k: &mut usize,
    n: usize,
    n1: usize,
    d: &mut [f64],
    q: &mut [Vec<f64>],
    ldq: usize,
    indxq: &mut [usize],
    rho: &mut f64,
    z: &mut [f64],
    dlamda: &mut [f64],
    w: &mut [f64],
    q2: &mut [Vec<f64>],
    indx: &mut [usize],
    indxc: &mut [usize],
    indxp: &mut [usize],
    coltyp: &mut [usize],
) -> i32 {

    let mut info = 0;

    if n == 0 {
        return info;
    }

    let n2 = n - n1;
    let n1p1 = n1 + 1;

    // Scale z
    if *rho < 0.0 {
        let alpha = -1.0;
        dscal(n2, alpha, &mut z[n1..], 1);
    }

    // Normalize z
    let t = 1.0 / 2.0f64.sqrt();
    dscal(n, t, z, 1);

    // Update rho
    *rho = (2.0 * *rho).abs();


    // Sort eigenvalues into increasing order
    for i in n1..n {
        indxq[i] += n1;
    }

    // Re-integrate deflated parts
    for i in 0..n {
        dlamda[i] = d[indxq[i]];
    }
    dlamrg(n1, n2, dlamda, 1, 1, indxc);
    for i in 0..n {
        indx[i] = indxq[indxc[i]];
    }

    // Calculate deflation tolerance
    let imax = idamax(n, z, 1);
    let jmax = idamax(n, d, 1);
    let eps = dlamch('E');
    let tol = 8.0 * eps * d[jmax].abs().max(z[imax].abs());

    if *rho * z[imax].abs() <= tol {
        *k = 0;
        for j in 0..n {
            let i = indx[j];
            dcopy(n, &q[i], ldq as i32, &mut q2[j], 1);
            dlamda[j] = d[i];
        }

        dlacpy('A', n, n, q2, ldq, q, ldq);
        dcopy(n, dlamda, 1, d, 1);
        return info;
    }

    // Check for multiple eigenvalues and deflate
    for i in 0..n1 {
        coltyp[i] = 1;
    }
    for i in n1..n {
        coltyp[i] = 3;
    }

    *k = 0;
    let mut k2 = n + 1;
    let mut j = 0;
    while j < n {
        let nj = indx[j];
        if *rho * z[nj].abs() <= tol {
            k2 -= 1;
            coltyp[nj] = 4;
            indxp[k2-1] = nj;
            j += 1;
        } else {
            let mut pj = nj;
            j += 1;
            while j < n {
                let nj = indx[j];
                if *rho * z[nj].abs() <= tol {
                    k2 -= 1;
                    coltyp[nj] = 4;
                    indxp[k2-1] = nj;
                    j+=1;
                } else {
                    // Check if eigenvalues are close enough to allow deflation
                    let mut s = z[pj];
                    let mut c = z[nj];
                    let tau = dlapy2(c, s);
                    let t = d[nj] - d[pj];
                    c /= tau;
                    s = -s / tau;
                    if (t * c * s).abs() <= tol {
                        z[nj] = tau;
                        z[pj] = 0.0;

                        if coltyp[nj] != coltyp[pj] {
                            coltyp[nj] = 2;
                        }
                        coltyp[pj] = 4;

                        if pj < nj {
                            let mut q_pj = q[pj].clone();
                            drot(n, &mut q_pj, 1, &mut q[nj], 1, c, s);
                            q[pj] = q_pj;
                        } else {
                            let mut q_nj = q[nj].clone();
                            drot(n, &mut q_nj, 1, &mut q[pj], 1, c, s);
                            q[nj] = q_nj;
                        }
                        
                        let temp = d[pj] * c * c + d[nj] * s * s;
                        d[nj] = d[pj] * s * s + d[nj] * c * c;
                        d[pj] = temp;

                        k2 -= 1;
                        let mut i = 1;

                        while k2 - 1 + i <= n {
                           if d[pj] < d[indxp[k2 - 1 + i - 1]] {
                               indxp[k2 - 1 + i - 2] = indxp[k2 - 1 + i - 1];
                               indxp[k2 - 1 + i - 1] = pj;
                           } else {
                               indxp[k2 - 1 + i - 2] = pj;
                               break;
                           }
                           i += 1;
                        }
                        pj = nj;
                        j += 1;
                    } else {
                        *k += 1;
                        dlamda[*k-1] = d[pj];
                        w[*k-1] = z[pj];
                        indxp[*k-1] = pj;
                        pj = nj;
                        j += 1;
                    }
                }
                if j >= n {
                    break;
                }
            }
            if j >= n {
                break;
            }
        }
        if j >= n {
            break;
        }
    }

    *k += 1;
    dlamda[*k-1] = d[indx[j-1]];
    w[*k-1] = z[indx[j-1]];
    indxp[*k-1] = indx[j-1];

    info
}


/// Creates a permutation list to merge two sorted sets into a single sorted set.
/// This function corresponds to LAPACK's DLAMRG subroutine.
pub fn dlamrg(n1: usize, n2: usize, a: &[f64], dtrd1: i32, dtrd2: i32, index: &mut [usize]) {
    let mut n1sv = n1;
    let mut n2sv = n2;
    let mut ind1 = if dtrd1 > 0 { 0 } else { n1 - 1 };
    let mut ind2 = if dtrd2 > 0 { n1 } else { n1 + n2 - 1 };
    let mut i = 0;
    let n = n1 + n2;

    while n1sv > 0 && n2sv > 0 {
        if a[ind1] <= a[ind2] {
            index[i] = ind1;
            i += 1;
            if dtrd1 > 0 {
                ind1 += 1;
            } else {
                ind1 -=1;
            }
            n1sv -= 1;
        } else {
            index[i] = ind2;
            i += 1;
             if dtrd2 > 0 {
                ind2 += 1;
            } else {
                ind2 -=1;
            }
            n2sv -= 1;
        }
    }

    // Copy remaining elements
    if n1sv > 0 {
        for _ in 0..n1sv {
            index[i] = ind1;
            i += 1;
             if dtrd1 > 0 {
                ind1 += 1;
            } else {
                ind1 -=1;
            }
        }
    } else {
        for _ in 0..n2sv {
            index[i] = ind2;
            i += 1;
             if dtrd2 > 0 {
                ind2 += 1;
            } else {
                ind2 -=1;
            }
        }
    }
}


/// Computes the i-th eigenvalue and eigenvector of a symmetric rank-one
/// modification of a 2-by-2 diagonal matrix.
/// Corresponds to LAPACK's DLAED5
pub fn dlaed5(
    i: usize,
    d: &[f64],
    z: &[f64],
    delta: &mut [f64],
    rho: f64,
    dlam: &mut f64,
) -> Result<(), i32> {
    if i == 1 {
        let del = d[1] - d[0];
        let w = 1.0 + 2.0 * rho * (z[1] * z[1] - z[0] * z[0]) / del;

        if w > 0.0 {
            let b = del + rho * (z[0] * z[0] + z[1] * z[1]);
            let c = rho * z[0] * z[0] * del;
            let tau = 2.0 * c / (b + (b * b - 4.0 * c).abs().sqrt());
            *dlam = d[0] + tau;
            delta[0] = -z[0] / tau;
            delta[1] = z[1] / (del - tau);
        } else {
            let b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
            let c = rho * z[1] * z[1] * del;
            let tau =
                -2.0 * c / (b + (b * b + 4.0 * c * (if b < 0.0 { -1.0 } else { 1.0 })).sqrt());
            *dlam = d[1] + tau;
            delta[0] = -z[0] / (del + tau);
            delta[1] = -z[1] / tau;
        }

        let temp = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
        delta[0] /= temp;
        delta[1] /= temp;
    } else if i == 2 {
        let del = d[1] - d[0];
        let b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
        let c = rho * z[1] * z[1] * del;

        let tau = if b > 0.0 {
            (b + (b * b + 4.0 * c).sqrt()) / 2.0
        } else {
            2.0 * c / (-b + (b * b + 4.0 * c).sqrt())
        };

        *dlam = d[1] + tau;
        delta[0] = -z[0] / (del + tau);
        delta[1] = -z[1] / tau;
        let temp = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
        delta[0] /= temp;
        delta[1] /= temp;
    }

    Ok(())
}


pub fn dlaed6(
    kniter: i32,
    orgati: bool,
    rho: f64,
    d: &mut [f64],
    z: &mut [f64],
    finit: f64,
    tau: &mut f64,
    info: &mut i32,
) {
    *info = 0;
    let mut lbd = if orgati { d[1] } else { d[0] };
    let mut ubd = if orgati { d[2] } else { d[1] };
    
    if finit < 0.0 {
        lbd = 0.0;
    } else {
        ubd = 0.0;
    }
    
    
    *tau = 0.0;
    if kniter == 2 {
        if orgati {
            let temp = (d[2] - d[1]) / 2.0;
            let c = rho + z[0] / ((d[0] - d[1]) - temp);
            let a = c * (d[1] + d[2]) + z[1] + z[2];
            let b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
            *tau = if c == 0.0 {
                b / a
            } else if a <= 0.0 {
                (a - (a * a - 4.0 * b * c).abs().sqrt()) / (2.0 * c)
            } else {
                2.0 * b / (a + (a * a - 4.0 * b * c).abs().sqrt())
            };
        } else {
            let temp = (d[0] - d[1]) / 2.0;
            let c = rho + z[2] / ((d[2] - d[1]) - temp);
            let a = c * (d[0] + d[1]) + z[0] + z[1];
            let b = c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0];
             *tau = if c == 0.0 {
                b / a
            } else if a <= 0.0 {
                (a - (a * a - 4.0 * b * c).abs().sqrt()) / (2.0 * c)
            } else {
                2.0 * b / (a + (a * a - 4.0 * b * c).abs().sqrt())
            };
        }

        if *tau < lbd || *tau > ubd {
            *tau = (lbd + ubd) / 2.0;
        }

        if d[0] == *tau || d[1] == *tau || d[2] == *tau {
            *tau = 0.0;
        } else {
            let temp = finit + *tau*z[0]/(d[0]*(d[0] - *tau)) + *tau*z[1]/(d[1]*(d[1]-*tau)) + *tau*z[2]/(d[2]*(d[2]-*tau));
            if temp <= 0.0 { lbd = *tau; } else { ubd = *tau; };
        }
        
    }


    let eps = dlamch('E');
    let base = dlamch('B');
    let small1 = base.powf(dlamch('S').log(base) / 3.0);
    let sminv1 = 1.0 / small1;
    let small2 = small1 * small1;
    let sminv2 = sminv1 * sminv1;

    let mut dscale = d.to_vec();
    let mut zscale = z.to_vec();
    let mut scalfac = 1.0;
    let mut scale = false;

    let temp = if orgati {
        (d[1] - *tau).abs().min((d[2] - *tau).abs())
    } else {
        (d[0] - *tau).abs().min((d[1] - *tau).abs())
    };

    if temp <= small1 {
        scale = true;
        let (scalfac, sclinv) = if temp <= small2 {
            (sminv2, small2)
        } else {
            (sminv1, small1)
        };

        for i in 0..3 {
            dscale[i] *= scalfac;
            zscale[i] *= scalfac;
        }
        *tau *= scalfac;
        lbd *= scalfac;
        ubd *= scalfac;
    }

    let mut f = finit;
    let mut df = 0.0;
    let mut ddf = 0.0;
    for j in 0..3 {
        let temp = if scale { 1.0 / (dscale[j] - *tau) } else { 1.0 / (d[j] - *tau) };
        if !temp.is_finite() { continue; }
        let temp1 = if scale { zscale[j] * temp } else { z[j] * temp };
        f += *tau * temp1 / if scale { dscale[j] } else { d[j] };
        df += temp1 * temp;
        ddf += df * temp;
    }
    
    let maxit = 20;
    for _ in 0..maxit {
         if f.abs() <= eps * finit.abs() || ubd-lbd <= 2.0*eps { break; };

         // Gragg-Thornton-Warner cubic convergent scheme
         let temp1 = if orgati { dscale[1] - *tau } else { dscale[0] - *tau };
         let temp2 = if orgati { dscale[2] - *tau } else { dscale[1] - *tau };

         let a = (temp1 + temp2) * f - temp1 * temp2 * df;
         let b = temp1 * temp2 * f;
         let c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;

         let temp = a.abs().max(b.abs()).max(c.abs());
         let a = a/temp; let b = b/temp; let c=c/temp;
        
        let eta = if c==0.0 {
             b/a
         } else if a <= 0.0 {
             (a - (a*a - 4.0*b*c).sqrt()) / (2.0*c)
         } else {
             2.0*b / (a + (a*a-4.0*b*c).sqrt())
         };
        
        if f*eta >= 0.0 {
            *tau = -f/df;
        }

        *tau += eta;
        if *tau < lbd || *tau > ubd { *tau = (lbd + ubd) / 2.0 };

        // Update f, df, ddf for next iteration
        f = finit; df = 0.0; ddf=0.0;
        for j in 0..3 {
             let temp = if scale { 1.0 / (dscale[j] - *tau) } else { 1.0 / (d[j] - *tau) };
             if !temp.is_finite() { continue; }
             let temp1 = if scale { zscale[j] * temp } else { z[j] * temp };
             f += *tau * temp1 / if scale { dscale[j] } else { d[j] };
             df += temp1 * temp;
             ddf += df * temp;
        }
    }

    if *tau < lbd || *tau > ubd {
        *info = 1;
    };

    if scale {
        *tau /= scalfac;
    }
}


/// Finds the roots of the secular equation and updates the eigenvectors.
/// This function corresponds to LAPACK's DLAED3 subroutine.  It's used when the original matrix is tridiagonal.
pub fn dlaed3(
    k: usize,
    n: usize,
    n1: usize,
    d: &mut [f64],
    q: &mut [Vec<f64>],
    ldq: usize,
    rho: f64,
    dlamda: &mut [f64],
    q2: &[Vec<f64>],
    indx: &[usize],
    ctot: &[usize],
    w: &mut [f64],
    s: &mut [Vec<f64>],
) -> i32 {
    let mut info = 0;

    if k == 0 {
        return info;
    }

    // Adjust dlamda for better accuracy
    for i in 0..k {
        dlamda[i] = dlamc3(dlamda[i], dlamda[i]) - dlamda[i];
    }

    for j in 0..k {
        let mut info_dlaed4 = 0;  // Initialize info before calling dlaed4

        let mut q_col = vec![0.0; k];
        for i in 0..k {
            q_col[i] = q[i][j];
        }
        info_dlaed4 = dlaed4(&dlamda[..k], &w[..k], &q_col, rho, d, q);
        
        // If dlaed4 failed, set info and return
        if info_dlaed4 != 0 {
            info = 1; // Or potentially more specific error code
            return info; 
        }
    }

    if k == 1 {
        // Return early for k=1 case
        return info;
    }

    if k == 2 {
        // Handle the 2x2 case by sorting eigenvectors
        for j in 0..k {
            let ii1 = indx[0]-1;
            let ii2 = indx[1]-1;
            let w1 = q[0][j];
            let w2 = q[1][j];
            q[0][j] = w1.min(w2); // Ascending order
            q[1][j] = w1.max(w2);
        }

         return info;
    }


    // Update w
    dcopy(k, w, 1, &mut s[0], 1);

    for j in 0..k {
        for i in 0..k {
            if i != j {
                w[i] *= q[i][j] / (dlamda[i] - dlamda[j]);
            }
        }
    }

    for i in 0..k {
        w[i] = -w[i].sqrt().copysign(s[0][i]);
    }


    for j in 0..k {
        for i in 0..k {
            s[0][i] = w[i] / q[i][j];
        }
        let temp = dnrm2(k, &s[0], 1);
        for i in 0..k {
            let ii = indx[i] - 1;
            q[i][j] = s[0][ii] / temp;
        }
    }


    // Compute the updated eigenvectors
    let n2 = n - n1;
    let n12 = ctot[0] + ctot[1];
    let n23 = ctot[1] + ctot[2];


    dlacpy('A', n23, k, &q[ctot[0]..], ldq, &mut s[..n23], n23);


    if n23 > 0 {
        let result = dgemm(&q2[n1 * n12..], &s[..n23]);
        for (i, row) in result.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                q[n1 + i][j] = *val;
            }
        }
    } else {
        dlaset('A', n2, k, 0.0, 0.0, &mut q[n1..]);
    }

    dlacpy('A', n12, k, q, ldq, &mut s[..n12], n12);
    let result = dgemm(q2, &mut s[..n12]);


    info
}


/// Finds the roots of the secular equation and updates the eigenvectors.
/// Used when the original matrix is dense.
pub fn dlaed9(
    k: usize,
    kstart: usize,
    kstop: usize,
    n: usize,
    d: &mut [f64],
    q: &mut [Vec<f64>],
    ldq: usize,
    rho: f64,
    dlamda: &mut [f64],
    w: &mut [f64],
    s: &mut [Vec<f64>],
    lds: usize,
) -> Result<(), i32> {
    // Test the input parameters
    if k > n {
        return Err(-4); // N must be >= K
    }
    if kstart < 1 || kstart > k.max(1) {
        return Err(-2);
    }
    if kstop < kstart || kstop > k.max(1) {
        return Err(-3);
    }
    if ldq < n.max(1) {
        return Err(-7);
    }
    if lds < k.max(1) {
        return Err(-12);
    }

    // Quick return if possible
    if k == 0 {
        return Ok(());
    }

    // Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can
    // be computed with high relative accuracy. This is needed to combat
    // problems with machines that lack a guard digit in add/subtract. Lol.
    for i in 0..n {
        dlamda[i] = dlamc3(dlamda[i], dlamda[i]) - dlamda[i];
    }

    // Compute eigenvalues of the modified secular equation
    for j in kstart-1..kstop {
        let mut d_sub = vec![0.0; k];
        let mut z_out = vec![vec![0.0; k]; k];
        
        // Call dlaed4 with the correct arguments
        // Note: dlaed4 takes the secular equation parameters and returns eigenvalue/vectors
        if dlaed4(&dlamda, &dlamda, w, rho, &mut d_sub, &mut z_out) != 0 {
            return Err(1); // Eigenvalue did not converge
        }
        
        // Copy the computed eigenvalue
        d[j] = d_sub[0];
        
        // Copy the computed eigenvector to Q
        for i in 0..k {
            q[i][j] = z_out[i][0];
        }
    }

    // If we only have a small problem (k ≤ 2), we're done after copying
    if k == 1 || k == 2 {
        for i in 0..k {
            for j in 0..k {
                s[j][i] = q[j][i];
            }
        }
        return Ok(());
    }

    // For larger problems, we need to compute updated W
    // First, copy W to S (we'll use first column of S as workspace)
    dcopy(k, w, 1, &mut s[0], 1);

    // Initialize W(I) = Q(I,I)
    for i in 0..k {
        w[i] = q[i][i];
    }

    // Compute updated W values
    for j in 0..k {
        for i in 0..j {
            w[i] *= q[i][j] / (dlamda[i] - dlamda[j]);
        }
        for i in j+1..k {
            w[i] *= q[i][j] / (dlamda[i] - dlamda[j]);
        }
    }

    // Update eigenvectors
    for i in 0..k {
        let sign = if s[0][i] >= 0.0 { 1.0 } else { -1.0 };
        w[i] = sign * (-w[i]).abs().sqrt();
    }

    // Compute eigenvectors of the modified rank-1 modification
    for j in 0..k {
        for i in 0..k {
            q[i][j] = w[i] / q[i][j];
        }
        // Normalize using DNRM2
        let temp = dnrm2(k, &q[j], 1);
        for i in 0..k {
            s[i][j] = q[i][j] / temp;
        }
    }

    Ok(())
}

/// Merges eigenvalues and deflates secular equation. Used when the original matrix is dense.
pub fn dlaed8(
    icompq: i32,
    k: &mut usize,
    n: usize, 
    qsiz: usize,
    d: &mut [f64],
    q: &mut [Vec<f64>],
    ldq: usize,
    indxq: &mut [usize],
    rho: &mut f64,
    cutpnt: usize,
    z: &mut [f64],
    dlamda: &mut [f64],
    q2: &mut [Vec<f64>],
    ldq2: usize,
    w: &mut [f64],
    perm: &mut [usize],
    givptr: &mut usize,
    givcol: &mut Vec<Vec<usize>>,
    givnum: &mut Vec<Vec<f64>>,
    indxp: &mut [usize],
    indx: &mut [usize],
) -> Result<(), i32> {
    // Constants
    const MONE: f64 = -1.0;
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const EIGHT: f64 = 8.0;

    // Test the input parameters
    if icompq < 0 || icompq > 1 {
        return Err(-1);
    }
    if n < 0 {
        return Err(-3);
    }
    if icompq == 1 && qsiz < n {
        return Err(-4);
    }
    if ldq < n.max(1) {
        return Err(-7);
    }
    if cutpnt < 1.min(n) || cutpnt > n {
        return Err(-10);
    }
    if ldq2 < n.max(1) {
        return Err(-14);
    }

    *givptr = 0;

    // Quick return if possible
    if n == 0 {
        return Ok(());
    }

    let n1 = cutpnt;
    let n2 = n - n1;
    let n1p1 = n1 + 1;

    // Normalize z vector
    if *rho < ZERO {
        dscal(n2, MONE, &mut z[n1p1..], 1);
    }

    let t = ONE / (TWO.sqrt());
    for j in 0..n {
        indx[j] = j;
    }
    dscal(n, t, z, 1);
    *rho = (TWO * *rho).abs();

    // Sort eigenvalues into increasing order
    for i in cutpnt..n {
        indxq[i] += cutpnt;
    }
    
    // Copy values to work arrays
    for i in 0..n {
        dlamda[i] = d[indxq[i]];
        w[i] = z[indxq[i]];
    }

    // Merge sorted subsets
    dlamrg(n1, n2, dlamda, 1, 1, indx);

    // Reorder based on sorted indices
    for i in 0..n {
        d[i] = dlamda[indx[i]];
        z[i] = w[indx[i]];
    }

    // Calculate deflation tolerance
    let imax = idamax(n, z, 1);
    let jmax = idamax(n, d, 1);
    let eps = dlamch('E');
    let tol = EIGHT * eps * d[jmax].abs();

    // If rank-1 modifier is small enough, only reorder Q
    if *rho * z[imax].abs() <= tol {
        *k = 0;
        if icompq == 0 {
            for j in 0..n {
                perm[j] = indxq[indx[j]];
            }
        } else {
            for j in 0..n {
                perm[j] = indxq[indx[j]];
                for i in 0..qsiz {
                    q2[i][j] = q[i][perm[j]];
                }
            }
            dlacpy('A', qsiz, n, q2, ldq2, q, ldq);
        }
        return Ok(());
    }

    // Handle deflation of eigenvalues
    *k = 0;
    let mut k2 = n + 1;
    let mut jlam = 0;
    let mut j = 0;

    // Main deflation detection loop
    'outer: while j < n {
        if *rho * z[j].abs() <= tol {
            // Deflate due to small z component
            k2 -= 1;
            indxp[k2 - 1] = j;
            if j == n - 1 {
                break;
            }
        } else {
            jlam = j;
            j += 1;

            while j < n {
                if *rho * z[j].abs() <= tol {
                    k2 -= 1;
                    indxp[k2 - 1] = j;
                } else {
                    // Check if eigenvalues are close enough to deflate
                    let s = z[jlam];
                    let c = z[j];
                    let tau = dlapy2(c, s);
                    let t = d[j] - d[jlam];
                    let c = c / tau;
                    let s = -s / tau;

                    if (t * c * s).abs() <= tol {
                        // Deflation is possible
                        z[j] = tau;
                        z[jlam] = ZERO;

                        // Record Givens rotation
                        *givptr += 1;
                        givcol[0][*givptr - 1] = indxq[indx[jlam]];
                        givcol[1][*givptr - 1] = indxq[indx[j]];
                        givnum[0][*givptr - 1] = c;
                        givnum[1][*givptr - 1] = s;

                        if icompq == 1 {
                            // Apply rotation column by column
                            let col1 = indxq[indx[jlam]];
                            let col2 = indxq[indx[j]];
                            for i in 0..qsiz {
                                let temp = c * q[i][col1] + s * q[i][col2];
                                q[i][col2] = -s * q[i][col1] + c * q[i][col2];
                                q[i][col1] = temp;
                            }
                        }

                        let t = d[jlam] * c * c + d[j] * s * s;
                        d[j] = d[jlam] * s * s + d[j] * c * c;
                        d[jlam] = t;

                        k2 -= 1;
                        let mut i = 1;
                        while k2 + i <= n {
                            if d[jlam] < d[indxp[k2 + i - 1]] {
                                indxp[k2 + i - 2] = indxp[k2 + i - 1];
                                indxp[k2 + i - 1] = jlam;
                                i += 1;
                            } else {
                                indxp[k2 + i - 2] = jlam;
                                break;
                            }
                        }
                        if k2 + i > n {
                            indxp[k2 + i - 2] = jlam;
                        }

                        jlam = j;
                    } else {
                        *k += 1;
                        w[*k - 1] = z[jlam];
                        dlamda[*k - 1] = d[jlam];
                        indxp[*k - 1] = jlam;
                        jlam = j;
                    }
                }
                j += 1;
            }
            
            // Record last eigenvalue
            *k += 1;
            w[*k - 1] = z[jlam];
            dlamda[*k - 1] = d[jlam];
            indxp[*k - 1] = jlam;
            
            break 'outer;
        }
        j += 1;
    }

    // Sort eigenvalues and eigenvectors into dlamda and q2
    if icompq == 0 {
        for j in 0..n {
            let jp = indxp[j];
            dlamda[j] = d[jp];
            perm[j] = indxq[indx[jp]];
        }
    } else {
        for j in 0..n {
            let jp = indxp[j];
            dlamda[j] = d[jp];
            perm[j] = indxq[indx[jp]];
            for i in 0..qsiz {
                q2[i][j] = q[i][perm[j]];
            }
        }
    }

    // Copy deflated eigenvalues and vectors back
    if *k < n {
        if icompq == 0 {
            dcopy(n - *k, &dlamda[*k..], 1, &mut d[*k..], 1);
        } else {
            dcopy(n - *k, &dlamda[*k..], 1, &mut d[*k..], 1);
            dlacpy('A', qsiz, n - *k, &q2[*k..], ldq2, &mut q[*k..], ldq);
        }
    }

    Ok(())
}


/// Computes the updated eigensystem of a diagonal matrix after modification by a 
/// rank-one symmetric matrix. Used when the original matrix is dense.
pub fn dlaed7(
    icompq: i32,
    n: usize,
    qsiz: usize,
    tlvls: usize,
    curlvl: usize,
    curpbm: usize,
    d: &mut [f64],
    q: &mut [Vec<f64>],
    ldq: usize,
    indxq: &mut [usize],
    rho: &mut f64,
    cutpnt: usize,
    qstore: &mut [Vec<f64>],
    qptr: &mut [usize],
    prmptr: &mut [usize],
    perm: &mut [usize],
    givptr: &mut [usize],
    givcol: &mut Vec<Vec<usize>>,
    givnum: &mut Vec<Vec<f64>>,
    work: &mut [f64],
    iwork: &mut [usize],
) -> Result<(), i32> {
    // Test the input parameters
    if icompq < 0 || icompq > 1 {
        return Err(-1);
    }
    if n < 0 {
        return Err(-2);
    }
    if icompq == 1 && qsiz < n {
        return Err(-4);
    }
    if ldq < n.max(1) {
        return Err(-9);
    }
    if cutpnt < 1.min(n) || cutpnt > n {
        return Err(-12); 
    }

    // Quick return if possible
    if n == 0 {
        return Ok(());
    }

    let ldq2 = if icompq == 1 { qsiz } else { n };

    // Set up workspace ranges 
    let iz = 0;
    let idlmda = iz + n;
    let iw = idlmda + n;
    let iq2 = iw + n;
    let is = iq2 + n * ldq2;

    let indx = 0;
    let indxp = indx + n;
    let coltyp = indxp + n;

    // Set up pointers into workspace
    let (work_left, work_right) = work.split_at_mut(idlmda);
    let (work_mid, work_end) = work_right.split_at_mut(iw - idlmda);
    let z = &mut work_left[iz..iz+n];
    let dlamda = work_mid;
    let w = &mut work_end[..n];

    // Split iwork
    let (iwork_left, iwork_right) = iwork.split_at_mut(indxp);
    let indx_p = &mut iwork_left[indx..];
    let indxp_p = &mut iwork_right[..n];

    // Form z-vector
    let mut ptr = 1 + 2_usize.pow(tlvls as u32);
    for i in 1..curlvl {
        ptr += 2_usize.pow((tlvls - i) as u32);
    }
    let curr = ptr + curpbm;
    
    if curlvl == tlvls {
        qptr[curr] = 1;
        prmptr[curr] = 1;
        givptr[curr] = 1;
    }
    
    // Execute DLAED8 to merge and deflate eigenvalues
    let mut k = 0;
    let result = dlaed8(
        icompq,
        &mut k,
        n,
        qsiz,
        d,
        q,
        ldq,
        indxq,
        rho,
        cutpnt,
        z,
        dlamda,
        qstore,
        ldq2,
        w,
        &mut perm[prmptr[curr]..],
        &mut givptr[curr+1],
        givcol,
        givnum,
        indxp_p,
        indx_p,
    );

    if let Err(e) = result {
        return Err(e);
    }

    prmptr[curr + 1] = prmptr[curr] + n;
    givptr[curr + 1] = givptr[curr + 1] + givptr[curr];

    // Solve Secular Equation if k is positive
    if k > 0 {
        // DLAED9: Need to construct proper matrices from workspace
        let mut q_temp = vec![vec![0.0; k]; k];
        let result = dlaed9(
            k,
            1,
            k,
            n,
            d,
            &mut q_temp,
            k,
            *rho,
            dlamda,
            w,
            qstore,
            k,
        );

        if let Err(e) = result {
            return Err(e);
        }

        if icompq == 1 {
            // Copy q_temp to proper locations
            for i in 0..qsiz {
                for j in 0..k {
                    q[i][j] = q_temp[i][j];
                }
            }
        }

        qptr[curr + 1] = qptr[curr] + k * k;

        // Prepare INDXQ sorting permutation
        let n1 = k;
        let n2 = n - k;
        dlamrg(n1, n2, d, 1, -1, indxq);
    } else {
        qptr[curr + 1] = qptr[curr];
        for i in 0..n {
            indxq[i] = i;
        }
    }

    Ok(())
}


/*
Not yet implemented functions:

- DLAEDA
  - Description: Computes the Z vector determining the rank-one modification of the diagonal matrix. It effectively prepares the data needed for the rank-one update in the divide step of the algorithm.
  - When it's called: Used within `dlaed0` during the divide and conquer process to set up vectors necessary for merging eigenvalues and eigenvectors from subproblems.

- DLASR
  - Description: Applies a sequence of plane rotations to a general rectangular matrix. This function can apply rotations from the left or the right, in forward or backward order, and with different pivot strategies.
  - When it's called: Used in `dsteqr` to apply accumulated rotations to eigenvectors during the QR iteration process for computing eigenvalues and eigenvectors of a tridiagonal matrix.
*/
