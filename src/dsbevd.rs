// src/dsbevd.rs

// We ONLY care about large matrices, and we ALWAYS want both eigenvectors and eigenvalues

use rayon::prelude::*;
use std::cmp::{max, min};

/// Represents a real symmetric banded matrix.
#[derive(Clone)]
pub struct SymmetricBandedMatrix {
    n: usize,          // Order of the matrix
    kd: usize,         // Number of subdiagonals (assuming 'L' storage)
    ab: Vec<Vec<f64>>, // The lower triangle of the symmetric band matrix A, stored in banded format
}

pub struct EigenResults {
    pub eigenvalues: Vec<f64>,       // Eigenvalues in ascending order
    pub eigenvectors: Vec<Vec<f64>>, // Eigenvectors corresponding to the eigenvalues
}

#[derive(Debug)]
pub struct Error(i32);

impl From<&'static str> for Error {
    fn from(_: &'static str) -> Error {
        Error(-1) // Simple conversion of all string errors to Error(-1)
    }
}

impl From<i32> for Error {
    fn from(e: i32) -> Error {
        Error(e)
    }
}

impl SymmetricBandedMatrix {
    /// Creates a new symmetric banded matrix.
    pub fn new(n: usize, kd: usize, ab: Vec<Vec<f64>>) -> Self {
        assert!(
            ab.len() == kd + 1,
            "Incorrect number of rows in 'ab' matrix"
        );
        assert!(
            ab[0].len() == n,
            "Incorrect number of columns in 'ab' matrix"
        );
        SymmetricBandedMatrix { n, kd, ab }
    }

    /// Computes all eigenvalues and eigenvectors of the symmetric banded matrix.
    pub fn dsbevd(&self) -> Result<EigenResults, Error> {
        let safmin = f64::MIN_POSITIVE;
        let eps = f64::EPSILON;
        let smlnum = safmin / eps;
        let bignum = 1.0 / smlnum;
        let rmin = smlnum.sqrt();
        let rmax = bignum.sqrt();

        // Extract diagonal and subdiagonal elements from banded matrix
        let mut d = vec![0.0; self.n];
        let mut d = vec![0.0; self.n];
        let mut e = vec![0.0; self.n - 1];
        for i in 0..self.n {
            d[i] = self.ab[0][i]; // Diagonal is in first row of banded storage
            if i < self.n - 1 && self.kd > 0 {  // Only access ab[1] if we have off-diagonal elements
                e[i] = self.ab[1][i];
            }
        }
        let anrm = dlanst('M', self.n, &d, &e);

        let mut scale = 1.0;
        let mut iscale = 0;

        if anrm > 0.0 && anrm < rmin {
            iscale = 1;
            scale = rmin / anrm;
        } else if anrm > rmax {
            iscale = 2;
            scale = rmax / anrm;
        }

        let mut working_matrix = self.clone();
        if scale != 1.0 {
            dlascl(&mut working_matrix.ab, anrm, scale)?;
        }

        // Call dsbtrd with explicit arguments
        let mut d = vec![0.0; self.n];
        let mut e = vec![0.0; self.n - 1];
        let mut q = vec![vec![0.0; self.n]; self.n];

        // Initialize Q to identity matrix before calling dsbtrd
        for i in 0..self.n {
            q[i][i] = 1.0;
        }

        dsbtrd(
            'U',
            working_matrix.n,
            working_matrix.kd,
            &mut working_matrix.ab,
            &mut d,
            &mut e,
            &mut q,
        );

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
}

/// Generates a vector of plane rotations for 2-by-2 matrices.
///
/// For i = 1,2,...,n:
///    [  c(i)  s(i) ] [ x(i) ] = [ a(i) ]
///    [ -s(i)  c(i) ] [ y(i) ] = [  0  ]
///
/// Returns updated x values (now containing a(i)) and c,s rotation values
pub fn dlargv(
    n: usize,
    x: &mut [f64],
    incx: usize,
    y: &mut [f64],
    incy: usize,
    c: &mut [f64],
    incc: usize,
) {
    if n == 0 {
        return;
    }
    
    debug_assert!(incx > 0, "incx must be positive");
    debug_assert!(incy > 0, "incy must be positive");
    debug_assert!(incc > 0, "incc must be positive");
    debug_assert!(x.len() >= 1 + (n - 1) * incx, "x array too small");
    debug_assert!(y.len() >= 1 + (n - 1) * incy, "y array too small");
    debug_assert!(c.len() >= 1 + (n - 1) * incc, "c array too small");

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

pub fn dlartv(
    n: usize,
    x: &mut [f64],
    incx: usize,
    y: &mut [f64],
    incy: usize,
    c: &[f64],
    s: &[f64],
    incc: usize,
) {
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
pub fn drot(n: usize, dx: &mut [f64], incx: i32, dy: &mut [f64], incy: i32, c: f64, s: f64) {
    if n == 0 {
        return;
    }

    let mut ix = if incx > 0 { 0 } else { (1 - n as i32) * incx };
    let mut iy = if incy > 0 { 0 } else { (1 - n as i32) * incy };

    for _ in 0..n {
        let temp = c * dx[ix as usize] + s * dy[iy as usize];
        dy[iy as usize] = c * dy[iy as usize] - s * dx[ix as usize];
        dx[ix as usize] = temp;

        ix += incx;
        iy += incy;
    }
}

pub fn dlar2v(
    n: usize,
    x: &mut [f64],
    y: &mut [f64], 
    z: &mut [f64],
    incx: usize,
    c: &[f64],
    s: &[f64], 
    incc: usize,
) -> Result<(), Error> {
    // Early return for empty case to avoid underflow
    if n == 0 {
        return Ok(());
    }
    
    // Validate inputs
    if incx == 0 {
        return Err(Error(-1)); // INCX must be positive
    }
    if incc == 0 { 
        return Err(Error(-1)); // INCC must be positive
    }
    
    // Validate array lengths
    let min_len = 1 + (n.saturating_sub(1)) * incx;
    if x.len() < min_len || y.len() < min_len || z.len() < min_len {
        return Err(Error(-1));
    }
    let min_c_len = 1 + (n.saturating_sub(1)) * incc;
    if c.len() < min_c_len || s.len() < min_c_len {
        return Err(Error(-1)); 
    }
    
    let mut ix = 0;
    let mut ic = 0;
    
    // Process each 2x2 rotation
    for _ in 0..n {
        // Load current values
        let xi = x[ix];
        let yi = y[ix];
        let zi = z[ix];
        let ci = c[ic];
        let si = s[ic];
        
        // Compute intermediates exactly as in LAPACK
        let t1 = si * zi; 
        let t2 = ci * zi;
        let t3 = t2 - si * xi;
        let t4 = t2 + si * yi;
        let t5 = ci * xi + t1;
        let t6 = ci * yi - t1;
        
        // Store results
        x[ix] = ci * t5 + si * t4;
        y[ix] = ci * t6 - si * t3;
        z[ix] = ci * t4 - si * t5;
        
        // Update indices
        ix += incx;
        ic += incc;
    }
    
    Ok(())
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
/// * `Err(Error)` if the algorithm failed to compute eigenvalues
pub fn dstedc(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>]) -> Result<(), Error> {
    let n = d.len();
    
    // Handle base cases first
    if n == 0 {
        return Ok(());
    }
    
    if n == 1 {
        z[0][0] = 1.0;
        return Ok(());
    }

    // Parameters
    let smlsiz = 25; // Minimum size for divide-conquer

    if n <= smlsiz {
        // Use QR algorithm for small matrices
        let mut work = vec![0.0; 2 * n - 1];
        dsteqr('I', n, d, e, z, &mut work)?;
        return Ok(());
    }

    // Scale the matrix if necessary
    let orgnrm = dlanst('M', n, d, e);
    if orgnrm == 0.0 {
        return Ok(());  // Zero matrix case
    }

    let mut subpbs = 1;
    let mut tlvls = 0;
    let mut iwork = vec![0usize; 4 * n];
    let mut work_matrix = vec![vec![0.0; n]; n];

    // Initialize z to identity
    for i in 0..n {
        for j in 0..n {
            z[i][j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    let mut start = 0;
    let mut info = 0;

    // Main divide and conquer loop
    while start < n {
        // Find the end of the current submatrix (look for small subdiagonal)
        let mut end = start;
        let eps = dlamch('E'); // Get machine epsilon
        while end < n - 1 {
            let tiny = eps * (d[end].abs().sqrt() * d[end + 1].abs().sqrt());
            if e[end].abs() <= tiny {
                e[end] = 0.0; // Deflate
                break;
            }
            end += 1;
        }

        // 'end' does not exceed 'n - 1' when used later
        if end >= n {
            end = n - 1;
        }

        // Process submatrix from start to end
        let m = end - start + 1;

        if m == 1 {
            // 1x1 block, already diagonal
            start = end + 1;
            continue;
        }

        if m <= smlsiz {
            // Small subproblem - use STEQR
            let mut d_sub = d[start..=end].to_vec();
            let mut e_sub = e[start..end].to_vec();
            let mut z_sub = vec![vec![0.0; m]; m];
            for i in 0..m {
                z_sub[i][i] = 1.0;
            }
            let mut work_sub = vec![0.0; 4 * m];
            dsteqr('I', m, &mut d_sub, &mut e_sub, &mut z_sub, &mut work_sub)?;
            // Copy results back
            for i in 0..m {
                d[start + i] = d_sub[i];
                for j in 0..m {
                    z[start + i][start + j] = z_sub[i][j];
                }
            }
        } else {
            // Large subproblem - use divide and conquer
            let mid = (start + end) / 2;
            e[mid] = 0.0; // Split the matrix

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
            let right_size = end - mid;
            let mut d_right = d[mid + 1..=end].to_vec();
            let mut e_right = e[mid + 1..end].to_vec();
            let mut z_right = vec![vec![0.0; right_size]; right_size];
            for i in 0..right_size {
                z_right[i][i] = 1.0;
            }
            dstedc(&mut d_right, &mut e_right, &mut z_right)?;

            // Merge solutions using dlaed1
            let k = m;
            let n1 = left_size;
            let n2 = right_size;
            let cutpnt = n1;

            // Form z-vector
            let mut z_vector = vec![0.0; k];
            z_vector[..n1].clone_from_slice(&z_left[n1 - 1]);
            z_vector[n1..].clone_from_slice(&z_right[0]);

            // Copy eigenvalues
            let mut d_work = vec![0.0; k];
            d_work[..n1].copy_from_slice(&d_left);
            d_work[n1..].copy_from_slice(&d_right);

            // Initialize q_work
            let mut q_work = vec![vec![0.0; k]; k];
            for i in 0..n1 {
                for j in 0..n1 {
                    q_work[i][j] = z_left[i][j];
                }
            }
            for i in 0..n2 {
                for j in 0..n2 {
                    q_work[n1 + i][n1 + j] = z_right[i][j];
                }
            }

            // Prepare indxq
            let mut indxq: Vec<usize> = vec![0; n]; // Initialize with appropriate size
            for i in 0..k {
                indxq[i] = i;
            }

            // Workspace arrays
            let mut work_work = vec![0.0; 4 * k + k * k];
            let mut iwork_work = vec![0usize; 3 * k];

            // Define rho (the subdiagonal element used to create the rank-one modification)
            let mut rho = e[mid];

            // Call dlaed1
            dlaed1(
                k,
                &mut d_work,
                &mut q_work,
                k,
                &mut indxq,
                &mut rho,
                cutpnt,
                &mut work_work,
                &mut iwork_work,
            )?;

            // Copy the merged eigenvalues and eigenvectors back
            for i in 0..k {
                d[start + i] = d_work[i];
                for j in 0..k {
                    z[start + i][start + j] = q_work[i][j];
                }
            }
        }

        start = end + 1;
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
pub fn dlaed4(
    d1: &[f64],
    d2: &[f64],
    z: &[f64],
    rho: f64,
    d: &mut [f64], 
    z_out: &mut [Vec<f64>],
) -> Result<(), Error> {
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

            for i in 0..n-1 {
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
                break; // Is this wrong?
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
        for i in 0..n-1 {
            let denom = dlamda_sorted[i] - lambda;
            let temp = if denom.abs() < eps {
                0.0
            } else {
                z_sorted[i] / denom
            };
            z_out[i][j] = if i == j { 1.0 } else { temp };
            norm += temp * temp;
        }
        norm += 1.0; // WRONG (probably)

        // Normalize
        norm = norm.sqrt();
        for i in 0..n-1 {
            z_out[i][j] /= norm;
        }
    }

    Ok(())
}

/// Multiplies q and z matrices to get the eigenvectors of the original matrix.
pub fn dgemm(q: &[Vec<f64>], z: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = q.len();
    let mut result = vec![vec![0.0; n]; n];

    // Use parallelism for large matrices
    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..n {
            row[j] = (0..n).map(|k| q[i][k] * z[k][j]).sum();
        }
    });

    result
}

/// Force a and b to be stored prior to addition
pub fn dlamc3(a: f64, b: f64) -> f64 {
    a + b
}

/// Compute the norm of a symmetric tridiagonal matrix
pub fn dlanst(norm_type: char, n: usize, d: &[f64], e: &[f64]) -> f64 {
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
        }
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
            for i in 1..n - 1 {
                work[i] = e[i - 1].abs() + d[i].abs() + e[i].abs();
            }
            // Last row
            work[n - 1] = e[n - 2].abs() + d[n - 1].abs();

            let mut max_val = 0.0_f64;
            for val in work.iter() {
                max_val = max_val.max(*val);
            }
            max_val
        }
        'F' | 'f' | 'E' | 'e' => {
            // Frobenius norm
            let mut scale = 0.0_f64;
            let mut sumsq = 1.0_f64;

            // Add diagonal elements
            dlassq(n, d, 1, &mut scale, &mut sumsq);

            // Add off-diagonal elements
            if n > 1 {
                dlassq(n - 1, e, 1, &mut scale, &mut sumsq);
                sumsq *= 2.0_f64;
            }

            scale * sumsq.sqrt()
        }
        _ => return 0.0, // Return zero for invalid norm type, matching LAPACK behavior
    }
}

pub fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64, f64) {
    if b == 0.0 {
        // Handle diagonal case specially
        if a.abs() >= c.abs() {
            return (a, c, 1.0, 0.0, 0.0);
        } else {
            return (c, a, 0.0, 1.0, 0.0);
        }
    }
    
    // Compute the eigenvalues
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

    let rt = if adf > ab {
        adf * (1.0 + (ab/adf).powi(2)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf/ab).powi(2)).sqrt()
    } else {
        ab * 2.0_f64.sqrt()
    };

    let (rt1, rt2, sgn1) = if sm < 0.0 {
        let rt1_val = 0.5 * (sm - rt);
        let rt2_val = (acmx / rt1_val) * acmn - (b / rt1_val) * b;
        (rt1_val, rt2_val, -1)
    } else if sm > 0.0 {
        let rt1_val = 0.5 * (sm + rt);
        let rt2_val = (acmx / rt1_val) * acmn - (b / rt1_val) * b;
        (rt1_val, rt2_val, 1)
    } else {
        (0.5 * rt, -0.5 * rt, 1)
    };

    let (cs, sgn2) = if df >= 0.0 {
        (df + rt, 1)
    } else {
        (df - rt, -1)
    };

    let acs = cs.abs();
    let (cs1, sn1) = if acs > ab {
        let ct = -tb / cs;
        let sn1_val = 1.0 / (1.0 + ct * ct).sqrt();
        (ct * sn1_val, sn1_val)
    } else if ab == 0.0 {
        (1.0, 0.0)
    } else {
        let tn = -cs / tb;
        let cs1_val = 1.0 / (1.0 + tn * tn).sqrt();
        (cs1_val, tn * cs1_val)
    };

    let (cs1_final, sn1_final) = if sgn1 == sgn2 {
        (-sn1, cs1)
    } else {
        (cs1, sn1)
    };

    (rt1, rt2, cs1_final, sn1_final, 0.0)
}


/// Safe computation of sqrt(x*x + y*y)
pub fn dlapy2(x: f64, y: f64) -> f64 {
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
pub fn dlaset(uplo: char, m: usize, n: usize, alpha: f64, beta: f64, a: &mut [Vec<f64>]) {
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
        }
        'L' | 'l' => {
            // Lower triangle
            for j in 0..n {
                for i in j + 1..m {
                    a[i][j] = alpha;
                }
                if j < m {
                    a[j][j] = beta;
                }
            }
        }
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
pub fn dgemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[Vec<f64>],
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
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
    for i in (0..n * incx).step_by(incx) {
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
    for i in (0..n * incx).step_by(incx) {
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

    for i in (incx..n * incx).step_by(incx) {
        let abs_val = x[i].abs();
        if abs_val > max_val {
            max_idx = i / incx;
            max_val = abs_val;
        }
    }
    max_idx
}

// Helper function to pass workspace arrays and handle error conditions safely
pub fn dsbtrd_wrapper(
    uplo: char,
    n: usize,
    kd: usize,
    ab: &mut [Vec<f64>],
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    assert!(n > 0, "Matrix dimension must be positive");
    assert!(kd >= 0 && kd < n, "Band width must be valid");
    assert!(ab.len() >= kd + 1, "Invalid array dimension for ab");
    assert!(ab[0].len() >= n, "Invalid array dimension for ab");

    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n - 1];
    let mut q = vec![vec![0.0; n]; n];

    dsbtrd(uplo, n, kd, ab, &mut d, &mut e, &mut q);

    (d, e, q)
}

// Helper function for safe mutable band access
pub fn get_mut_bands(
    ab: &mut [Vec<f64>],
    k1: usize,
    k2: usize,
    start: usize,
    len: usize,
) -> (&mut [f64], &mut [f64]) {
    assert!(k1 != k2, "Cannot borrow same band twice");
    let (min_k, max_k) = if k1 < k2 { (k1, k2) } else { (k2, k1) };
    let (lower, upper) = ab.split_at_mut(max_k);
    if k1 < k2 {
        (
            &mut lower[k1][start..start + len],
            &mut upper[0][start..start + len],
        )
    } else {
        (
            &mut upper[0][start..start + len],
            &mut lower[k2][start..start + len],
        )
    }
}

pub fn dlartg(f: f64, g: f64) -> (f64, f64) {
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

pub fn dsbtrd(
    uplo: char,
    n: usize,
    kd: usize,
    ab: &mut [Vec<f64>],
    d: &mut [f64],
    e: &mut [f64],
    q: &mut [Vec<f64>],
) -> Result<(), Error> {
    let kd1 = kd + 1;
    let kdm1 = if kd >= 1 { kd - 1 } else { 0 };
    let mut nr: usize = 0;
    let mut j1: usize = kd1;
    let mut j2: usize = 1;
    let mut work = vec![0.0; n];
    let mut rotations = Vec::new();

    // Initialize Q to identity
    dlaset('F', n, n, 0.0, 1.0, q);

    if uplo == 'U' {
        for j in 0..n - 2 {
            for k in (0..kd - 1).rev() {
                j1 = if j1 > kd1 { j1 - kd1 } else { 0 };
                j2 = if j2 > kd1 { j2 - kd1 } else { 0 };

                if nr > 0 {
                    let j_start = j1.saturating_sub(1);
                    let len = nr.min(ab[0].len() - j_start);

                    // Store rotations
                    rotations.clear();
                    for idx in 0..nr {
                        j1.checked_sub(kd)
                            .and_then(|x| x.checked_sub(1))
                            .and_then(|x| x.checked_add(idx.checked_mul(kd)?))
                            .ok_or(Error(-1))?;


                        // j is within the bounds of BOTH ab[kd] AND ab[kd-1]
                        if j < ab[kd].len() && j < ab[kd - 1].len() {
                            // bound check
                            rotations.push((ab[kd][j], ab[kd - 1][j]));
                        }
                    }

                    // Apply stored rotations
                    for (idx, &(x, y)) in rotations.iter().enumerate() {
                        let (cs, sn) = dlartg(x, y);
                        let start = j_start + idx;
                        if start + 1 < ab[0].len() {
                            let (band1, band2) = get_mut_bands(ab, k, k + 1, start, 1);
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
                                v1.push(ab[kd - l][j]);
                                v2.push(ab[kd - l + 1][j]);
                            }
                        }

                        for i in 0..v1.len() {
                            let (cs, sn) = dlartg(v1[i], v2[i]);
                            let j = j_start + i;
                            if j < ab[0].len() {
                                ab[kd - l][j] = cs * v1[i] + sn * v2[i];
                                ab[kd - l + 1][j] = -sn * v1[i] + cs * v2[i];
                            }
                        }
                    }
                }

                if k < kdm1 && j + k < n && j + k + 1 < n {  // Bounds check for matrix edges
                    let x = ab[kd - k][j + k];
                    let y = ab[kd - k - 1][j + k + 1];
                    let (cs, sn) = dlartg(x, y);

                    // Apply rotation
                    let len = k.min(n - (j + k) - 1);
                    let mut temp_row1 = Vec::new();
                    let mut temp_row2 = Vec::new();

                    for i in 0..len {
                        temp_row1.push(ab[kd - k][j + k + i]);
                        temp_row2.push(ab[kd - k - 1][j + k + 1 + i]);
                    }

                    for i in 0..len {
                        ab[kd - k][j + k + i] = cs * temp_row1[i] + sn * temp_row2[i];
                        ab[kd - k - 1][j + k + 1 + i] = -sn * temp_row1[i] + cs * temp_row2[i];
                    }

                    ab[kd - k - 1][j + k + 1] = 0.0;
                }

                nr += 1;
            }

            if j < n - 1 {
                d[j] = ab[kd][j];
                e[j] = ab[kdm1][j + 1];
            }
        }
        d[n - 1] = ab[kd][n - 1];
    } else {
        for j in 0..n - 2 {
            for k in (0..kd - 1).rev() {
                j1 = if j1 > kd1 { j1 - kd1 } else { 0 };
                j2 = if j2 > kd1 { j2 - kd1 } else { 0 };

                if nr > 0 {
                    let j_start = j1;
                    let len = nr.min(ab[0].len() - j_start);

                    let mut temp_bands = Vec::new();
                    for l in 0..=k {
                        let mut band = Vec::new();
                        for idx in 0..len {
                            band.push(ab[l][j_start + idx]);
                        }
                        temp_bands.push(band);
                    }

                    for l in 0..k {
                        for i in 0..len {
                            let (cs, sn) = dlartg(temp_bands[l][i], temp_bands[l + 1][i]);
                            if j_start + i < ab[0].len() {
                                ab[l][j_start + i] =
                                    cs * temp_bands[l][i] + sn * temp_bands[l + 1][i];
                                ab[l + 1][j_start + i] =
                                    -sn * temp_bands[l][i] + cs * temp_bands[l + 1][i];
                            }
                        }
                    }
                }

                if k < kdm1 {
                    let (cs, sn) = dlartg(ab[k + 1][j], ab[k + 2][j]);

                    let len = k.min(n - j - 1);
                    let mut temp1 = Vec::new();
                    let mut temp2 = Vec::new();

                    for i in 0..len {
                        temp1.push(ab[k + 2][j + i]);
                        temp2.push(ab[k + 1][j + 1 + i]);
                    }

                    for i in 0..len {
                        ab[k + 2][j + i] = cs * temp1[i] + sn * temp2[i];
                        ab[k + 1][j + 1 + i] = -sn * temp1[i] + cs * temp2[i];
                    }

                    ab[k + 2][j] = 0.0;
                }

                nr += 1;
            }

            if j < n - 1 {
                d[j] = ab[0][j];
                e[j] = ab[1][j];
            }
        }
        d[n - 1] = ab[0][n - 1];
    }
    Ok(())
}


pub fn dsteqr(
    compz: char,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    z: &mut [Vec<f64>],
    work: &mut [f64],
) -> Result<(), Error> {
    // Validate inputs
    if compz != 'I' {
        return Err(Error(-1));
    }
    if n == 0 {
        return Ok(());
    }
    if d.len() < n || (n > 1 && e.len() < n-1) || z.len() < n || 
       work.len() < 2*n || z.iter().any(|row| row.len() < n) {
        return Err(Error(-2));
    }

    if n == 1 {
        z[0][0] = 1.0;
        return Ok(());
    }

    // Machine constants
    let eps = dlamch('E');
    let eps2 = eps * eps;
    let safmin = dlamch('S');
    let safmax = 1.0 / safmin;
    let ssfmax = (safmax).sqrt() / 3.0;
    let ssfmin = (safmin).sqrt() / eps2;

    let nm1 = n - 1;
    let nmaxit = n * 30;
    let mut jtot = 0;

    // Initialize Z to identity
    dlaset('F', n, n, 0.0, 1.0, z);

    let mut l1 = 1;  // FORTRAN 1-based indexing ???
    while l1 <= n {  // Use <= for 1-based comparison
        // Zero superdiagonal element
        if l1 > 1 {
            e[l1-2] = 0.0;  // Convert to 0-based
        }

        // Find next block
        let mut m = l1;
        if l1 <= nm1 {
            for m_iter in l1..=nm1 {
                m = m_iter;
                let tst = e[m-1].abs();  // Convert to 0-based
                let tol = (d[m-1].abs() * d[m].abs()).sqrt() * eps;
                if tst <= tol {
                    e[m-1] = 0.0;
                    break;
                }
            }
        }

        let mut l = l1 - 1;  // Convert to 0-based
        let mut lend = m - 1;  // Convert to 0-based
        let lsv = l;
        let lendsv = lend;

        if lend == l {
            l1 = m + 1;  // Next block
            continue;
        }

        // Scale submatrix
        let anorm = dlanst('M', lend-l+1, &d[l..=lend], &e[l..lend]);
        let mut iscale = 0;

        if anorm > 0.0 {
            if anorm > ssfmax {
                iscale = 1;
                // Scale down
                let mut d_mat = vec![vec![0.0; 1]; lend-l+1];
                let mut e_mat = vec![vec![0.0; 1]; lend-l];
                for i in 0..=lend-l {
                    d_mat[i][0] = d[l+i];
                }
                for i in 0..lend-l {
                    e_mat[i][0] = e[l+i];
                }
                dlascl(&mut d_mat, anorm, ssfmax)?;
                dlascl(&mut e_mat, anorm, ssfmax)?;
                for i in 0..=lend-l {
                    d[l+i] = d_mat[i][0];
                }
                for i in 0..lend-l {
                    e[l+i] = e_mat[i][0];
                }
            } else if anorm < ssfmin {
                iscale = 2;
                // Scale up
                let mut d_mat = vec![vec![0.0; 1]; lend-l+1];
                let mut e_mat = vec![vec![0.0; 1]; lend-l];
                for i in 0..=lend-l {
                    d_mat[i][0] = d[l+i];
                }
                for i in 0..lend-l {
                    e_mat[i][0] = e[l+i];
                }
                dlascl(&mut d_mat, anorm, ssfmin)?;
                dlascl(&mut e_mat, anorm, ssfmin)?;
                for i in 0..=lend-l {
                    d[l+i] = d_mat[i][0];
                }
                for i in 0..lend-l {
                    e[l+i] = e_mat[i][0];
                }
            }
        }

        // Choose between QL/QR iteration
        if d[lend].abs() < d[l].abs() {
            lend = lsv;
            l = lendsv;
        }

        if lend >= l {
            // QL Iteration
            loop {
                if l == lend {
                    break;
                }
                if jtot >= nmaxit {
                    return Err(Error((l + 1) as i32));
                }
                jtot += 1;

                let mut g = (d[l+1] - d[l]) / (2.0 * e[l]);
                let mut r = dlapy2(g, 1.0);
                g = d[m-1] - d[l] + e[l] / (g + r.copysign(g));
                let mut s = 1.0;
                let mut c = 1.0;
                let mut p = 0.0;

                for i in (l..=m-2).rev() {
                    let f = s * e[i];
                    let b = c * e[i];
                    let (cs, sn) = dlartg(g, f);
                    c = cs;
                    s = sn;

                    if i < m-2 {
                        e[i+1] = r;
                    }
                    g = d[i+1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i+1] = g + p;
                    g = c * r - b;

                    // Store rotation
                    work[i] = c;
                    work[n-1+i] = -s;
                }

                // Apply rotations
                let mm = m - l;
                dlasr(
                    'R', 'V', 'B',
                    n, mm,
                    &work[l..l+mm],
                    &work[n-1+l..n-1+l+mm],
                    z,
                    n,
                )?;

                d[l] = d[l] - p;
                e[l] = g;

                if e[l].abs() <= eps * (d[l].abs() + d[l+1].abs()) {
                    e[l] = 0.0;
                    break;
                }
            }
        } else {
            // QR Iteration
            loop {
                if l == lend {
                    break;
                }
                if jtot >= nmaxit {
                    return Err(Error((l + 1) as i32));
                }
                jtot += 1;

                let mut g = (d[l-1] - d[l]) / (2.0 * e[l-1]);
                let mut r = dlapy2(g, 1.0);
                g = d[m-1] - d[l] + e[l-1] / (g + r.copysign(g));
                let mut s = 1.0;
                let mut c = 1.0;
                let mut p = 0.0;

                for i in m-1..l {
                    let f = s * e[i];
                    let b = c * e[i];
                    let (cs, sn) = dlartg(g, f);
                    c = cs;
                    s = sn;

                    if i > m-1 {
                        e[i-1] = r;
                    }
                    g = d[i] - p;
                    r = (d[i+1] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i] = g + p;
                    g = c * r - b;

                    work[i] = c;
                    work[n-1+i] = s;
                }

                // Apply rotations
                let mm = l - m + 1;
                dlasr(
                    'R', 'V', 'F',
                    n, mm,
                    &work[m-1..m-1+mm],
                    &work[n-1+m-1..n-1+m-1+mm],
                    z,
                    n,
                )?;

                d[l] = d[l] - p;
                e[l-1] = g;

                if e[l-1].abs() <= eps * (d[l-1].abs() + d[l].abs()) {
                    e[l-1] = 0.0;
                    break;
                }
            }
        }

        // Undo scaling
        if iscale == 1 {
            let mut d_mat = vec![vec![0.0; 1]; lendsv-lsv+1];
            let mut e_mat = vec![vec![0.0; 1]; lendsv-lsv];
            for i in 0..=lendsv-lsv {
                d_mat[i][0] = d[lsv+i];
            }
            for i in 0..lendsv-lsv {
                e_mat[i][0] = e[lsv+i];
            }
            dlascl(&mut d_mat, ssfmax, anorm)?;
            dlascl(&mut e_mat, ssfmax, anorm)?;
            for i in 0..=lendsv-lsv {
                d[lsv+i] = d_mat[i][0];
            }
            for i in 0..lendsv-lsv {
                e[lsv+i] = e_mat[i][0];
            }
        } else if iscale == 2 {
            let mut d_mat = vec![vec![0.0; 1]; lendsv-lsv+1];
            let mut e_mat = vec![vec![0.0; 1]; lendsv-lsv];
            for i in 0..=lendsv-lsv {
                d_mat[i][0] = d[lsv+i];
            }
            for i in 0..lendsv-lsv {
                e_mat[i][0] = e[lsv+i];
            }
            dlascl(&mut d_mat, ssfmin, anorm)?;
            dlascl(&mut e_mat, ssfmin, anorm)?;
            for i in 0..=lendsv-lsv {
                d[lsv+i] = d_mat[i][0];
            }
            for i in 0..lendsv-lsv {
                e[lsv+i] = e_mat[i][0];
            }
        }

        l1 = l + 2;
    }

    // Sort eigenvalues and eigenvectors
    for ii in 2..=n {
        let i = ii - 1;  // i is 1-based
        let mut k = i;   // k is 1-based
        let mut p = d[i-1];  // Convert to 0-based for array access
        for j in ii..=n {
            if d[j-1] < p {
                k = j;
                p = d[j-1];
            }
        }
        if k != i {
            d[k-1] = d[i-1];
            d[i-1] = p;
            for row in 0..n {
                let temp = z[row][k-1];
                z[row][k-1] = z[row][i-1];
                z[row][i-1] = temp;
            }
        }
    }

    Ok(())
}


/// Scales a matrix by cto/cfrom without over/underflow.
/// Translated from LAPACK's DLASCL for the general matrix case (type 'G').
pub fn dlascl(a: &mut [Vec<f64>], cfrom: f64, cto: f64) -> Result<(), Error> {
    if cfrom == 0.0 {
        return Err(Error(-1));
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
pub fn dswap(n: usize, dx: &mut [f64], incx: usize, dy: &mut [f64], incy: usize) {
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
pub fn dlassq(n: usize, x: &[f64], incx: usize, scale: &mut f64, sumsq: &mut f64) {
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
pub fn ilaenv(ispec: i32, name: &str, opts: &str, n1: i32, n2: i32, n3: i32, n4: i32) -> i32 {
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
///     - 0: Compute eigenvalues only. We always want also eigenvectors too.
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
pub fn dlaed0(
    icompq: i32,
    n: usize,
    qsiz: usize,
    tlvls: usize,
    curlvl: usize,
    curpbm: usize,
    d: &mut [f64],
    e: &mut [f64],
    q: &mut [Vec<f64>],
    ldq: usize,
    qstore: &mut [Vec<f64>],
    qptr: &mut [usize],
    prmptr: &mut [usize],
    perm: &mut [usize],
    givptr: &mut [usize],
    givcol: &mut Vec<Vec<usize>>,
    givnum: &mut Vec<Vec<f64>>,
    work: &mut [f64],
    iwork: &mut [usize],
) -> Result<(), Error> {
    // Constants
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;

    // Quick return if possible
    if n == 0 {
        return Ok(());
    }

    // Get problem size cutoff for small subproblems
    let smlsiz = 25; // From ILAENV(9)

    // Initialize IWORK array for submatrix sizes
    iwork[0] = n;
    let mut subpbs: usize = 1;
    let mut tlvls: usize = 0;

    // Determine number of levels and subproblem sizes
    while iwork[subpbs.wrapping_sub(1)] > smlsiz {
        // Check if we have enough space for next iteration
        if subpbs.wrapping_mul(2) >= iwork.len() {
            break;
        }

        for j in (0..subpbs).rev() {
            // Check array bounds before indexing
            let j2 = j.wrapping_mul(2);
            if j2 < iwork.len() {
                iwork[j2] = (iwork[j] + 1) / 2;
                if j2 > 0 {
                    iwork[j2.wrapping_sub(1)] = iwork[j] / 2;
                }
            }
        }
        tlvls += 1;
        subpbs = subpbs.wrapping_mul(2);
    }

    // Calculate cumulative sizes with bounds checking
    for j in 1..subpbs.min(iwork.len()) {
        if j > 0 && j < iwork.len() {
            let prev = iwork[j.wrapping_sub(1)];
            iwork[j] = iwork[j].wrapping_add(prev);
        }
    }

    // Set up workspaces - avoid overflow
    let indxq = 0;

    // Different workspace setup based on ICOMPQ
    let (iprmpt, iperm, iqptr, igivpt, igivcl, igivnm, iq, iwrem) = if icompq != 2 {
        // Compute workspace sizes for eigenvalues/accumulate vectors
        let temp = (n as f64).ln() / TWO.ln();
        let mut lgn = temp as i32;
        if 2i32.pow(lgn as u32) < n as i32 {
            lgn += 1;
        }
        if 2i32.pow(lgn as u32) < n as i32 {
            lgn += 1;
        }

        let iprmpt = indxq + n + 1;
        let iperm = iprmpt + n * lgn as usize;
        let iqptr = iperm + n * lgn as usize;
        let igivpt = iqptr + n + 2;
        let igivcl = igivpt + n * lgn as usize;
        let igivnm = 1;
        let iq = igivnm + 2 * n * lgn as usize;
        let iwrem = iq + n * n + 1;

        // Initialize pointers
        for i in 0..=subpbs {
            iwork[iprmpt + i] = 1;
            iwork[igivpt + i] = 1;
        }
        iwork[iqptr] = 1;

        (iprmpt, iperm, iqptr, igivpt, igivcl, igivnm, iq, iwrem)
    } else {
        (0, 0, 0, 0, 0, 0, 0, 0)
    };

    // Solve each subproblem at the bottom of divide-conquer tree
    let mut curr = 0;
    let spm1 = subpbs - 1;

    // Divide matrix using rank-1 modifications
    for i in 0..spm1 {
        let submat = if i < iwork.len() { iwork[i] } else { 0 } + 1;
        let smm1 = submat - 1;
        d[smm1] = d[smm1] - e[smm1].abs();
        d[submat] = d[submat] - e[smm1].abs();
    }

    // Process each submatrix
    for i in 0..=spm1 {
        let (submat, matsiz) = if i == 0 {
            (1, iwork[0])
        } else {
            (iwork[i] + 1, iwork[i + 1] - iwork[i])
        };

        if icompq == 2 {
            // Compute eigenvalues and vectors
            dsteqr(
                'I',
                matsiz,
                &mut d[submat - 1..],
                &mut e[submat - 1..],
                &mut q[submat - 1..],
                work,
            )?;
        } else {
            // Compute eigenvalues only or with original vectors
            if iq > 0 {
                let work_offset = iq - 1 + iwork[iqptr + curr];
            }
            let mut z = vec![vec![0.0; matsiz]; matsiz];
            dsteqr(
                'I',
                matsiz,
                &mut d[submat - 1..],
                &mut e[submat - 1..],
                &mut z,
                work,
            )?;

            if icompq == 1 {
                // Multiply by original vectors if needed
                let result = dgemm(&q[submat - 1..], &z);
                for (i, row) in result.iter().enumerate() {
                    qstore[submat - 1 + i] = row.clone();
                }

                if iqptr + curr + 1 < iwork.len() {
                    iwork[iqptr + curr + 1] = iwork[iqptr + curr] + matsiz * matsiz;
                }
                curr += 1;
            }
        }

        // Set up index mapping
        let mut k = 1;
        for j in submat..=iwork[i + 1] {
            iwork[indxq + j - 1] = k;
            k += 1;
        }
    }

    // Merge eigensystems of adjacent submatrices
    let mut curlvl = 1;
    while subpbs > 1 {
        let spm2 = subpbs - 2;
        for i in (0..=spm2).step_by(2) {
            let (submat, matsiz, msd2, curprb) = if i == 0 {
                (1, iwork[1], iwork[0], 0)
            } else {
                (
                    iwork[i] + 1,
                    iwork[i + 2] - iwork[i],
                    (iwork[i + 2] - iwork[i]) / 2,
                    i / 2 + 1,
                )
            };

            if icompq == 2 {
                let split_point = indxq + submat - 1;
                let (iwork_left, iwork_right) = iwork.split_at_mut(split_point);

                // Adjust the indices relative to the split slices
                let iwork_slice1 = &mut iwork_left[(submat - 1)..];
                let iwork_slice2 = &mut iwork_right[(subpbs - split_point)..];

                dlaed1(
                    matsiz,
                    &mut d[submat - 1..],
                    &mut q[submat - 1..],
                    ldq,
                    iwork_slice1,
                    &mut e[submat + msd2 - 2],
                    msd2,
                    work,
                    iwork_slice2,
                );
            } else {
                let cutpnt = msd2;
                let rho_index = submat + msd2 - 2; // Zero-based indexing
                let mut rho = e[rho_index];
                let mut info = 0;
                let mut givptr: usize = 0;
                let mut qstore_flat: Vec<f64> = qstore.iter().flatten().cloned().collect();
                let mut indxq = vec![0usize; n]; // Is this right?

                dlaed7(
                    icompq,
                    n,
                    qsiz,
                    tlvls,
                    curlvl,
                    curpbm,
                    d,
                    q,
                    ldq,
                    &mut indxq[..],
                    &mut rho,
                    cutpnt,
                    &mut qstore_flat, // &mut [f64] (flattened qstore)
                    qptr,
                    prmptr,
                    perm,
                    &mut givptr, // &mut usize
                    givcol,
                    givnum,
                    work,
                    iwork,
                )?;
            }

            iwork[i / 2 + 1] = iwork[i + 2];
        }
        subpbs /= 2;
        curlvl += 1;
    }

    // Re-merge deflated eigenvalues/vectors
    match icompq {
        1 => {
            for i in 0..n-1 {
                // Use checked_add to safely handle potential overflow
                if let Some(idx) = indxq.checked_add(i) {
                    if idx < iwork.len() {
                        let j = iwork[idx];
                        if j > 0 && j <= d.len() {  // Bounds check for d array
                            work[i] = d[j - 1];
                            dcopy(qsiz, &qstore[j - 1], 1, &mut q[i], 1);
                        }
                    }
                }
            }
            dcopy(n, work, 1, d, 1);
        }
        2 => {
            for i in 0..n-1 {
                // Use checked_add and bounds checking
                if let Some(idx) = indxq.checked_add(i) {
                    if idx < iwork.len() {
                        let j = iwork[idx];
                        if j > 0 && j <= d.len() {  // Bounds check for d array
                            work[i] = d[j - 1];
                            // Use a separate vector for the work to avoid borrowing conflicts
                            let start = n * i;
                            let end = start + n;
                            if j <= q.len() {  // Bounds check for q array
                                work[start..end].copy_from_slice(&q[j - 1][..n]);
                            }
                        }
                    }
                }
            }
            dcopy(n, work, 1, d, 1);
            
            // Declare `work_matrix` with limited scope
            {
                let mut work_matrix = vec![vec![0.0; n]; n];
                dlacpy('A', n, n, &q, ldq, &mut work_matrix, n);
            }
        }
        _ => {
            for i in 0..n-1 {
                // Use checked_add and bounds checking
                if let Some(idx) = indxq.checked_add(i) {
                    if idx < iwork.len() {
                        let j = iwork[idx];
                        if j > 0 && j <= d.len() {  // Bounds check for d array
                            work[i] = d[j - 1];
                        }
                    }
                }
            }
            dcopy(n, work, 1, d, 1);
        }
    }

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
        'B' | 'b' => 2.0f64,       // radix(zero). We assume base 2.
        'P' | 'p' => eps * 2.0f64, // eps * radix(zero)
        'N' | 'n' => f64::MANTISSA_DIGITS as f64, // digits(zero)
        'R' | 'r' => rnd,
        'M' | 'm' => f64::MIN_EXP as f64, // minexponent(zero)
        'U' | 'u' => f64::MIN_POSITIVE,   // tiny(zero)
        'L' | 'l' => f64::MAX_EXP as f64, // maxexponent(zero)
        'O' | 'o' => f64::MAX,            // huge(zero)
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
pub fn dlacpy(
    uplo: char,
    m: usize,
    n: usize,
    a: &[Vec<f64>],
    lda: usize,
    b: &mut [Vec<f64>],
    ldb: usize,
) {
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
    ldq: usize, // Now used for validation
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
) -> Result<i32, Error> {
    // Input Validation
    if n == 0 {
        return Ok(0);
    }
    if n1 < 1 || n1 > n {
        return Err(Error::from("Invalid N1 value in dlaed2"));
    }
    if ldq < n {
        return Err(Error::from("Invalid LDQ value in dlaed2"));
    }
    if d.len() != n
        || q.len() != n
        || indxq.len() != n
        || z.len() != n
        || dlamda.len() != n
        || w.len() != n
        || q2.len() != n
        || indx.len() != n
        || indxc.len() != n
        || indxp.len() != n
        || coltyp.len() != n
    {
        return Err(Error::from("Incorrect array dimensions in dlaed2"));
    }
    for q_vec in q.iter() {
        if q_vec.len() != n {
            return Err(Error::from("Incorrect nested vector dimension in dlaed2"));
        }
    }

    let n2 = n - n1;

    // If rho is negative, negate the second part of z
    if *rho < 0.0 {
        for zi in z[n1..n].iter_mut() {
            *zi = -*zi;
        }
    }

    // Normalize z so that norm(z) = 1 / sqrt(2)
    let t = 1.0 / (2.0f64).sqrt();
    dscal(n, t, z, 1);
    *rho = 2.0 * (*rho).abs();

    // Update indxq: increment indices in second half
    for i in n1..n {
        indxq[i] += n1;
    }

    // Initialize dlamda by permuting d according to indxq
    for i in 0..n {
        dlamda[i] = d[indxq[i]];
    }

    // Merge the two sorted sublists in dlamda into indxc
    dlamrg(n1, n2, &dlamda, 1, 1, indxc);

    // Apply the permutation from indxc to indxq to get indx
    for i in 0..n {
        if i < n && indxc[i] < indxq.len() {
            indx[i] = indxq[indxc[i]];
        }
    }

    // Calculate the allowable deflation tolerance
    let imax = idamax(n, z, 1);
    let jmax = idamax(n, d, 1);
    let eps = dlamch('E');
    let tol = 8.0 * eps * d[jmax].abs().max(z[imax].abs());

    // If the rank-1 modifier is small enough, reorganize Q and D
    if *rho * z[imax].abs() <= tol {
        *k = 0;
        for (j, &i) in indx.iter().enumerate() {
            dlamda[j] = d[i];
            q2[j] = q[i].clone();
        }
        d.copy_from_slice(&dlamda[0..n]);
        for i in 0..n {
            q[i] = q2[i].clone();
        }
        return Ok(0);
    }

    // Initialize coltyp
    for i in 0..n {
        if i < n1 {
            coltyp[i] = 1; // Type 1
        } else {
            coltyp[i] = 3; // Type 3
        }
    }

    *k = 0;
    let mut k2 = n;
    let mut j = 0;
    while j < n {
        let pj = indx[j];

        if *rho * z[pj].abs() <= tol {
            // Deflation due to small z component
            k2 -= 1;
            coltyp[pj] = 4; // Type 4
            indxp[k2] = pj;
            j += 1;
            continue;
        }

        let mut deflate = false;

        if j + 1 < n {
            let nj = indx[j + 1];

            if *rho * z[nj].abs() <= tol {
                // Deflation due to small z component at nj
                k2 -= 1;
                coltyp[nj] = 4; // Type 4
                indxp[k2] = nj;
                // Process pj in next iteration
                j += 2;
                continue;
            }

            // Check if eigenvalues are close enough to allow deflation
            let s = z[pj];
            let c = z[nj];
            let tau = dlapy2(c, s);
            let t = d[nj] - d[pj];
            let c_norm = c / tau;
            let s_norm = -s / tau;

            if (t * c_norm * s_norm).abs() <= tol {
                // Deflation is possible
                z[nj] = tau;
                z[pj] = 0.0;
                if coltyp[nj] != coltyp[pj] {
                    coltyp[nj] = 2; // Type 2
                }
                coltyp[pj] = 4; // Deflated

                // Apply rotation to eigenvectors
                if pj != nj {
                    let (q_pj, q_nj) = if pj < nj {
                        let (q_left, q_right) = q.split_at_mut(nj);
                        (&mut q_left[pj], &mut q_right[0])
                    } else {
                        let (q_left, q_right) = q.split_at_mut(pj);
                        (&mut q_right[0], &mut q_left[nj])
                    };
                    drot(n, q_pj, 1, q_nj, 1, c_norm, s_norm);
                }

                // Update d
                let d_pj = d[pj];
                let d_nj = d[nj];
                let temp = d_pj * c_norm * c_norm + d_nj * s_norm * s_norm;
                d[nj] = d_pj * s_norm * s_norm + d_nj * c_norm * c_norm;
                d[pj] = temp;

                // Place deflated eigenvalue at the end
                k2 -= 1;
                indxp[k2] = pj; // Any bounds issues?

                // The eigenvalues remain sorted
                let mut i = 1;
                while k2 + i < n && d[pj] < d[indxp[k2 + i]] {
                    indxp[k2 + i - 1] = indxp[k2 + i];
                    indxp[k2 + i] = pj;
                    i += 1;
                }
                if k2 + i - 1 < n {
                    indxp[k2 + i - 1] = pj;
                }

                // Deflation occurred
                deflate = true;
            }
        }

        if !deflate {
            // No deflation, record this eigenvalue
            *k += 1;
            dlamda[*k - 1] = d[pj];
            w[*k - 1] = z[pj];
            indxp[*k - 1] = pj;
        }

        j += 1;
    }

    // Count up the total number of the various types of columns
    let mut ctot = [0usize; 4];
    for &i in indxp.iter().take(n) {
        if i < n && i < coltyp.len() {  // Add bounds check for coltyp
            let ct = coltyp[i];
            // Validate ct before using it as an index
            if ct > 0 && ct <= 4 {
                ctot[ct - 1] += 1;
            }
        }
    }

    // Form a permutation which positions the four column types into four groups
    let mut psm = [0usize; 5]; // psm[0] is unused to match Fortran 1-based indexing
    psm[1] = 0;
    for i in 2..5 {
        psm[i] = psm[i - 1] + ctot[i - 2];
    }

    // Fill indx and indxc according to the types
    let mut psm_fill = psm.clone();

    for &i in indxp.iter().take(n) {
        if i < n && i < coltyp.len() {
            let ct = coltyp[i];
            if ct > 0 && ct <= 4 {
                if psm_fill[ct] < n {  // Bounds check
                    indx[psm_fill[ct]] = i;
                    indxc[psm_fill[ct]] = i;
                }
                psm_fill[ct] += 1;
            }
        }
    }

    // Sort the eigenvalues and corresponding eigenvectors into dlamda and q2
    // The eigenvalues/vectors which were not deflated go into the first k slots
    let mut iq1 = 0usize;
    let mut iq2 = (ctot[0] + ctot[1]) * n1;

    for i in 0..*k {
        let js = indx[i];
        match coltyp[js] {
            1 => {
                dlamda[i] = d[js];
                w[i] = z[js];
                // Copy the first n1 components of q[js]
                q2[i].resize(n1, 0.0);
                q2[i].copy_from_slice(&q[js][..n1]);
                iq1 += n1;
            }
            2 => {
                dlamda[i] = d[js];
                w[i] = z[js];
                // Copy the first n1 components
                q2[i].resize(n1, 0.0);
                q2[i].copy_from_slice(&q[js][..n1]);
                // Copy the last n2 components
                q2[i + *k].resize(n2, 0.0);
                q2[i + *k].copy_from_slice(&q[js][n1..]);
                iq1 += n1;
                iq2 += n2;
            }
            3 => {
                dlamda[i] = d[js];
                w[i] = z[js];
                // Copy the last n2 components
                q2[i + *k].resize(n2, 0.0);
                q2[i + *k].copy_from_slice(&q[js][n1..]);
                iq2 += n2;
            }
            _ => {} // Ignore deflated
        }
    }

    // The deflated eigenvalues and their corresponding vectors go back into the last N - K slots
    if *k < n {
        for i in *k..n {
            d[i] = d[indx[i]];
            q[i] = q[indx[i]].clone();
        }
    }

    // The counts in ctot are needed in dlaed3. Do they need to be copied?

    Ok(0)
}


/// Creates a permutation list to merge two sorted sets into a single sorted set.
/// This function corresponds to LAPACK's DLAMRG subroutine.
pub fn dlamrg(
    n1: usize,
    n2: usize,
    a: &[f64],
    dtrd1: i32,
    dtrd2: i32,
    index: &mut [usize],
) {
    // Initialize variables as isize to handle negative strides
    let mut n1sv = n1 as isize;
    let mut n2sv = n2 as isize;

    // Initialize indices exactly as LAPACK does
    let mut ind1 = if dtrd1 > 0 {
        0isize  // Start of first array
    } else {
        n1 as isize - 1  // End of first array
    };
    let mut ind2 = if dtrd2 > 0 {
        n1 as isize  // Start of second array
    } else {
        n1 as isize + n2 as isize - 1  // End of second array
    };

    let mut i = 0;
    while n1sv > 0 && n2sv > 0 {
        // Indices are within bounds
        if ind1 < 0 || ind1 >= a.len() as isize || ind2 < 0 || ind2 >= a.len() as isize {
            return;  // Early return on invalid indices rather than panic
        }

        if a[ind1 as usize] <= a[ind2 as usize] {
            index[i] = ind1 as usize;
            i += 1;
            ind1 += dtrd1 as isize;  // Move forward or backward depending on stride
            n1sv -= 1;
        } else {
            index[i] = ind2 as usize;
            i += 1;
            ind2 += dtrd2 as isize;  // Move forward or backward depending on stride
            n2sv -= 1;
        }
    }

    // Handle remaining elements
    while n1sv > 0 {
        if ind1 < 0 || ind1 >= a.len() as isize {
            panic!("Index out of bounds in dlamrg");
        }
        index[i] = ind1 as usize;
        i += 1;
        ind1 += dtrd1 as isize;
        n1sv -= 1;
    }

    while n2sv > 0 {
        if ind2 < 0 || ind2 >= a.len() as isize {
            panic!("Index out of bounds in dlamrg");
        }
        index[i] = ind2 as usize;
        i += 1;
        ind2 += dtrd2 as isize;
        n2sv -= 1;
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
) -> Result<(), Error> {
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
            let temp = finit
                + *tau * z[0] / (d[0] * (d[0] - *tau))
                + *tau * z[1] / (d[1] * (d[1] - *tau))
                + *tau * z[2] / (d[2] * (d[2] - *tau));
            if temp <= 0.0 {
                lbd = *tau;
            } else {
                ubd = *tau;
            };
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
        let temp = if scale {
            1.0 / (dscale[j] - *tau)
        } else {
            1.0 / (d[j] - *tau)
        };
        if !temp.is_finite() {
            continue;
        }
        let temp1 = if scale { zscale[j] * temp } else { z[j] * temp };
        f += *tau * temp1 / if scale { dscale[j] } else { d[j] };
        df += temp1 * temp;
        ddf += df * temp;
    }

    let maxit = 20;
    for _ in 0..maxit {
        if f.abs() <= eps * finit.abs() || ubd - lbd <= 2.0 * eps {
            break;
        };

        // Gragg-Thornton-Warner cubic convergent scheme
        let temp1 = if orgati {
            dscale[1] - *tau
        } else {
            dscale[0] - *tau
        };
        let temp2 = if orgati {
            dscale[2] - *tau
        } else {
            dscale[1] - *tau
        };

        let a = (temp1 + temp2) * f - temp1 * temp2 * df;
        let b = temp1 * temp2 * f;
        let c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;

        let temp = a.abs().max(b.abs()).max(c.abs());
        let a = a / temp;
        let b = b / temp;
        let c = c / temp;

        let eta = if c == 0.0 {
            b / a
        } else if a <= 0.0 {
            (a - (a * a - 4.0 * b * c).sqrt()) / (2.0 * c)
        } else {
            2.0 * b / (a + (a * a - 4.0 * b * c).sqrt())
        };

        if f * eta >= 0.0 {
            *tau = -f / df;
        }

        *tau += eta;
        if *tau < lbd || *tau > ubd {
            *tau = (lbd + ubd) / 2.0
        };

        // Update f, df, ddf for next iteration
        f = finit;
        df = 0.0;
        ddf = 0.0;
        for j in 0..3 {
            let temp = if scale {
                1.0 / (dscale[j] - *tau)
            } else {
                1.0 / (d[j] - *tau)
            };
            if !temp.is_finite() {
                continue;
            }
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
) -> Result<i32, Error> {
   // Validate all input dimensions
   if k > n || n1 > n/2 || n1 == 0 || ldq < n || 
      d.len() < n || q.len() < n || dlamda.len() < k ||
      w.len() < k || indx.len() < n || ctot.len() < 4 ||
      q2.len() < n || s.len() < n {
       return Err(Error(-1));
   }

   for row in q.iter() {
       if row.len() < n {
           return Err(Error(-1));
       }
   }
   for row in q2.iter() {
       if row.len() < n {
           return Err(Error(-1)); 
       }
   }
   for row in s.iter() {
       if row.len() < k {
           return Err(Error(-1));
       }
   }

   // Quick return if possible
   if k == 0 {
       return Ok(0);
   }

   // Adjust dlamda for better accuracy
   for i in 0..k {
       dlamda[i] = dlamc3(dlamda[i], dlamda[i]) - dlamda[i];
   }

   // Find roots of secular equation
   for j in 0..k {
       let mut q_col = vec![0.0; k];
       for i in 0..k {
           if i < q.len() && j < q[i].len() {
               q_col[i] = q[i][j];
           }
       }
       
       if let Err(e) = dlaed4(&dlamda[..k], &w[..k], &q_col, rho, &mut d[..k], q) {
           return Ok(1);
       }
   }

   if k == 1 {
       return Ok(0);
   }

   if k == 2 {
       // Handle 2x2 case
       for j in 0..k {
           if j >= q[0].len() || j >= q[1].len() {
               continue;
           }
           let w1 = q[0][j];
           let w2 = q[1][j];
           
           // Safe indexing with bounds checks
           if let Some(&i1) = indx.get(0) {
               if let Some(&i2) = indx.get(1) {
                   // Convert from 1-based to 0-based safely
                   let ii1 = i1.checked_sub(1).unwrap_or(0);
                   let ii2 = i2.checked_sub(1).unwrap_or(0);
                   q[0][j] = w1.min(w2);
                   q[1][j] = w1.max(w2);
               }
           }
       }
       return Ok(0);
   }

   // Update w
   if !s.is_empty() && !s[0].is_empty() {
       for (i, &wi) in w.iter().take(k).enumerate() {
           s[0][i] = wi;
       }
   }

   // Initialize w
   for i in 0..k {
       if i < q.len() && i < q[i].len() {
           w[i] = q[i][i];
       }
   }

   // Update w values
   for j in 0..k {
       for i in 0..j {
           if i < w.len() && j < dlamda.len() {
               let denom = dlamda[i] - dlamda[j];
               if denom != 0.0 {
                   if let Some(qi) = q.get(i).and_then(|row| row.get(j)) {
                       w[i] *= qi / denom;
                   }
               }
           }
       }
       for i in (j+1)..k {
           if i < w.len() && j < dlamda.len() {
               let denom = dlamda[i] - dlamda[j];
               if denom != 0.0 {
                   if let Some(qi) = q.get(i).and_then(|row| row.get(j)) {
                       w[i] *= qi / denom;
                   }
               }
           }
       }
   }

   // Update w with sqrt
   if !s.is_empty() && !s[0].is_empty() {
       for i in 0..k {
           if i < w.len() && i < s[0].len() {
               w[i] = (-w[i]).sqrt().copysign(s[0][i]);
           }
       }
   }

   // Compute eigenvectors
   for j in 0..k {
       for i in 0..k {
           if i < s.len() && i < q.len() && j < q[i].len() {
               s[0][i] = w[i] / q[i][j];
           }
       }

       let temp = dnrm2(k, &s[0], 1);
       if temp != 0.0 {
           for i in 0..k {
               if i < q.len() && j < q[i].len() && i < indx.len() {
                   let ii = indx[i].saturating_sub(1);
                   if ii < s[0].len() {
                       q[i][j] = s[0][ii] / temp;
                   }
               }
           }
       }
   }

   // Update eigenvectors
   let n2 = n.saturating_sub(n1);
   let n12 = ctot.get(0).unwrap_or(&0) + ctot.get(1).unwrap_or(&0);
   let n23 = ctot.get(1).unwrap_or(&0) + ctot.get(2).unwrap_or(&0);

   let ctot_0 = ctot.get(0).copied().unwrap_or(0);
   
   // Safe matrix operations with bounds checking
   if n23 > 0 && ctot_0 < q.len() {
       for i in 0..n23.min(q.len() - ctot_0) {
           for j in 0..k {
               if i < s.len() && j < s[i].len() && 
                  i + ctot_0 < q.len() && j < q[i + ctot_0].len() {
                   s[i][j] = q[i + ctot_0][j];
               }
           }
       }
   }

   let iq2 = n1.saturating_mul(n12);
   
   if n23 > 0 {
       let result = dgemm(&q2[iq2..], &s[..n23.min(s.len())]);
       for (i, row) in result.iter().enumerate() {
           for (j, &val) in row.iter().enumerate() {
               if i + n1 < q.len() && j < q[i + n1].len() {
                   q[i + n1][j] = val;
               }
           }
       }
   } else {
       for i in n1..q.len() {
           for j in 0..k.min(q[i].len()) {
               q[i][j] = 0.0;
           }
       }
   }

   if n12 > 0 {
       for i in 0..n12.min(q.len()) {
           for j in 0..k {
               if i < s.len() && j < s[i].len() && j < q[i].len() {
                   s[i][j] = q[i][j];
               }
           }
       }
       
       let result = dgemm(q2, &s[..n12.min(s.len())]);
       for (i, row) in result.iter().enumerate().take(n1) {
           for (j, &val) in row.iter().enumerate().take(k) {
               if i < q.len() && j < q[i].len() {
                   q[i][j] = val;
               }
           }
       }
   } else {
       for i in 0..n1.min(q.len()) {
           for j in 0..k.min(q[i].len()) {
               q[i][j] = 0.0;
           }
       }
   }

   Ok(0)
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
) -> Result<(), Error> {
    // Test the input parameters
    if k > n {
        return Err(Error(-4)); // N must be >= K
    }
    if kstart < 1 || kstart > k.max(1) {
        return Err(Error(-2));
    }
    if kstop < kstart || kstop > k.max(1) {
        return Err(Error(-3));
    }
    if ldq < n.max(1) {
        return Err(Error(-7));
    }
    if lds < k.max(1) {
        return Err(Error(-12));
    }

    // Quick return if possible
    if k == 0 {
        return Ok(());
    }

    // Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can
    // be computed with high relative accuracy. This is needed to combat
    // problems with machines that lack a guard digit in add/subtract. Lol.
    for i in 0..n-1 {
        dlamda[i] = dlamc3(dlamda[i], dlamda[i]) - dlamda[i];
    }

    // Compute eigenvalues of the modified secular equation
    for j in kstart - 1..kstop {
        let mut d_sub = vec![0.0; k];
        let mut z_out = vec![vec![0.0; k]; k];

        // Call dlaed4 with the correct arguments
        // Note: dlaed4 takes the secular equation parameters and returns eigenvalue/vectors
        dlaed4(&dlamda, &dlamda, w, rho, &mut d_sub, &mut z_out)?;

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
        for i in j + 1..k {
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
) -> Result<(), Error> {
    // Constants
    const MONE: f64 = -1.0;
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const EIGHT: f64 = 8.0;

    // Test the input parameters
    if icompq < 0 || icompq > 1 {
        return Err(Error(-1));
    }
    if n < 0 {
        return Err(Error(-3));
    }
    if icompq == 1 && qsiz < n {
        return Err(Error(-4));
    }
    if ldq < n.max(1) {
        return Err(Error(-7));
    }
    if cutpnt < 1.min(n) || cutpnt > n {
        return Err(Error(-10));
    }
    if ldq2 < n.max(1) {
        return Err(Error(-14));
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
    for i in 0..n-1 {
        dlamda[i] = d[indxq[i]];
        w[i] = z[indxq[i]];
    }

    // Merge sorted subsets
    dlamrg(n1, n2, dlamda, 1, 1, indx);

    // Reorder based on sorted indices
    for i in 0..n-1 {
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
    if n >= isize::MAX as usize - 1 {
        return Err(Error(-1));
    }
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
///
/// Corresponds to LAPACK's DLAED7 subroutine.
///
/// # Arguments
/// - `icompq`: Specifies whether to compute eigenvectors.
///     - 0: Compute eigenvalues only.
///     - 1: Compute eigenvectors of the original dense symmetric matrix.
/// - `n`: The order of the matrix.
/// - `qsiz`: The dimension of the orthogonal matrix used to reduce
///           the full matrix to tridiagonal form.
/// - `tlvls`: The total number of merging levels in the overall divide and conquer tree.
/// - `curlvl`: The current level in the overall merge routine.
/// - `curpbm`: The current problem in the current level.
/// - `d`: On entry, the diagonal elements of the tridiagonal matrix.
///        On exit, contains the eigenvalues.
/// - `q`: On entry, if `icompq = 1`, the orthogonal matrix used to reduce the original matrix
///        to tridiagonal form. On exit, if `icompq = 1`, contains the eigenvectors.
/// - `indxq`: On entry, the permutation which separately sorts the two subproblems
///            in D into ascending order. On exit, the permutation which will reintegrate
///            the subproblems back into sorted order.
/// - `rho`: The subdiagonal entry used to create the rank-one modification.
///          On exit, destroyed.
/// - `cutpnt`: The location of the last eigenvalue in the leading submatrix.
/// - `qstore`: Workspace array to store the eigenvectors.
/// - `qptr`, `prmptr`, `perm`, `givptr`, `givcol`, `givnum`: Workspace arrays for DLAED7.
/// - `work`: Workspace array.
/// - `iwork`: Integer workspace array.
///
/// # Returns
/// - `Result<(), Error>` indicating success or an error.
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
    qstore: &mut [f64],
    qptr: &mut [usize],
    prmptr: &mut [usize],
    perm: &mut [usize],
    givptr: &mut usize,
    givcol: &mut [Vec<usize>],
    givnum: &mut [Vec<f64>],
    work: &mut [f64],
    iwork: &mut [usize],
) -> Result<(), Error> {
    // Test the input parameters
    if icompq < 0 || icompq > 1 {
        return Err(Error(-1));
    }
    if n < 0 {
        return Err(Error(-2));
    }
    if icompq == 1 && qsiz < n {
        return Err(Error(-4));
    }

    // Quick return if possible
    if n == 0 {
        return Ok(());
    }

    // Constants
    const ONE: f64 = 1.0;
    const ZERO: f64 = 0.0;

    // Workspace indices - adjust as per actual usage
    let iz = 0;
    let idlmda = iz + n;
    let iw = idlmda + n;
    let iq2 = iw + n;
    let is = iq2 + n * n; // Adjusted size for q2

    // Integer workspace indices
    let indx = 0;
    let indxc = indx + n;
    let coltyp = indxc + n;
    let indxp = coltyp + n;

    // Extract slices from work arrays
    // Split `work` into non-overlapping slices
    let (work_iz, work_rest) = work.split_at_mut(idlmda);
    let (work_dlamda, work_rest) = work_rest.split_at_mut(iw - idlmda);
    let (work_w, work_q2) = work_rest.split_at_mut(iq2 - iw);

    // Now assign the slices accordingly
    let z = work_iz;
    let dlamda = work_dlamda;
    let w = work_w;
    let q2 = work_q2;

    let indxv = &mut iwork[indx..indx + n];
    let indxcv = &mut iwork[indxc..indxc + n];
    let coltypv = &mut iwork[coltyp..coltyp + n];
    let indxpv = &mut iwork[indxp..indxp + n];

    // Initialize Givens rotation count
    *givptr = 0;

    // Variables
    let n1 = cutpnt;
    let n2 = n - n1;

    // Form the z-vector which consists of the last row of Q_1 and the first row of Q_2
    for i in 0..n-1 {
        z[i] = if i < n1 { q[n1 - 1][i] } else { q[i][i] };
    }

    // Deflate eigenvalues
    let mut k = 0;

    // Declarations of givcol and givnum
    let mut givcol = vec![vec![0usize; 2]; n]; // Are these appropriate dimensions?
    let mut givnum = vec![vec![0.0; 2]; n]; // Are these appropriate dimensions?

    let mut indxp = vec![0usize; n];
    let mut indx = vec![0usize; n];
    let mut q2 = vec![vec![0.0; n]; n]; // Are dimensions correct?

    let ldq2 = if icompq == 1 { qsiz } else { n };
    dlaed8(
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
        &mut q2,
        ldq2,
        w,
        perm,
        givptr,
        &mut givcol,
        &mut givnum,
        &mut indxp,
        &mut indx,
    )?;

    if k != 0 {
        // Solve the secular equation
        let mut s = vec![vec![0.0; k]; k];
        dlaed9(
            k, 1, // kstart
            k, // kstop
            n, d, q, ldq, *rho, dlamda, w, &mut s, k,
        )?;

        if icompq == 1 {
            // Multiply Q by the updated eigenvectors
            let mut q2_mat = vec![vec![0.0; k]; qsiz];
            for i in 0..qsiz {
                for j in 0..k {
                    q2_mat[i][j] = q[i][perm[j]];
                }
            }

            let result = dgemm(&q2_mat, &s);
            for i in 0..qsiz {
                for j in 0..k {
                    q[i][j] = result[i][j];
                }
            }
        }

        // Prepare the INDXQ sorting permutation
        let n1 = k;
        let n2 = n - k;
        dlamrg(n1, n2, d, 1, -1, indxq);
    } else {
        for i in 0..n-1 {
            indxq[i] = i;
        }
    }

    Ok(())
}

/// Computes the Z vector determining the rank-one modification of the diagonal matrix
/// during the merge phase of the divide and conquer algorithm.
///
/// This function is used by DSTEDC when computing eigenvectors of a symmetric
/// tridiagonal matrix using divide and conquer. It specifically handles computing
/// the parts of eigenvectors needed during merging of subproblems.
///
/// # Arguments
/// * `n` - The dimension of the symmetric tridiagonal matrix
/// * `tlvls` - The total number of levels in the divide & conquer tree
/// * `curlvl` - The current level in the divide & conquer tree (0 <= curlvl <= tlvls)
/// * `curpbm` - The current problem in the current level
/// * `prmptr` - Array of pointers to permutations at each level
/// * `perm` - The permutations used to form deflation sets
/// * `givptr` - Array of pointers to Givens rotations
/// * `givcol` - The columns rotated by Givens rotations
/// * `givnum` - The cosines and sines of the Givens rotations
/// * `q` - The eigenvectors of the subproblems at the current level
/// * `qptr` - Array of pointers into Q for the different splits
/// * `z` - The final Z vector on output
/// * `ztemp` - Temporary workspace array for Z computations
///
/// # Returns
/// * `Result<(), i32>` - Ok(()) on success, Err(info) if an error occurred
pub fn dlaeda(
    n: usize,
    tlvls: usize,
    curlvl: usize,
    curpbm: usize,
    prmptr: &[usize],
    perm: &[usize],
    givptr: &[usize],
    givcol: &[Vec<usize>],
    givnum: &[Vec<f64>],
    q: &[f64],
    qptr: &[usize],
    z: &mut [f64],
    ztemp: &mut [f64],
) -> Result<(), Error> {
    // Parameter validation
    if n == 0 {
        return Ok(());
    }

    // Determine location of first number in second half
    let mid = n / 2 + 1;

    let curr = if curlvl == 0 {
        curpbm
    } else {
        let base: usize = 2;
        let shift1 = base.saturating_pow(curlvl as u32); 
        let shift2 = base.saturating_pow((curlvl - 1) as u32);
        curpbm.saturating_mul(shift1).saturating_add(shift2)
    };
    
    // Verify array bounds before accessing qptr
    if curr + 2 >= qptr.len() {
        return Err(Error(-1));
    }
    
    // Calculate sizes of the two eigenblocks
    let bsiz1 = ((qptr[curr + 1].saturating_sub(qptr[curr])) as f64).sqrt() as usize;
    let bsiz2 = ((qptr[curr + 2].saturating_sub(qptr[curr + 1])) as f64).sqrt() as usize;

    // Initialize z to zero
    for k in 0..mid - bsiz1 {
        z[k] = 0.0;
    }

    // Copy last row of first eigenblock
    for k in 0..bsiz1 {
        z[mid - bsiz1 + k] = q[qptr[curr] + bsiz1 * k + bsiz1 - 1];
    }

    // Copy first row of second eigenblock
    for k in 0..bsiz2 {
        z[mid + k] = q[qptr[curr + 1] + bsiz2 * k];
    }

    // Zero out remainder
    for k in mid + bsiz2..n {
        z[k] = 0.0;
    }

    // Apply permutations and Givens rotations for each level
    let mut ptr = 1 << tlvls;
    for k in 1..=curlvl {
        // Calculate indices for current level
        let level_curr = ptr + curpbm * (1 << (curlvl - k)) + (1 << (curlvl - k - 1)) - 1;

        // Get sizes for the blocks at this level
        let psiz1 = prmptr[level_curr + 1] - prmptr[level_curr];
        let psiz2 = prmptr[level_curr + 2] - prmptr[level_curr + 1];
        let zptr1 = mid - psiz1;

        // Apply Givens rotations
        for i in givptr[level_curr]..givptr[level_curr + 1] {
            // Get rotation columns
            let c1 = givcol[0][i - 1] - 1;
            let c2 = givcol[1][i - 1] - 1;

            // Apply rotation
            let temp = z[zptr1 + c1];
            z[zptr1 + c1] = givnum[0][i - 1] * temp + givnum[1][i - 1] * z[zptr1 + c2];
            z[zptr1 + c2] = -givnum[1][i - 1] * temp + givnum[0][i - 1] * z[zptr1 + c2];
        }

        for i in givptr[level_curr + 1]..givptr[level_curr + 2] {
            // Get rotation columns for second block
            let c1 = givcol[0][i - 1] - 1;
            let c2 = givcol[1][i - 1] - 1;

            // Apply rotation to second block
            let temp = z[mid - 1 + c1];
            z[mid - 1 + c1] = givnum[0][i - 1] * temp + givnum[1][i - 1] * z[mid - 1 + c2];
            z[mid - 1 + c2] = -givnum[1][i - 1] * temp + givnum[0][i - 1] * z[mid - 1 + c2];
        }

        // Permute the vector by copying to temporary space
        for i in 0..psiz1 {
            ztemp[i] = z[zptr1 + perm[prmptr[level_curr] + i] - 1];
        }
        for i in 0..psiz2 {
            ztemp[psiz1 + i] = z[mid + perm[prmptr[level_curr + 1] + i] - 1];
        }

        // Transform first block
        let mut bsiz = ((qptr[level_curr + 1] - qptr[level_curr]) as f64).sqrt() as usize;
        if bsiz > 0 {
            // Create matrix view for multiplication
            let mut qmat = vec![vec![0.0; bsiz]; bsiz];
            for i in 0..bsiz {
                for j in 0..bsiz {
                    qmat[i][j] = q[qptr[level_curr] + i * bsiz + j];
                }
            }

            // Perform matrix-vector multiplication
            let mut temp = vec![0.0; bsiz];
            for i in 0..bsiz {
                temp[i] = 0.0;
                for j in 0..bsiz {
                    temp[i] += qmat[j][i] * ztemp[j];
                }
            }

            // Copy result back
            for i in 0..bsiz {
                z[zptr1 + i] = temp[i];
            }
        }

        // Copy any remaining elements
        for i in bsiz..psiz1 {
            z[zptr1 + i] = ztemp[i];
        }

        // Transform second block
        bsiz = ((qptr[level_curr + 2] - qptr[level_curr + 1]) as f64).sqrt() as usize;
        if bsiz > 0 {
            // Create matrix view for multiplication
            let mut qmat = vec![vec![0.0; bsiz]; bsiz];
            for i in 0..bsiz {
                for j in 0..bsiz {
                    qmat[i][j] = q[qptr[level_curr + 1] + i * bsiz + j];
                }
            }

            // Perform matrix-vector multiplication
            let mut temp = vec![0.0; bsiz];
            for i in 0..bsiz {
                temp[i] = 0.0;
                for j in 0..bsiz {
                    temp[i] += qmat[j][i] * ztemp[psiz1 + j];
                }
            }

            // Copy result back
            for i in 0..bsiz {
                z[mid + i] = temp[i];
            }
        }

        // Copy any remaining elements
        for i in bsiz..psiz2 {
            z[mid + i] = ztemp[psiz1 + i];
        }

        // Update pointer for next level
        ptr += 1 << (tlvls - k);
    }

    Ok(())
}

/// Applies a sequence of plane rotations to a real matrix A from either the left or right.
/// This function corresponds to LAPACK's DLASR routine.
///
/// # Arguments
/// * `side` - Specifies whether P is applied from the left ('L') or right ('R')
/// * `pivot` - Specifies rotation plane:
///   - 'V': Variable pivot (k,k+1)
///   - 'T': Top pivot (1,k+1)
///   - 'B': Bottom pivot (k,z)
/// * `direct` - Specifies order of rotations:
///   - 'F': Forward P = P(z-1)*...*P(2)*P(1)
///   - 'B': Backward P = P(1)*P(2)*...*P(z-1)
/// * `m` - Number of rows in matrix A
/// * `n` - Number of columns in matrix A
/// * `c` - Cosines of the rotations
/// * `s` - Sines of the rotations
/// * `a` - The matrix to be transformed
/// * `lda` - Leading dimension of A
pub fn dlasr(
    side: char,
    pivot: char,
    direct: char,
    m: usize,
    n: usize,
    c: &[f64],
    s: &[f64],
    a: &mut [Vec<f64>],
    lda: usize,
) -> Result<(), Error> {
    // Parameter validation
    if !matches!(side, 'L' | 'R') {
        return Err(Error(1));
    }
    if !matches!(pivot, 'V' | 'T' | 'B') {
        return Err(Error(2));
    }
    if !matches!(direct, 'F' | 'B') {
        return Err(Error(3));
    }
    if m < 0 {
        return Err(Error(4));
    }
    if n < 0 {
        return Err(Error(5));
    }
    if lda < m.max(1) {
        return Err(Error(9));
    }

    // Quick return if possible
    if m == 0 || n == 0 {
        return Ok(());
    }

    // Constants
    const ONE: f64 = 1.0;
    const ZERO: f64 = 0.0;

    match side {
        'L' => {
            // Form P * A
            match pivot {
                'V' => {
                    // Variable pivot
                    if direct == 'F' {
                        // Forward sequence
                        for j in 0..m - 1 {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..n {
                                    let temp = a[j + 1][i];
                                    a[j + 1][i] = ctemp * temp - stemp * a[j][i];
                                    a[j][i] = stemp * temp + ctemp * a[j][i];
                                }
                            }
                        }
                    } else {
                        // Backward sequence
                        for j in (0..m - 1).rev() {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..n {
                                    let temp = a[j + 1][i];
                                    a[j + 1][i] = ctemp * temp - stemp * a[j][i];
                                    a[j][i] = stemp * temp + ctemp * a[j][i];
                                }
                            }
                        }
                    }
                }
                'T' => {
                    // Top pivot
                    if direct == 'F' {
                        for j in 1..m {
                            let ctemp = c[j - 1];
                            let stemp = s[j - 1];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..n {
                                    let temp = a[j][i];
                                    a[j][i] = ctemp * temp - stemp * a[0][i];
                                    a[0][i] = stemp * temp + ctemp * a[0][i];
                                }
                            }
                        }
                    } else {
                        for j in (1..m).rev() {
                            let ctemp = c[j - 1];
                            let stemp = s[j - 1];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..n {
                                    let temp = a[j][i];
                                    a[j][i] = ctemp * temp - stemp * a[0][i];
                                    a[0][i] = stemp * temp + ctemp * a[0][i];
                                }
                            }
                        }
                    }
                }
                'B' => {
                    // Bottom pivot
                    if direct == 'F' {
                        for j in 0..m - 1 {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..n {
                                    let temp = a[j][i];
                                    a[j][i] = stemp * a[m - 1][i] + ctemp * temp;
                                    a[m - 1][i] = ctemp * a[m - 1][i] - stemp * temp;
                                }
                            }
                        }
                    } else {
                        for j in (0..m - 1).rev() {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..n {
                                    let temp = a[j][i];
                                    a[j][i] = stemp * a[m - 1][i] + ctemp * temp;
                                    a[m - 1][i] = ctemp * a[m - 1][i] - stemp * temp;
                                }
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        'R' => {
            // Form A * P^T
            match pivot {
                'V' => {
                    // Variable pivot
                    if direct == 'F' {
                        for j in 0..n - 1 {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..m {
                                    let temp = a[i][j + 1];
                                    a[i][j + 1] = ctemp * temp - stemp * a[i][j];
                                    a[i][j] = stemp * temp + ctemp * a[i][j];
                                }
                            }
                        }
                    } else {
                        for j in (0..n - 1).rev() {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..m {
                                    let temp = a[i][j + 1];
                                    a[i][j + 1] = ctemp * temp - stemp * a[i][j];
                                    a[i][j] = stemp * temp + ctemp * a[i][j];
                                }
                            }
                        }
                    }
                }
                'T' => {
                    // Top pivot
                    if direct == 'F' {
                        for j in 1..n {
                            let ctemp = c[j - 1];
                            let stemp = s[j - 1];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..m {
                                    let temp = a[i][j];
                                    a[i][j] = ctemp * temp - stemp * a[i][0];
                                    a[i][0] = stemp * temp + ctemp * a[i][0];
                                }
                            }
                        }
                    } else {
                        for j in (1..n).rev() {
                            let ctemp = c[j - 1];
                            let stemp = s[j - 1];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..m {
                                    let temp = a[i][j];
                                    a[i][j] = ctemp * temp - stemp * a[i][0];
                                    a[i][0] = stemp * temp + ctemp * a[i][0];
                                }
                            }
                        }
                    }
                }
                'B' => {
                    // Bottom pivot
                    if direct == 'F' {
                        for j in 0..n - 1 {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..m {
                                    let temp = a[i][j];
                                    a[i][j] = stemp * a[i][n - 1] + ctemp * temp;
                                    a[i][n - 1] = ctemp * a[i][n - 1] - stemp * temp;
                                }
                            }
                        }
                    } else {
                        for j in (0..n - 1).rev() {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != ONE || stemp != ZERO {
                                for i in 0..m {
                                    let temp = a[i][j];
                                    a[i][j] = stemp * a[i][n - 1] + ctemp * temp;
                                    a[i][n - 1] = ctemp * a[i][n - 1] - stemp * temp;
                                }
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    }

    Ok(())
}



/// Computes the updated eigensystem of a diagonal matrix after modification by a rank-one symmetric matrix.
///
/// # Parameters
/// - `n`: The dimension of the symmetric tridiagonal matrix (`n >= 0`).
/// - `d`: On entry, the eigenvalues of the rank-1-perturbed matrix. On exit, the eigenvalues of the repaired matrix (`d.len() == n`).
/// - `q`: On entry, the eigenvectors of the rank-1-perturbed matrix. On exit, the eigenvectors of the repaired tridiagonal matrix (`q.len() == n`, `q[i].len() == n`).
/// - `ldq`: The leading dimension of the array `q` (`ldq >= n`).
/// - `indxq`: On entry, the permutation which separately sorts the two subproblems in `d` into ascending order.
///            On exit, the permutation which will reintegrate the subproblems back into sorted order (`indxq.len() == n`).
/// - `rho`: The subdiagonal entry used to create the rank-1 modification.
/// - `cutpnt`: The location of the last eigenvalue in the leading sub-matrix (`min(1,n) <= cutpnt <= n/2`).
/// - `work`: Workspace array (`work.len() >= 4*n + n^2`).
/// - `iwork`: Workspace array (`iwork.len() >= 4*n`).
///
/// # Returns
/// - `Result<(), Error>` indicating success or an error.
pub fn dlaed1(
    n: usize,
    d: &mut [f64],
    q: &mut [Vec<f64>],
    ldq: usize,
    indxq: &mut [usize],
    rho: &mut f64,
    cutpnt: usize,
    work: &mut [f64],
    iwork: &mut [usize],
) -> Result<(), Error> {
    // Input validation
    if n == 0 {
        return Ok(());
    }
    if ldq < n {
        return Err(Error(-4));
    }
    if cutpnt < 1.min(n) || cutpnt > n / 2 {
        return Err(Error(-7));
    }
    if work.len() < 4 * n + n * n {
        return Err(Error(-8));
    }
    if iwork.len() < 4 * n {
        return Err(Error(-9));
    }

    // Indices for work array slices
    let iz = 0;
    let idlmda = iz + n;
    let iw = idlmda + n;
    let iq2 = iw + n;

    // Indices for integer work array slices
    let indx = 0;
    let indxc = indx + n;
    let coltyp = indxc + n;
    let indxp = coltyp + n;

    // Split work array into non-overlapping mutable slices
    let (work_iz, work_rest) = work.split_at_mut(idlmda);
    let z = &mut work_iz[iz..];

    let (work_dlamda, work_rest) = work_rest.split_at_mut(n);
    let dlamda = &mut work_dlamda[..];

    let (work_w, work_rest) = work_rest.split_at_mut(n);
    let w = &mut work_w[..];

    let (q2_storage, _work_unused) = work_rest.split_at_mut(n * n);
    // q2_storage has size n*n

    // Form the z-vector which consists of the last row of Q1 and the first row of Q2
    // Copy the last row of Q1 (up to cutpnt)
    for i in 0..cutpnt {
        z[i] = q[cutpnt - 1][i];
    }
    // Copy the first row of Q2 (from cutpnt to n)
    for i in cutpnt..n {
        z[i] = q[i][i];
    }

    // Prepare q2 as a 2D array using q2_storage
    let mut q2: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        q2.push(q2_storage[i * n..(i + 1) * n].to_vec());
    }

    // Split iwork array into non-overlapping mutable slices
    let (iwork_indx, iwork_rest) = iwork.split_at_mut(indxc);
    let indxv = &mut iwork_indx[indx..];

    let (iwork_indxc, iwork_rest) = iwork_rest.split_at_mut(n);
    let indxcv = &mut iwork_indxc[..];

    let (iwork_coltyp, iwork_indxp) = iwork_rest.split_at_mut(n);
    let coltypv = &mut iwork_coltyp[..];
    let indxpv = &mut iwork_indxp[..];

    // Initialize k (number of non-deflated eigenvalues)
    let mut k = 0;

    // Call dlaed2 to deflate eigenvalues
    let result = dlaed2(
        &mut k,
        n,
        cutpnt,
        d,
        q,
        ldq,
        indxq,
        rho,
        z,
        dlamda,
        w,
        &mut q2,
        indxv,
        indxcv,
        indxpv,
        coltypv,
    )?;

    if k != 0 {
        // Solve the secular equation
        // Prepare s for dlaed3 (size k * k)
        // We can reuse q2_storage (size n*n) to store s (size k*k)
        let s_storage_size = k * k;
        if q2_storage.len() < s_storage_size {
            return Err(Error(-8)); // Storage is sufficient
        }
        let s_storage = &mut q2_storage[..s_storage_size];
        let mut s_matrix: Vec<Vec<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            s_matrix.push(s_storage[i * k..(i + 1) * k].to_vec());
        }

        let result = dlaed3(
            k,
            n,
            cutpnt,
            d,
            q,
            ldq,
            *rho,
            dlamda,
            &q2,
            indxv,
            coltypv,
            w,
            &mut s_matrix,
        );

        // Prepare the indxq sorting permutation
        let n1 = k;
        let n2 = n - k;
        dlamrg(n1, n2, d, 1, -1, indxq);

        // Adjust indxq to be 0-based (since dlamrg may produce 1-based indices)
        for i in 0..n {
            indxq[i] = indxq[i].saturating_sub(1); // No underflow
        }
    } else {
        // All eigenvalues were deflated; set indxq to identity permutation
        for i in 0..n {
            indxq[i] = i;
        }
    }

    Ok(())
}
