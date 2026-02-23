use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::io;

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn norm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn normalize(v: &mut [f64]) {
    let n = norm2(v);
    if n > 0.0 {
        for x in v {
            *x /= n;
        }
    }
}

pub fn lanczos_smallest<F>(
    n: usize,
    k: usize,
    extra_iters: usize,
    mut matvec: F,
) -> io::Result<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: FnMut(&[f64], &mut [f64]),
{
    if n == 0 || k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let m = (k + extra_iters).max(k + 2).min(n);
    let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_CAFEu64);
    let mut q = vec![0.0; n];
    for v in &mut q {
        *v = rng.gen_range(-1.0..1.0);
    }
    normalize(&mut q);
    if norm2(&q) == 0.0 {
        q[0] = 1.0;
    }

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut alphas: Vec<f64> = Vec::with_capacity(m);
    let mut betas: Vec<f64> = Vec::with_capacity(m.saturating_sub(1));
    let mut prev_q = vec![0.0; n];
    let mut w = vec![0.0; n];

    for iter in 0..m {
        matvec(&q, &mut w);
        if iter > 0 {
            let beta_prev = betas[iter - 1];
            for i in 0..n {
                w[i] -= beta_prev * prev_q[i];
            }
        }

        let alpha = dot(&q, &w);
        for i in 0..n {
            w[i] -= alpha * q[i];
        }

        // Full re-orthogonalization for stability.
        for b in &basis {
            let proj = dot(b, &w);
            for i in 0..n {
                w[i] -= proj * b[i];
            }
        }

        let beta = norm2(&w);
        basis.push(q.clone());
        alphas.push(alpha);

        if iter + 1 >= m || beta < 1e-12 {
            break;
        }

        betas.push(beta);
        prev_q.copy_from_slice(&q);
        for i in 0..n {
            q[i] = w[i] / beta;
        }
    }

    let t_dim = alphas.len();
    if t_dim == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Build tridiagonal T and solve exactly (small matrix).
    let mut t = vec![vec![0.0f64; t_dim]; t_dim];
    for i in 0..t_dim {
        t[i][i] = alphas[i];
        if i + 1 < t_dim {
            t[i][i + 1] = betas[i];
            t[i + 1][i] = betas[i];
        }
    }

    // Use simple Jacobi for small dense symmetric T.
    let (evals, evecs_t) = jacobi_symmetric(t, 256, 1e-12);
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_unstable_by(|&a, &b| evals[a].total_cmp(&evals[b]));
    let out_k = k.min(order.len());

    let mut out_vals = Vec::with_capacity(out_k);
    let mut out_vecs = Vec::with_capacity(out_k);
    for &idx in order.iter().take(out_k) {
        out_vals.push(evals[idx]);
        let mut v = vec![0.0f64; n];
        for b in 0..t_dim {
            let coeff = evecs_t[b][idx];
            for (row, vrow) in v.iter_mut().enumerate().take(n) {
                *vrow += basis[b][row] * coeff;
            }
        }
        normalize(&mut v);
        out_vecs.push(v);
    }

    Ok((out_vals, out_vecs))
}

fn jacobi_symmetric(
    mut a: Vec<Vec<f64>>,
    max_sweeps: usize,
    tol: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0f64; n]; n];
    for (i, row) in v.iter_mut().enumerate().take(n) {
        row[i] = 1.0;
    }

    for _ in 0..max_sweeps {
        let mut p = 0usize;
        let mut q = 1usize.min(n.saturating_sub(1));
        let mut max_val = 0.0f64;
        for (i, row) in a.iter().enumerate().take(n) {
            for (j, &val) in row.iter().enumerate().skip(i + 1).take(n.saturating_sub(i + 1)) {
                let val = val.abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < tol || n < 2 {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let phi = 0.5 * (aqq - app).atan2(2.0 * apq);
        let (s, c) = phi.sin_cos();

        for row in a.iter_mut().take(n) {
            let aip = row[p];
            let aiq = row[q];
            row[p] = c * aip - s * aiq;
            row[q] = s * aip + c * aiq;
        }
        let old_p = a[p].clone();
        let old_q = a[q].clone();
        for j in 0..n {
            a[p][j] = c * old_p[j] - s * old_q[j];
            a[q][j] = s * old_p[j] + c * old_q[j];
        }
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for row in v.iter_mut().take(n) {
            let vip = row[p];
            let viq = row[q];
            row[p] = c * vip - s * viq;
            row[q] = s * vip + c * viq;
        }
    }

    let evals = (0..n).map(|i| a[i][i]).collect::<Vec<_>>();
    (evals, v)
}

fn chebyshev_coefficients<F>(degree: usize, mut f: F) -> Vec<f64>
where
    F: FnMut(f64) -> f64,
{
    let m = degree + 1;
    let mut coeffs = vec![0.0f64; m];
    for (j, coeff) in coeffs.iter_mut().enumerate().take(m) {
        let mut sum = 0.0;
        for i in 0..m {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / (m as f64);
            let x = theta.cos();
            sum += f(x) * (j as f64 * theta).cos();
        }
        *coeff = 2.0 * sum / (m as f64);
    }
    coeffs
}

pub fn estimate_ngec_hutchinson<F>(
    n: usize,
    trace_l: f64,
    lambda_max: f64,
    degree: usize,
    samples: usize,
    mut matvec: F,
) -> io::Result<f64>
where
    F: FnMut(&[f64], &mut [f64]),
{
    if n == 0 || n == 1 || trace_l <= 0.0 || lambda_max <= 0.0 {
        return Ok(0.0);
    }

    // Map t in [-1,1] -> x in [0, lambda_max].
    let coeffs = chebyshev_coefficients(degree, |t| {
        let x = 0.5 * lambda_max * (t + 1.0);
        if x <= 1e-15 {
            0.0
        } else {
            x * x.ln()
        }
    });

    let mut rng = ChaCha8Rng::seed_from_u64(0xBAD5EEDu64);
    let mut trace_est = 0.0f64;
    let mut z = vec![0.0f64; n];
    let mut t0 = vec![0.0f64; n];
    let mut t1 = vec![0.0f64; n];
    let mut t2 = vec![0.0f64; n];
    let mut az = vec![0.0f64; n];
    let mut bz = vec![0.0f64; n];
    let mut pz = vec![0.0f64; n];

    for _ in 0..samples.max(1) {
        for zi in &mut z {
            *zi = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        }

        t0.copy_from_slice(&z);
        for i in 0..n {
            pz[i] = 0.5 * coeffs[0] * t0[i];
        }
        if degree == 0 {
            trace_est += dot(&z, &pz);
            continue;
        }

        matvec(&t0, &mut az);
        for i in 0..n {
            bz[i] = (2.0 / lambda_max) * az[i] - t0[i];
            t1[i] = bz[i];
            pz[i] += coeffs[1] * t1[i];
        }

        for coeff in coeffs.iter().take(degree + 1).skip(2) {
            matvec(&t1, &mut az);
            for i in 0..n {
                bz[i] = (2.0 / lambda_max) * az[i] - t1[i];
                t2[i] = 2.0 * bz[i] - t0[i];
                pz[i] += *coeff * t2[i];
            }
            t0.copy_from_slice(&t1);
            t1.copy_from_slice(&t2);
        }

        trace_est += dot(&z, &pz);
    }

    let trace_l_log_l = trace_est / (samples.max(1) as f64);
    let ngec = (trace_l.ln() - trace_l_log_l / trace_l) / (n as f64).ln();
    Ok(ngec.max(0.0))
}
