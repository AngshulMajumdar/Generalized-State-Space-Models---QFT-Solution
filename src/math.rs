
use nalgebra::{DMatrix, DVector, SymmetricEigen};

pub fn inv(m: &DMatrix<f64>) -> DMatrix<f64> {
    m.clone().try_inverse().expect("matrix inversion failed")
}

pub fn kron_eye(block: &DMatrix<f64>, nblocks: usize) -> DMatrix<f64> {
    let br = block.nrows();
    let bc = block.ncols();
    let mut out = DMatrix::<f64>::zeros(nblocks * br, nblocks * bc);
    for b in 0..nblocks {
        for i in 0..br {
            for j in 0..bc {
                out[(b * br + i, b * bc + j)] = block[(i, j)];
            }
        }
    }
    out
}

pub fn block_lag_profile(m: &DMatrix<f64>, d: usize, nt: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(nt);
    for lag in 0..nt {
        let mut sum = 0.0;
        let mut count = 0usize;
        for t in 0..(nt - lag) {
            let s = t + lag;
            let mut ss = 0.0;
            for i in 0..d {
                for j in 0..d {
                    let v = m[(t * d + i, s * d + j)];
                    ss += v * v;
                }
            }
            sum += ss.sqrt();
            count += 1;
        }
        out.push(sum / count as f64);
    }
    out
}

pub fn project_block_range(m: &DMatrix<f64>, d: usize, nt: usize, range: usize) -> DMatrix<f64> {
    let mut out = DMatrix::<f64>::zeros(m.nrows(), m.ncols());
    for t in 0..nt {
        let s0 = t.saturating_sub(range);
        let s1 = usize::min(nt - 1, t + range);
        for s in s0..=s1 {
            for i in 0..d {
                for j in 0..d {
                    out[(t * d + i, s * d + j)] = m[(t * d + i, s * d + j)];
                }
            }
        }
    }
    out
}

pub fn frobenius_norm(m: &DMatrix<f64>) -> f64 {
    m.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn relative_frobenius(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    let mut d = a.clone();
    d -= b;
    frobenius_norm(&d) / frobenius_norm(b)
}

pub fn relative_l2(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    let mut d = a.clone();
    d -= b;
    d.norm() / b.norm()
}

pub fn min_eigenvalue_symmetric(m: &DMatrix<f64>) -> f64 {
    let e = SymmetricEigen::new((m.clone() + m.transpose()) * 0.5);
    e.eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

pub fn spectral_norm_symmetric(m: &DMatrix<f64>) -> f64 {
    let e = SymmetricEigen::new((m.clone() + m.transpose()) * 0.5);
    e.eigenvalues.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
}

pub fn fit_log_log(x: &[f64], y: &[f64]) -> (f64, f64) {
    let lx: Vec<f64> = x.iter().map(|v| v.ln()).collect();
    let ly: Vec<f64> = y.iter().map(|v| v.ln()).collect();
    linear_fit(&lx, &ly)
}

pub fn fit_semilog(x: &[f64], y: &[f64]) -> (f64, f64) {
    let ly: Vec<f64> = y.iter().map(|v| v.ln()).collect();
    linear_fit(x, &ly)
}

pub fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let denom = n * sxx - sx * sx;
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

pub fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (mean, var.sqrt())
}

pub fn inverse_2x2(a: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-18 {
        return None;
    }
    Some([
        [a[1][1] / det, -a[0][1] / det],
        [-a[1][0] / det, a[0][0] / det],
    ])
}

pub fn cholesky_2x2(a: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    if a[0][0] <= 0.0 {
        return None;
    }
    let l00 = a[0][0].sqrt();
    let l10 = a[1][0] / l00;
    let rem = a[1][1] - l10 * l10;
    if rem <= 0.0 {
        return None;
    }
    Some([[l00, 0.0], [l10, rem.sqrt()]])
}

pub fn gh_nodes_weights(order: usize) -> (Vec<f64>, Vec<f64>) {
    // Golub-Welsch for physicists' Hermite weight exp(-x^2).
    let mut j = DMatrix::<f64>::zeros(order, order);
    for i in 0..order.saturating_sub(1) {
        let b = (((i + 1) as f64) / 2.0).sqrt();
        j[(i, i + 1)] = b;
        j[(i + 1, i)] = b;
    }
    let eig = SymmetricEigen::new(j);
    let mut pairs: Vec<(f64, f64)> = (0..order)
        .map(|k| {
            let node = eig.eigenvalues[k];
            let weight_normalized = eig.eigenvectors[(0, k)].powi(2);
            (node, weight_normalized)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    (
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
    )
}

pub fn standardize_rows(rows: &[Vec<f64>]) -> Vec<(f64, f64)> {
    if rows.is_empty() {
        return Vec::new();
    }
    let cols = rows[0].len();
    (0..cols)
        .map(|j| {
            let v: Vec<f64> = rows.iter().map(|r| r[j]).collect();
            mean_sd(&v)
        })
        .collect()
}

// Symmetric lower-banded Cholesky for Figure 6.
pub fn banded_cholesky(mut a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let bw = a.len() - 1;
    let n = a[0].len();
    for j in 0..n {
        let k0 = j.saturating_sub(bw);
        let mut diag = a[0][j];
        for k in k0..j {
            let ljk = a[j - k][k];
            diag -= ljk * ljk;
        }
        assert!(diag > 0.0, "band matrix lost positive definiteness");
        let ljj = diag.sqrt();
        a[0][j] = ljj;

        let imax = usize::min(n - 1, j + bw);
        for i in (j + 1)..=imax {
            let kk0 = usize::max(i.saturating_sub(bw), k0);
            let mut s = a[i - j][j];
            for k in kk0..j {
                let lik = a[i - k][k];
                let ljk = a[j - k][k];
                s -= lik * ljk;
            }
            a[i - j][j] = s / ljj;
        }
    }
    a
}

pub fn banded_cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let bw = l.len() - 1;
    let n = b.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let k0 = i.saturating_sub(bw);
        let mut s = b[i];
        for k in k0..i {
            s -= l[i - k][k] * y[k];
        }
        y[i] = s / l[0][i];
    }

    let mut x = vec![0.0; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let kmax = usize::min(n - 1, i + bw);
        let mut s = y[i];
        for k in (i + 1)..=kmax {
            s -= l[k - i][i] * x[k];
        }
        x[i] = s / l[0][i];
    }
    x
}
