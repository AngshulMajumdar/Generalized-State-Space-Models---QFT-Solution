
use anyhow::{bail, Result};

#[derive(Clone, Debug)]
pub struct LbfgsOptions {
    pub max_iter: usize,
    pub grad_tol: f64,
    pub function_tol: f64,
    pub history: usize,
    pub c1: f64,
    pub min_step: f64,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        Self {
            max_iter: 80,
            grad_tol: 1e-7,
            function_tol: 1e-10,
            history: 10,
            c1: 1e-4,
            min_step: 1e-12,
        }
    }
}

pub struct LbfgsResult {
    pub x: Vec<f64>,
    pub value: f64,
    pub grad_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub fn minimize<F>(mut x: Vec<f64>, mut fg: F, opt: &LbfgsOptions) -> Result<LbfgsResult>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
{
    let (mut f, mut g) = fg(&x)?;
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    for iter in 0..opt.max_iter {
        let gnorm = l2(&g);
        if gnorm <= opt.grad_tol {
            return Ok(LbfgsResult {
                x,
                value: f,
                grad_norm: gnorm,
                iterations: iter,
                converged: true,
            });
        }

        let mut q = g.clone();
        let mut alpha = vec![0.0; s_hist.len()];
        for idx in (0..s_hist.len()).rev() {
            alpha[idx] = rho_hist[idx] * dot(&s_hist[idx], &q);
            axpy(&mut q, -alpha[idx], &y_hist[idx]);
        }

        let scale = if let (Some(s), Some(y)) = (s_hist.last(), y_hist.last()) {
            let yy = dot(y, y);
            if yy > 0.0 { dot(s, y) / yy } else { 1.0 }
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|v| scale * v).collect();
        for idx in 0..s_hist.len() {
            let beta = rho_hist[idx] * dot(&y_hist[idx], &r);
            axpy(&mut r, alpha[idx] - beta, &s_hist[idx]);
        }
        let mut direction: Vec<f64> = r.iter().map(|v| -v).collect();
        if dot(&direction, &g) >= 0.0 {
            direction = g.iter().map(|v| -v).collect();
        }

        let directional = dot(&g, &direction);
        let mut step = 1.0;
        let mut accepted = None;
        while step >= opt.min_step {
            let trial: Vec<f64> = x
                .iter()
                .zip(direction.iter())
                .map(|(a, d)| a + step * d)
                .collect();
            let (ft, gt) = fg(&trial)?;
            if ft.is_finite() && ft <= f + opt.c1 * step * directional {
                accepted = Some((trial, ft, gt));
                break;
            }
            step *= 0.5;
        }
        let Some((x_new, f_new, g_new)) = accepted else {
            bail!("L-BFGS line search failed");
        };

        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-14 {
            if s_hist.len() == opt.history {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
        }

        let rel = (f - f_new).abs() / (1.0 + f.abs());
        x = x_new;
        f = f_new;
        g = g_new;
        if rel <= opt.function_tol {
            return Ok(LbfgsResult {
                x,
                value: f,
                grad_norm: l2(&g),
                iterations: iter + 1,
                converged: true,
            });
        }
    }

    Ok(LbfgsResult {
        x,
        value: f,
        grad_norm: l2(&g),
        iterations: opt.max_iter,
        converged: false,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yy, xx) in y.iter_mut().zip(x.iter()) {
        *yy += a * xx;
    }
}

fn l2(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
