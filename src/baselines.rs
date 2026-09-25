
use crate::lbfgs::{minimize, LbfgsOptions};
use crate::math::{cholesky_2x2, gh_nodes_weights, inverse_2x2, inv, min_eigenvalue_symmetric};
use crate::model::{amplitude, inverse_transform_scalar, transformed_sample, PaperModel};
use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

pub fn laplace(model: &PaperModel, mu: &DVector<f64>, eps: f64) -> Result<(DVector<f64>, DMatrix<f64>)> {
    let a = amplitude(eps);
    let mut u = mu.clone();
    let mut hu = model.pu.clone();

    for _ in 0..60 {
        let mut hp = DVector::<f64>::zeros(model.n);
        let mut hpp = DVector::<f64>::zeros(model.n);
        for i in 0..model.n {
            let d = 1.0 + a * u[i].cos();
            hp[i] = -a * u[i].sin() / d;
            hpp[i] = (-a * u[i].cos() - a * a) / (d * d);
        }

        let mut du = u.clone();
        du -= mu;
        let grad = &model.pu * du + hp;
        hu = model.pu.clone();
        for i in 0..model.n {
            hu[(i, i)] += hpp[i];
        }
        let delta = inv(&hu) * grad;
        let next = &u - &delta;
        let stop = delta.norm() < 1e-11 * (1.0 + u.norm());
        u = next;
        if stop {
            break;
        }
    }

    if min_eigenvalue_symmetric(&hu) <= 0.0 {
        bail!("Laplace Hessian is not positive definite");
    }
    let hu_inv = inv(&hu);
    let mut mean_x = DVector::<f64>::zeros(model.n);
    let mut dvec = DVector::<f64>::zeros(model.n);
    for i in 0..model.n {
        dvec[i] = 1.0 + a * u[i].cos();
        mean_x[i] = u[i] + a * u[i].sin();
    }
    let mut cov = DMatrix::<f64>::zeros(model.n, model.n);
    for i in 0..model.n {
        for j in 0..model.n {
            cov[(i, j)] = dvec[i] * hu_inv[(i, j)] * dvec[j];
        }
    }
    Ok((mean_x, (cov.clone() + cov.transpose()) * 0.5))
}

pub struct VbResult {
    pub mean: DVector<f64>,
    pub cov: DMatrix<f64>,
    pub converged: bool,
    pub iterations: usize,
}

pub fn meanfield_vb(model: &PaperModel, mu: &DVector<f64>, eps: f64) -> Result<VbResult> {
    let (m0, g0) = laplace(model, mu, eps)?;
    let mut theta = vec![0.0; 2 * model.n];
    for i in 0..model.n {
        theta[i] = m0[i];
        theta[model.n + i] = g0[(i, i)].max(1e-10).sqrt().ln();
    }
    let hu = &model.pu * mu;
    let (nodes, weights) = gh_nodes_weights(20);
    let a = amplitude(eps);
    let n = model.n;

    let fg = |th: &[f64]| -> Result<(f64, Vec<f64>)> {
        let mut eu = DVector::<f64>::zeros(n);
        let mut eu2 = DVector::<f64>::zeros(n);
        let mut ejac = DVector::<f64>::zeros(n);

        let mut deu_m = vec![0.0; n];
        let mut deu_l = vec![0.0; n];
        let mut deu2_m = vec![0.0; n];
        let mut deu2_l = vec![0.0; n];
        let mut dej_m = vec![0.0; n];
        let mut dej_l = vec![0.0; n];

        for i in 0..n {
            let m = th[i];
            let s = th[n + i].exp();
            for (&node, &w) in nodes.iter().zip(weights.iter()) {
                let dxdl = std::f64::consts::SQRT_2 * s * node;
                let x = m + dxdl;
                let u = inverse_transform_scalar(x, eps);
                let d = 1.0 + a * u.cos();
                let dudx = 1.0 / d;
                let dlogjdx = -a * u.sin() / (d * d);

                eu[i] += w * u;
                eu2[i] += w * u * u;
                ejac[i] += w * d.ln();

                deu_m[i] += w * dudx;
                deu_l[i] += w * dudx * dxdl;
                deu2_m[i] += w * 2.0 * u * dudx;
                deu2_l[i] += w * 2.0 * u * dudx * dxdl;
                dej_m[i] += w * dlogjdx;
                dej_l[i] += w * dlogjdx * dxdl;
            }
        }

        let peu = &model.pu * &eu;
        let mut value = 0.5 * eu.dot(&peu) - hu.dot(&eu);
        for i in 0..n {
            value += 0.5 * model.pu[(i, i)] * (eu2[i] - eu[i] * eu[i]);
            value += ejac[i] - th[n + i];
        }

        let mut grad = vec![0.0; 2 * n];
        for i in 0..n {
            let coef = peu[i] - model.pu[(i, i)] * eu[i] - hu[i];
            grad[i] = coef * deu_m[i]
                + 0.5 * model.pu[(i, i)] * deu2_m[i]
                + dej_m[i];
            grad[n + i] = coef * deu_l[i]
                + 0.5 * model.pu[(i, i)] * deu2_l[i]
                + dej_l[i]
                - 1.0;
        }
        Ok((value, grad))
    };

    let res = minimize(theta, fg, &LbfgsOptions::default())?;
    let mut mean = DVector::<f64>::zeros(n);
    let mut cov = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        mean[i] = res.x[i];
        let s = res.x[n + i].exp();
        cov[(i, i)] = s * s;
    }
    Ok(VbResult {
        mean,
        cov,
        converged: res.converged,
        iterations: res.iterations,
    })
}

#[derive(Clone)]
struct PairSite {
    i: usize,
    j: usize,
    pij: f64,
    k: [[f64; 2]; 2],
    h: [f64; 2],
}

fn pair_min_eigenvalue(a: [[f64; 2]; 2]) -> f64 {
    let tr = a[0][0] + a[1][1];
    let diff = a[0][0] - a[1][1];
    0.5 * (tr - (diff * diff + 4.0 * a[0][1] * a[1][0]).sqrt())
}

fn mat2_vec(a: [[f64; 2]; 2], x: [f64; 2]) -> [f64; 2] {
    [a[0][0] * x[0] + a[0][1] * x[1], a[1][0] * x[0] + a[1][1] * x[1]]
}

fn assemble_ep(tau: &[f64], eta: &[f64], pairs: &[PairSite]) -> (DMatrix<f64>, DVector<f64>) {
    let n = tau.len();
    let mut p = DMatrix::<f64>::zeros(n, n);
    let mut h = DVector::<f64>::from_column_slice(eta);
    for i in 0..n {
        p[(i, i)] = tau[i];
    }
    for site in pairs {
        p[(site.i, site.i)] += site.k[0][0];
        p[(site.i, site.j)] += site.k[0][1];
        p[(site.j, site.i)] += site.k[1][0];
        p[(site.j, site.j)] += site.k[1][1];
        h[site.i] += site.h[0];
        h[site.j] += site.h[1];
    }
    (p, h)
}

pub fn ep_12_sweep(model: &PaperModel, mu: &DVector<f64>, eps: f64) -> Result<(DVector<f64>, DMatrix<f64>)> {
    let a = amplitude(eps);
    let hu = &model.pu * mu;
    let n = model.n;

    let mut tau: Vec<f64> = (0..n).map(|i| model.pu[(i, i)]).collect();
    let mut eta: Vec<f64> = hu.iter().copied().collect();
    let mut pairs = Vec::<PairSite>::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let pij = model.pu[(i, j)];
            if pij.abs() > 1e-12 {
                pairs.push(PairSite {
                    i,
                    j,
                    pij,
                    k: [[0.0, pij], [pij, 0.0]],
                    h: [0.0, 0.0],
                });
            }
        }
    }

    let (gh1x, gh1w) = gh_nodes_weights(24);
    let (gh2x, gh2w) = gh_nodes_weights(10);
    let (mut p, mut h) = assemble_ep(&tau, &eta, &pairs);
    let mut g = inv(&p);
    let mut m = &g * &h;

    for _ in 0..12 {
        let mut nt = tau.clone();
        let mut ne = eta.clone();
        let mut npairs = pairs.clone();

        // Unary tilted moments.
        for i in 0..n {
            let v = g[(i, i)];
            let mtau = 1.0 / v;
            let meta = m[i] / v;
            let ct = mtau - tau[i];
            let ce = meta - eta[i];
            if ct <= 1e-10 {
                continue;
            }
            let vc = 1.0 / ct;
            let mc = ce / ct;
            let mut xs = Vec::with_capacity(gh1x.len());
            let mut logw = Vec::with_capacity(gh1x.len());
            for (&node, &qw) in gh1x.iter().zip(gh1w.iter()) {
                let x = mc + (2.0 * vc).sqrt() * node;
                let u = inverse_transform_scalar(x, eps);
                let d = 1.0 + a * u.cos();
                let lf = -0.5 * model.pu[(i, i)] * u * u + hu[i] * u - d.ln();
                xs.push(x);
                logw.push(qw.ln() + lf);
            }
            let maxlw = logw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut ww: Vec<f64> = logw.iter().map(|v| (v - maxlw).exp()).collect();
            let sw: f64 = ww.iter().sum();
            for w in &mut ww { *w /= sw; }
            let mt: f64 = ww.iter().zip(xs.iter()).map(|(w, x)| w * x).sum();
            let vt: f64 = ww.iter().zip(xs.iter()).map(|(w, x)| w * (x - mt).powi(2)).sum();
            if vt > 1e-12 {
                nt[i] = 1.0 / vt - ct;
                ne[i] = mt / vt - ce;
            }
        }

        // Pair tilted moments.
        for (idx, site) in pairs.iter().enumerate() {
            let i = site.i;
            let j = site.j;
            let gs = [[g[(i, i)], g[(i, j)]], [g[(j, i)], g[(j, j)]]];
            let Some(qm) = inverse_2x2(gs) else { continue; };
            let qh = mat2_vec(qm, [m[i], m[j]]);
            let kc = [
                [qm[0][0] - site.k[0][0], qm[0][1] - site.k[0][1]],
                [qm[1][0] - site.k[1][0], qm[1][1] - site.k[1][1]],
            ];
            let hc = [qh[0] - site.h[0], qh[1] - site.h[1]];
            if pair_min_eigenvalue(kc) <= 1e-9 {
                continue;
            }
            let Some(vc) = inverse_2x2(kc) else { continue; };
            let mc = mat2_vec(vc, hc);
            let Some(lc) = cholesky_2x2(vc) else { continue; };

            let mut points: Vec<[f64; 2]> = Vec::with_capacity(100);
            let mut logw: Vec<f64> = Vec::with_capacity(100);
            for (ii, &x1) in gh2x.iter().enumerate() {
                for (jj, &x2) in gh2x.iter().enumerate() {
                    let z0 = std::f64::consts::SQRT_2 * x1;
                    let z1 = std::f64::consts::SQRT_2 * x2;
                    let xx0 = mc[0] + lc[0][0] * z0;
                    let xx1 = mc[1] + lc[1][0] * z0 + lc[1][1] * z1;
                    let u0 = inverse_transform_scalar(xx0, eps);
                    let u1 = inverse_transform_scalar(xx1, eps);
                    let lf = -site.pij * u0 * u1;
                    points.push([xx0, xx1]);
                    logw.push((gh2w[ii] * gh2w[jj]).ln() + lf);
                }
            }
            let maxlw = logw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut ww: Vec<f64> = logw.iter().map(|v| (v - maxlw).exp()).collect();
            let sw: f64 = ww.iter().sum();
            for w in &mut ww { *w /= sw; }
            let mt0: f64 = ww.iter().zip(points.iter()).map(|(w, x)| w * x[0]).sum();
            let mt1: f64 = ww.iter().zip(points.iter()).map(|(w, x)| w * x[1]).sum();

            let mut ct = [[0.0; 2]; 2];
            for (w, x) in ww.iter().zip(points.iter()) {
                let d0 = x[0] - mt0;
                let d1 = x[1] - mt1;
                ct[0][0] += w * d0 * d0;
                ct[0][1] += w * d0 * d1;
                ct[1][0] += w * d1 * d0;
                ct[1][1] += w * d1 * d1;
            }
            ct[0][0] += 1e-11;
            ct[1][1] += 1e-11;
            let Some(kt) = inverse_2x2(ct) else { continue; };
            let ht = mat2_vec(kt, [mt0, mt1]);
            npairs[idx].k = [
                [kt[0][0] - kc[0][0], kt[0][1] - kc[0][1]],
                [kt[1][0] - kc[1][0], kt[1][1] - kc[1][1]],
            ];
            npairs[idx].h = [ht[0] - hc[0], ht[1] - hc[1]];
        }

        let mut damp = 0.15;
        let mut accepted = None;
        while damp > 1e-5 {
            let tt: Vec<f64> = tau.iter().zip(nt.iter()).map(|(x, y)| (1.0 - damp) * x + damp * y).collect();
            let ee: Vec<f64> = eta.iter().zip(ne.iter()).map(|(x, y)| (1.0 - damp) * x + damp * y).collect();
            let mut pp = pairs.clone();
            for k in 0..pp.len() {
                for r in 0..2 {
                    for c in 0..2 {
                        pp[k].k[r][c] = (1.0 - damp) * pairs[k].k[r][c] + damp * npairs[k].k[r][c];
                    }
                    pp[k].h[r] = (1.0 - damp) * pairs[k].h[r] + damp * npairs[k].h[r];
                }
            }
            let (pt, ht) = assemble_ep(&tt, &ee, &pp);
            if pt.clone().cholesky().is_some() {
                accepted = Some((tt, ee, pp, pt, ht));
                break;
            }
            damp *= 0.5;
        }
        let Some((tt, ee, pp, pt, ht)) = accepted else { break; };
        tau = tt;
        eta = ee;
        pairs = pp;
        p = pt;
        h = ht;
        g = inv(&p);
        m = &g * &h;
    }

    Ok((m, g))
}

pub fn sample_exact(
    model: &PaperModel,
    mu: &DVector<f64>,
    eps: f64,
    n_samples: usize,
    seed: u64,
) -> Result<Vec<DVector<f64>>> {
    let chol = model
        .gu
        .clone()
        .cholesky()
        .ok_or_else(|| anyhow::anyhow!("G_u is not positive definite"))?
        .l()
        .clone_owned();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let z = DVector::from_iterator(model.n, (0..model.n).map(|_| -> f64 { normal.sample(&mut rng) }));
        out.push(transformed_sample(mu, &chol, &z, eps));
    }
    Ok(out)
}

pub fn sample_mean(samples: &[DVector<f64>]) -> DVector<f64> {
    let n = samples[0].len();
    let mut m = DVector::<f64>::zeros(n);
    for x in samples {
        m += x.clone();
    }
    m / samples.len() as f64
}

pub fn sample_covariance(samples: &[DVector<f64>], center: &DVector<f64>, divisor: f64) -> DMatrix<f64> {
    let n = center.len();
    let mut g = DMatrix::<f64>::zeros(n, n);
    for x in samples {
        let mut d = x.clone();
        d -= center;
        g += &d * d.transpose();
    }
    g / divisor
}
