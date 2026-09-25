
use crate::io;
use crate::math::{inv, kron_eye};
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use std::path::Path;

#[derive(Clone)]
pub struct PaperModel {
    pub d: usize,
    pub t: usize,
    pub nt: usize,
    pub n: usize,
    pub a: DMatrix<f64>,
    pub u: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub ry: DMatrix<f64>,
    pub p0: DMatrix<f64>,
    pub pu: DMatrix<f64>,
    pub gu: DMatrix<f64>,
    pub mus: Vec<DVector<f64>>,
}

impl PaperModel {
    pub fn load(repo_root: &Path, t: usize) -> Result<Self> {
        let inp = repo_root.join("reference_inputs");
        let a = io::read_matrix(&inp.join("A.csv"))?;
        let u = io::read_matrix(&inp.join("U.csv"))?;
        let c = io::read_matrix(&inp.join("C.csv"))?;
        let q = io::read_matrix(&inp.join("Q.csv"))?;
        let ry = io::read_matrix(&inp.join("R_y.csv"))?;
        let p0 = io::read_matrix(&inp.join("P0.csv"))?;
        let d = a.nrows();
        let nt = t + 1;
        let n = nt * d;
        let mus = io::read_rows(&inp.join(format!("conditioned_mu_T{t}.csv")))?;
        if mus.len() != 10 || mus.iter().any(|m| m.len() != n) {
            anyhow::bail!("conditioned mean file has wrong shape for T={t}");
        }

        let qi = inv(&q);
        let ryi = inv(&ry);
        let p0i = inv(&p0);
        let mut pz = DMatrix::<f64>::zeros(n, n);

        for tt in 0..nt {
            let i0 = tt * d;
            if tt == 0 {
                add_block(&mut pz, i0, i0, &p0i, 1.0);
            }
            if tt >= 1 {
                let im = (tt - 1) * d;
                add_block(&mut pz, i0, i0, &qi, 1.0);
                let ata = a.transpose() * &qi * &a;
                add_block(&mut pz, im, im, &ata, 1.0);
                let atq = a.transpose() * &qi;
                add_block(&mut pz, im, i0, &atq, -1.0);
                let qa = &qi * &a;
                add_block(&mut pz, i0, im, &qa, -1.0);
            }
            let ctrc = c.transpose() * &ryi * &c;
            add_block(&mut pz, i0, i0, &ctrc, 1.0);
        }

        let ui = inv(&u);
        let bui = kron_eye(&ui, nt);
        let pu = bui.transpose() * pz * bui;
        let gu = inv(&pu);

        Ok(Self { d, t, nt, n, a, u, c, q, ry, p0, pu, gu, mus })
    }

    pub fn hu(&self, run: usize) -> DVector<f64> {
        &self.pu * &self.mus[run]
    }
}

fn add_block(dst: &mut DMatrix<f64>, row: usize, col: usize, block: &DMatrix<f64>, scale: f64) {
    for i in 0..block.nrows() {
        for j in 0..block.ncols() {
            dst[(row + i, col + j)] += scale * block[(i, j)];
        }
    }
}

pub fn amplitude(eps: f64) -> f64 {
    1.0 - (-eps).exp()
}

pub fn transform_scalar(u: f64, eps: f64) -> f64 {
    u + amplitude(eps) * u.sin()
}

pub fn inverse_transform_scalar(x: f64, eps: f64) -> f64 {
    let a = amplitude(eps);
    let mut u = x;
    for _ in 0..30 {
        let d = 1.0 + a * u.cos();
        let delta = (u + a * u.sin() - x) / d;
        u -= delta;
        if delta.abs() < 1e-14 * (1.0 + u.abs()) {
            break;
        }
    }
    u
}

pub fn inverse_transform_vec(x: &DVector<f64>, eps: f64) -> DVector<f64> {
    DVector::from_iterator(x.len(), x.iter().map(|&v| inverse_transform_scalar(v, eps)))
}

pub struct ExactMoments {
    pub mean: DVector<f64>,
    pub cov: DMatrix<f64>,
    pub b: DVector<f64>,
    pub c: DVector<f64>,
    pub c1: DMatrix<f64>,
    pub c2: DMatrix<f64>,
}

pub fn exact_moments(model: &PaperModel, mu: &DVector<f64>, eps: f64) -> ExactMoments {
    let a = amplitude(eps);
    let n = model.n;
    let mut b = DVector::<f64>::zeros(n);
    let mut c = DVector::<f64>::zeros(n);
    let mut v = DVector::<f64>::zeros(n);
    for i in 0..n {
        v[i] = model.gu[(i, i)];
        let damp = (-0.5 * v[i]).exp();
        b[i] = damp * mu[i].sin();
        c[i] = damp * mu[i].cos();
    }

    let mut mean = mu.clone();
    mean += b.clone() * a;
    let mut xi = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let gij = model.gu[(i, j)];
            let e1 = (-0.5 * (v[i] + v[j] - 2.0 * gij)).exp();
            let e2 = (-0.5 * (v[i] + v[j] + 2.0 * gij)).exp();
            xi[(i, j)] = 0.5
                * (e1 * (mu[i] - mu[j]).cos()
                    - e2 * (mu[i] + mu[j]).cos());
        }
    }

    let mut c1 = DMatrix::<f64>::zeros(n, n);
    let mut c2 = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            c1[(i, j)] = model.gu[(i, j)] * (c[i] + c[j]);
            c2[(i, j)] = xi[(i, j)] - b[i] * b[j];
        }
    }

    let mut cov = model.gu.clone();
    cov += c1.clone() * a;
    cov += c2.clone() * (a * a);
    ExactMoments { mean, cov, b, c, c1, c2 }
}

pub fn dressed_series(model: &PaperModel, mu: &DVector<f64>, eps: f64, order: usize) -> (DVector<f64>, DMatrix<f64>) {
    let ex = exact_moments(model, mu, eps);
    let (aa, bb) = match order {
        0 => (0.0, 0.0),
        1 => (eps, 0.0),
        2 => (eps - 0.5 * eps * eps, eps * eps),
        _ => panic!("only orders 0,1,2 are used by the paper"),
    };
    let mut mean = mu.clone();
    mean += ex.b.clone() * aa;
    let mut cov = model.gu.clone();
    cov += ex.c1.clone() * aa;
    cov += ex.c2.clone() * bb;
    (mean, cov)
}

pub fn action_hessian_at_mean(model: &PaperModel, mean_x: &DVector<f64>, mu: &DVector<f64>, eps: f64) -> DMatrix<f64> {
    let a = amplitude(eps);
    let u = inverse_transform_vec(mean_x, eps);
    let mut du = u.clone();
    du -= mu;
    let q = &model.pu * du;

    let mut r = DVector::<f64>::zeros(model.n);
    let mut extra = DVector::<f64>::zeros(model.n);
    for i in 0..model.n {
        let d = 1.0 + a * u[i].cos();
        r[i] = 1.0 / d;
        let udd = a * u[i].sin() / d.powi(3);
        let jac2 = -a * u[i].cos() / d.powi(3)
            - 2.0 * a * a * u[i].sin().powi(2) / d.powi(4);
        extra[i] = udd * q[i] + jac2;
    }

    let mut h = DMatrix::<f64>::zeros(model.n, model.n);
    for i in 0..model.n {
        for j in 0..model.n {
            h[(i, j)] = r[i] * model.pu[(i, j)] * r[j];
        }
        h[(i, i)] += extra[i];
    }
    (h.clone() + h.transpose()) * 0.5
}

pub fn transformed_sample(mu: &DVector<f64>, chol_gu: &DMatrix<f64>, z: &DVector<f64>, eps: f64) -> DVector<f64> {
    let mut u = mu.clone();
    u += chol_gu * z;
    DVector::from_iterator(
        u.len(),
        u.iter().map(|&x| transform_scalar(x, eps)),
    )
}
