
use crate::baselines;
use crate::io;
use crate::math::{
    banded_cholesky, banded_cholesky_solve, block_lag_profile, fit_log_log, fit_semilog,
    frobenius_norm, inv, mean_sd, min_eigenvalue_symmetric, project_block_range,
    relative_frobenius, relative_l2, spectral_norm_symmetric,
};
use crate::model::{action_hessian_at_mean, dressed_series, exact_moments, PaperModel};
use crate::plot::{self, AxisScale, Curve, PlotSpec};
use anyhow::{Context, Result};
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MASTER_SEED: u64 = 20260924;
const N_RUNS: usize = 10;

const NATIVE_EPS: [f64; 12] = [
    0.001, 0.002, 0.004, 0.007, 0.01, 0.02, 0.04, 0.07, 0.10, 0.16, 0.25, 0.36,
];
const BASELINE_EPS: [f64; 8] = [0.01, 0.02, 0.04, 0.07, 0.10, 0.16, 0.25, 0.36];
const LAMBDAS: [f64; 4] = [0.25, 0.40, 0.60, 0.80];
const SAMPLE_COUNTS: [usize; 9] = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

fn repo_root() -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    if cwd.join("reference_inputs").exists() && cwd.join("paper_reference").exists() {
        Ok(cwd)
    } else {
        anyhow::bail!("run the binary from the repository root")
    }
}

pub fn reproduce_all(out: &Path) -> Result<()> {
    run_native(out)?;
    run_baselines(out)?;
    render_computed_figures(out)?;
    validate_against_paper(out)?;
    Ok(())
}

pub fn run_native(out: &Path) -> Result<()> {
    let root = repo_root()?;
    let data_dir = out.join("data");
    io::ensure_dir(&data_dir)?;
    let model = PaperModel::load(&root, 30)?;

    figure1(&model, &data_dir)?;
    figure2(&model, &data_dir)?;
    figure3(&model, &data_dir)?;
    figure4(&model, &data_dir)?;
    figure5(&model, &data_dir)?;
    figure6(&data_dir)?;
    Ok(())
}

fn figure1(model: &PaperModel, data_dir: &Path) -> Result<()> {
    let eps = 0.60_f64.powi(2);
    let mut rows = Vec::<Vec<String>>::new();
    for run in 0..N_RUNS {
        let mu = &model.mus[run];
        let ex = exact_moments(model, mu, eps);
        let h = action_hessian_at_mean(model, &ex.mean, mu, eps);
        let gi = inv(&ex.cov);
        let sigma = &h - &gi;
        let ph = block_lag_profile(&h, model.d, model.nt);
        let pg = block_lag_profile(&gi, model.d, model.nt);
        let ps = block_lag_profile(&sigma, model.d, model.nt);
        let scale = ph[0];
        for lag in 0..model.nt {
            rows.push(vec![
                run.to_string(),
                lag.to_string(),
                format!("{:.17e}", ph[lag] / scale),
                format!("{:.17e}", pg[lag] / scale),
                format!("{:.17e}", ps[lag] / scale),
            ]);
        }
    }
    io::write_table(
        &data_dir.join("figure1.csv"),
        &["run", "lag", "bare", "inverse_two_point", "self_energy"],
        &rows,
    )
}

fn figure2(model: &PaperModel, data_dir: &Path) -> Result<()> {
    let mut rows = Vec::<Vec<String>>::new();
    for &lambda in &LAMBDAS {
        let eps = lambda * lambda;
        for run in 0..N_RUNS {
            let mu = &model.mus[run];
            let ex = exact_moments(model, mu, eps);
            let h = action_hessian_at_mean(model, &ex.mean, mu, eps);
            let gi = inv(&ex.cov);
            let sigma = &h - &gi;
            let ph = block_lag_profile(&h, model.d, model.nt);
            let ps = block_lag_profile(&sigma, model.d, model.nt);
            let scale = ph[0];
            for lag in 0..model.nt {
                rows.push(vec![
                    run.to_string(),
                    format!("{lambda:.17e}"),
                    lag.to_string(),
                    format!("{:.17e}", ps[lag] / scale),
                ]);
            }
        }
    }
    io::write_table(
        &data_dir.join("figure2.csv"),
        &["run", "lambda", "lag", "normalized_self_energy"],
        &rows,
    )
}

fn figure3(model: &PaperModel, data_dir: &Path) -> Result<()> {
    let eps = 0.80_f64.powi(2);
    let mut rows = Vec::<Vec<String>>::new();
    for run in 0..N_RUNS {
        let mu = &model.mus[run];
        let ex = exact_moments(model, mu, eps);
        let h = action_hessian_at_mean(model, &ex.mean, mu, eps);
        let gi = inv(&ex.cov);
        let sigma = &h - &gi;
        for range in 1..=10 {
            let sr = project_block_range(&sigma, model.d, model.nt, range);
            let gr = inv(&(h.clone() - sr));
            let err = relative_frobenius(&gr, &ex.cov);
            rows.push(vec![run.to_string(), range.to_string(), format!("{err:.17e}")]);
        }
    }
    io::write_table(
        &data_dir.join("figure3.csv"),
        &["run", "range", "relative_covariance_error"],
        &rows,
    )
}

fn figure4(model: &PaperModel, data_dir: &Path) -> Result<()> {
    let mut rows = Vec::<Vec<String>>::new();
    for run in 0..N_RUNS {
        let mu = &model.mus[run];
        for &eps in &NATIVE_EPS {
            let ex = exact_moments(model, mu, eps);
            let mut vals = Vec::<f64>::new();
            for p in 0..=2 {
                let (m, _) = dressed_series(model, mu, eps, p);
                vals.push(relative_l2(&m, &ex.mean));
            }
            for p in 0..=2 {
                let (_, g) = dressed_series(model, mu, eps, p);
                vals.push(relative_frobenius(&g, &ex.cov));
            }
            let mut row = vec![run.to_string(), format!("{eps:.17e}")];
            row.extend(vals.iter().map(|x| format!("{x:.17e}")));
            rows.push(row);
        }
    }
    io::write_table(
        &data_dir.join("figure4.csv"),
        &["run", "epsilon", "mean_p0", "mean_p1", "mean_p2", "cov_p0", "cov_p1", "cov_p2"],
        &rows,
    )
}

fn figure5(model: &PaperModel, data_dir: &Path) -> Result<()> {
    let eps = 0.80_f64.powi(2);
    let mut rows = Vec::<Vec<String>>::new();
    for run in 0..N_RUNS {
        let mu = &model.mus[run];
        let ex = exact_moments(model, mu, eps);
        let h = action_hessian_at_mean(model, &ex.mean, mu, eps);
        let gi = inv(&ex.cov);
        let sigma = &h - &gi;
        let hmin = min_eigenvalue_symmetric(&h);
        for range in 0..=10 {
            let sr = project_block_range(&sigma, model.d, model.nt, range);
            let sn = spectral_norm_symmetric(&sr);
            rows.push(vec![
                run.to_string(),
                range.to_string(),
                format!("{hmin:.17e}"),
                format!("{sn:.17e}"),
                format!("{:.17e}", hmin - sn),
            ]);
        }
    }
    io::write_table(
        &data_dir.join("figure5.csv"),
        &["run", "range", "h", "sigma_norm", "margin"],
        &rows,
    )
}

fn make_banded_spd(t: usize, d: usize, range: usize, seed: u64) -> Vec<Vec<f64>> {
    let nt = t + 1;
    let n = nt * d;
    let bw = (range + 1) * d - 1;
    let mut a = vec![vec![0.0; n]; bw + 1];
    let mut rowsum = vec![0.0; n];
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;

    for tt in 0..nt {
        for i0 in 0..d {
            let i = tt * d + i0;
            for j0 in 0..i0 {
                let j = tt * d + j0;
                let v: f64 = normal.sample(&mut rng);
                let v = 0.004 * v;
                a[i - j][j] = v;
                rowsum[i] += v.abs();
                rowsum[j] += v.abs();
            }
        }
    }
    for lag in 1..=range {
        let sc = 0.010 / lag as f64;
        for tt in 0..(nt - lag) {
            let ss = tt + lag;
            for i0 in 0..d {
                let i = ss * d + i0;
                for j0 in 0..d {
                    let j = tt * d + j0;
                    let v: f64 = normal.sample(&mut rng);
                    let v = sc * v;
                    a[i - j][j] = v;
                    rowsum[i] += v.abs();
                    rowsum[j] += v.abs();
                }
            }
        }
    }
    for i in 0..n {
        a[0][i] = 1.0 + 1.05 * rowsum[i];
    }
    a
}

fn figure6(data_dir: &Path) -> Result<()> {
    let horizons = [200usize, 400, 800, 1600, 3200, 6400];
    let d = 8;
    let range = 3;
    let mut rows = Vec::<Vec<String>>::new();
    for (it, &t) in horizons.iter().enumerate() {
        let n = (t + 1) * d;
        for rep in 0..N_RUNS {
            let a = make_banded_spd(t, d, range, MASTER_SEED + 10000 * it as u64 + rep as u64);
            let mut rng = ChaCha20Rng::seed_from_u64(MASTER_SEED + 50000 * it as u64 + rep as u64);
            let normal = StandardNormal;
            let rhs: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();

            let lwarm = banded_cholesky(a.clone());
            black_box(banded_cholesky_solve(&lwarm, &rhs));

            let start = Instant::now();
            let l = banded_cholesky(a);
            let x = banded_cholesky_solve(&l, &rhs);
            black_box(x);
            let sec = start.elapsed().as_secs_f64();
            rows.push(vec![t.to_string(), rep.to_string(), format!("{sec:.17e}")]);
        }
    }
    io::write_table(
        &data_dir.join("figure6.csv"),
        &["horizon", "repeat", "seconds"],
        &rows,
    )
}

pub fn run_baselines(out: &Path) -> Result<()> {
    let root = repo_root()?;
    let data_dir = out.join("data");
    io::ensure_dir(&data_dir)?;
    let model = PaperModel::load(&root, 10)?;

    let mut rows78 = Vec::<Vec<String>>::new();
    let mut fig9_store: Vec<BTreeMap<String, Vec<f64>>> = Vec::new();

    for run in 0..N_RUNS {
        let mu = &model.mus[run];
        for (j, &eps) in BASELINE_EPS.iter().enumerate() {
            let ex = exact_moments(&model, mu, eps);
            let (mq, gq) = dressed_series(&model, mu, eps, 2);
            let vb = baselines::meanfield_vb(&model, mu, eps)?;
            let (mep, gep) = baselines::ep_12_sweep(&model, mu, eps)?;
            let (ml, gl) = baselines::laplace(&model, mu, eps)?;
            let samples = baselines::sample_exact(
                &model,
                mu,
                eps,
                2000,
                MASTER_SEED + 900000 + 1000 * run as u64 + j as u64,
            )?;
            let ms = baselines::sample_mean(&samples);
            let gs = baselines::sample_covariance(&samples, &ex.mean, samples.len() as f64);

            let methods = [
                ("QFT2", mq, gq),
                ("VB", vb.mean, vb.cov),
                ("EP", mep, gep),
                ("Laplace", ml, gl),
                ("OracleSampling", ms, gs),
            ];
            let mut row = vec![run.to_string(), format!("{eps:.17e}")];
            for (_, m, g) in &methods {
                row.push(format!("{:.17e}", relative_l2(m, &ex.mean)));
                row.push(format!("{:.17e}", relative_frobenius(g, &ex.cov)));
            }
            rows78.push(row);

            if (eps - 0.36).abs() < 1e-12 {
                let scale = block_lag_profile(&ex.cov, model.d, model.nt)[0];
                let mut map = BTreeMap::<String, Vec<f64>>::new();
                map.insert("Exact".to_string(), normalize_profile(block_lag_profile(&ex.cov, model.d, model.nt), scale));
                for (name, _, g) in &methods {
                    map.insert((*name).to_string(), normalize_profile(block_lag_profile(g, model.d, model.nt), scale));
                }
                fig9_store.push(map);
            }
        }
    }

    io::write_table(
        &data_dir.join("figure7_8.csv"),
        &[
            "run", "epsilon",
            "mean_QFT2", "cov_QFT2",
            "mean_VB", "cov_VB",
            "mean_EP", "cov_EP",
            "mean_Laplace", "cov_Laplace",
            "mean_OracleSampling", "cov_OracleSampling",
        ],
        &rows78,
    )?;

    let mut rows9 = Vec::<Vec<String>>::new();
    for run in 0..N_RUNS {
        let map = &fig9_store[run];
        for lag in 0..model.nt {
            rows9.push(vec![
                run.to_string(),
                lag.to_string(),
                format!("{:.17e}", map["Exact"][lag]),
                format!("{:.17e}", map["QFT2"][lag]),
                format!("{:.17e}", map["VB"][lag]),
                format!("{:.17e}", map["EP"][lag]),
                format!("{:.17e}", map["Laplace"][lag]),
                format!("{:.17e}", map["OracleSampling"][lag]),
            ]);
        }
    }
    io::write_table(
        &data_dir.join("figure9.csv"),
        &["run", "lag", "Exact", "QFT2", "VB", "EP", "Laplace", "OracleSampling"],
        &rows9,
    )?;

    sampling_budget_curves(&model, &data_dir)?;
    Ok(())
}

fn normalize_profile(mut p: Vec<f64>, scale: f64) -> Vec<f64> {
    for v in &mut p { *v /= scale; }
    p
}

fn sampling_budget_curves(model: &PaperModel, data_dir: &Path) -> Result<()> {
    let eps = 0.36;
    let nmax = *SAMPLE_COUNTS.last().unwrap();
    let mut rows10 = Vec::<Vec<String>>::new();
    let mut rows11 = Vec::<Vec<String>>::new();

    for run in 0..N_RUNS {
        let mu = &model.mus[run];
        let ex = exact_moments(model, mu, eps);

        let samples_mean = baselines::sample_exact(
            model,
            mu,
            eps,
            nmax,
            MASTER_SEED + 700000 + run as u64,
        )?;
        for &nn in &SAMPLE_COUNTS {
            let m = baselines::sample_mean(&samples_mean[..nn]);
            rows10.push(vec![
                run.to_string(),
                nn.to_string(),
                format!("{:.17e}", relative_l2(&m, &ex.mean)),
            ]);
        }

        let samples_cov = baselines::sample_exact(
            model,
            mu,
            eps,
            nmax,
            MASTER_SEED + 810000 + run as u64,
        )?;
        for &nn in &SAMPLE_COUNTS {
            let g = baselines::sample_covariance(&samples_cov[..nn], &ex.mean, nn as f64);
            rows11.push(vec![
                run.to_string(),
                nn.to_string(),
                format!("{:.17e}", relative_frobenius(&g, &ex.cov)),
            ]);
        }
    }
    io::write_table(&data_dir.join("figure10.csv"), &["run", "N", "error"], &rows10)?;
    io::write_table(&data_dir.join("figure11.csv"), &["run", "N", "error"], &rows11)?;
    Ok(())
}

pub fn render_computed_figures(out: &Path) -> Result<()> {
    let data = out.join("data");
    let figs = out.join("figures");
    io::ensure_dir(&figs)?;
    render_1(&data, &figs)?;
    render_2(&data, &figs)?;
    render_3(&data, &figs)?;
    render_4(&data, &figs)?;
    render_5(&data, &figs)?;
    render_6(&data, &figs)?;
    render_7_8(&data, &figs)?;
    render_9(&data, &figs)?;
    render_10_11(&data, &figs)?;
    Ok(())
}

fn aggregate_by_x(rows: &[Vec<f64>], xcol: usize, ycol: usize) -> Vec<(f64, f64, f64)> {
    let mut map: BTreeMap<String, (f64, Vec<f64>)> = BTreeMap::new();
    for r in rows {
        let x = r[xcol];
        let key = format!("{x:.12e}");
        map.entry(key).or_insert((x, Vec::new())).1.push(r[ycol]);
    }
    let mut v: Vec<(f64, f64, f64)> = map
        .values()
        .map(|(x, ys)| {
            let (m, s) = mean_sd(ys);
            (*x, m, s)
        })
        .collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    v
}

fn curve_from_agg(label: &str, agg: &[(f64, f64, f64)]) -> Curve {
    Curve {
        label: label.to_string(),
        x: agg.iter().map(|v| v.0).collect(),
        y: agg.iter().map(|v| v.1).collect(),
        sd: Some(agg.iter().map(|v| v.2).collect()),
    }
}

fn render_1(data: &Path, figs: &Path) -> Result<()> {
    let (_, rows) = io::read_numeric_csv(&data.join("figure1.csv"))?;
    let labels = [(2, "Bare Hessian H(m)"), (3, "Inverse two-point G^-1"), (4, "Self-energy Sigma")];
    let curves = labels.iter().map(|(c, l)| curve_from_agg(l, &aggregate_by_x(&rows, 1, *c))).collect();
    plot::write_svg(&figs.join("figure1.svg"), &PlotSpec {
        title: "Generated temporal nonlocality".into(),
        x_label: "Temporal lag |t-s|".into(),
        y_label: "Normalized mean block Frobenius norm".into(),
        x_scale: AxisScale::Linear, y_scale: AxisScale::Log10, curves,
    })
}

fn render_2(data: &Path, figs: &Path) -> Result<()> {
    let (_, rows) = io::read_numeric_csv(&data.join("figure2.csv"))?;
    let mut curves = Vec::new();
    for &lam in &LAMBDAS {
        let sub: Vec<Vec<f64>> = rows.iter().filter(|r| (r[1] - lam).abs() < 1e-9 && r[2] >= 2.0 && r[2] <= 10.0).cloned().collect();
        curves.push(curve_from_agg(&format!("lambda={lam:.2}"), &aggregate_by_x(&sub, 2, 3)));
    }
    plot::write_svg(&figs.join("figure2.svg"), &PlotSpec {
        title: "Exact self-energy localization".into(),
        x_label: "Temporal lag |t-s|".into(),
        y_label: "Normalized self-energy block norm".into(),
        x_scale: AxisScale::Linear, y_scale: AxisScale::Log10, curves,
    })
}

fn render_3(data: &Path, figs: &Path) -> Result<()> {
    let (_, rows) = io::read_numeric_csv(&data.join("figure3.csv"))?;
    let agg = aggregate_by_x(&rows, 1, 2);
    let mut curve = curve_from_agg("Range truncation error", &agg);
    let xfit: Vec<f64> = agg.iter().filter(|v| v.1 >= 1e-11).map(|v| v.0).collect();
    let yfit: Vec<f64> = agg.iter().filter(|v| v.1 >= 1e-11).map(|v| v.1).collect();
    let (slope, intercept) = fit_semilog(&xfit, &yfit);
    let fit = Curve {
        label: format!("fit slope {slope:.3}"),
        x: agg.iter().map(|v| v.0).collect(),
        y: agg.iter().map(|v| (intercept + slope * v.0).exp()).collect(),
        sd: None,
    };
    plot::write_svg(&figs.join("figure3.svg"), &PlotSpec {
        title: "Exact self-energy range truncation".into(),
        x_label: "Self-energy half-bandwidth R".into(),
        y_label: "Relative covariance error".into(),
        x_scale: AxisScale::Linear, y_scale: AxisScale::Log10,
        curves: vec![curve, fit],
    })
}

fn render_4(data: &Path, figs: &Path) -> Result<()> {
    let (_, rows) = io::read_numeric_csv(&data.join("figure4.csv"))?;
    for (kind, basecol, filename) in [("mean", 2usize, "figure4a.svg"), ("covariance", 5usize, "figure4b.svg")] {
        let mut curves = Vec::new();
        for p in 0..=2 {
            let agg = aggregate_by_x(&rows, 1, basecol + p);
            let xfit: Vec<f64> = agg.iter().take(5).map(|v| v.0).collect();
            let yfit: Vec<f64> = agg.iter().take(5).map(|v| v.1).collect();
            let (slope, _) = fit_log_log(&xfit, &yfit);
            curves.push(curve_from_agg(&format!("p={p}, slope={slope:.3}"), &agg));
        }
        plot::write_svg(&figs.join(filename), &PlotSpec {
            title: format!("Perturbative consistency: {kind}"),
            x_label: "epsilon".into(),
            y_label: "Relative error".into(),
            x_scale: AxisScale::Log10, y_scale: AxisScale::Log10, curves,
        })?;
    }
    Ok(())
}

fn render_5(data: &Path, figs: &Path) -> Result<()> {
    let (_, rows) = io::read_numeric_csv(&data.join("figure5.csv"))?;
    let curves = [
        (2usize, "h = lambda_min(H)"),
        (3usize, "||Sigma^(R)||_2"),
        (4usize, "margin"),
    ].iter().map(|(c,l)| curve_from_agg(l, &aggregate_by_x(&rows, 1, *c))).collect();
    plot::write_svg(&figs.join("figure5.svg"), &PlotSpec {
        title: "Pointwise spectral admissibility".into(),
        x_label: "Range R".into(), y_label: "Spectral quantity".into(),
        x_scale: AxisScale::Linear, y_scale: AxisScale::Linear, curves,
    })
}

fn render_6(data: &Path, figs: &Path) -> Result<()> {
    let (_, rows) = io::read_numeric_csv(&data.join("figure6.csv"))?;
    let agg = aggregate_by_x(&rows, 0, 2);
    let x: Vec<f64> = agg.iter().map(|v| v.0).collect();
    let y: Vec<f64> = agg.iter().map(|v| v.1 * 1e3).collect();
    let sd: Vec<f64> = agg.iter().map(|v| v.2 * 1e3).collect();
    let (slope, intercept) = fit_log_log(&x, &y);
    let fit = Curve { label: format!("fit slope={slope:.3}"), x: x.clone(), y: x.iter().map(|xx| (intercept + slope*xx.ln()).exp()).collect(), sd: None };
    plot::write_svg(&figs.join("figure6.svg"), &PlotSpec {
        title: "Fixed-range banded-solve scaling".into(),
        x_label: "Trajectory horizon T".into(), y_label: "Solve time (ms)".into(),
        x_scale: AxisScale::Log10, y_scale: AxisScale::Log10,
        curves: vec![Curve { label: "Rust banded solve".into(), x, y, sd: Some(sd) }, fit],
    })
}

fn render_7_8(data: &Path, figs: &Path) -> Result<()> {
    let (_, rows) = io::read_numeric_csv(&data.join("figure7_8.csv"))?;
    let names = ["QFT2", "VB", "EP", "Laplace", "OracleSampling"];
    for (kind, offset, file) in [("mean", 2usize, "figure7.svg"), ("covariance", 3usize, "figure8.svg")] {
        let mut curves = Vec::new();
        for (mi, name) in names.iter().enumerate() {
            let col = offset + 2 * mi;
            curves.push(curve_from_agg(name, &aggregate_by_x(&rows, 1, col)));
        }
        plot::write_svg(&figs.join(file), &PlotSpec {
            title: format!("Baseline comparison: {kind} error"),
            x_label: "epsilon".into(), y_label: "Relative error".into(),
            x_scale: AxisScale::Log10, y_scale: AxisScale::Log10, curves,
        })?;
    }
    Ok(())
}

fn render_9(data: &Path, figs: &Path) -> Result<()> {
    let (_, rows) = io::read_numeric_csv(&data.join("figure9.csv"))?;
    let names = ["Exact", "QFT2", "VB", "EP", "Laplace", "OracleSampling"];
    let mut curves = Vec::new();
    for (i, name) in names.iter().enumerate() {
        curves.push(curve_from_agg(name, &aggregate_by_x(&rows, 1, 2+i)));
    }
    plot::write_svg(&figs.join("figure9.svg"), &PlotSpec {
        title: "Temporal covariance profile at epsilon=0.36".into(),
        x_label: "Temporal lag |t-s|".into(),
        y_label: "Covariance block norm / exact lag-0 norm".into(),
        x_scale: AxisScale::Linear, y_scale: AxisScale::Log10, curves,
    })
}

fn render_10_11(data: &Path, figs: &Path) -> Result<()> {
    for (num, title) in [(10, "Finite-N mean error"), (11, "Finite-N covariance error")] {
        let (_, rows) = io::read_numeric_csv(&data.join(format!("figure{num}.csv")))?;
        let agg = aggregate_by_x(&rows, 1, 2);
        let x: Vec<f64> = agg.iter().map(|v| v.0).collect();
        let y: Vec<f64> = agg.iter().map(|v| v.1).collect();
        let (slope, intercept) = fit_log_log(&x, &y);
        let fit = Curve { label: format!("fit slope={slope:.3}"), x: x.clone(), y: x.iter().map(|xx| (intercept + slope*xx.ln()).exp()).collect(), sd: None };
        plot::write_svg(&figs.join(format!("figure{num}.svg")), &PlotSpec {
            title: title.into(), x_label: "Exact iid samples N".into(), y_label: "Relative error".into(),
            x_scale: AxisScale::Log10, y_scale: AxisScale::Log10,
            curves: vec![curve_from_agg("Oracle iid", &agg), fit],
        })?;
    }
    Ok(())
}

pub fn render_reference_figures(out: &Path) -> Result<()> {
    let root = repo_root()?;
    let src = root.join("paper_reference").join("figures");
    let dst = out.join("paper_reference_figures");
    io::copy_tree(&src, &dst)?;
    Ok(())
}

pub fn validate_against_paper(out: &Path) -> Result<()> {
    let root = repo_root()?;
    let refdir = root.join("paper_reference").join("data");
    let datadir = out.join("data");
    io::ensure_dir(out)?;
    let mut report = String::new();
    report.push_str("Dressed Trajectory Fields Rust validation\n");
    report.push_str("========================================\n\n");
    report.push_str("Deterministic exact-family diagnostics should agree closely with the paper reference.\n");
    report.push_str("Sampling and timing are informational because Rust uses ChaCha20 and the local machine timer,\n");
    report.push_str("whereas the paper used NumPy PCG64 and SciPy/OpenBLAS on a different machine.\n\n");

    for name in ["figure1.csv", "figure2.csv", "figure3.csv", "figure4.csv", "figure5.csv", "figure7_8.csv", "figure9.csv"] {
        let comp = datadir.join(name);
        let reference = refdir.join(name);
        if !comp.exists() {
            report.push_str(&format!("{name}: NOT RUN\n"));
            continue;
        }
        let (hc, rc) = io::read_numeric_csv(&comp)?;
        let (hr, rr) = io::read_numeric_csv(&reference)?;
        if hc != hr || rc.len() != rr.len() {
            report.push_str(&format!("{name}: shape/header mismatch\n"));
            continue;
        }
        let mut max_rel = 0.0_f64;
        let mut mean_rel = 0.0_f64;
        let mut count = 0usize;
        for (a, b) in rc.iter().zip(rr.iter()) {
            for j in 0..a.len() {
                // run/lag/range/epsilon columns are identifiers; differences there are exact anyway.
                let denom = b[j].abs().max(1e-12);
                let rel = (a[j] - b[j]).abs() / denom;
                max_rel = max_rel.max(rel);
                mean_rel += rel;
                count += 1;
            }
        }
        mean_rel /= count.max(1) as f64;
        report.push_str(&format!("{name}: mean relative difference={mean_rel:.3e}, max={max_rel:.3e}\n"));
    }

    for name in ["figure6.csv", "figure10.csv", "figure11.csv"] {
        if datadir.join(name).exists() {
            report.push_str(&format!("{name}: INFORMATIONAL (machine/RNG dependent)\n"));
        }
    }
    fs::write(out.join("validation.txt"), report)?;
    Ok(())
}
