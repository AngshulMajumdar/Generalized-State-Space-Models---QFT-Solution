
use dressed_trajectory_fields::math::{block_lag_profile, inv};
use dressed_trajectory_fields::model::{action_hessian_at_mean, exact_moments, PaperModel};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn manufactured_family_has_generated_lag_two_inverse_two_point() {
    let model = PaperModel::load(&root(), 30).unwrap();
    let mu = &model.mus[0];
    let eps = 0.60_f64.powi(2);
    let ex = exact_moments(&model, mu, eps);
    let h = action_hessian_at_mean(&model, &ex.mean, mu, eps);
    let gi = inv(&ex.cov);
    let ph = block_lag_profile(&h, model.d, model.nt);
    let pg = block_lag_profile(&gi, model.d, model.nt);
    assert!(ph[2].abs() < 1e-12);
    assert!(pg[2] / ph[0] > 1e-7);
}

#[test]
fn exact_covariance_is_positive_definite() {
    let model = PaperModel::load(&root(), 10).unwrap();
    for mu in &model.mus {
        let ex = exact_moments(&model, mu, 0.36);
        assert!(ex.cov.clone().cholesky().is_some());
    }
}
