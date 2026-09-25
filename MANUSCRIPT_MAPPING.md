# Manuscript-to-code map

This file maps the numerical statements of the manuscript to the Rust implementation.

| Paper item | Rust implementation | Output |
|---|---|---|
| Exact transformed mean/covariance | `src/model.rs::exact_moments` | used throughout |
| Order-p dressed Taylor series | `src/model.rs::dressed_series` | Figure 4 and QFT2 curves |
| Physical action Hessian | `src/model.rs::action_hessian_at_mean` | Figures 1–3, 5 |
| Exact self-energy | `H - G^{-1}` in `src/experiment.rs` | Figures 1–3, 5 |
| Range projection | `src/math.rs::project_block_range` | Figures 3 and 5 |
| Banded Cholesky/solve | `src/math.rs::banded_cholesky*` | Figure 6 |
| Laplace | `src/baselines.rs::laplace` | Figures 7–9 |
| Mean-field Gaussian VB | `src/baselines.rs::meanfield_vb` | Figures 7–9 |
| 12-sweep damped Gaussian EP | `src/baselines.rs::ep_12_sweep` | Figures 7–9 |
| iid exact sampling | `src/baselines.rs::sample_exact` | Figures 7–11 |
| Oracle-centred covariance estimator | `src/baselines.rs::sample_covariance` with exact centre | Figures 9 and 11 |
| Figure orchestration | `src/experiment.rs` | `results/data`, `results/figures` |
| SVG rendering | `src/plot.rs` | all Rust-rendered figures |
| Paper-reference validation | `src/experiment.rs::validate_against_paper` | `results/validation.txt` |

The exact paper model matrices and conditioned target means are pinned in
`reference_inputs/`. The final paper arrays are retained separately in
`paper_reference/data/` and are not read during recomputation.
