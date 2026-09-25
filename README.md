# Dressed Trajectory Fields — Rust reproduction repository

This repository reproduces the numerical results for:

**Angshul Majumdar, “Dressed Trajectory Fields: Generated Temporal Nonlocality in Conditioned Stochastic Dynamics” (2026).**

The implementation is intentionally Rust-first. No Python, JAX, NumPy, SciPy, MATLAB, R, shell pipeline, or notebook is required to recompute the experiments.

## What is reproduced

The `reproduce` command recomputes the numerical data for all eleven paper figures:

- Figure 1 — generated off-band inverse-two-point structure.
- Figure 2 — exact self-energy localization for four nonlinearities.
- Figure 3 — covariance error under exact self-energy range truncation.
- Figure 4a/4b — perturbative mean/covariance error for orders `p=0,1,2`.
- Figure 5 — pointwise spectral-admissibility quantities.
- Figure 6 — fixed-range block-banded Cholesky/solve scaling.
- Figures 7–8 — order-2 dressed series against mean-field VB, 12-sweep damped Gaussian EP, Laplace, and iid exact sampling.
- Figure 9 — temporal covariance profile at `epsilon=0.36`.
- Figures 10–11 — finite-sample mean and oracle-centred covariance errors.

The Rust code implements the model and baselines described in the final manuscript:

- exact transformed-Gaussian moments;
- order-0/1/2 dressed Taylor series;
- exact physical-state Hessian and self-energy;
- range projection;
- block-banded Cholesky/solve kernel;
- Laplace trajectory approximation with the transformation Jacobian;
- mean-field Gaussian VB using 20-point Gauss–Hermite quadrature and a local L-BFGS implementation;
- the finite **12-sweep damped Gaussian EP calculation** used by the paper, with 24-point unary and `10 x 10` pair Gauss–Hermite rules;
- exact iid trajectory sampling.

## Important reproducibility detail

The repository contains the exact deterministic model matrices and the ten conditioned Gaussian means used by the paper in `reference_inputs/`.

Those fixtures were generated once using the paper’s stated NumPy master seed `20260924`. Pinning them is deliberate: it removes irrelevant cross-language differences between NumPy PCG64 and Rust RNG implementations while preserving exactly the conditioned targets used by the manuscript.

For fresh Monte Carlo draws and the timing experiment, the Rust recomputation uses deterministic `ChaCha20Rng` seeds derived from the same integer seed schedule. Therefore:

- Figures 1–5 and the deterministic parts of Figures 7–9 should closely reproduce the paper curves.
- Figures 10–11 preserve the `N^{-1/2}` Monte Carlo law but are not expected to be bit-for-bit identical to NumPy PCG64 draws.
- Figure 6 is machine dependent by construction; the quantity of interest is the fitted scaling exponent, not the absolute milliseconds.

The exact paper arrays and paper PDF figures are retained under `paper_reference/` for comparison.

## Requirements

- Rust stable, edition 2021.
- BLAS is **not** required. Linear algebra uses `nalgebra`.
- The output plots are SVG generated directly by Rust.

## Quick start

From the repository root:

```text
cargo run --release -- reproduce
```

This creates:

```text
results/
  data/
    figure1.csv
    ...
    figure11.csv
  figures/
    figure1.svg
    ...
    figure11.svg
  validation.txt
```

The repository intentionally contains no dot-prefixed files or folders.

## Commands

Recompute everything:

```text
cargo run --release -- reproduce
```

Only native trajectory-field experiments (Figures 1–6):

```text
cargo run --release -- native
cargo run --release -- figures
```

Only baseline experiments (Figures 7–11):

```text
cargo run --release -- baselines
cargo run --release -- figures
```

Compare a completed Rust run with the paper reference arrays:

```text
cargo run --release -- validate
```

Copy the exact paper PDF figures into the selected output directory:

```text
cargo run --release -- reference-figures
```

Use a different output directory by supplying it as the second argument:

```text
cargo run --release -- reproduce my_results
```

## Model

The latent model is

```text
z_0 ~ N(0, P0)
z_t = A z_{t-1} + xi_t,       xi_t ~ N(0,Q)
w_t = C z_t + eta_t,          eta_t ~ N(0,R_y)
u_t = U z_t
x_t = phi_epsilon(u_t)
phi_epsilon(u) = u + a_epsilon sin(u)
a_epsilon = 1 - exp(-epsilon)
```

with the observation-coordinate transformation

```text
y_t = sinh(beta w_t)/beta.
```

The weak-coupling family is analytic on `epsilon in (-log 2, infinity)`. The numerical experiments use `epsilon=lambda^2 >= 0`.

For the stacked conditional Gaussian field

```text
u | y ~ N(mu, G_u),
```

the exact physical-state moments are evaluated analytically in `src/model.rs`.

## Paper parameters

State dimension:

```text
d = 8
```

Master seed used to generate the fixed paper target:

```text
20260924
```

Native experiment horizon:

```text
T = 30
```

Baseline comparison horizon:

```text
T = 10
```

Native perturbative grid:

```text
0.001, 0.002, 0.004, 0.007, 0.01, 0.02,
0.04, 0.07, 0.10, 0.16, 0.25, 0.36
```

Baseline grid:

```text
0.01, 0.02, 0.04, 0.07, 0.10, 0.16, 0.25, 0.36
```

Self-energy localization sweep:

```text
lambda = 0.25, 0.40, 0.60, 0.80
epsilon = lambda^2
```

## Repository layout

```text
Cargo.toml
README.md
CITATION.cff
src/
  main.rs
  lib.rs
  model.rs
  math.rs
  lbfgs.rs
  baselines.rs
  experiment.rs
  io.rs
  plot.rs
tests/
  reproduction.rs
reference_inputs/
  A.csv
  U.csv
  C.csv
  Q.csv
  R_y.csv
  P0.csv
  conditioned_mu_T10.csv
  conditioned_mu_T30.csv
paper_reference/
  manuscript.pdf
  data/
  figures/
```

## Validation philosophy

`paper_reference/data` contains the final arrays used by the submitted manuscript. The Rust implementation does **not** consume them when recomputing the experiments; they are only used by `validate`.

`reference_inputs`, by contrast, are part of the definition of the exact paper target. They pin the model matrices and conditioned means so that language-specific RNG details do not silently change the problem being solved.

## Numerical implementation notes

### Mean-field VB

The code uses the same family and numerical protocol as the paper:

- diagonal Gaussian `q(x)`;
- 20-point Gauss–Hermite quadrature;
- initialization at the Laplace mean and marginal Laplace standard deviations;
- optimization in `(m, log s)`;
- maximum 80 L-BFGS iterations;
- function tolerance `1e-10`;
- gradient tolerance `1e-7`.

`src/lbfgs.rs` provides the L-BFGS implementation, including an Armijo backtracking line search.

### EP

The EP calculation is deliberately the finite calculation reported in the paper rather than a claimed fixed point:

- full global Gaussian;
- initialization at the `epsilon=0` Gaussian reference;
- 24-point one-dimensional Gauss–Hermite unary tilted moments;
- `10 x 10` product Gauss–Hermite pair tilted moments;
- synchronous site proposals;
- global damping `0.15`;
- positive-definiteness backtracking;
- exactly 12 sweeps.

### Sampling covariance

The covariance budget experiment uses the oracle-centred estimator

```text
Ghat_N = (1/N) sum_i (X_i - m_star)(X_i - m_star)^T,
```

matching the theorem in the manuscript.

## Output differences to expect

Absolute timing values depend on CPU/compiler/system load.

Fresh sampling curves vary because this repository uses a Rust RNG rather than NumPy PCG64. The asymptotic slope and scale should agree statistically.

Small floating-point deviations can also arise from different QR/eigensolver/linear-algebra implementations, but the model matrices and conditioned means are pinned to the paper target, so those effects are limited to the numerical algorithms themselves.

## No hidden files

This distribution intentionally contains no dot-prefixed file or directory. It also contains no editor metadata, notebook checkpoints, cache directories, build outputs, or OS metadata.
