# Fixed paper target

These files pin the deterministic target used in the final manuscript.

`A.csv`, `U.csv`, `C.csv`, `Q.csv`, `R_y.csv`, and `P0.csv` are the matrices
generated from the manuscript's master NumPy seed 20260924.

`conditioned_mu_T10.csv` and `conditioned_mu_T30.csv` contain the ten stacked
conditional Gaussian means used by the baseline and native-theorem experiments,
respectively.

They are inputs, not precomputed outputs. Exact moments, Hessians, self-energies,
approximations, sampling errors, and figures are recomputed by Rust.
