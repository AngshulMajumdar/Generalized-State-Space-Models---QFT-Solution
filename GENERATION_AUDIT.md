# Generation audit

This repository was assembled from the final manuscript source and its retained
numerical reproduction assets.

Checks performed during repository generation:

- No dot-prefixed file or folder exists in the repository.
- The archive was scanned after ZIP creation for dot-prefixed entries.
- The pinned Rust model equations were independently evaluated against the
  final paper reference arrays.
- Figures 1, 2, 3, 4 and 5 matched the final paper arrays to floating-point
  roundoff (maximum absolute discrepancies approximately 1e-14 or smaller).
- The analytic gradient implemented for mean-field Gaussian VB was checked
  against centred finite differences on representative mean and log-standard-
  deviation coordinates; discrepancies were below 1e-8.
- The Rust source includes smoke tests in `tests/reproduction.rs`.

The artifact-generation runtime did not provide `cargo` or `rustc`, and attempts
to install the Rust toolchain in that runtime did not complete. Therefore the
repository could not be compiled inside the artifact-generation environment.
Run the following immediately after cloning:

    cargo test --release
    cargo run --release -- reproduce

The code is written against stable Rust 2021 and uses only crates declared in
`Cargo.toml`.
