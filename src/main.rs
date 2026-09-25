use anyhow::{bail, Result};
use dressed_trajectory_fields::experiment;
use std::env;
use std::path::PathBuf;

fn usage() {
    eprintln!(
        "Usage:
  cargo run --release -- reproduce [output_dir]
  cargo run --release -- native [output_dir]
  cargo run --release -- baselines [output_dir]
  cargo run --release -- figures [output_dir]
  cargo run --release -- validate [output_dir]
  cargo run --release -- reference-figures [output_dir]

Default output_dir: results"
    );
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "reproduce".to_string());
    let out = PathBuf::from(args.next().unwrap_or_else(|| "results".to_string()));

    match command.as_str() {
        "reproduce" => experiment::reproduce_all(&out),
        "native" => experiment::run_native(&out),
        "baselines" => experiment::run_baselines(&out),
        "figures" => experiment::render_computed_figures(&out),
        "validate" => experiment::validate_against_paper(&out),
        "reference-figures" => experiment::render_reference_figures(&out),
        "-h" | "--help" | "help" => {
            usage();
            Ok(())
        }
        other => {
            usage();
            bail!("unknown command: {other}")
        }
    }
}
