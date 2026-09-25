
use anyhow::{Context, Result};
use csv::{ReaderBuilder, WriterBuilder};
use nalgebra::{DMatrix, DVector};
use std::fs;
use std::path::Path;

pub fn ensure_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("create directory {}", path.display()))
}

pub fn read_matrix(path: &Path) -> Result<DMatrix<f64>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(path)?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let row: Result<Vec<f64>, _> = rec.iter().map(|x| x.parse::<f64>()).collect();
        rows.push(row.with_context(|| format!("parse {}", path.display()))?);
    }
    let nrows = rows.len();
    let ncols = rows.first().map(|r| r.len()).unwrap_or(0);
    if rows.iter().any(|r| r.len() != ncols) {
        anyhow::bail!("ragged matrix in {}", path.display());
    }
    Ok(DMatrix::from_fn(nrows, ncols, |i, j| rows[i][j]))
}

pub fn read_rows(path: &Path) -> Result<Vec<DVector<f64>>> {
    let m = read_matrix(path)?;
    Ok((0..m.nrows())
        .map(|i| DVector::from_iterator(m.ncols(), (0..m.ncols()).map(|j| m[(i, j)])))
        .collect())
}

pub fn write_table(path: &Path, header: &[&str], rows: &[Vec<String>]) -> Result<()> {
    if let Some(parent) = path.parent() {
        ensure_dir(parent)?;
    }
    let mut w = WriterBuilder::new().from_path(path)?;
    w.write_record(header)?;
    for row in rows {
        w.write_record(row)?;
    }
    w.flush()?;
    Ok(())
}

pub fn read_numeric_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let header = rdr.headers()?.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let mut rows = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let parsed = rec
            .iter()
            .map(|x| x.parse::<f64>().with_context(|| format!("parse numeric CSV {}", path.display())))
            .collect::<Result<Vec<_>>>()?;
        rows.push(parsed);
    }
    Ok((header, rows))
}

pub fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    ensure_dir(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let p = entry.path();
        let out = dst.join(entry.file_name());
        if p.is_dir() {
            copy_tree(&p, &out)?;
        } else {
            fs::copy(&p, &out)?;
        }
    }
    Ok(())
}
