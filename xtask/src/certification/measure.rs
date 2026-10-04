use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::spec::CertResult;

#[path = "measure_child.rs"]
mod measure_child;
#[path = "measure_process.rs"]
mod measure_process;

pub(crate) use measure_child::run as run_child;
pub(crate) use measure_process::run;

#[derive(Debug, Deserialize, Serialize)]
struct ChildMeasurement {
    vector: Vec<f32>,
    batch: Vec<BatchComparison>,
    installed_manifest_sha256: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct VectorComparison {
    byte_equal: bool,
    max_absolute_difference: f64,
    minimum_cosine: f64,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct BatchComparison {
    position: usize,
    comparison: VectorComparison,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct MeasurementRecord {
    schema_version: u32,
    kind: String,
    source_commit: String,
    source_dirty: bool,
    model_id: String,
    model_revision: String,
    profile: String,
    spec_sha256: String,
    installed_manifest_sha256: Option<String>,
    configured_threads: usize,
    comparison_threads: usize,
    batch: Vec<BatchComparison>,
    threads: VectorComparison,
}

fn compare_vectors(left: &[f32], right: &[f32]) -> CertResult<VectorComparison> {
    if left.is_empty() || left.len() != right.len() {
        return Err(
            "measurement_shape_mismatch: vectors must have equal nonzero dimensions".into(),
        );
    }
    let mut byte_equal = true;
    let (mut maximum, mut dot, mut left_norm, mut right_norm) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (left, right) in left.iter().zip(right) {
        if !left.is_finite() || !right.is_finite() {
            return Err("measurement_non_finite: vectors must contain finite values".into());
        }
        byte_equal &= left.to_bits() == right.to_bits();
        let left = f64::from(*left);
        let right = f64::from(*right);
        maximum = maximum.max((left - right).abs());
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm <= 0.0 || right_norm <= 0.0 {
        return Err("measurement_zero_norm: cosine requires nonzero vectors".into());
    }
    Ok(VectorComparison {
        byte_equal,
        max_absolute_difference: maximum,
        minimum_cosine: dot / (left_norm.sqrt() * right_norm.sqrt()),
    })
}

fn record_path(repository: &Path, model: &str, invocation: u128) -> PathBuf {
    repository
        .join(".tessera/cert-evidence")
        .join(model)
        .join(format!("{invocation}-{}.measure", std::process::id()))
}

fn write_record(path: &Path, record: &MeasurementRecord) -> CertResult<()> {
    let bytes = serde_json::to_vec_pretty(record)?;
    fs::create_dir_all(path.parent().ok_or("measurement path has no parent")?)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let written = file.write_all(&bytes).and_then(|()| file.sync_all());
    if let Err(error) = written {
        drop(file);
        fs::remove_file(path).map_err(|cleanup| {
            format!("measurement write failed: {error}; cleanup failed: {cleanup}")
        })?;
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/measure.rs"]
mod tests;
