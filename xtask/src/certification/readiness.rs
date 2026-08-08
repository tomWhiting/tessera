use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::artifacts;
use super::spec::{CertResult, LoadedSpec};

const EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
struct EvidenceSummary {
    schema_version: u32,
    model_id: String,
    spec_sha256: String,
    profile: String,
    device: String,
    status: String,
    source_commit: String,
    source_dirty: bool,
    peak_rss: PeakRssSummary,
}

#[derive(Debug, Clone, Deserialize)]
struct PeakRssSummary {
    enforced: bool,
    bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ReadinessReport {
    pub(crate) model_id: String,
    pub(crate) ready: bool,
    pub(crate) successful_runs: usize,
    pub(crate) required_runs: usize,
    pub(crate) source_commit: Option<String>,
    pub(crate) reasons: Vec<String>,
}

pub(crate) fn evaluate(repository: &Path, loaded: &LoadedSpec) -> ReadinessReport {
    let mut reasons = Vec::new();
    if let Err(error) = artifacts::verify_cached(repository, loaded) {
        reasons.push(format!("artifact integrity is not ready: {error}"));
    }
    if loaded.spec.promotion.official_reference_sha256.is_none() {
        reasons.push("no pinned official-reference fingerprint is checked in".to_string());
    }

    let evidence = match load_evidence(repository, loaded) {
        Ok(evidence) => evidence,
        Err(error) => {
            reasons.push(format!("working evidence is unreadable: {error}"));
            Vec::new()
        }
    };
    let valid = evidence
        .into_iter()
        .filter(|entry| evidence_matches(entry, loaded))
        .collect::<Vec<_>>();
    let required = loaded.spec.promotion.minimum_successful_runs;
    if valid.len() < required {
        reasons.push(format!(
            "{} matching successful runs found; {required} required",
            valid.len()
        ));
    }
    if loaded.spec.promotion.require_clean_source && valid.iter().any(|entry| entry.source_dirty) {
        reasons.push("at least one matching run used a dirty source tree".to_string());
    }
    if loaded.spec.promotion.require_enforced_rss
        && valid
            .iter()
            .any(|entry| !entry.peak_rss.enforced || entry.peak_rss.bytes.is_none())
    {
        reasons.push("at least one matching run lacks enforceable peak-RSS evidence".to_string());
    }

    let source_commit = common_source_commit(&valid);
    if !valid.is_empty() && source_commit.is_none() {
        reasons.push("matching runs do not share one source commit".to_string());
    }
    ReadinessReport {
        model_id: loaded.spec.model.id.clone(),
        ready: reasons.is_empty(),
        successful_runs: valid.len(),
        required_runs: required,
        source_commit,
        reasons,
    }
}

fn load_evidence(repository: &Path, loaded: &LoadedSpec) -> CertResult<Vec<EvidenceSummary>> {
    let directory = repository
        .join(".tessera/cert-evidence")
        .join(&loaded.spec.model.id);
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        entries.push(serde_json::from_slice(&fs::read(path)?)?);
    }
    Ok(entries)
}

fn evidence_matches(evidence: &EvidenceSummary, loaded: &LoadedSpec) -> bool {
    evidence.schema_version == EVIDENCE_SCHEMA_VERSION
        && evidence.model_id == loaded.spec.model.id
        && evidence.spec_sha256 == loaded.sha256
        && evidence.profile == "smoke"
        && evidence.device == "cpu"
        && evidence.status == "passed"
}

fn common_source_commit(evidence: &[EvidenceSummary]) -> Option<String> {
    let first = evidence.first()?.source_commit.clone();
    evidence
        .iter()
        .all(|entry| entry.source_commit == first)
        .then_some(first)
}

#[cfg(test)]
#[path = "tests/readiness.rs"]
mod tests;
