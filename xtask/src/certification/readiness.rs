use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::artifacts::{self, VerifiedArtifact};
use super::evidence;
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
    #[serde(default)]
    verified_artifacts: Vec<VerifiedArtifact>,
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
    let current_source = match evidence::source_state(repository) {
        Ok(source) => Some(source),
        Err(error) => {
            reasons.push(format!("current source state is unavailable: {error}"));
            None
        }
    };
    if loaded.spec.promotion.require_clean_source
        && current_source
            .as_ref()
            .is_some_and(|(_, source_dirty)| *source_dirty)
    {
        reasons.push("current source tree is dirty".to_string());
    }

    let evidence = match load_evidence(repository, loaded) {
        Ok(evidence) => evidence,
        Err(error) => {
            reasons.push(format!("working evidence is unreadable: {error}"));
            Vec::new()
        }
    };
    let matching = evidence
        .iter()
        .filter(|entry| evidence_matches(entry, loaded))
        .collect::<Vec<_>>();
    let required = loaded.spec.promotion.minimum_successful_runs;
    let (source_commit, successful_runs) =
        current_source
            .as_ref()
            .map_or((None, 0), |(current_commit, _)| {
                eligible_current_cohort(
                    &matching,
                    current_commit,
                    loaded.spec.promotion.require_clean_source,
                    loaded.spec.promotion.require_enforced_rss,
                )
            });
    if successful_runs < required {
        reasons.push(format!(
            "eligible current-HEAD cohort has {successful_runs} successful runs; {required} required"
        ));
    }
    ReadinessReport {
        model_id: loaded.spec.model.id.clone(),
        ready: reasons.is_empty(),
        successful_runs,
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
        && artifact_manifest_matches(evidence, loaded)
}

fn artifact_manifest_matches(evidence: &EvidenceSummary, loaded: &LoadedSpec) -> bool {
    evidence.verified_artifacts.len() == loaded.spec.artifacts.len()
        && loaded.spec.artifacts.iter().all(|expected| {
            evidence.verified_artifacts.iter().any(|observed| {
                observed.path == expected.path
                    && observed.size_bytes == expected.size_bytes
                    && observed.sha256 == expected.sha256
            })
        })
}

fn eligible_current_cohort(
    evidence: &[&EvidenceSummary],
    current_commit: &str,
    require_clean_source: bool,
    require_enforced_rss: bool,
) -> (Option<String>, usize) {
    let count = evidence
        .iter()
        .filter(|entry| entry.source_commit == current_commit)
        .filter(|entry| !require_clean_source || !entry.source_dirty)
        .filter(|entry| {
            !require_enforced_rss || (entry.peak_rss.enforced && entry.peak_rss.bytes.is_some())
        })
        .count();
    if count == 0 {
        (None, 0)
    } else {
        (Some(current_commit.to_string()), count)
    }
}

#[cfg(test)]
#[path = "tests/readiness.rs"]
mod tests;
