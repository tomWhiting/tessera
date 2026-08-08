use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::artifacts::{self, VerifiedArtifact};
use super::evidence;
use super::reference::{self, LoadedReference, ReferenceComparison};
use super::spec::{CapabilityScope, CertResult, LoadedSpec};

const EVIDENCE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize)]
struct EvidenceSummary {
    schema_version: u32,
    model_id: String,
    spec_sha256: String,
    profile: String,
    device: String,
    capability: CapabilityScope,
    status: String,
    source_commit: String,
    source_dirty: bool,
    peak_rss: PeakRssSummary,
    #[serde(default)]
    verified_artifacts: Vec<VerifiedArtifact>,
    reference_comparison: ReferenceComparison,
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
    pub(crate) profiles: Vec<ProfileReadiness>,
    pub(crate) reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ProfileReadiness {
    pub(crate) profile: String,
    pub(crate) capability: CapabilityScope,
    pub(crate) successful_runs: usize,
    pub(crate) required_runs: usize,
    pub(crate) ready: bool,
    pub(crate) reasons: Vec<String>,
}

pub(crate) fn evaluate(repository: &Path, loaded: &LoadedSpec) -> ReadinessReport {
    let mut reasons = Vec::new();
    if let Err(error) = artifacts::verify_cached(repository, loaded) {
        reasons.push(format!("artifact integrity is not ready: {error}"));
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
    let required = loaded.spec.promotion.minimum_successful_runs;
    let mut profiles = Vec::new();
    for profile_name in &loaded.spec.promotion.required_profiles {
        let profile = loaded
            .spec
            .profile(profile_name)
            .expect("required profiles are validated while loading specifications");
        let mut profile_reasons = Vec::new();
        let checked_reference =
            match reference::load_optional(repository, &loaded.spec, profile_name) {
                Ok(Some(reference)) => Some(reference),
                Ok(None) => {
                    profile_reasons.push("no checked official reference is configured".to_string());
                    None
                }
                Err(error) => {
                    profile_reasons.push(format!("checked official reference is invalid: {error}"));
                    None
                }
            };
        let matching = evidence
            .iter()
            .filter(|entry| {
                checked_reference.as_ref().is_some_and(|reference| {
                    evidence_matches(entry, loaded, profile_name, reference)
                })
            })
            .collect::<Vec<_>>();
        let successful_runs = current_source.as_ref().map_or(0, |(current_commit, _)| {
            eligible_current_cohort(
                &matching,
                current_commit,
                loaded.spec.promotion.require_clean_source,
                loaded.spec.promotion.require_enforced_rss,
            )
            .1
        });
        if successful_runs < required {
            profile_reasons.push(format!(
                "eligible current-HEAD cohort has {successful_runs} passed official comparisons; {required} required"
            ));
        }
        for reason in &profile_reasons {
            reasons.push(format!("profile '{profile_name}': {reason}"));
        }
        profiles.push(ProfileReadiness {
            profile: profile_name.clone(),
            capability: profile.capability.clone(),
            successful_runs,
            required_runs: required,
            ready: profile_reasons.is_empty(),
            reasons: profile_reasons,
        });
    }
    let successful_runs = profiles
        .iter()
        .map(|profile| profile.successful_runs)
        .min()
        .unwrap_or(0);
    let source_commit = current_source
        .as_ref()
        .filter(|_| successful_runs > 0)
        .map(|(commit, _)| commit.clone());
    ReadinessReport {
        model_id: loaded.spec.model.id.clone(),
        ready: reasons.is_empty(),
        successful_runs,
        required_runs: required,
        source_commit,
        profiles,
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

fn evidence_matches(
    evidence: &EvidenceSummary,
    loaded: &LoadedSpec,
    profile_name: &str,
    reference: &LoadedReference,
) -> bool {
    let Ok(profile) = loaded.spec.profile(profile_name) else {
        return false;
    };
    evidence.schema_version == EVIDENCE_SCHEMA_VERSION
        && evidence.model_id == loaded.spec.model.id
        && evidence.spec_sha256 == loaded.sha256
        && evidence.profile == profile_name
        && evidence.device == profile.capability.device.label()
        && evidence.capability == profile.capability
        && evidence.status == "passed"
        && reference::comparison_is_complete(&evidence.reference_comparison, reference)
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
