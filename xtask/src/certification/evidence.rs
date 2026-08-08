use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::artifacts::VerifiedArtifact;
use super::reference::ReferenceComparison;
use super::spec::{CapabilityScope, CertResult, LoadedSpec, ProcessLimits, ResourceLimits};

pub(crate) const EVIDENCE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct CheckEvidence {
    pub(crate) name: String,
    pub(crate) passed: bool,
    pub(crate) detail: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct SmokeObservation {
    pub(crate) representation: String,
    pub(crate) primary_shape: Vec<usize>,
    pub(crate) batch_shapes: Vec<Vec<usize>>,
    pub(crate) finite: bool,
    pub(crate) norm_min: Option<f32>,
    pub(crate) norm_max: Option<f32>,
    pub(crate) non_zero: Option<usize>,
    pub(crate) repeat_similarity: f32,
    pub(crate) relevant_score: f32,
    pub(crate) unrelated_score: f32,
    pub(crate) score_margin: f32,
    pub(crate) checks: Vec<CheckEvidence>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ChildOutcome {
    pub(crate) status: String,
    pub(crate) error: Option<String>,
    pub(crate) verified_artifacts: Vec<VerifiedArtifact>,
    pub(crate) observation: Option<SmokeObservation>,
    pub(crate) reference_comparison: ReferenceComparison,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct PeakRssEvidence {
    pub(crate) enforced: bool,
    pub(crate) bytes: Option<u64>,
    pub(crate) limit_bytes: u64,
    pub(crate) method: String,
    pub(crate) sample_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EvidenceRecord {
    pub(crate) schema_version: u32,
    pub(crate) model_id: String,
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) spec_sha256: String,
    pub(crate) profile: String,
    pub(crate) device: String,
    pub(crate) capability: CapabilityScope,
    pub(crate) repetition: usize,
    pub(crate) status: String,
    pub(crate) error: Option<String>,
    pub(crate) source_commit: String,
    pub(crate) source_dirty: bool,
    pub(crate) host_os: String,
    pub(crate) host_arch: String,
    pub(crate) process_id: u32,
    pub(crate) started_unix_ms: u128,
    pub(crate) completed_unix_ms: u128,
    pub(crate) duration_ms: u128,
    pub(crate) resource_policy: ResourceLimits,
    pub(crate) process_limits: ProcessLimits,
    pub(crate) peak_rss: PeakRssEvidence,
    pub(crate) verified_artifacts: Vec<VerifiedArtifact>,
    pub(crate) observation: Option<SmokeObservation>,
    pub(crate) reference_comparison: ReferenceComparison,
}

pub(crate) struct RecordInput<'a> {
    pub(crate) loaded: &'a LoadedSpec,
    pub(crate) profile: &'a str,
    pub(crate) repetition: usize,
    pub(crate) child_pid: u32,
    pub(crate) started_unix_ms: u128,
    pub(crate) completed_unix_ms: u128,
    pub(crate) peak_rss: PeakRssEvidence,
    pub(crate) outcome: ChildOutcome,
}

pub(crate) fn build_record(
    repository: &Path,
    input: RecordInput<'_>,
) -> CertResult<EvidenceRecord> {
    let profile = input.loaded.spec.profile(input.profile)?;
    let (source_commit, source_dirty) = source_state(repository)?;
    Ok(EvidenceRecord {
        schema_version: EVIDENCE_SCHEMA_VERSION,
        model_id: input.loaded.spec.model.id.clone(),
        repository: input.loaded.spec.model.repository.clone(),
        revision: input.loaded.spec.model.revision.clone(),
        spec_sha256: input.loaded.sha256.clone(),
        profile: input.profile.to_string(),
        device: profile.capability.device.label().to_string(),
        capability: profile.capability.clone(),
        repetition: input.repetition,
        status: input.outcome.status,
        error: input.outcome.error,
        source_commit,
        source_dirty,
        host_os: std::env::consts::OS.to_string(),
        host_arch: std::env::consts::ARCH.to_string(),
        process_id: input.child_pid,
        started_unix_ms: input.started_unix_ms,
        completed_unix_ms: input.completed_unix_ms,
        duration_ms: input
            .completed_unix_ms
            .saturating_sub(input.started_unix_ms),
        resource_policy: profile.resource_policy.clone(),
        process_limits: profile.process.clone(),
        peak_rss: input.peak_rss,
        verified_artifacts: input.outcome.verified_artifacts,
        observation: input.outcome.observation,
        reference_comparison: input.outcome.reference_comparison,
    })
}

pub(crate) fn evidence_path(
    repository: &Path,
    model_id: &str,
    repetition: usize,
    timestamp_ms: u128,
) -> PathBuf {
    repository
        .join(".tessera/cert-evidence")
        .join(model_id)
        .join(format!("{timestamp_ms}-run-{repetition}.json"))
}

pub(crate) fn write_record(path: &Path, record: &EvidenceRecord) -> CertResult<()> {
    let parent = path.parent().ok_or("evidence path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(record)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub(crate) fn now_unix_ms() -> CertResult<u128> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}

pub(crate) fn source_state(repository: &Path) -> CertResult<(String, bool)> {
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repository)
        .output()?;
    if !commit.status.success() {
        return Err("failed to resolve the source commit for certification evidence".into());
    }
    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repository)
        .output()?;
    if !status.status.success() {
        return Err("failed to inspect source cleanliness for certification evidence".into());
    }
    Ok((
        String::from_utf8(commit.stdout)?.trim().to_string(),
        !status.stdout.is_empty(),
    ))
}

#[cfg(test)]
#[path = "tests/evidence.rs"]
mod tests;
