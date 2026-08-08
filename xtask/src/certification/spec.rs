use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tessera::model_registry::{get_model, ModelType};

pub(crate) type CertResult<T> = Result<T, Box<dyn Error>>;

const SPEC_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct CertificationSpec {
    pub(crate) schema_version: u32,
    pub(crate) model: ModelSpec,
    pub(crate) artifacts: Vec<ArtifactSpec>,
    pub(crate) profiles: BTreeMap<String, ProfileSpec>,
    pub(crate) smoke: SmokeSpec,
    pub(crate) promotion: PromotionSpec,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ModelSpec {
    pub(crate) id: String,
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) representation: Representation,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ArtifactSpec {
    pub(crate) path: String,
    pub(crate) size_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ProfileSpec {
    pub(crate) resource_policy: ResourceLimits,
    pub(crate) process: ProcessLimits,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ResourceLimits {
    pub(crate) max_sequence_tokens: usize,
    pub(crate) max_batch_items: usize,
    pub(crate) max_batch_tokens: usize,
    pub(crate) max_model_bytes: usize,
    pub(crate) max_input_bytes_per_sequence: usize,
    pub(crate) max_attention_cells: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ProcessLimits {
    pub(crate) cpu_threads: usize,
    pub(crate) timeout_seconds: u64,
    pub(crate) max_artifact_bytes: u64,
    pub(crate) min_free_disk_bytes: u64,
    pub(crate) max_peak_rss_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct SmokeSpec {
    pub(crate) fixture: RetrievalFixture,
    pub(crate) expected_dimension: usize,
    pub(crate) expected_vocabulary_size: Option<usize>,
    pub(crate) normalized: bool,
    pub(crate) minimum_score_margin: f32,
    pub(crate) repeat_similarity_minimum: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct RetrievalFixture {
    pub(crate) query: String,
    pub(crate) positive: String,
    pub(crate) negative: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct PromotionSpec {
    pub(crate) minimum_successful_runs: usize,
    pub(crate) require_clean_source: bool,
    pub(crate) require_enforced_rss: bool,
    pub(crate) official_reference_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Representation {
    Dense,
    MultiVector,
    Sparse,
    Vision,
}

#[derive(Debug, Clone)]
pub(crate) struct LoadedSpec {
    pub(crate) sha256: String,
    pub(crate) spec: CertificationSpec,
}

impl CertificationSpec {
    pub(crate) fn profile(&self, name: &str) -> CertResult<&ProfileSpec> {
        self.profiles
            .get(name)
            .ok_or_else(|| format!("model '{}' has no profile '{name}'", self.model.id).into())
    }

    pub(crate) fn expected_artifact_bytes(&self) -> CertResult<u64> {
        self.artifacts.iter().try_fold(0_u64, |total, artifact| {
            total.checked_add(artifact.size_bytes).ok_or_else(|| {
                format!("artifact byte total overflow for model '{}'", self.model.id).into()
            })
        })
    }
}

pub(crate) fn load_all(repository: &Path) -> CertResult<Vec<LoadedSpec>> {
    let directory = repository.join("certification/specs");
    let mut paths = fs::read_dir(&directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| path.extension().and_then(|value| value.to_str()) == Some("json"));
    paths.sort();
    paths.into_iter().map(|path| load(&path)).collect()
}

pub(crate) fn load_model(repository: &Path, model_id: &str) -> CertResult<LoadedSpec> {
    validate_identifier(model_id)?;
    load(
        &repository
            .join("certification/specs")
            .join(format!("{model_id}.json")),
    )
}

fn load(path: &Path) -> CertResult<LoadedSpec> {
    let bytes = fs::read(path)?;
    let spec: CertificationSpec = serde_json::from_slice(&bytes)?;
    validate(path, &spec)?;
    validate_registry(&spec)?;
    Ok(LoadedSpec {
        sha256: digest_bytes(&bytes),
        spec,
    })
}

fn validate(path: &Path, spec: &CertificationSpec) -> CertResult<()> {
    if spec.schema_version != SPEC_SCHEMA_VERSION {
        return Err(format!(
            "{} uses unsupported schema version {}",
            path.display(),
            spec.schema_version
        )
        .into());
    }
    validate_identifier(&spec.model.id)?;
    let stem = path.file_stem().and_then(|value| value.to_str());
    if stem != Some(spec.model.id.as_str()) {
        return Err(format!("spec filename must match model id '{}'", spec.model.id).into());
    }
    if !spec.model.repository.contains('/') {
        return Err(format!("model '{}' has an invalid Hub repository", spec.model.id).into());
    }
    validate_hex(&spec.model.revision, 40, "revision")?;
    if spec.artifacts.is_empty() {
        return Err(format!("model '{}' has no checked artifacts", spec.model.id).into());
    }
    let mut paths = HashSet::new();
    for artifact in &spec.artifacts {
        validate_artifact_path(&artifact.path)?;
        if !paths.insert(&artifact.path) {
            return Err(format!(
                "model '{}' repeats artifact '{}'",
                spec.model.id, artifact.path
            )
            .into());
        }
        if artifact.size_bytes == 0 {
            return Err(format!("artifact '{}' has zero size", artifact.path).into());
        }
        validate_hex(&artifact.sha256, 64, "artifact SHA-256")?;
    }
    let profile = spec.profile("smoke")?;
    validate_limits(spec, profile)?;
    if spec.smoke.expected_dimension == 0 || spec.promotion.minimum_successful_runs < 2 {
        return Err(format!(
            "model '{}' has incomplete smoke or promotion criteria",
            spec.model.id
        )
        .into());
    }
    if let Some(reference) = &spec.promotion.official_reference_sha256 {
        validate_hex(reference, 64, "official reference SHA-256")?;
    }
    Ok(())
}

fn validate_limits(spec: &CertificationSpec, profile: &ProfileSpec) -> CertResult<()> {
    let resource = &profile.resource_policy;
    let process = &profile.process;
    if resource.max_sequence_tokens == 0
        || resource.max_batch_items == 0
        || resource.max_batch_tokens == 0
        || resource.max_model_bytes == 0
        || resource.max_input_bytes_per_sequence == 0
        || resource.max_attention_cells == 0
        || process.cpu_threads == 0
        || process.timeout_seconds == 0
        || process.max_peak_rss_bytes == 0
    {
        return Err(format!("model '{}' has a zero resource limit", spec.model.id).into());
    }
    if process.cpu_threads > 2 {
        return Err(format!(
            "model '{}' exceeds the two-thread certification ceiling",
            spec.model.id
        )
        .into());
    }
    let artifact_bytes = spec.expected_artifact_bytes()?;
    if artifact_bytes > process.max_artifact_bytes {
        return Err(format!(
            "model '{}' artifacts exceed its artifact-byte cap",
            spec.model.id
        )
        .into());
    }
    Ok(())
}

fn validate_registry(spec: &CertificationSpec) -> CertResult<()> {
    let registered = get_model(&spec.model.id)
        .ok_or_else(|| format!("model '{}' is absent from the registry", spec.model.id))?;
    if registered.huggingface_id != spec.model.repository
        || registered.revision != Some(spec.model.revision.as_str())
    {
        return Err(format!(
            "model '{}' spec repository or revision drifted from models.json",
            spec.model.id
        )
        .into());
    }
    let representation_matches = matches!(
        (spec.model.representation, registered.model_type),
        (Representation::Dense, ModelType::Dense)
            | (Representation::MultiVector, ModelType::Colbert)
            | (Representation::Sparse, ModelType::Sparse)
            | (Representation::Vision, ModelType::VisionLanguage)
    );
    if !representation_matches || !registered.is_runnable() {
        return Err(format!(
            "model '{}' spec does not match a runnable registry entry",
            spec.model.id
        )
        .into());
    }
    Ok(())
}

fn validate_identifier(value: &str) -> CertResult<()> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_'))
    {
        return Err(format!("invalid model identifier '{value}'").into());
    }
    Ok(())
}

fn validate_artifact_path(value: &str) -> CertResult<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe artifact path '{value}'").into());
    }
    Ok(())
}

fn validate_hex(value: &str, length: usize, label: &str) -> CertResult<()> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{label} must be {length} lowercase hexadecimal characters").into());
    }
    Ok(())
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
#[path = "tests/spec.rs"]
mod tests;
