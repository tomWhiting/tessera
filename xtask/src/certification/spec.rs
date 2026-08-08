use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tessera::model_registry::{get_model, ModelType};

use super::reference::{self, ReferencePointer};

pub(crate) type CertResult<T> = Result<T, Box<dyn Error>>;

const SPEC_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CertificationSpec {
    pub(crate) schema_version: u32,
    pub(crate) model: ModelSpec,
    pub(crate) artifacts: Vec<ArtifactSpec>,
    pub(crate) profiles: BTreeMap<String, ProfileSpec>,
    pub(crate) smoke: SmokeSpec,
    pub(crate) promotion: PromotionSpec,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelSpec {
    pub(crate) id: String,
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) representation: Representation,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtifactSpec {
    pub(crate) path: String,
    pub(crate) size_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProfileSpec {
    pub(crate) kind: ProfileKind,
    pub(crate) capability: CapabilityScope,
    pub(crate) resource_policy: ResourceLimits,
    pub(crate) process: ProcessLimits,
    pub(crate) official_reference: Option<ReferencePointer>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResourceLimits {
    pub(crate) max_sequence_tokens: usize,
    pub(crate) max_batch_items: usize,
    pub(crate) max_batch_tokens: usize,
    pub(crate) max_model_bytes: usize,
    pub(crate) max_input_bytes_per_sequence: usize,
    pub(crate) max_attention_cells: usize,
    pub(crate) max_job_items: usize,
    pub(crate) max_job_input_bytes: usize,
    pub(crate) max_output_bytes: usize,
    pub(crate) max_activation_bytes: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProcessLimits {
    pub(crate) cpu_threads: usize,
    pub(crate) timeout_seconds: u64,
    pub(crate) max_artifact_bytes: u64,
    pub(crate) min_free_disk_bytes: u64,
    pub(crate) max_peak_rss_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SmokeSpec {
    pub(crate) fixture: RetrievalFixture,
    pub(crate) expected_dimension: usize,
    pub(crate) expected_vocabulary_size: Option<usize>,
    pub(crate) normalized: bool,
    pub(crate) minimum_score_margin: f32,
    pub(crate) repeat_similarity_minimum: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RetrievalFixture {
    pub(crate) query: String,
    pub(crate) positive: String,
    pub(crate) negative: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PromotionSpec {
    pub(crate) minimum_successful_runs: usize,
    pub(crate) required_profiles: Vec<String>,
    pub(crate) require_clean_source: bool,
    pub(crate) require_enforced_rss: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProfileKind {
    Smoke,
    LongContext,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapabilityScope {
    pub(crate) device: CertificationDevice,
    pub(crate) dtype: CertificationDtype,
    pub(crate) semantic_mode: SemanticMode,
    pub(crate) max_sequence_tokens: usize,
    pub(crate) context_window_tokens: usize,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CertificationDevice {
    Cpu,
}

impl CertificationDevice {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CertificationDtype {
    F32,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SemanticMode {
    Query,
    Document,
    SparseQuery,
    SparseDocument,
    LateInteractionQuery,
    LateInteractionDocument,
    VisionDocument,
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
    validate_references(path, &spec)?;
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
    let smoke_profile = spec.profile("smoke")?;
    if smoke_profile.kind != ProfileKind::Smoke {
        return Err(format!("model '{}' smoke profile has the wrong kind", spec.model.id).into());
    }
    for (name, profile) in &spec.profiles {
        validate_identifier(name)?;
        validate_limits(spec, profile)?;
    }
    if spec.smoke.expected_dimension == 0 || spec.promotion.minimum_successful_runs < 2 {
        return Err(format!(
            "model '{}' has incomplete smoke or promotion criteria",
            spec.model.id
        )
        .into());
    }
    if spec.promotion.required_profiles.is_empty()
        || !spec
            .promotion
            .required_profiles
            .iter()
            .any(|name| name == "smoke")
    {
        return Err(format!("model '{}' must require its smoke profile", spec.model.id).into());
    }
    let mut required_profiles = HashSet::new();
    for name in &spec.promotion.required_profiles {
        if !required_profiles.insert(name) {
            return Err(format!(
                "model '{}' repeats required profile '{name}'",
                spec.model.id
            )
            .into());
        }
        spec.profile(name)?;
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
        || resource.max_job_items == 0
        || resource.max_job_input_bytes == 0
        || resource.max_output_bytes == 0
        || resource.max_activation_bytes == 0
        || process.cpu_threads == 0
        || process.timeout_seconds == 0
        || process.max_artifact_bytes == 0
        || process.min_free_disk_bytes == 0
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
    let registered = get_model(&spec.model.id)
        .ok_or_else(|| format!("model '{}' is absent from the registry", spec.model.id))?;
    if profile.capability.max_sequence_tokens != resource.max_sequence_tokens
        || profile.capability.context_window_tokens != registered.context_length
        || resource.max_sequence_tokens > registered.context_length
        || resource.max_batch_tokens < resource.max_sequence_tokens
    {
        return Err(format!(
            "model '{}' profile capability does not match its enforced token limits or registry context",
            spec.model.id
        )
        .into());
    }
    let semantic_mode_matches = matches!(
        (spec.model.representation, profile.capability.semantic_mode),
        (
            Representation::Dense,
            SemanticMode::Query | SemanticMode::Document
        ) | (
            Representation::Sparse,
            SemanticMode::SparseQuery | SemanticMode::SparseDocument
        ) | (
            Representation::MultiVector,
            SemanticMode::LateInteractionQuery | SemanticMode::LateInteractionDocument
        ) | (Representation::Vision, SemanticMode::VisionDocument)
    );
    if !semantic_mode_matches {
        return Err(format!(
            "model '{}' profile semantic mode does not match its representation",
            spec.model.id
        )
        .into());
    }
    let minimum_attention_cells = resource
        .max_sequence_tokens
        .checked_mul(resource.max_sequence_tokens)
        .ok_or("profile attention-cell requirement overflowed")?;
    if resource.max_attention_cells < minimum_attention_cells {
        return Err(format!(
            "model '{}' profile cannot admit one maximum-length sequence",
            spec.model.id
        )
        .into());
    }
    let configured_batch_input_bytes = resource
        .max_batch_items
        .checked_mul(resource.max_input_bytes_per_sequence)
        .ok_or("profile batch input-byte requirement overflowed")?;
    if resource.max_job_items < resource.max_batch_items
        || resource.max_job_input_bytes < configured_batch_input_bytes
    {
        return Err(format!(
            "model '{}' profile job limits cannot admit one configured batch",
            spec.model.id
        )
        .into());
    }
    if profile.kind == ProfileKind::LongContext && resource.max_sequence_tokens < 1024 {
        return Err(format!(
            "model '{}' long-context profile is below 1024 tokens",
            spec.model.id
        )
        .into());
    }
    let declared_live_bytes = resource
        .max_model_bytes
        .checked_add(resource.max_activation_bytes)
        .ok_or("profile live-memory requirement overflowed")?;
    let declared_live_bytes = u64::try_from(declared_live_bytes)
        .map_err(|_| "profile live-memory requirement does not fit u64")?;
    if declared_live_bytes > process.max_peak_rss_bytes {
        return Err(format!(
            "model '{}' model-plus-activation budget exceeds its RSS watchdog",
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

fn validate_references(path: &Path, spec: &CertificationSpec) -> CertResult<()> {
    let repository = path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or("certification spec is not inside certification/specs")?;
    for (name, profile) in &spec.profiles {
        if let Some(pointer) = &profile.official_reference {
            reference::load_checked(repository, spec, name, profile, pointer)?;
        }
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
