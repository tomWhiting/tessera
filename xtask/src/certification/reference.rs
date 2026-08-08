use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::spec::{
    CapabilityScope, CertResult, CertificationSpec, ProfileKind, ProfileSpec, Representation,
};

const REFERENCE_SCHEMA_VERSION: u32 = 1;
const MAX_REFERENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PROBE_IMAGE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReferencePointer {
    pub(crate) path: String,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReferenceDocument {
    pub(crate) schema_version: u32,
    pub(crate) model_id: String,
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) profile: String,
    pub(crate) capability: CapabilityScope,
    pub(crate) provenance: ReferenceProvenance,
    pub(crate) probe: ReferenceProbe,
    pub(crate) tolerance: NumericTolerance,
    pub(crate) expected: ReferenceOutput,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReferenceProvenance {
    pub(crate) producer: String,
    pub(crate) framework: String,
    pub(crate) framework_version: String,
    pub(crate) source_repository: String,
    pub(crate) source_revision: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ReferenceProbe {
    Text {
        text: String,
        token_count: usize,
    },
    Image {
        path: String,
        sha256: String,
        query: String,
        query_token_count: usize,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct NumericTolerance {
    pub(crate) absolute: f32,
    pub(crate) relative: f32,
    pub(crate) minimum_cosine: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "representation", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ReferenceOutput {
    Dense {
        values: Vec<f32>,
    },
    Sparse {
        vocabulary_size: usize,
        indices: Vec<usize>,
        values: Vec<f32>,
    },
    MultiVector {
        rows: usize,
        columns: usize,
        values: Vec<f32>,
    },
    Vision {
        rows: usize,
        columns: usize,
        values: Vec<f32>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct LoadedReference {
    pub(crate) path: String,
    pub(crate) sha256: String,
    pub(crate) document: ReferenceDocument,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ComparisonStatus {
    NotConfigured,
    Passed,
    Failed,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReferenceComparison {
    pub(crate) status: ComparisonStatus,
    pub(crate) reference_path: Option<String>,
    pub(crate) reference_sha256: Option<String>,
    pub(crate) expected_output_sha256: Option<String>,
    pub(crate) observed_output_sha256: Option<String>,
    pub(crate) probe_tokens: Option<usize>,
    pub(crate) observed_shape: Vec<usize>,
    pub(crate) compared_values: usize,
    pub(crate) max_absolute_error: Option<f32>,
    pub(crate) max_relative_error: Option<f32>,
    pub(crate) minimum_cosine: Option<f32>,
    pub(crate) detail: String,
}

impl ReferenceComparison {
    pub(crate) fn not_configured() -> Self {
        Self {
            status: ComparisonStatus::NotConfigured,
            reference_path: None,
            reference_sha256: None,
            expected_output_sha256: None,
            observed_output_sha256: None,
            probe_tokens: None,
            observed_shape: Vec::new(),
            compared_values: 0,
            max_absolute_error: None,
            max_relative_error: None,
            minimum_cosine: None,
            detail: "profile has no checked official reference".to_string(),
        }
    }

    pub(crate) fn not_run(reference: &LoadedReference, detail: &str) -> Self {
        Self {
            status: ComparisonStatus::Failed,
            reference_path: Some(reference.path.clone()),
            reference_sha256: Some(reference.sha256.clone()),
            expected_output_sha256: None,
            observed_output_sha256: None,
            probe_tokens: Some(reference.document.probe.token_count()),
            observed_shape: Vec::new(),
            compared_values: 0,
            max_absolute_error: None,
            max_relative_error: None,
            minimum_cosine: None,
            detail: detail.to_string(),
        }
    }
}

pub(crate) fn load_optional(
    repository: &Path,
    spec: &CertificationSpec,
    profile_name: &str,
) -> CertResult<Option<LoadedReference>> {
    let profile = spec.profile(profile_name)?;
    profile
        .official_reference
        .as_ref()
        .map(|pointer| load_checked(repository, spec, profile_name, profile, pointer))
        .transpose()
}

pub(crate) fn load_checked(
    repository: &Path,
    spec: &CertificationSpec,
    profile_name: &str,
    profile: &ProfileSpec,
    pointer: &ReferencePointer,
) -> CertResult<LoadedReference> {
    validate_hex(&pointer.sha256)?;
    let path = resolve_checked(&reference_root(repository), &pointer.path)?;
    ensure_size(&path, MAX_REFERENCE_BYTES, "official reference")?;
    let bytes = fs::read(&path).map_err(|error| {
        format!(
            "official reference '{}' is unreadable: {error}",
            path.display()
        )
    })?;
    let actual_sha256 = digest(&bytes);
    if actual_sha256 != pointer.sha256 {
        return Err(format!(
            "official reference '{}' SHA-256 mismatch: got {actual_sha256}, expected {}",
            pointer.path, pointer.sha256
        )
        .into());
    }
    let document: ReferenceDocument = serde_json::from_slice(&bytes)?;
    validate_document(repository, spec, profile_name, profile, &document)?;
    Ok(LoadedReference {
        path: pointer.path.clone(),
        sha256: actual_sha256,
        document,
    })
}

pub(crate) fn resolve_image(repository: &Path, probe: &ReferenceProbe) -> CertResult<PathBuf> {
    let ReferenceProbe::Image { path, sha256, .. } = probe else {
        return Err("reference probe is not an image".into());
    };
    validate_hex(sha256)?;
    let resolved = resolve_checked(&repository.join("certification/fixtures"), path)?;
    ensure_size(&resolved, MAX_PROBE_IMAGE_BYTES, "reference image")?;
    let bytes = fs::read(&resolved)?;
    let actual = digest(&bytes);
    if actual != *sha256 {
        return Err(format!(
            "reference image '{}' SHA-256 mismatch: got {actual}, expected {sha256}",
            resolved.display()
        )
        .into());
    }
    Ok(resolved)
}

pub(crate) fn compare(
    reference: &LoadedReference,
    observed: &ReferenceOutput,
) -> CertResult<ReferenceComparison> {
    super::reference_compare::compare(reference, observed)
}

pub(crate) fn comparison_is_complete(
    comparison: &ReferenceComparison,
    reference: &LoadedReference,
) -> bool {
    let expected = &reference.document.expected;
    let Ok(expected_shape) = expected.shape() else {
        return false;
    };
    let Ok(expected_bytes) = serde_json::to_vec(expected) else {
        return false;
    };
    let expected_sha256 = digest(&expected_bytes);
    let tolerance = reference.document.tolerance;
    let maximum_expected = expected
        .values()
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let maximum_allowed = tolerance.absolute + tolerance.relative * maximum_expected;
    comparison.status == ComparisonStatus::Passed
        && comparison.reference_path.as_deref() == Some(reference.path.as_str())
        && comparison.reference_sha256.as_deref() == Some(reference.sha256.as_str())
        && comparison.expected_output_sha256.as_deref() == Some(expected_sha256.as_str())
        && comparison
            .observed_output_sha256
            .as_deref()
            .is_some_and(|value| validate_hex(value).is_ok())
        && comparison.probe_tokens == Some(reference.document.probe.token_count())
        && comparison.observed_shape == expected_shape
        && comparison.compared_values == expected.values().len()
        && comparison
            .max_absolute_error
            .is_some_and(|value| value.is_finite() && value <= maximum_allowed)
        && comparison.max_relative_error.is_some_and(f32::is_finite)
        && comparison.minimum_cosine.is_some_and(|value| {
            value.is_finite() && value >= reference.document.tolerance.minimum_cosine
        })
}

fn validate_document(
    repository: &Path,
    spec: &CertificationSpec,
    profile_name: &str,
    profile: &ProfileSpec,
    document: &ReferenceDocument,
) -> CertResult<()> {
    if document.schema_version != REFERENCE_SCHEMA_VERSION {
        return Err("unsupported official-reference schema version".into());
    }
    if document.model_id != spec.model.id
        || document.repository != spec.model.repository
        || document.revision != spec.model.revision
        || document.profile != profile_name
        || document.capability != profile.capability
    {
        return Err("official reference identity or capability does not match its profile".into());
    }
    let provenance = &document.provenance;
    if provenance.producer.trim().is_empty()
        || provenance.framework.trim().is_empty()
        || provenance.framework_version.trim().is_empty()
        || provenance.source_repository != spec.model.repository
        || provenance.source_revision != spec.model.revision
    {
        return Err("official reference provenance is incomplete or unpinned".into());
    }
    validate_tolerance(document.tolerance)?;
    validate_probe(repository, profile, &document.probe)?;
    super::reference_compare::validate_output(&document.expected)?;
    if document.expected.representation() != spec.model.representation {
        return Err("official reference representation does not match the model".into());
    }
    Ok(())
}

fn validate_probe(
    repository: &Path,
    profile: &ProfileSpec,
    probe: &ReferenceProbe,
) -> CertResult<()> {
    let tokens = probe.token_count();
    if tokens == 0 || tokens > profile.capability.max_sequence_tokens {
        return Err("reference probe token count is outside the capability scope".into());
    }
    match probe {
        ReferenceProbe::Text { text, .. } if text.trim().is_empty() => {
            return Err("reference text probe is empty".into());
        }
        ReferenceProbe::Image { query, .. } if query.trim().is_empty() => {
            return Err("reference image probe query is empty".into());
        }
        ReferenceProbe::Image { .. } => {
            resolve_image(repository, probe)?;
        }
        ReferenceProbe::Text { .. } => {}
    }
    if profile.kind == ProfileKind::LongContext
        && tokens < profile.capability.max_sequence_tokens.saturating_mul(7) / 8
    {
        return Err(
            "long-context reference probe must exercise at least 87.5% of the profile limit".into(),
        );
    }
    Ok(())
}

fn validate_tolerance(tolerance: NumericTolerance) -> CertResult<()> {
    if !tolerance.absolute.is_finite()
        || !tolerance.relative.is_finite()
        || !tolerance.minimum_cosine.is_finite()
        || tolerance.absolute < 0.0
        || tolerance.relative < 0.0
        || !(0.0..=1.0).contains(&tolerance.minimum_cosine)
        || tolerance.absolute + tolerance.relative <= 0.0
        || tolerance.absolute > 0.001
        || tolerance.relative > 0.01
        || tolerance.minimum_cosine < 0.999
    {
        return Err("official reference tolerance is invalid".into());
    }
    Ok(())
}

impl ReferenceProbe {
    pub(crate) const fn token_count(&self) -> usize {
        match self {
            Self::Text { token_count, .. } => *token_count,
            Self::Image {
                query_token_count, ..
            } => *query_token_count,
        }
    }
}

impl ReferenceOutput {
    pub(crate) const fn representation(&self) -> Representation {
        match self {
            Self::Dense { .. } => Representation::Dense,
            Self::Sparse { .. } => Representation::Sparse,
            Self::MultiVector { .. } => Representation::MultiVector,
            Self::Vision { .. } => Representation::Vision,
        }
    }

    pub(super) fn shape(&self) -> CertResult<Vec<usize>> {
        match self {
            Self::Dense { values } => Ok(vec![values.len()]),
            Self::Sparse {
                vocabulary_size, ..
            } => Ok(vec![*vocabulary_size]),
            Self::MultiVector {
                rows,
                columns,
                values,
            }
            | Self::Vision {
                rows,
                columns,
                values,
            } => {
                if rows.checked_mul(*columns) != Some(values.len()) {
                    return Err("reference matrix shape does not match its values".into());
                }
                Ok(vec![*rows, *columns])
            }
        }
    }

    pub(super) fn values(&self) -> &[f32] {
        match self {
            Self::Dense { values }
            | Self::Sparse { values, .. }
            | Self::MultiVector { values, .. }
            | Self::Vision { values, .. } => values,
        }
    }

    pub(super) fn sparse_indices(&self) -> Option<&[usize]> {
        match self {
            Self::Sparse { indices, .. } => Some(indices),
            _ => None,
        }
    }

    pub(super) fn row_width(&self) -> Option<usize> {
        match self {
            Self::Dense { values } => Some(values.len()),
            Self::Sparse { values, .. } => Some(values.len()),
            Self::MultiVector { columns, .. } | Self::Vision { columns, .. } => Some(*columns),
        }
    }
}

fn reference_root(repository: &Path) -> PathBuf {
    repository.join("certification/references")
}

fn validate_relative_path(value: &str) -> CertResult<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe certification fixture path '{value}'").into());
    }
    Ok(())
}

fn resolve_checked(root: &Path, relative: &str) -> CertResult<PathBuf> {
    validate_relative_path(relative)?;
    let canonical_root = fs::canonicalize(root)?;
    let resolved = fs::canonicalize(root.join(relative))?;
    if !resolved.starts_with(&canonical_root) {
        return Err(format!("certification fixture '{relative}' escapes its checked root").into());
    }
    Ok(resolved)
}

fn ensure_size(path: &Path, maximum: u64, label: &str) -> CertResult<()> {
    let bytes = fs::metadata(path)?.len();
    if bytes > maximum {
        return Err(format!(
            "{label} '{}' is {bytes} bytes; limit is {maximum}",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn validate_hex(value: &str) -> CertResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("reference SHA-256 must be 64 lowercase hexadecimal characters".into());
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
#[path = "tests/reference.rs"]
mod tests;
