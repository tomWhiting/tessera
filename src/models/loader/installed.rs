use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::models::registry::{self, ModelInfo};

/// A failure to validate an installed model, without exposing artifact contents.
#[derive(Debug, Error)]
pub enum InstalledModelError {
    /// The registry weight metadata has no supported runtime format.
    #[error(transparent)]
    WeightMetadata(#[from] crate::error::TesseraError),
    /// The requested model is absent from the registry.
    #[error("installed_model_not_registered: manifest.json: unknown model {model_id:?}")]
    ModelNotRegistered {
        /// Requested registry identifier.
        model_id: String,
    },
    /// The manifest cannot be decoded with the required members.
    #[error(
        "invalid_manifest: {filename:?}: invalid or unknown member at line {line}, column {column}"
    )]
    InvalidManifest {
        /// Manifest filename.
        filename: String,
        /// Position of the decoding failure.
        line: usize,
        /// Position within the failing line.
        column: usize,
    },
    /// The manifest schema version is unsupported.
    #[error("unsupported_manifest_schema: {filename:?}: schema_version must be 1")]
    UnsupportedSchema {
        /// Manifest filename.
        filename: String,
    },
    /// The manifest identifies a different registry entry.
    #[error("installed_model_id_mismatch: {filename:?}: model id does not match the requested registry model")]
    ModelIdMismatch {
        /// Manifest filename.
        filename: String,
    },
    /// The manifest names a different repository.
    #[error("installed_repository_mismatch: {filename:?}: repository does not match the registry")]
    RepositoryMismatch {
        /// Manifest filename.
        filename: String,
    },
    /// The manifest names a different immutable revision.
    #[error("installed_revision_mismatch: {filename:?}: revision does not match the registry pin")]
    RevisionMismatch {
        /// Manifest filename.
        filename: String,
    },
    /// The registry cannot establish an immutable revision.
    #[error("installed_revision_unpinned: {filename:?}: registry has no pinned revision")]
    RevisionUnpinned {
        /// Manifest filename.
        filename: String,
    },
    /// An artifact was not declared in the manifest.
    #[error("installed_artifact_not_listed: {filename:?}: required artifact is absent from the manifest")]
    ArtifactNotListed {
        /// Required artifact filename.
        filename: String,
    },
    /// An artifact path can escape the installed directory or is not a filename.
    #[error("invalid_installed_artifact_path: {filename:?}: expected a plain filename without parent components or separators")]
    InvalidArtifactPath {
        /// Rejected artifact path.
        filename: String,
    },
    /// An artifact was declared more than once.
    #[error("duplicate_installed_artifact: {filename:?}: artifact occurs more than once")]
    DuplicateArtifact {
        /// Repeated artifact filename.
        filename: String,
    },
    /// An artifact is a symbolic link.
    #[error("installed_artifact_symlink: {filename:?}: symbolic links are refused")]
    ArtifactSymlink {
        /// Rejected filename.
        filename: String,
    },
    /// An artifact is not a regular file.
    #[error("installed_artifact_not_file: {filename:?}: expected a regular file")]
    ArtifactNotFile {
        /// Rejected filename.
        filename: String,
    },
    /// An artifact could not be opened or read.
    #[error("installed_artifact_io: {filename:?}: {source}")]
    ArtifactIo {
        /// Unreadable filename.
        filename: String,
        /// Filesystem failure.
        #[source]
        source: std::io::Error,
    },
    /// An artifact has a different byte count.
    #[error(
        "installed_artifact_size_mismatch: {filename:?}: expected {expected} bytes, found {actual}"
    )]
    ArtifactSizeMismatch {
        /// Artifact filename.
        filename: String,
        /// Manifest byte count.
        expected: u64,
        /// Observed byte count.
        actual: u64,
    },
    /// An artifact has a different digest.
    #[error("installed_artifact_hash_mismatch: {filename:?}: SHA-256 does not match the manifest")]
    ArtifactHashMismatch {
        /// Artifact filename.
        filename: String,
    },
    /// An artifact digest is not a hexadecimal SHA-256 value.
    #[error(
        "invalid_installed_artifact_hash: {filename:?}: SHA-256 must contain 64 hexadecimal digits"
    )]
    InvalidArtifactHash {
        /// Artifact filename.
        filename: String,
    },
    /// The registered weight artifact requires pickle or a sharded index.
    #[error("installed_safetensors_required: {filename:?}: installed loading requires a single safetensors weights file")]
    SafetensorsRequired {
        /// Incompatible registered weights filename.
        filename: String,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    model: ManifestModel,
    artifacts: Vec<Artifact>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestModel {
    id: String,
    repository: String,
    revision: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: String,
    size_bytes: u64,
    sha256: String,
}

/// An installed source whose entire artifact list has passed integrity checks.
pub struct InstalledModel {
    directory: PathBuf,
    artifacts: BTreeSet<String>,
    manifest_sha256: String,
}

impl InstalledModel {
    /// Validates the installed files against the requested immutable registry entry.
    ///
    /// # Errors
    /// Returns a named validation failure before any model is constructed.
    pub fn open(model_id: &str, directory: impl AsRef<Path>) -> Result<Self, InstalledModelError> {
        let model = registry::get_model(model_id).ok_or_else(|| {
            InstalledModelError::ModelNotRegistered {
                model_id: model_id.to_string(),
            }
        })?;
        Self::open_for_model(model, directory.as_ref())
    }

    pub(super) fn open_for_model(
        model: &ModelInfo,
        directory: &Path,
    ) -> Result<Self, InstalledModelError> {
        let declared_weight = super::supported_weight_filename(model)?;
        let weights = model
            .safetensors_file
            .filter(|filename| {
                Path::new(filename)
                    .extension()
                    .is_some_and(|extension| extension == "safetensors")
            })
            .ok_or_else(|| InstalledModelError::SafetensorsRequired {
                filename: declared_weight.to_string(),
            })?;
        let filename = "manifest.json";
        let mut file = regular_file(directory, filename)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|source| io_error(filename, source))?;
        let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|error| {
            InstalledModelError::InvalidManifest {
                filename: filename.to_string(),
                line: error.line(),
                column: error.column(),
            }
        })?;
        validate_model(&manifest, model)?;
        let mut artifacts = BTreeSet::new();
        for artifact in &manifest.artifacts {
            validate_filename(&artifact.path)?;
            if !artifacts.insert(artifact.path.clone()) {
                return Err(InstalledModelError::DuplicateArtifact {
                    filename: artifact.path.clone(),
                });
            }
        }
        for required in [model.config_file, model.tokenizer_file, weights] {
            if !artifacts.contains(required) {
                return Err(InstalledModelError::ArtifactNotListed {
                    filename: required.to_string(),
                });
            }
        }
        for artifact in &manifest.artifacts {
            validate_artifact(directory, artifact)?;
        }
        Ok(Self {
            directory: directory.to_path_buf(),
            artifacts,
            manifest_sha256: format!("{:x}", Sha256::digest(&bytes)),
        })
    }

    /// Returns the lowercase SHA-256 of the exact manifest bytes that were read.
    #[must_use]
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub(super) fn get(&self, filename: &str) -> Result<PathBuf, InstalledModelError> {
        if !self.artifacts.contains(filename) {
            return Err(InstalledModelError::ArtifactNotListed {
                filename: filename.to_string(),
            });
        }
        Ok(self.directory.join(filename))
    }
}

fn validate_model(manifest: &Manifest, model: &ModelInfo) -> Result<(), InstalledModelError> {
    let filename = "manifest.json".to_string();
    if manifest.schema_version != 1 {
        return Err(InstalledModelError::UnsupportedSchema { filename });
    }
    if manifest.model.id != model.id {
        return Err(InstalledModelError::ModelIdMismatch { filename });
    }
    if manifest.model.repository != model.huggingface_id {
        return Err(InstalledModelError::RepositoryMismatch { filename });
    }
    let revision = model
        .revision
        .ok_or_else(|| InstalledModelError::RevisionUnpinned {
            filename: filename.clone(),
        })?;
    if manifest.model.revision != revision {
        return Err(InstalledModelError::RevisionMismatch { filename });
    }
    Ok(())
}

fn validate_filename(filename: &str) -> Result<(), InstalledModelError> {
    let mut components = Path::new(filename).components();
    if filename.contains("..")
        || filename.contains(['/', '\\', '\0', ':'])
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(InstalledModelError::InvalidArtifactPath {
            filename: filename.to_string(),
        });
    }
    Ok(())
}

fn io_error(filename: &str, source: std::io::Error) -> InstalledModelError {
    InstalledModelError::ArtifactIo {
        filename: filename.to_string(),
        source,
    }
}

fn regular_file(directory: &Path, filename: &str) -> Result<File, InstalledModelError> {
    let path = directory.join(filename);
    let metadata = std::fs::symlink_metadata(&path).map_err(|source| io_error(filename, source))?;
    if metadata.file_type().is_symlink() {
        return Err(InstalledModelError::ArtifactSymlink {
            filename: filename.to_string(),
        });
    }
    if !metadata.is_file() {
        return Err(InstalledModelError::ArtifactNotFile {
            filename: filename.to_string(),
        });
    }
    File::open(path).map_err(|source| io_error(filename, source))
}

fn validate_artifact(directory: &Path, artifact: &Artifact) -> Result<(), InstalledModelError> {
    if artifact.sha256.len() != 64 || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(InstalledModelError::InvalidArtifactHash {
            filename: artifact.path.clone(),
        });
    }
    let mut file = regular_file(directory, &artifact.path)?;
    let actual = file
        .metadata()
        .map_err(|source| io_error(&artifact.path, source))?
        .len();
    if actual != artifact.size_bytes {
        return Err(InstalledModelError::ArtifactSizeMismatch {
            filename: artifact.path.clone(),
            expected: artifact.size_bytes,
            actual,
        });
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 8192];
    let mut total = 0u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| io_error(&artifact.path, source))?;
        if count == 0 {
            break;
        }
        total += count as u64;
        hash.update(&buffer[..count]);
    }
    if total != artifact.size_bytes {
        return Err(InstalledModelError::ArtifactSizeMismatch {
            filename: artifact.path.clone(),
            expected: artifact.size_bytes,
            actual: total,
        });
    }
    if !format!("{:x}", hash.finalize()).eq_ignore_ascii_case(&artifact.sha256) {
        return Err(InstalledModelError::ArtifactHashMismatch {
            filename: artifact.path.clone(),
        });
    }
    Ok(())
}
