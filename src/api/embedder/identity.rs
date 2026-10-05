//! The model identity an embedding worker reports, and the named failures it maps.

use std::fmt;

use crate::core::tokenizer::{CutConfigurationError, PromptConfigurationError};
use crate::error::TesseraError;
use crate::models::registry::{Distance, ModelInfo};
use crate::models::InstalledModelError;

#[cfg(test)]
mod tests;

/// Longest `Display` text of an [`EmbedFailure`], in characters.
pub const EMBED_FAILURE_MAX_CHARS: usize = 1024;

/// The exact model a dense embedder loaded, captured once when it was built.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelIdentity {
    /// Registry identifier.
    pub name: String,
    /// Hugging Face repository the registry pins.
    pub repository: String,
    /// Immutable 40-digit revision the registry pins.
    pub revision: String,
    /// Lowercase SHA-256 of the installed manifest, when loaded from one.
    pub manifest_sha256: Option<String>,
    /// Length of every returned vector.
    pub dimensions: usize,
    /// Registry context length, bounded by the model's position table.
    pub max_tokens: usize,
    /// Special tokens the tokenizer adds to one sequence.
    pub special_tokens: usize,
    /// Token count of the longer role prefix, excluding special tokens.
    pub prefix_tokens: usize,
    /// Whether returned vectors are L2-normalised.
    pub normalised: bool,
    /// How two vectors from this model are compared.
    pub distance: Distance,
}

/// Facts about a loaded model that the registry entry does not hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadedFacts {
    pub dimensions: usize,
    pub special_tokens: usize,
    pub prefix_tokens: usize,
    pub normalised: bool,
    pub position_table: Option<usize>,
}

impl ModelIdentity {
    pub(crate) fn new(
        model: &ModelInfo,
        facts: LoadedFacts,
        manifest_sha256: Option<String>,
    ) -> anyhow::Result<Self> {
        let revision = model
            .revision
            .ok_or_else(|| anyhow::anyhow!("Model '{}' has no pinned revision", model.id))?;
        let distance = model
            .distance
            .ok_or_else(|| anyhow::anyhow!("Model '{}' has no distance", model.id))?;
        let max_tokens = facts.position_table.map_or(model.context_length, |table| {
            table.min(model.context_length)
        });
        Ok(Self {
            name: model.id.to_string(),
            repository: model.huggingface_id.to_string(),
            revision: revision.to_string(),
            manifest_sha256,
            dimensions: facts.dimensions,
            max_tokens,
            special_tokens: facts.special_tokens,
            prefix_tokens: facts.prefix_tokens,
            normalised: facts.normalised,
            distance,
        })
    }
}

/// A failure an embedding worker reports by its stable code.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmbedFailure {
    /// An installed file or manifest is present but does not match the registry.
    ModelMismatch {
        /// File that failed.
        file: String,
        /// The validation failure, naming the file.
        reason: String,
    },
    /// An installed file or the manifest is absent.
    ModelMissing {
        /// File that is absent.
        file: String,
        /// The validation failure, naming the file.
        reason: String,
    },
    /// A configured limit cannot be met by this model.
    Limits {
        /// Limit that was refused.
        limit: &'static str,
        /// The configuration failure, naming the limit.
        message: String,
    },
    /// The model returned a vector that cannot be sent.
    OutputInvalid {
        /// Position of the item in the caller's input.
        index: usize,
        /// What is wrong with its vector.
        reason: String,
    },
}

impl EmbedFailure {
    /// Returns the stable code for this failure.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::ModelMismatch { .. } => "embed_model_mismatch",
            Self::ModelMissing { .. } => "embed_model_missing",
            Self::Limits { .. } => "embed_limits",
            Self::OutputInvalid { .. } => "embed_output_invalid",
        }
    }

    pub(crate) fn limits(limit: &'static str, message: String) -> TesseraError {
        TesseraError::Other(anyhow::Error::new(Self::Limits { limit, message }))
    }

    pub(crate) fn from_cut_configuration(error: anyhow::Error) -> TesseraError {
        if error.downcast_ref::<CutConfigurationError>().is_some()
            || error.downcast_ref::<PromptConfigurationError>().is_some()
        {
            Self::limits("max_sequence_tokens", error.to_string())
        } else {
            TesseraError::ConfigError(error.to_string())
        }
    }

    pub(crate) fn from_installed(error: &InstalledModelError) -> Option<Self> {
        use InstalledModelError as E;
        let (file, missing) = match error {
            E::ArtifactHashMismatch { filename }
            | E::ArtifactSizeMismatch { filename, .. }
            | E::ModelIdMismatch { filename }
            | E::RepositoryMismatch { filename }
            | E::RevisionMismatch { filename }
            | E::UnsupportedSchema { filename } => (filename.as_str(), false),
            E::ModelNotRegistered { .. } => ("manifest.json", false),
            E::ArtifactNotListed { filename } => (filename.as_str(), true),
            E::ArtifactIo { filename, source } if source.kind() == std::io::ErrorKind::NotFound => {
                (filename.as_str(), true)
            }
            _ => return None,
        };
        let file = truncate(file);
        let reason = truncate(&error.to_string());
        Some(if missing {
            Self::ModelMissing { file, reason }
        } else {
            Self::ModelMismatch { file, reason }
        })
    }

    fn from_source(error: &(dyn std::error::Error + 'static)) -> Option<Self> {
        if let Some(failure) = error.downcast_ref::<Self>() {
            return Some(failure.clone());
        }
        error
            .downcast_ref::<InstalledModelError>()
            .and_then(Self::from_installed)
    }
}

impl fmt::Display for EmbedFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::ModelMismatch { reason, .. } | Self::ModelMissing { reason, .. } => {
                format!("{}: {reason}", self.code())
            }
            Self::Limits { message, .. } => message.clone(),
            Self::OutputInvalid { index, reason } => {
                format!("{}: item {index}: {reason}", self.code())
            }
        };
        formatter.write_str(&truncate(&text))
    }
}

impl std::error::Error for EmbedFailure {}

impl TesseraError {
    /// Returns the [`EmbedFailure`] this error carries anywhere in its chain.
    ///
    /// Installed-model validation failures are named by their worker code;
    /// any other error returns `None`.
    #[must_use]
    pub fn embed_failure(&self) -> Option<EmbedFailure> {
        let (Self::ModelLoadError { source, .. }
        | Self::EncodingError { source, .. }
        | Self::Other(source)) = self
        else {
            return None;
        };
        source.chain().find_map(EmbedFailure::from_source)
    }
}

fn truncate(text: &str) -> String {
    text.chars().take(EMBED_FAILURE_MAX_CHARS).collect()
}
