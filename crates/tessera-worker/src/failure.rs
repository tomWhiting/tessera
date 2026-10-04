use std::path::Path;

use haem_frames::embedding::{Failed, FailedCode};
use tessera::{EmbedFailure, InstalledModelError, ResourcePolicyError, TesseraError};

pub struct Failure {
    code: FailedCode,
    message: String,
}

impl Failure {
    pub fn new(code: FailedCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn limits(message: impl Into<String>) -> Self {
        Self::new(FailedCode::EmbedLimits, message)
    }

    pub fn installed(error: &InstalledModelError, directory: &Path) -> Self {
        match error {
            InstalledModelError::ArtifactIo { filename, source } => Self::new(
                FailedCode::EmbedModelMissing,
                format!("{}: {source}", directory.join(filename).display()),
            ),
            _ => Self::new(
                FailedCode::EmbedModelMismatch,
                format!("{}: {error}", directory.display()),
            ),
        }
    }

    pub fn model_load(error: &TesseraError, directory: &Path) -> Self {
        if let TesseraError::ModelLoadError { source, .. }
        | TesseraError::EncodingError { source, .. }
        | TesseraError::Other(source) = error
        {
            for cause in source.chain() {
                if let Some(installed) = cause.downcast_ref::<InstalledModelError>() {
                    return Self::installed(installed, directory);
                }
                if cause.downcast_ref::<ResourcePolicyError>().is_some() {
                    return Self::limits(error.to_string());
                }
                if let Some(io) = cause.downcast_ref::<std::io::Error>() {
                    return Self::new(
                        FailedCode::EmbedModelMissing,
                        format!("{}: {error}: {io}", directory.display()),
                    );
                }
            }
        }
        if let Some(EmbedFailure::Limits { .. }) = error.embed_failure() {
            return Self::limits(error.to_string());
        }
        match error {
            TesseraError::IoError(source) => Self::new(
                FailedCode::EmbedModelMissing,
                format!("{}: {source}", directory.display()),
            ),
            TesseraError::ConfigError(_) => Self::limits(error.to_string()),
            TesseraError::ModelLoadError { .. } | TesseraError::ModelNotFound { .. } => Self::new(
                FailedCode::EmbedModelMismatch,
                format!("{}: {error}", directory.display()),
            ),
            _ => Self::new(FailedCode::EmbedInferenceFailed, error.to_string()),
        }
    }

    pub fn inference(error: &TesseraError) -> Self {
        let code = match error.embed_failure() {
            Some(EmbedFailure::Limits { .. }) => FailedCode::EmbedLimits,
            Some(EmbedFailure::OutputInvalid { .. }) => FailedCode::EmbedOutputInvalid,
            _ => FailedCode::EmbedInferenceFailed,
        };
        let limit_error = match error {
            TesseraError::EncodingError { source, .. } | TesseraError::Other(source) => source
                .chain()
                .any(|cause| cause.downcast_ref::<ResourcePolicyError>().is_some()),
            _ => false,
        };
        Self::new(
            if limit_error {
                FailedCode::EmbedLimits
            } else {
                code
            },
            error.to_string(),
        )
    }

    pub fn into_message(self) -> Failed {
        Failed {
            code: self.code,
            message: self
                .message
                .chars()
                .take(haem_frames::embedding::FAILED_MESSAGE_CHARS)
                .collect(),
        }
    }
}

impl From<haem_frames::embedding::Error> for Failure {
    fn from(error: haem_frames::embedding::Error) -> Self {
        Self::new(error.code, error.message)
    }
}
