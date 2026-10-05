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

    /// Builds `text_longer_than_model {"items":[{"id":"...","tokens_total":N,"tokens_limit":M},...],"omitted":K}`.
    ///
    /// Whole items form a request-order prefix; omitted counts the remaining
    /// overlength items. The complete message fits the shared character bound.
    pub fn overlength(items: &[(&str, usize, usize)]) -> Self {
        let mut message = String::from("text_longer_than_model {\"items\":[");
        let mut characters = message.chars().count();
        let mut included = 0;
        for &(id, tokens_total, tokens_limit) in items {
            let item = format!(
                "{{\"id\":{},\"tokens_total\":{tokens_total},\"tokens_limit\":{tokens_limit}}}",
                json_string(id)
            );
            let separator = usize::from(included != 0);
            let suffix = format!("],\"omitted\":{}}}", items.len() - included - 1);
            let added = item.chars().count() + separator;
            if characters + added + suffix.chars().count()
                > haem_frames::embedding::FAILED_MESSAGE_CHARS
            {
                break;
            }
            if included != 0 {
                message.push(',');
            }
            message.push_str(&item);
            characters += added;
            included += 1;
        }
        message.push_str(&format!("],\"omitted\":{}}}", items.len() - included));
        Self::limits(message)
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

fn json_string(value: &str) -> String {
    const HEX: [char; 16] = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    ];
    let mut quoted = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            character if character < ' ' => {
                let mut encoded = [0; 4];
                let code = usize::from(character.encode_utf8(&mut encoded).as_bytes()[0]);
                quoted.push_str("\\u00");
                quoted.push(HEX[code / 16]);
                quoted.push(HEX[code % 16]);
            }
            character => quoted.push(character),
        }
    }
    quoted.push('"');
    quoted
}
