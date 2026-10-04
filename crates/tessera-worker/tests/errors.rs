use std::path::Path;

use haem_frames::embedding::FailedCode;
use tessera::{InstalledModelError, TesseraError};

#[path = "../src/failure.rs"]
mod failure;

#[test]
fn an_unclassified_inference_error_has_the_new_wire_code() {
    let error = TesseraError::TokenizationError("tokenizer refused the input".into());
    let failed = failure::Failure::inference(&error).into_message();
    assert_eq!(failed.code, FailedCode::EmbedInferenceFailed);
    assert!(failed.message.contains("tokenizer refused"));
}

#[test]
fn every_unreadable_file_is_missing_even_when_it_exists() {
    let error = InstalledModelError::ArtifactIo {
        filename: "config.json".into(),
        source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
    };
    let failed = failure::Failure::installed(&error, Path::new("/installed")).into_message();
    assert_eq!(failed.code, FailedCode::EmbedModelMissing);
    assert!(failed.message.contains("/installed/config.json"));
    assert!(failed.message.contains("permission denied"));
}

#[test]
fn a_folder_contents_refusal_is_model_mismatch() {
    let error = InstalledModelError::ArtifactSymlink {
        filename: "model.safetensors".into(),
    };
    let failed = failure::Failure::installed(&error, Path::new("/installed")).into_message();
    assert_eq!(failed.code, FailedCode::EmbedModelMismatch);
    assert!(failed.message.contains("model.safetensors"));
}

#[test]
fn failures_bound_unicode_characters_without_breaking_the_message() {
    let failed = failure::Failure::new(FailedCode::EmbedProtocol, "é".repeat(2048)).into_message();
    assert_eq!(failed.message.chars().count(), 1024);
    assert!(failed.message.chars().all(|character| character == 'é'));
}

#[test]
fn model_load_io_errors_keep_the_missing_code_and_os_reason() {
    let error = TesseraError::IoError(std::io::Error::from(std::io::ErrorKind::NotFound));
    let failed = failure::Failure::model_load(&error, Path::new("/installed")).into_message();
    assert_eq!(failed.code, FailedCode::EmbedModelMissing);
    assert!(failed.message.contains("/installed"));
    assert!(failed.message.contains("entity not found"));
}

#[test]
fn model_configuration_limits_keep_the_limits_code() {
    let error = TesseraError::ConfigError("derived Start budget cannot fit".into());
    let failed = failure::Failure::model_load(&error, Path::new("/installed")).into_message();
    assert_eq!(failed.code, FailedCode::EmbedLimits);
}
