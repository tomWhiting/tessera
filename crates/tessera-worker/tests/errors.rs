//! Checks how library errors map to the worker's failure codes.

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

/// A message over the frame's character cap is written whole to the worker's
/// log first; the frame keeps whole characters and names how many it left out.
#[test]
fn an_over_cap_message_is_logged_whole_and_names_its_cut() {
    let whole = "é".repeat(2048);
    let mut log = Vec::new();
    let failed = failure::Failure::new(FailedCode::EmbedProtocol, whole.clone())
        .into_message_logged(&mut log);
    assert!(String::from_utf8(log).unwrap().contains(&whole));
    let count = failed.message.chars().count();
    assert!(count <= haem_frames::embedding::FAILED_MESSAGE_CHARS);
    let (kept, named) = failed.message.split_once('…').unwrap();
    assert!(kept.chars().all(|character| character == 'é'));
    let left = kept.chars().count();
    assert_eq!(
        named,
        format!(
            " {} more characters (full text on the worker's stderr)",
            2048 - left
        )
    );
}

#[test]
fn a_message_within_the_cap_is_sent_whole_and_not_logged() {
    let mut log = Vec::new();
    let whole = "é".repeat(haem_frames::embedding::FAILED_MESSAGE_CHARS);
    let failed = failure::Failure::new(FailedCode::EmbedProtocol, whole.clone())
        .into_message_logged(&mut log);
    assert_eq!(failed.message, whole);
    assert!(log.is_empty());
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

#[test]
fn overlength_messages_keep_json_whole_through_the_failure_frame_bound() {
    let ids = ["x".repeat(512), "y".repeat(512), "z".repeat(512)];
    let items: Vec<(&str, usize, usize)> = ids.iter().map(|id| (id.as_str(), 26, 16)).collect();
    let failed = failure::Failure::overlength(&items).into_message();
    assert_eq!(failed.code, FailedCode::EmbedLimits);
    assert!(failed.message.chars().count() <= haem_frames::embedding::FAILED_MESSAGE_CHARS);
    let object: serde_json::Value = serde_json::from_str(
        failed
            .message
            .strip_prefix("text_longer_than_model ")
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        object["items"],
        serde_json::json!([{"id":ids[0], "tokens_total":26, "tokens_limit":16}])
    );
    assert_eq!(object["omitted"], 2);
}
