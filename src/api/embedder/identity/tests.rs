use std::fs;

use serde_json::{json, Value};
use tempfile::TempDir;

use super::{EmbedFailure, LoadedFacts, ModelIdentity, EMBED_FAILURE_MAX_CHARS};
use crate::core::tokenizer::tests::{cut_tokenizer_with_policy, tokenizer};
use crate::error::TesseraError;
use crate::models::registry::{get_model, Distance};
use crate::models::InstalledModel;
use crate::runtime::ResourcePolicy;

const BGE_BASE_REVISION: &str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a";
const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

fn facts(position_table: Option<usize>) -> LoadedFacts {
    LoadedFacts {
        dimensions: 768,
        special_tokens: 2,
        normalised: true,
        position_table,
    }
}

#[test]
fn special_tokens_count_the_template_added_to_one_sequence() {
    let policy = ResourcePolicy::new(8, 16, 2048, usize::MAX);
    assert_eq!(cut_tokenizer_with_policy(policy).cut_special_tokens(), 2);
    assert_eq!(tokenizer(policy).cut_special_tokens(), 0);
}

#[test]
fn bge_base_identity_without_a_manifest() {
    let model = get_model("bge-base-en-v1.5").unwrap();
    let identity = ModelIdentity::new(model, facts(Some(512)), None).unwrap();
    assert_eq!(
        identity,
        ModelIdentity {
            name: "bge-base-en-v1.5".to_string(),
            repository: "BAAI/bge-base-en-v1.5".to_string(),
            revision: BGE_BASE_REVISION.to_string(),
            manifest_sha256: None,
            dimensions: 768,
            max_tokens: 512,
            special_tokens: 2,
            normalised: true,
            distance: Distance::Dot,
        }
    );
}

#[test]
fn bge_base_identity_with_a_manifest() {
    let model = get_model("bge-base-en-v1.5").unwrap();
    let digest = "27045b6bfabdfa66541a7e24f5a0fd4774f4ea055b91fe17a8b7c35e05eda6df";
    let identity = ModelIdentity::new(model, facts(Some(512)), Some(digest.to_string())).unwrap();
    assert_eq!(identity.manifest_sha256.as_deref(), Some(digest));
    assert_eq!(identity.revision, BGE_BASE_REVISION);
    assert_eq!(identity.name, "bge-base-en-v1.5");
}

#[test]
fn max_tokens_never_exceeds_the_position_table() {
    let model = get_model("bge-base-en-v1.5").unwrap();
    for (table, expected) in [(Some(256), 256), (Some(514), 512), (None, 512)] {
        let identity = ModelIdentity::new(model, facts(table), None).unwrap();
        assert_eq!(identity.max_tokens, expected, "{table:?}");
    }
}

fn fixture() -> (TempDir, Value) {
    let dir = tempfile::tempdir().unwrap();
    let artifacts: Vec<Value> = ["config.json", "tokenizer.json", "model.safetensors"]
        .into_iter()
        .map(|path| {
            fs::write(dir.path().join(path), b"abc").unwrap();
            json!({"path": path, "size_bytes": 3, "sha256": ABC_SHA256})
        })
        .collect();
    let manifest = json!({"schema_version": 1,
        "model": {"id": "bge-base-en-v1.5", "repository": "BAAI/bge-base-en-v1.5",
            "revision": BGE_BASE_REVISION},
        "artifacts": artifacts});
    (dir, manifest)
}

fn failure(model_id: &str, dir: &TempDir, manifest: Option<&Value>) -> EmbedFailure {
    if let Some(manifest) = manifest {
        fs::write(
            dir.path().join("manifest.json"),
            serde_json::to_vec(manifest).unwrap(),
        )
        .unwrap();
    }
    let Err(error) = InstalledModel::open(model_id, dir.path()) else {
        panic!("invalid installed folder was accepted");
    };
    let failure = EmbedFailure::from_installed(&error).expect("failure must be named");
    assert!(failure.to_string().contains(&error.to_string()));
    failure
}

fn assert_mismatch(failure: &EmbedFailure, file: &str) {
    assert_eq!(failure.code(), "embed_model_mismatch");
    assert!(matches!(failure, EmbedFailure::ModelMismatch { file: f, .. } if f == file));
}

fn assert_missing(failure: &EmbedFailure, file: &str) {
    assert_eq!(failure.code(), "embed_model_missing");
    assert!(matches!(failure, EmbedFailure::ModelMissing { file: f, .. } if f == file));
}

#[test]
fn hash_mismatch_is_a_model_mismatch() {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"][0]["sha256"] = json!("0".repeat(64));
    assert_mismatch(
        &failure("bge-base-en-v1.5", &dir, Some(&manifest)),
        "config.json",
    );
}

#[test]
fn size_mismatch_is_a_model_mismatch() {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"][1]["size_bytes"] = json!(4);
    assert_mismatch(
        &failure("bge-base-en-v1.5", &dir, Some(&manifest)),
        "tokenizer.json",
    );
}

#[test]
fn model_id_mismatch_is_a_model_mismatch() {
    let (dir, mut manifest) = fixture();
    manifest["model"]["id"] = json!("another");
    assert_mismatch(
        &failure("bge-base-en-v1.5", &dir, Some(&manifest)),
        "manifest.json",
    );
}

#[test]
fn repository_mismatch_is_a_model_mismatch() {
    let (dir, mut manifest) = fixture();
    manifest["model"]["repository"] = json!("another/repository");
    assert_mismatch(
        &failure("bge-base-en-v1.5", &dir, Some(&manifest)),
        "manifest.json",
    );
}

#[test]
fn revision_mismatch_is_a_model_mismatch() {
    let (dir, mut manifest) = fixture();
    manifest["model"]["revision"] = json!("0".repeat(40));
    assert_mismatch(
        &failure("bge-base-en-v1.5", &dir, Some(&manifest)),
        "manifest.json",
    );
}

#[test]
fn unregistered_model_is_a_model_mismatch() {
    let (dir, manifest) = fixture();
    assert_mismatch(
        &failure("not-a-model", &dir, Some(&manifest)),
        "manifest.json",
    );
}

#[test]
fn unknown_manifest_schema_is_a_model_mismatch() {
    let (dir, mut manifest) = fixture();
    manifest["schema_version"] = json!(2);
    assert_mismatch(
        &failure("bge-base-en-v1.5", &dir, Some(&manifest)),
        "manifest.json",
    );
}

#[test]
fn absent_manifest_is_a_missing_model() {
    let (dir, _) = fixture();
    assert_missing(&failure("bge-base-en-v1.5", &dir, None), "manifest.json");
}

#[test]
fn unlisted_artifact_is_a_missing_model() {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"]
        .as_array_mut()
        .unwrap()
        .retain(|artifact| artifact["path"] != "model.safetensors");
    assert_missing(
        &failure("bge-base-en-v1.5", &dir, Some(&manifest)),
        "model.safetensors",
    );
}

#[test]
fn listed_but_absent_artifact_is_a_missing_model() {
    let (dir, manifest) = fixture();
    fs::remove_file(dir.path().join("config.json")).unwrap();
    assert_missing(
        &failure("bge-base-en-v1.5", &dir, Some(&manifest)),
        "config.json",
    );
}

#[test]
fn other_installed_failures_are_not_named() {
    let (dir, mut manifest) = fixture();
    manifest["extra"] = json!(true);
    fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let Err(error) = InstalledModel::open("bge-base-en-v1.5", dir.path()) else {
        panic!("invalid manifest was accepted");
    };
    assert_eq!(EmbedFailure::from_installed(&error), None);
}

#[test]
fn builder_reports_an_installed_mismatch_by_its_code() {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"][0]["size_bytes"] = json!(4);
    fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let Err(error) = crate::TesseraDenseBuilder::new()
        .model("bge-base-en-v1.5")
        .model_dir(dir.path())
        .device(candle_core::Device::Cpu)
        .build()
    else {
        panic!("a size mismatch must be refused");
    };
    assert!(matches!(error, TesseraError::ModelLoadError { .. }));
    assert_mismatch(&error.embed_failure().unwrap(), "config.json");
}

#[test]
fn every_code_is_found_in_an_error_chain() {
    let missing = TesseraError::ModelLoadError {
        model_id: "bge-base-en-v1.5".to_string(),
        source: anyhow::Error::new(crate::InstalledModelError::ArtifactNotListed {
            filename: "tokenizer.json".to_string(),
        })
        .context("Loading tokenizer"),
    };
    let mismatch = TesseraError::ModelLoadError {
        model_id: "bge-base-en-v1.5".to_string(),
        source: anyhow::Error::new(crate::InstalledModelError::ArtifactHashMismatch {
            filename: "config.json".to_string(),
        }),
    };
    let limits = EmbedFailure::limits("batch_items", "made-up limit".to_string());
    let output = TesseraError::EncodingError {
        context: "made-up chunk".to_string(),
        source: anyhow::Error::new(EmbedFailure::OutputInvalid {
            index: 7,
            reason: "made-up reason".to_string(),
        }),
    };
    let codes: Vec<_> = [missing, mismatch, limits, output]
        .iter()
        .map(|error| error.embed_failure().unwrap().code())
        .collect();
    assert_eq!(
        codes,
        [
            "embed_model_missing",
            "embed_model_mismatch",
            "embed_limits",
            "embed_output_invalid"
        ]
    );
}

#[test]
fn errors_without_a_named_failure_return_none() {
    let unreadable = TesseraError::ModelLoadError {
        model_id: "bge-base-en-v1.5".to_string(),
        source: anyhow::Error::new(crate::InstalledModelError::ArtifactIo {
            filename: "config.json".to_string(),
            source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        }),
    };
    for error in [
        TesseraError::ConfigError("made-up".to_string()),
        TesseraError::Other(anyhow::anyhow!("made-up")),
        unreadable,
    ] {
        assert_eq!(error.embed_failure(), None, "{error}");
    }
}

#[test]
fn display_names_the_subject_within_the_length_cap() {
    let long = "x".repeat(4 * EMBED_FAILURE_MAX_CHARS);
    let failures = [
        EmbedFailure::from_installed(&crate::InstalledModelError::ArtifactHashMismatch {
            filename: long.clone(),
        })
        .unwrap(),
        EmbedFailure::from_installed(&crate::InstalledModelError::ArtifactNotListed {
            filename: "config.json".to_string(),
        })
        .unwrap(),
        EmbedFailure::Limits {
            limit: "batch_items",
            message: long.clone(),
        },
        EmbedFailure::OutputInvalid {
            index: 3,
            reason: long,
        },
    ];
    for failure in &failures {
        assert!(failure.to_string().chars().count() <= EMBED_FAILURE_MAX_CHARS);
        assert!(
            failure.to_string().starts_with(failure.code()) || failure.code() == "embed_limits"
        );
    }
    assert!(failures[1].to_string().contains("\"config.json\""));
    assert!(failures[3].to_string().contains("item 3"));
}
