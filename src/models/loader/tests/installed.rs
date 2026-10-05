use std::fs;

use serde_json::{json, Value};
use tempfile::TempDir;

use crate::models::loader::{InstalledModel, ModelFileResolver};

fn fixture() -> (TempDir, Value) {
    let dir = tempfile::tempdir().unwrap();
    let artifacts: Vec<Value> = ["config.json", "tokenizer.json", "model.safetensors"]
        .into_iter()
        .map(|path| {
            fs::write(dir.path().join(path), b"abc").unwrap();
            json!({"path": path, "size_bytes": 3,
                "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"})
        })
        .collect();
    let manifest = json!({"schema_version": 1,
        "model": {"id": "bge-base-en-v1.5", "repository": "BAAI/bge-base-en-v1.5",
            "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"},
        "artifacts": artifacts});
    (dir, manifest)
}

fn write_manifest(dir: &TempDir, manifest: &Value) {
    fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn installed_registry_id_comes_from_the_manifest() {
    let (dir, manifest) = fixture();
    write_manifest(&dir, &manifest);
    assert_eq!(
        InstalledModel::registry_id(dir.path()).unwrap(),
        "bge-base-en-v1.5"
    );
    fs::remove_file(dir.path().join("model.safetensors")).unwrap();
    assert_eq!(
        InstalledModel::registry_id(dir.path()).unwrap(),
        "bge-base-en-v1.5"
    );
}

#[test]
fn installed_registry_id_rejects_unknown_manifest_members() {
    let (dir, mut manifest) = fixture();
    manifest["model"]["extra"] = json!(true);
    write_manifest(&dir, &manifest);
    let error = InstalledModel::registry_id(dir.path()).unwrap_err();
    assert!(error.to_string().starts_with("invalid_manifest"));
}

#[test]
fn installed_registry_id_requires_a_registered_identity() {
    let cases = [
        ("id", "unregistered", "installed_model_not_registered"),
        (
            "repository",
            "other/repository",
            "installed_repository_mismatch",
        ),
        ("revision", "other-revision", "installed_revision_mismatch"),
    ];
    for (member, value, code) in cases {
        let (dir, mut manifest) = fixture();
        manifest["model"][member] = json!(value);
        write_manifest(&dir, &manifest);
        let error = InstalledModel::registry_id(dir.path()).unwrap_err();
        assert!(error.to_string().starts_with(code), "{error}");
    }
}

#[test]
fn installed_registry_id_rejects_an_unsupported_schema() {
    let (dir, mut manifest) = fixture();
    manifest["schema_version"] = json!(2);
    write_manifest(&dir, &manifest);
    let error = InstalledModel::registry_id(dir.path()).unwrap_err();
    assert!(error.to_string().starts_with("unsupported_manifest_schema"));
}

#[test]
fn installed_registry_id_names_a_missing_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let error = InstalledModel::registry_id(dir.path()).unwrap_err();
    assert!(error.to_string().starts_with("installed_artifact_io"));
    assert!(error.to_string().contains("manifest.json"));
}

#[test]
fn both_manifest_readers_refuse_an_oversized_file_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let file = fs::File::create(dir.path().join("manifest.json")).unwrap();
    file.set_len(1_048_577).unwrap();
    let errors = [
        InstalledModel::registry_id(dir.path()).unwrap_err(),
        InstalledModel::open("bge-base-en-v1.5", dir.path())
            .err()
            .expect("oversized manifest must not open"),
    ];
    for error in errors {
        let message = error.to_string();
        assert!(
            message.starts_with("installed_manifest_too_large"),
            "{message}"
        );
        assert!(message.contains("manifest.json"));
        assert!(message.contains("1048577"));
        assert!(message.contains("1048576"));
    }
}

#[test]
fn a_valid_manifest_at_the_byte_ceiling_is_accepted_by_both_readers() {
    let (dir, manifest) = fixture();
    let mut bytes = serde_json::to_vec(&manifest).unwrap();
    bytes.resize(1_048_576, b' ');
    fs::write(dir.path().join("manifest.json"), bytes).unwrap();
    assert_eq!(
        InstalledModel::registry_id(dir.path()).unwrap(),
        "bge-base-en-v1.5"
    );
    assert!(InstalledModel::open("bge-base-en-v1.5", dir.path()).is_ok());
}

fn refusal(dir: &TempDir, manifest: &Value, name: &str, filename: &str) {
    write_manifest(dir, manifest);
    let Err(error) = InstalledModel::open("bge-base-en-v1.5", dir.path()) else {
        panic!("invalid installed folder was accepted");
    };
    let message = error.to_string();
    assert!(message.starts_with(name), "{message}");
    assert!(message.contains(&format!("{filename:?}")), "{message}");
}

#[test]
fn installed_valid_folder_opens() {
    let (dir, manifest) = fixture();
    write_manifest(&dir, &manifest);
    let installed = InstalledModel::open("bge-base-en-v1.5", dir.path()).unwrap();
    assert_eq!(installed.manifest_sha256().len(), 64);
}

#[test]
fn installed_manifest_digest_known() {
    let (dir, manifest) = fixture();
    write_manifest(&dir, &manifest);
    let installed = InstalledModel::open("bge-base-en-v1.5", dir.path()).unwrap();
    assert_eq!(
        installed.manifest_sha256(),
        "27045b6bfabdfa66541a7e24f5a0fd4774f4ea055b91fe17a8b7c35e05eda6df"
    );
    assert!(installed
        .manifest_sha256()
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
}

#[test]
fn installed_resolver_keeps_digest_from_artifact_open() {
    let (dir, manifest) = fixture();
    write_manifest(&dir, &manifest);
    let model = crate::models::registry::get_model("bge-base-en-v1.5").unwrap();
    let files = ModelFileResolver::installed(model, dir.path()).unwrap();
    fs::remove_file(dir.path().join("manifest.json")).unwrap();
    assert_eq!(
        files.installed_manifest_sha256(),
        Some("27045b6bfabdfa66541a7e24f5a0fd4774f4ea055b91fe17a8b7c35e05eda6df")
    );
    assert_eq!(
        files.get("config.json").unwrap(),
        dir.path().join("config.json")
    );
}

#[cfg(feature = "fetch")]
#[test]
fn ordinary_resolver_has_no_installed_manifest_digest() {
    let dir = tempfile::tempdir().unwrap();
    let model = crate::models::registry::get_model("bge-base-en-v1.5").unwrap();
    let source = super::super::ArtifactSource::Offline(
        hf_hub::Cache::new(dir.path().to_path_buf())
            .repo(super::super::validated_repo(model).unwrap()),
    );
    let files = ModelFileResolver { model, source };
    assert_eq!(files.installed_manifest_sha256(), None);
}

#[test]
fn installed_unknown_top_level_member_refused() {
    let (dir, mut manifest) = fixture();
    manifest["extra"] = json!("private contents");
    refusal(&dir, &manifest, "invalid_manifest", "manifest.json");
}

#[test]
fn installed_unknown_model_member_refused() {
    let (dir, mut manifest) = fixture();
    manifest["model"]["extra"] = json!(true);
    refusal(&dir, &manifest, "invalid_manifest", "manifest.json");
}

#[test]
fn installed_unknown_artifact_member_refused() {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"][0]["extra"] = json!(true);
    refusal(&dir, &manifest, "invalid_manifest", "manifest.json");
}

#[test]
fn installed_wrong_schema_refused() {
    let (dir, mut manifest) = fixture();
    manifest["schema_version"] = json!(2);
    refusal(
        &dir,
        &manifest,
        "unsupported_manifest_schema",
        "manifest.json",
    );
}

#[test]
fn installed_wrong_id_refused() {
    let (dir, mut manifest) = fixture();
    manifest["model"]["id"] = json!("another");
    refusal(
        &dir,
        &manifest,
        "installed_model_id_mismatch",
        "manifest.json",
    );
}

#[test]
fn installed_wrong_repository_refused() {
    let (dir, mut manifest) = fixture();
    manifest["model"]["repository"] = json!("another/repository");
    refusal(
        &dir,
        &manifest,
        "installed_repository_mismatch",
        "manifest.json",
    );
}

#[test]
fn installed_wrong_revision_refused() {
    let (dir, mut manifest) = fixture();
    manifest["model"]["revision"] = json!("different");
    refusal(
        &dir,
        &manifest,
        "installed_revision_mismatch",
        "manifest.json",
    );
}

fn missing_artifact(filename: &str) {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"]
        .as_array_mut()
        .unwrap()
        .retain(|artifact| artifact["path"] != filename);
    refusal(&dir, &manifest, "installed_artifact_not_listed", filename);
}

#[test]
fn installed_config_not_listed() {
    missing_artifact("config.json");
}
#[test]
fn installed_tokenizer_not_listed() {
    missing_artifact("tokenizer.json");
}
#[test]
fn installed_weights_not_listed() {
    missing_artifact("model.safetensors");
}

fn invalid_path(path: &str) {
    let (dir, mut manifest) = fixture();
    let mut artifact = manifest["artifacts"][0].clone();
    artifact["path"] = json!(path);
    manifest["artifacts"].as_array_mut().unwrap().push(artifact);
    refusal(&dir, &manifest, "invalid_installed_artifact_path", path);
}

#[test]
fn installed_parent_path_refused() {
    invalid_path("..");
}
#[test]
fn installed_slash_refused() {
    invalid_path("nested/config.json");
}
#[test]
fn installed_backslash_refused() {
    invalid_path("nested\\config.json");
}
#[test]
fn installed_absolute_path_refused() {
    invalid_path("/config.json");
}
#[test]
fn installed_empty_path_refused() {
    invalid_path("");
}
#[test]
fn installed_dot_path_refused() {
    invalid_path(".");
}
#[test]
fn installed_embedded_parent_path_refused() {
    invalid_path("config..json");
}

#[cfg(unix)]
#[test]
fn installed_artifact_symlink_refused() {
    let (dir, manifest) = fixture();
    fs::remove_file(dir.path().join("config.json")).unwrap();
    std::os::unix::fs::symlink("tokenizer.json", dir.path().join("config.json")).unwrap();
    refusal(&dir, &manifest, "installed_artifact_symlink", "config.json");
}

#[cfg(unix)]
#[test]
fn installed_manifest_symlink_refused() {
    let (dir, manifest) = fixture();
    fs::write(
        dir.path().join("real.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    std::os::unix::fs::symlink("real.json", dir.path().join("manifest.json")).unwrap();
    let Err(error) = InstalledModel::open("bge-base-en-v1.5", dir.path()) else {
        panic!("symlink accepted");
    };
    assert!(error.to_string().starts_with("installed_artifact_symlink"));
}

#[test]
fn installed_not_regular_file_refused() {
    let (dir, manifest) = fixture();
    fs::remove_file(dir.path().join("config.json")).unwrap();
    fs::create_dir(dir.path().join("config.json")).unwrap();
    refusal(
        &dir,
        &manifest,
        "installed_artifact_not_file",
        "config.json",
    );
}

#[test]
fn installed_missing_file_refused() {
    let (dir, manifest) = fixture();
    fs::remove_file(dir.path().join("config.json")).unwrap();
    refusal(&dir, &manifest, "installed_artifact_io", "config.json");
}

#[test]
fn installed_wrong_size_refused() {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"][0]["size_bytes"] = json!(4);
    refusal(
        &dir,
        &manifest,
        "installed_artifact_size_mismatch",
        "config.json",
    );
}

#[test]
fn installed_wrong_hash_refused() {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"][0]["sha256"] = json!("0".repeat(64));
    refusal(
        &dir,
        &manifest,
        "installed_artifact_hash_mismatch",
        "config.json",
    );
}

#[test]
fn installed_invalid_hash_refused() {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"][0]["sha256"] = json!("invalid");
    refusal(
        &dir,
        &manifest,
        "invalid_installed_artifact_hash",
        "config.json",
    );
}

#[test]
fn installed_duplicate_artifact_refused() {
    let (dir, mut manifest) = fixture();
    let duplicate = manifest["artifacts"][0].clone();
    manifest["artifacts"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    refusal(
        &dir,
        &manifest,
        "duplicate_installed_artifact",
        "config.json",
    );
}

#[test]
fn installed_extra_artifact_is_hashed() {
    let (dir, mut manifest) = fixture();
    fs::write(dir.path().join("extra.json"), b"xyz").unwrap();
    let mut extra = manifest["artifacts"][0].clone();
    extra["path"] = json!("extra.json");
    manifest["artifacts"].as_array_mut().unwrap().push(extra);
    refusal(
        &dir,
        &manifest,
        "installed_artifact_hash_mismatch",
        "extra.json",
    );
}

#[test]
fn installed_pickle_only_model_refused() {
    let (dir, manifest) = fixture();
    write_manifest(&dir, &manifest);
    let Err(error) = InstalledModel::open("splade-pp-en-v1", dir.path()) else {
        panic!("pickle accepted");
    };
    assert!(error
        .to_string()
        .starts_with("installed_safetensors_required"));
    assert!(error.to_string().contains("pytorch_model.bin"));
}

#[cfg(not(feature = "fetch"))]
#[test]
fn installed_fetch_not_built_in_is_named() {
    let model = crate::models::registry::get_model("bge-base-en-v1.5").unwrap();
    let Err(error) = crate::models::loader::ModelFileResolver::new(model) else {
        panic!("fetch accepted");
    };
    assert!(error.to_string().contains("fetching is not built in"));
}

#[cfg(not(feature = "fetch"))]
#[test]
fn installed_dense_builder_reports_fetch_not_built_in() {
    let result = crate::TesseraDenseBuilder::new()
        .model("bge-base-en-v1.5")
        .build();
    assert!(
        matches!(result, Err(crate::error::TesseraError::FetchingNotBuiltIn { model_id })
        if model_id == "bge-base-en-v1.5")
    );
}

#[test]
fn installed_dense_builder_checks_manifest_before_tokenizer_without_offline_environment() {
    let (dir, mut manifest) = fixture();
    manifest["artifacts"][0]["size_bytes"] = json!(4);
    write_manifest(&dir, &manifest);
    std::env::set_var("TESSERA_OFFLINE", "invalid");
    let result = crate::TesseraDenseBuilder::new()
        .model("bge-base-en-v1.5")
        .model_dir(dir.path())
        .device(candle_core::Device::Cpu)
        .build();
    let Err(crate::error::TesseraError::ModelLoadError { source, .. }) = result else {
        panic!("installed validation must fail before tokenizer loading");
    };
    assert!(
        matches!(source.downcast_ref::<crate::InstalledModelError>(),
        Some(crate::InstalledModelError::ArtifactSizeMismatch { filename, .. }) if filename == "config.json")
    );
}
