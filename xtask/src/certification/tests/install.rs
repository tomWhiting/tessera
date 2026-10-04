use std::fs;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

use super::{install, prepare_directory, require_dense};
use crate::certification::spec::{load_model, LoadedSpec, Representation};

struct Fixture {
    root: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!(
                "test fixture cleanup failed for {}: {error}",
                self.root.display()
            );
            assert!(
                std::thread::panicking(),
                "test fixture cleanup failed: {error}"
            );
        }
    }
}

fn fixture() -> (Fixture, LoadedSpec) {
    let root = std::env::temp_dir().join(format!(
        "tessera-cert-install-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::create_dir(&root).unwrap();
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let mut loaded = load_model(&repository, "bge-base-en-v1.5").unwrap();
    let snapshot = root
        .join(".tessera/cert-cache/hub/models--BAAI--bge-base-en-v1.5/snapshots")
        .join(&loaded.spec.model.revision);
    fs::create_dir_all(&snapshot).unwrap();
    let refs = snapshot.parent().unwrap().parent().unwrap().join("refs");
    fs::create_dir(&refs).unwrap();
    fs::write(
        refs.join(&loaded.spec.model.revision),
        &loaded.spec.model.revision,
    )
    .unwrap();
    for artifact in &mut loaded.spec.artifacts {
        artifact.size_bytes = 3;
        artifact.sha256 = format!("{:x}", Sha256::digest(b"abc"));
        fs::write(snapshot.join(&artifact.path), b"abc").unwrap();
    }
    (Fixture { root }, loaded)
}

#[test]
fn installs_verified_files_and_loader_compatible_manifest() {
    let (fixture, loaded) = fixture();
    let root = &fixture.root;
    let destination = root.join("installed");
    let digest = install(root, &loaded, &destination).unwrap();
    let bytes = fs::read(destination.join("manifest.json")).unwrap();
    assert_eq!(digest, format!("{:x}", Sha256::digest(&bytes)));
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["model"]["revision"], loaded.spec.model.revision);
    assert_eq!(manifest["model"].as_object().unwrap().len(), 3);
    assert_eq!(manifest["artifacts"].as_array().unwrap().len(), 3);
    for artifact in &loaded.spec.artifacts {
        assert_eq!(fs::read(destination.join(&artifact.path)).unwrap(), b"abc");
        assert!(!fs::symlink_metadata(destination.join(&artifact.path))
            .unwrap()
            .file_type()
            .is_symlink());
    }
}

#[test]
fn nonempty_destination_is_preserved() {
    let (fixture, loaded) = fixture();
    let root = &fixture.root;
    let destination = root.join("installed");
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("existing"), b"keep").unwrap();
    let error = install(root, &loaded, &destination)
        .unwrap_err()
        .to_string();
    assert!(error.contains("not empty"), "{error}");
    assert_eq!(fs::read(destination.join("existing")).unwrap(), b"keep");
    assert_eq!(fs::read_dir(&destination).unwrap().count(), 1);
}

#[test]
fn corrupt_cache_is_refused_before_destination_is_created() {
    let (fixture, loaded) = fixture();
    let root = &fixture.root;
    let source = root
        .join(".tessera/cert-cache/hub/models--BAAI--bge-base-en-v1.5/snapshots")
        .join(&loaded.spec.model.revision)
        .join("config.json");
    fs::write(source, b"bad").unwrap();
    let destination = root.join("installed");
    let error = install(root, &loaded, &destination)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("config.json") && error.contains("SHA-256 mismatch"),
        "{error}"
    );
    assert!(!destination.exists());
}

#[test]
fn incompatible_manifest_rolls_back_only_new_files() {
    let (fixture, mut loaded) = fixture();
    let root = &fixture.root;
    loaded.spec.model.revision = "0".repeat(40);
    let original = root.join(".tessera/cert-cache/hub/models--BAAI--bge-base-en-v1.5/snapshots");
    let old = fs::read_dir(&original)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    fs::rename(old, original.join(&loaded.spec.model.revision)).unwrap();
    fs::write(
        original
            .parent()
            .unwrap()
            .join("refs")
            .join(&loaded.spec.model.revision),
        &loaded.spec.model.revision,
    )
    .unwrap();
    let destination = root.join("installed");
    fs::create_dir(&destination).unwrap();
    let error = install(root, &loaded, &destination)
        .unwrap_err()
        .to_string();
    assert!(error.contains("installed_revision_mismatch"), "{error}");
    assert!(destination.is_dir());
    assert_eq!(fs::read_dir(&destination).unwrap().count(), 0);
}

#[test]
fn rejects_nondense_and_file_destinations() {
    assert!(require_dense(Representation::Sparse)
        .unwrap_err()
        .to_string()
        .contains("dense models only"));
    let root =
        std::env::temp_dir().join(format!("tessera-cert-install-file-{}", std::process::id()));
    fs::write(&root, b"keep").unwrap();
    assert!(prepare_directory(&root).is_err());
    assert_eq!(fs::read(&root).unwrap(), b"keep");
    fs::remove_file(root).unwrap();
}
