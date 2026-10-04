use std::path::Path;

use super::{evidence_path, now_unix_ms};

#[test]
fn evidence_paths_are_model_scoped() {
    let path = evidence_path(Path::new("/repo"), "bge-base-en-v1.5", 2, 1234);
    assert_eq!(
        path,
        Path::new("/repo/.tessera/cert-evidence/bge-base-en-v1.5/1234-run-2.json")
    );
    assert!(now_unix_ms().unwrap() > 0);
}

#[test]
fn older_outcome_without_installed_digest_remains_readable() {
    use crate::certification::evidence::ChildOutcome;
    use crate::certification::reference::ReferenceComparison;
    let old = serde_json::json!({
        "status": "passed", "error": null, "verified_artifacts": [], "observation": null,
        "reference_comparison": ReferenceComparison::not_configured()
    });
    let mut outcome: ChildOutcome = serde_json::from_value(old).unwrap();
    assert!(outcome.installed_manifest_sha256.is_none());
    let digest = "a".repeat(64);
    outcome.installed_manifest_sha256 = Some(digest.clone());
    let serialized = serde_json::to_value(&outcome).unwrap();
    assert_eq!(serialized["installed_manifest_sha256"], digest);
    let reread: ChildOutcome = serde_json::from_value(serialized).unwrap();
    assert_eq!(
        reread.installed_manifest_sha256.as_deref(),
        Some(digest.as_str())
    );
}
