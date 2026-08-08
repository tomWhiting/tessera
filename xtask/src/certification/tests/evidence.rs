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
