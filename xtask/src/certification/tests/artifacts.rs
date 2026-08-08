use std::fs;

use super::{required_fetch_free_bytes, sha256_file};

#[test]
fn hashes_files_without_loading_them_as_one_buffer() {
    let path = std::env::temp_dir().join(format!(
        "tessera-cert-sha256-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::write(&path, b"tessera").unwrap();
    let digest = sha256_file(&path).unwrap();
    fs::remove_file(path).unwrap();

    assert_eq!(
        digest,
        "2f1e83d30fff12f10f4a956d08bd6b200ae89e24621c2066c1a902aab2da7acb"
    );
}

#[test]
fn fetch_reserves_artifacts_plus_retained_free_space() {
    assert_eq!(required_fetch_free_bytes(500, 1_500).unwrap(), 2_000);
    assert!(required_fetch_free_bytes(u64::MAX, 1).is_err());
}
