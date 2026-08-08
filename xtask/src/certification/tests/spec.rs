use super::{digest_bytes, validate_artifact_path, validate_hex};

#[test]
fn hashes_spec_bytes_deterministically() {
    assert_eq!(
        digest_bytes(b"tessera"),
        "2f1e83d30fff12f10f4a956d08bd6b200ae89e24621c2066c1a902aab2da7acb"
    );
}

#[test]
fn rejects_unsafe_artifact_paths() {
    assert!(validate_artifact_path("model.safetensors").is_ok());
    assert!(validate_artifact_path("weights/model-00001.safetensors").is_ok());
    assert!(validate_artifact_path("../model.safetensors").is_err());
    assert!(validate_artifact_path("/tmp/model.safetensors").is_err());
}

#[test]
fn accepts_only_lowercase_fixed_width_hex() {
    assert!(validate_hex(&"a".repeat(40), 40, "revision").is_ok());
    assert!(validate_hex(&"A".repeat(40), 40, "revision").is_err());
    assert!(validate_hex(&"a".repeat(39), 40, "revision").is_err());
}
