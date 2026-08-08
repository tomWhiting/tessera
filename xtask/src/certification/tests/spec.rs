use std::path::Path;

use super::{
    digest_bytes, load_all, validate_artifact_path, validate_hex, ProfileKind, PromotionSpec,
};

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

#[test]
fn legacy_presence_only_reference_hash_is_rejected() {
    let value = serde_json::json!({
        "minimum_successful_runs": 2,
        "required_profiles": ["smoke"],
        "require_clean_source": true,
        "require_enforced_rss": true,
        "official_reference_sha256": "f".repeat(64)
    });
    assert!(serde_json::from_value::<PromotionSpec>(value).is_err());
}

#[test]
fn checked_specs_have_scoped_smoke_and_distinct_long_context_profiles() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let specs = load_all(repository).unwrap();
    assert_eq!(specs.len(), 10);
    for loaded in specs {
        let smoke = loaded.spec.profile("smoke").unwrap();
        assert_eq!(smoke.kind, ProfileKind::Smoke);
        assert_eq!(
            smoke.capability.max_sequence_tokens,
            smoke.resource_policy.max_sequence_tokens
        );
        if loaded.spec.model.id.starts_with("jina-embeddings")
            || matches!(
                loaded.spec.model.id.as_str(),
                "nomic-embed-v1.5" | "snowflake-arctic-l"
            )
        {
            let long = loaded.spec.profile("long-context-8k").unwrap();
            assert_eq!(long.kind, ProfileKind::LongContext);
            assert_eq!(long.capability.max_sequence_tokens, 8192);
            assert!(loaded
                .spec
                .promotion
                .required_profiles
                .contains(&"long-context-8k".to_string()));
        }
    }
}
