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
    let spec_files = std::fs::read_dir(repository.join("certification/specs"))
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .unwrap()
                .path()
                .extension()
                .and_then(|value| value.to_str())
                == Some("json")
        })
        .count();
    assert_eq!(specs.len(), spec_files);
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
                "nomic-embed-v1.5" | "snowflake-arctic-l" | "gte-modernbert-base"
            )
        {
            let long = loaded.spec.profile("long-context-2k").unwrap();
            assert_eq!(long.kind, ProfileKind::LongContext);
            assert_eq!(long.capability.max_sequence_tokens, 2048);
            assert!(loaded
                .spec
                .promotion
                .required_profiles
                .contains(&"long-context-2k".to_string()));
        }
    }
}

#[test]
fn added_dense_specs_bind_their_registry_retrieval_metadata() {
    use tessera::model_registry::{get_model, Distance, ModelType};

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let expected = [
        (
            "multilingual-e5-base",
            "query: ",
            "passage: ",
            "scores = (embeddings[:2] @ embeddings[2:].T) * 100",
            "https://huggingface.co/intfloat/multilingual-e5-base/blob/d128750597153bb5987e10b1c3493a34e5a4502a/README.md#L6821",
        ),
        (
            "multilingual-e5-large",
            "query: ",
            "passage: ",
            "scores = (embeddings[:2] @ embeddings[2:].T) * 100",
            "https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/README.md#L5994",
        ),
        (
            "snowflake-arctic-l",
            "query: ",
            "",
            "# Compute cosine similarity scores",
            "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0/blob/ac6544c8a46e00af67e330e85a9028c66b8cfd9a/README.md#L9126",
        ),
        (
            "nomic-embed-v1.5",
            "search_query: ",
            "search_document: ",
            "embeddings = F.normalize(embeddings, p=2, dim=1)",
            "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/README.md#L2699",
        ),
        (
            "gte-modernbert-base",
            "",
            "",
            "scores = (embeddings[:1] @ embeddings[1:].T) * 100",
            "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/blob/e7f32e3c00f91d699e8c43b53106206bcc72bb22/README.md#L77",
        ),
    ];
    for (id, query, document, words, url) in expected {
        let loaded = super::load_model(repository, id).unwrap();
        let registered = get_model(&loaded.spec.model.id).unwrap();
        assert_eq!(registered.model_type, ModelType::Dense, "{id}");
        let prompts = registered.prompts.unwrap();
        assert_eq!((prompts.query, prompts.document), (query, document), "{id}");
        assert_eq!(registered.distance, Some(Distance::Cosine), "{id}");
        let comparison = registered.card_comparison.unwrap();
        assert_eq!((comparison.words, comparison.url), (words, url), "{id}");
    }
}
