use std::ffi::OsStr;

use hf_hub::Cache;

use super::{
    download_model_file, parse_offline_flag, validated_repo, ArtifactSource, ModelFileResolver,
};
use crate::models::registry::get_model_by_hf_id;

const BGE_ID: &str = "BAAI/bge-base-en-v1.5";
const BGE_REVISION: &str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a";

#[test]
fn repository_descriptor_uses_the_exact_registry_pin() {
    let model = get_model_by_hf_id(BGE_ID).unwrap();
    let repo = validated_repo(model).unwrap();

    assert_eq!(model.id, "bge-base-en-v1.5");
    assert_eq!(repo.url(), BGE_ID);
    assert_eq!(repo.revision(), BGE_REVISION);
}

#[test]
fn unavailable_catalog_revision_fails_before_io() {
    let error = download_model_file("jinaai/jina-colbert-v2-96", "config.json").unwrap_err();

    assert!(error
        .to_string()
        .contains("has no pinned HuggingFace revision"));
}

#[test]
fn offline_cache_miss_is_explicit_and_network_free() {
    let model = get_model_by_hf_id(BGE_ID).unwrap();
    let repo = validated_repo(model).unwrap();
    let cache_root =
        std::env::temp_dir().join(format!("tessera-missing-hf-cache-{}", std::process::id()));
    let resolver = ModelFileResolver {
        model,
        source: ArtifactSource::Offline(Cache::new(cache_root).repo(repo)),
    };

    let error = resolver.get("config.json").unwrap_err();
    assert!(error.to_string().contains("TESSERA_OFFLINE=1"));
    assert!(error.to_string().contains(BGE_REVISION));
}

#[test]
fn offline_flag_accepts_only_the_documented_value() {
    assert!(!parse_offline_flag(None).unwrap());
    assert!(parse_offline_flag(Some(OsStr::new("1"))).unwrap());
    assert!(parse_offline_flag(Some(OsStr::new("true"))).is_err());
    assert!(parse_offline_flag(Some(OsStr::new("0"))).is_err());
}
