use candle_core::Device;

use super::preflight_registered_model;
use crate::models::registry::ModelType;
use crate::runtime::ResourcePolicy;

#[test]
fn unknown_models_are_rejected_before_runtime_initialization() {
    let error = preflight_registered_model(
        "example/unknown-model",
        512,
        ModelType::Dense,
        &Device::Cpu,
        &ResourcePolicy::default(),
    )
    .expect_err("unknown model metadata cannot support a safe preflight");

    assert!(error.to_string().contains("not registered"));
}

#[test]
fn catalog_only_models_are_rejected_before_runtime_initialization() {
    let error = preflight_registered_model(
        "jinaai/jina-colbert-v2",
        8192,
        ModelType::Colbert,
        &Device::Cpu,
        &ResourcePolicy::for_model_context(8192).with_max_model_bytes(4 * 1024 * 1024 * 1024),
    )
    .expect_err("catalog-only models do not have a runnable adapter");

    assert!(error.to_string().contains("catalog-only"));
}

#[test]
fn model_type_mismatches_are_rejected_before_runtime_initialization() {
    let error = preflight_registered_model(
        "BAAI/bge-base-en-v1.5",
        512,
        ModelType::Sparse,
        &Device::Cpu,
        &ResourcePolicy::default(),
    )
    .expect_err("a dense checkpoint cannot enter a sparse encoder");

    assert!(error.to_string().contains("requires 'Sparse'"));
}

#[test]
fn registered_context_cannot_be_bypassed_by_a_custom_config() {
    let error = preflight_registered_model(
        "BAAI/bge-base-en-v1.5",
        1024,
        ModelType::Dense,
        &Device::Cpu,
        &ResourcePolicy::default().with_max_sequence_tokens(513),
    )
    .expect_err("the registry context remains the hard model capability");

    assert!(error.to_string().contains("model context limit 512"));
}

#[test]
fn successful_preflight_returns_the_immutable_registry_entry() {
    let model = preflight_registered_model(
        "BAAI/bge-base-en-v1.5",
        512,
        ModelType::Dense,
        &Device::Cpu,
        &ResourcePolicy::default(),
    )
    .expect("BGE should pass preflight");

    assert_eq!(model.id, "bge-base-en-v1.5");
    assert_eq!(
        model.revision,
        Some("a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
    );
}
