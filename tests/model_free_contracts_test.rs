//! Network-free integration contracts for the public model-selection surface.
//!
//! Real model execution belongs to the local certification harness. These
//! tests stop at validation paths that must run before device selection,
//! artifact lookup, or model allocation.

use candle_core::Device;
use tessera::encoding::dense::CandleDenseEncoder;
use tessera::model_registry::{get_model, runnable_models, SupportTier, MODEL_REGISTRY};
use tessera::{
    ModelConfig, ResourcePolicy, TesseraDenseBuilder, TesseraMultiVectorBuilder,
    TesseraSparseBuilder, TesseraVisionBuilder,
};

fn error_message<T>(result: tessera::Result<T>) -> String {
    match result {
        Ok(_) => panic!("validation unexpectedly reached model construction"),
        Err(error) => error.to_string(),
    }
}

fn assert_contains_all(message: &str, expected: &[&str]) {
    for fragment in expected {
        assert!(
            message.contains(fragment),
            "expected {message:?} to contain {fragment:?}"
        );
    }
}

#[test]
fn registry_separates_runnable_models_from_catalog_only_metadata() {
    let runnable = runnable_models();
    assert!(!runnable.is_empty());
    assert!(runnable.iter().all(|model| model.is_runnable()));
    assert!(runnable
        .iter()
        .all(|model| model.support_tier != SupportTier::CatalogOnly));

    let catalog_only = get_model("jina-colbert-v2").expect("catalog entry");
    assert_eq!(catalog_only.support_tier, SupportTier::CatalogOnly);
    assert!(!catalog_only.is_runnable());
    assert!(!runnable.iter().any(|model| model.id == catalog_only.id));

    let unique_ids = MODEL_REGISTRY
        .iter()
        .map(|model| model.id)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique_ids.len(), MODEL_REGISTRY.len());
}

#[test]
fn model_config_rejects_catalog_only_entries_without_artifact_lookup() {
    let error = ModelConfig::from_registry("jina-colbert-v2")
        .expect_err("catalog-only metadata must not produce a load configuration");

    assert_contains_all(&error.to_string(), &["jina-colbert-v2", "catalog-only"]);
}

#[test]
fn every_builder_requires_a_model_id() {
    let messages = [
        error_message(TesseraDenseBuilder::new().build()),
        error_message(TesseraMultiVectorBuilder::new().build()),
        error_message(TesseraSparseBuilder::new().build()),
        error_message(TesseraVisionBuilder::new().build()),
    ];

    assert!(messages.iter().all(|message| message.contains("Model ID")));
}

#[test]
fn every_builder_rejects_unknown_models_before_loading() {
    let messages = [
        error_message(
            TesseraDenseBuilder::new()
                .model("not-a-real-tessera-model")
                .build(),
        ),
        error_message(
            TesseraMultiVectorBuilder::new()
                .model("not-a-real-tessera-model")
                .build(),
        ),
        error_message(
            TesseraSparseBuilder::new()
                .model("not-a-real-tessera-model")
                .build(),
        ),
        error_message(
            TesseraVisionBuilder::new()
                .model("not-a-real-tessera-model")
                .build(),
        ),
    ];

    assert!(messages.iter().all(|message| {
        message.contains("not-a-real-tessera-model") && message.contains("not found")
    }));
}

#[test]
fn typed_builders_reject_the_wrong_runnable_paradigm() {
    let cases = [
        (
            error_message(TesseraDenseBuilder::new().model("colbert-v2").build()),
            "colbert-v2",
            "not a dense model",
        ),
        (
            error_message(
                TesseraMultiVectorBuilder::new()
                    .model("bge-base-en-v1.5")
                    .build(),
            ),
            "bge-base-en-v1.5",
            "not Colbert",
        ),
        (
            error_message(
                TesseraSparseBuilder::new()
                    .model("bge-base-en-v1.5")
                    .build(),
            ),
            "bge-base-en-v1.5",
            "not Sparse",
        ),
        (
            error_message(
                TesseraVisionBuilder::new()
                    .model("bge-base-en-v1.5")
                    .build(),
            ),
            "bge-base-en-v1.5",
            "not VisionLanguage",
        ),
    ];

    for (message, model_id, expected) in cases {
        assert_contains_all(&message, &[model_id, expected]);
    }
}

#[test]
fn typed_builders_reject_catalog_only_entries() {
    let cases = [
        (
            error_message(
                TesseraDenseBuilder::new()
                    .model("jina-embeddings-v3")
                    .build(),
            ),
            "jina-embeddings-v3",
        ),
        (
            error_message(
                TesseraMultiVectorBuilder::new()
                    .model("gte-modern-colbert")
                    .build(),
            ),
            "gte-modern-colbert",
        ),
        (
            error_message(TesseraSparseBuilder::new().model("minicoil-v1").build()),
            "minicoil-v1",
        ),
        (
            error_message(TesseraVisionBuilder::new().model("colpali-v1.3-hf").build()),
            "colpali-v1.3-hf",
        ),
    ];

    for (message, model_id) in cases {
        assert_contains_all(&message, &[model_id, "catalog-only"]);
    }
}

#[test]
fn typed_builders_reject_over_context_policies() {
    let over_context = ResourcePolicy::default().with_max_sequence_tokens(513);
    let messages = [
        error_message(
            TesseraDenseBuilder::new()
                .model("bge-base-en-v1.5")
                .resource_policy(over_context)
                .build(),
        ),
        error_message(
            TesseraMultiVectorBuilder::new()
                .model("colbert-v2")
                .resource_policy(over_context)
                .build(),
        ),
        error_message(
            TesseraSparseBuilder::new()
                .model("splade-pp-en-v1")
                .resource_policy(over_context)
                .build(),
        ),
        error_message(
            TesseraVisionBuilder::new()
                .model("colpali-v1.2")
                .resource_policy(over_context)
                .build(),
        ),
    ];

    for message in messages {
        assert_contains_all(&message, &["sequence token limit 513", "context limit 512"]);
    }
}

#[test]
fn resource_and_builder_options_fail_before_model_loading() {
    let model_budget_error =
        error_message(TesseraVisionBuilder::new().model("colpali-v1.2").build());
    assert_contains_all(
        &model_budget_error,
        &["model parameter bytes", "resource policy limit"],
    );

    let dimension_error = error_message(
        TesseraDenseBuilder::new()
            .model("nomic-embed-v1.5")
            .dimension(999)
            .build(),
    );
    assert_contains_all(
        &dimension_error,
        &["Unsupported dimension 999", "Supported:"],
    );

    let batch_error = error_message(
        TesseraDenseBuilder::new()
            .model("bge-base-en-v1.5")
            .batch_size(0)
            .build(),
    );
    assert_contains_all(&batch_error, &["Batch size", "greater than zero"]);
}

#[test]
fn dense_encoder_requires_pooling_before_artifact_lookup() {
    let mut config = ModelConfig::from_registry("bge-base-en-v1.5").expect("runnable config");
    config.pooling_strategy = None;

    let error = match CandleDenseEncoder::new(config, Device::Cpu) {
        Ok(_) => panic!("validation unexpectedly reached model construction"),
        Err(error) => error.to_string(),
    };
    assert_contains_all(&error, &["pooling_strategy", "bge-base-en-v1.5"]);
}
