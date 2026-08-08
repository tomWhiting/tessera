use super::TesseraMultiVectorBuilder;
use crate::error::TesseraError;
use crate::runtime::ResourcePolicy;

#[test]
fn resource_policy_cannot_exceed_multi_vector_model_context() {
    let result = TesseraMultiVectorBuilder::new()
        .model("colbert-v2")
        .resource_policy(ResourcePolicy::default().with_max_sequence_tokens(513))
        .build();

    let Err(error) = result else {
        panic!("an over-context resource policy must be rejected");
    };
    assert!(matches!(
        error,
        TesseraError::ConfigError(message)
            if message.contains("Configured sequence token limit 513")
                && message.contains("model context limit 512")
    ));
}

#[test]
fn catalog_only_multi_vector_model_is_rejected_before_loading() {
    let result = TesseraMultiVectorBuilder::new()
        .model("gte-modern-colbert")
        .build();

    let Err(error) = result else {
        panic!("a catalog-only multi-vector model must be rejected");
    };
    assert!(matches!(
        error,
        TesseraError::ConfigError(message)
            if message.contains("gte-modern-colbert")
                && message.contains("catalog-only")
                && message.contains("ModernBERT")
    ));
}

#[test]
fn multi_vector_builder_rejects_a_runnable_dense_model() {
    let result = TesseraMultiVectorBuilder::new()
        .model("bge-base-en-v1.5")
        .build();

    let Err(error) = result else {
        panic!("a dense model must not be routed through the multi-vector loader");
    };
    assert!(matches!(
        error,
        TesseraError::ConfigError(message)
            if message.contains("bge-base-en-v1.5")
                && message.contains("not Colbert")
    ));
}
