use super::TesseraDenseBuilder;
use crate::error::TesseraError;
use crate::runtime::ResourcePolicy;

#[test]
fn zero_batch_size_is_rejected_before_model_loading() {
    let result = TesseraDenseBuilder::new()
        .model("bge-base-en-v1.5")
        .batch_size(0)
        .build();

    let Err(error) = result else {
        panic!("a zero batch size must be rejected");
    };

    assert!(matches!(
        error,
        TesseraError::ConfigError(message)
            if message == "Batch size must be greater than zero. Use .batch_size(1) or larger"
    ));
}

#[test]
fn resource_policy_cannot_exceed_dense_model_context() {
    let result = TesseraDenseBuilder::new()
        .model("bge-base-en-v1.5")
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
fn catalog_only_dense_model_is_rejected_before_loading() {
    let result = TesseraDenseBuilder::new()
        .model("jina-embeddings-v3")
        .build();

    let Err(error) = result else {
        panic!("a catalog-only dense model must be rejected");
    };
    assert!(matches!(
        error,
        TesseraError::ConfigError(message)
            if message.contains("jina-embeddings-v3")
                && message.contains("catalog-only")
                && message.contains("LoRA")
    ));
}
