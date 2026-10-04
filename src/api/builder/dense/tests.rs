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

    assert!(matches!(error, TesseraError::Other(_)));
    assert_eq!(
        error.to_string(),
        "Batch size must be greater than zero. Use .batch_size(1) or larger"
    );
    assert_eq!(error.embed_failure().unwrap().code(), "embed_limits");
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
    assert!(matches!(error, TesseraError::Other(_)));
    let message = error.to_string();
    assert!(message.starts_with("Invalid resource policy for model 'bge-base-en-v1.5': "));
    assert!(message.contains("Configured sequence token limit 513"));
    assert!(message.contains("model context limit 512"));
    assert!(matches!(
        error.embed_failure(),
        Some(crate::EmbedFailure::Limits {
            limit: "max_sequence_tokens",
            ..
        })
    ));
}

#[test]
fn batch_size_over_the_policy_is_a_limits_failure() {
    let result = TesseraDenseBuilder::new()
        .model("bge-base-en-v1.5")
        .resource_policy(ResourcePolicy::default().with_max_batch_items(2))
        .batch_size(3)
        .build();

    let Err(error) = result else {
        panic!("a batch size over the policy must be rejected");
    };
    assert!(error
        .to_string()
        .starts_with("Invalid dense batch size for model 'bge-base-en-v1.5': "));
    assert!(matches!(
        error.embed_failure(),
        Some(crate::EmbedFailure::Limits {
            limit: "batch_items",
            ..
        })
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
