use super::TesseraSparseBuilder;
use crate::error::TesseraError;
use crate::runtime::ResourcePolicy;

#[test]
fn resource_policy_cannot_exceed_sparse_model_context() {
    let result = TesseraSparseBuilder::new()
        .model("splade-pp-en-v1")
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
fn catalog_only_sparse_model_is_rejected_before_loading() {
    let result = TesseraSparseBuilder::new().model("minicoil-v1").build();

    let Err(error) = result else {
        panic!("a catalog-only sparse model must be rejected");
    };
    assert!(matches!(
        error,
        TesseraError::ConfigError(message)
            if message.contains("minicoil-v1")
                && message.contains("catalog-only")
                && message.contains("ONNX")
    ));
}
