use super::TesseraVisionBuilder;
use crate::error::TesseraError;
use crate::runtime::ResourcePolicy;

#[test]
fn resource_policy_cannot_exceed_vision_model_context() {
    let result = TesseraVisionBuilder::new()
        .model("colpali-v1.2")
        .resource_policy(ResourcePolicy::default().with_max_sequence_tokens(8193))
        .build();

    let Err(error) = result else {
        panic!("an over-context resource policy must be rejected");
    };
    assert!(matches!(
        error,
        TesseraError::ConfigError(message)
            if message.contains("Configured sequence token limit 8193")
                && message.contains("model context limit 8192")
    ));
}

#[test]
fn default_policy_rejects_f32_colpali_before_model_loading() {
    let result = TesseraVisionBuilder::new().model("colpali-v1.2").build();

    let Err(error) = result else {
        panic!("the default model byte budget must reject F32 ColPali");
    };
    assert!(matches!(
        error,
        TesseraError::ConfigError(message)
            if message.contains("Estimated model parameter bytes 12000000000")
                && message.contains("resource policy limit 2147483648")
    ));
}

#[test]
fn catalog_only_vision_model_is_rejected_before_loading() {
    let result = TesseraVisionBuilder::new().model("colpali-v1.3-hf").build();

    let Err(error) = result else {
        panic!("a catalog-only vision model must be rejected");
    };
    assert!(matches!(
        error,
        TesseraError::ConfigError(message)
            if message.contains("colpali-v1.3-hf")
                && message.contains("catalog-only")
                && message.contains("namespaces vlm.*")
    ));
}
