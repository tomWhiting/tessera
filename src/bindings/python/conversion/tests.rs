use super::{exception_kind, exception_message, ExceptionKind};
use crate::{EmbeddingRefusal, TesseraError};

#[test]
fn overlength_refusal_is_a_value_error_with_complete_token_counts() {
    let error = TesseraError::EncodingError {
        context: "Failed to encode text".to_owned(),
        source: anyhow::Error::new(EmbeddingRefusal::TextLongerThanModel {
            tokens_total: 513,
            tokens_limit: 512,
        })
        .context("Tokenizing complete input"),
    };
    assert_eq!(exception_kind(&error), ExceptionKind::Value);
    assert_eq!(
        exception_message(&error),
        "Encoding failed: Failed to encode text: Tokenizing complete input: text_longer_than_model: tokens_total=513, tokens_limit=512"
    );
}

#[test]
fn no_content_refusal_is_a_value_error() {
    let error = TesseraError::Other(anyhow::Error::new(EmbeddingRefusal::NoContentTokens));
    assert_eq!(exception_kind(&error), ExceptionKind::Value);
    assert_eq!(exception_message(&error), "embed_input_no_content_tokens");
}
