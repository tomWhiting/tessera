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

#[test]
fn wrapped_overlength_message_keeps_both_counts() {
    let error = TesseraError::EncodingError {
        context: "encode".to_owned(),
        source: anyhow::Error::new(EmbeddingRefusal::TextLongerThanModel {
            tokens_total: 513,
            tokens_limit: 512,
        })
        .context("tokenize"),
    };
    assert_eq!(
        exception_message(&error),
        "Encoding failed: encode: tokenize: text_longer_than_model: tokens_total=513, tokens_limit=512"
    );
}

#[test]
fn every_error_variant_keeps_its_message_and_exception_kind() {
    let errors = [
        (
            TesseraError::UnsupportedWeightsFormat {
                model_id: "model".to_owned(),
                format: "onnx",
            },
            ExceptionKind::Value,
        ),
        (
            TesseraError::FetchingNotBuiltIn {
                model_id: "model".to_owned(),
            },
            ExceptionKind::Runtime,
        ),
        (
            TesseraError::ModelNotFound {
                model_id: "model".to_owned(),
            },
            ExceptionKind::Runtime,
        ),
        (
            TesseraError::ModelLoadError {
                model_id: "model".to_owned(),
                source: anyhow::anyhow!("load failed"),
            },
            ExceptionKind::Runtime,
        ),
        (
            TesseraError::EncodingError {
                context: "encode".to_owned(),
                source: anyhow::anyhow!("inference failed"),
            },
            ExceptionKind::Runtime,
        ),
        (
            TesseraError::UnsupportedDimension {
                model_id: "model".to_owned(),
                requested: 8,
                supported: vec![4],
            },
            ExceptionKind::Value,
        ),
        (
            TesseraError::DeviceError("device".to_owned()),
            ExceptionKind::Runtime,
        ),
        (
            TesseraError::QuantizationError("quantization".to_owned()),
            ExceptionKind::Value,
        ),
        (
            TesseraError::TokenizationError(std::io::Error::other("tokenizer").into()),
            ExceptionKind::Runtime,
        ),
        (
            TesseraError::ConfigError("configuration".to_owned()),
            ExceptionKind::Value,
        ),
        (
            TesseraError::DimensionMismatch {
                expected: 4,
                actual: 8,
            },
            ExceptionKind::Value,
        ),
        (
            TesseraError::MatryoshkaError("dimensions".to_owned()),
            ExceptionKind::Value,
        ),
        (
            TesseraError::IoError(std::io::Error::other("read")),
            ExceptionKind::Io,
        ),
        (
            TesseraError::TensorError(candle_core::Error::Msg("tensor".to_owned())),
            ExceptionKind::Runtime,
        ),
        (
            TesseraError::Other(anyhow::anyhow!("other")),
            ExceptionKind::Runtime,
        ),
    ];
    assert_eq!(errors.len(), 15);
    for (error, expected) in errors {
        assert_eq!(exception_kind(&error), expected, "{error}");
        assert_eq!(exception_message(&error), error.to_string());
    }
}

#[test]
fn every_input_refusal_is_a_value_error_through_error_context() {
    for refusal in [
        EmbeddingRefusal::NoContentTokens,
        EmbeddingRefusal::Empty,
        EmbeddingRefusal::TooLarge {
            input_bytes: 4,
            limit: 3,
        },
        EmbeddingRefusal::TextLongerThanModel {
            tokens_total: 513,
            tokens_limit: 512,
        },
    ] {
        let error = TesseraError::Other(anyhow::Error::new(refusal).context("input"));
        assert_eq!(exception_kind(&error), ExceptionKind::Value);
        assert_eq!(exception_message(&error), format!("input: {refusal}"));
    }
}
