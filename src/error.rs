//! Error types for Tessera.
//!
//! This module provides structured error types for better error handling
//! and more informative error messages throughout the library.

use thiserror::Error;

/// Main error type for Tessera operations.
///
/// Provides structured errors for common failure modes with context-rich
/// error messages to aid debugging and programmatic error handling.
#[derive(Error, Debug)]
pub enum TesseraError {
    /// Model with the specified ID was not found in the registry.
    #[error("Model '{model_id}' not found in registry")]
    ModelNotFound {
        /// Model identifier that was not found
        model_id: String
    },

    /// Failed to load a model from disk or remote source.
    #[error("Failed to load model '{model_id}': {source}")]
    ModelLoadError {
        /// Model identifier that failed to load
        model_id: String,
        /// Underlying error cause
        #[source]
        source: anyhow::Error,
    },

    /// Encoding operation failed during inference.
    #[error("Encoding failed: {context}")]
    EncodingError {
        /// Context describing what encoding operation failed
        context: String,
        /// Underlying error cause
        #[source]
        source: anyhow::Error,
    },

    /// Requested embedding dimension is not supported by the model.
    #[error("Unsupported dimension {requested} for model '{model_id}'. Supported: {supported:?}")]
    UnsupportedDimension {
        /// Model identifier
        model_id: String,
        /// Requested dimension size
        requested: usize,
        /// List of supported dimension sizes
        supported: Vec<usize>,
    },

    /// Device-related error (GPU, Metal, CPU).
    #[error("Device error: {0}")]
    DeviceError(String),

    /// Quantization operation failed.
    #[error("Quantization error: {0}")]
    QuantizationError(String),

    /// Tokenization failed.
    #[error("Tokenization error: {0}")]
    TokenizationError(#[from] tokenizers::Error),

    /// Invalid configuration provided.
    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    /// Dimension mismatch between tensors or embeddings.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension size
        expected: usize,
        /// Actual dimension size received
        actual: usize
    },

    /// Matryoshka truncation error.
    #[error("Matryoshka truncation error: {0}")]
    MatryoshkaError(String),

    /// IO error when reading/writing files.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Candle tensor operation error.
    #[error("Tensor operation error: {0}")]
    TensorError(#[from] candle_core::Error),

    /// Catch-all for other errors.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type for Tessera operations.
///
/// Uses `TesseraError` as the error type for all public APIs.
pub type Result<T> = std::result::Result<T, TesseraError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TesseraError::ModelNotFound {
            model_id: "colbert-v2".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Model 'colbert-v2' not found in registry"
        );
    }

    #[test]
    fn test_unsupported_dimension() {
        let err = TesseraError::UnsupportedDimension {
            model_id: "jina-colbert-v2".to_string(),
            requested: 100,
            supported: vec![64, 96, 128],
        };
        assert!(err.to_string().contains("Unsupported dimension 100"));
        assert!(err.to_string().contains("jina-colbert-v2"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = TesseraError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        assert_eq!(
            err.to_string(),
            "Dimension mismatch: expected 128, got 64"
        );
    }
}
