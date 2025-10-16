//! Model configuration for BERT-based encoders.
//!
//! Provides pre-configured settings for popular models including:
//! - Standard BERT models (bert-base-uncased, distilbert-base-uncased)
//! - ColBERT models optimized for late-interaction retrieval
//!
//! Models can be loaded from the registry using `from_registry()`:
//!
//! ```no_run
//! use tessera::models::ModelConfig;
//!
//! // Load from registry by ID
//! let config = ModelConfig::from_registry("colbert-v2").unwrap();
//!
//! // Or use convenience methods
//! let config = ModelConfig::colbert_v2();
//! ```

use anyhow::{anyhow, Result};

/// Configuration for a BERT-based model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Name of the model on HuggingFace Hub
    pub model_name: String,
    /// Dimensionality of the embedding vectors
    pub embedding_dim: usize,
    /// Maximum sequence length the model can handle
    pub max_seq_length: usize,
}

// Model name constants

// ColBERT models (recommended)
pub const COLBERT_V2: &str = "colbert-ir/colbertv2.0";
pub const JINA_COLBERT_V2: &str = "jinaai/jina-colbert-v2";
pub const COLBERT_SMALL: &str = "answerdotai/answerai-colbert-small-v1";

// Standard BERT models for comparison
pub const DISTILBERT_BASE_UNCASED: &str = "distilbert-base-uncased";

impl ModelConfig {
    /// Creates a new model configuration.
    pub fn new(model_name: String, embedding_dim: usize, max_seq_length: usize) -> Self {
        Self {
            model_name,
            embedding_dim,
            max_seq_length,
        }
    }

    /// Creates a configuration from the model registry by ID.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tessera::models::ModelConfig;
    ///
    /// let config = ModelConfig::from_registry("colbert-v2")
    ///     .expect("Model not found in registry");
    /// ```
    pub fn from_registry(id: &str) -> Result<Self> {
        let model = super::registry::get_model(id)
            .ok_or_else(|| anyhow!("Model '{}' not found in registry", id))?;

        Ok(Self {
            model_name: model.huggingface_id.to_string(),
            embedding_dim: model.embedding_dim.default_dim(),
            max_seq_length: model.context_length,
        })
    }

    /// Creates a configuration for distilbert-base-uncased.
    ///
    /// This is recommended for prototyping with standard BERT models as it's faster than full BERT.
    pub fn distilbert_base_uncased() -> Self {
        Self::new(DISTILBERT_BASE_UNCASED.to_string(), 768, 512)
    }

    /// Creates a configuration for ColBERT v2.
    ///
    /// ColBERT v2 is a BERT-based model specifically trained for late-interaction retrieval.
    /// This is the original ColBERT implementation from Stanford.
    ///
    /// Model: colbert-ir/colbertv2.0
    /// Size: ~440MB (110M parameters)
    /// Embedding dim: 128 (after projection from 768-dim BERT)
    /// Max sequence length: 512 tokens
    pub fn colbert_v2() -> Self {
        Self::new(COLBERT_V2.to_string(), 128, 512)
    }

    /// Creates a configuration for Jina ColBERT v2.
    ///
    /// Jina's multilingual ColBERT model supporting 89 languages.
    /// Larger than standard ColBERT but provides excellent multilingual support.
    ///
    /// Model: jinaai/jina-colbert-v2
    /// Size: ~2.1GB (560M parameters)
    /// Languages: 89 languages
    /// Embedding dim: 768
    /// Max sequence length: 8192 tokens
    pub fn jina_colbert_v2() -> Self {
        Self::new(JINA_COLBERT_V2.to_string(), 768, 8192)
    }

    /// Creates a configuration for ColBERT Small.
    ///
    /// A smaller, faster ColBERT variant.
    /// Recommended for development and testing due to faster download and inference.
    ///
    /// Model: answerdotai/answerai-colbert-small-v1
    /// Size: ~130MB (33M parameters)
    /// Embedding dim: 96 (after projection from 384-dim DistilBERT)
    /// Max sequence length: 512 tokens
    pub fn colbert_small() -> Self {
        Self::new(COLBERT_SMALL.to_string(), 96, 512)
    }

    /// Creates a configuration for a custom model.
    pub fn custom(model_name: impl Into<String>, embedding_dim: usize, max_seq_length: usize) -> Self {
        Self::new(model_name.into(), embedding_dim, max_seq_length)
    }
}

impl Default for ModelConfig {
    /// Returns the default configuration (ColBERT Small).
    ///
    /// ColBERT Small is recommended as the default because:
    /// - It's a real ColBERT model optimized for retrieval
    /// - Small size (~130MB) means faster downloads
    /// - Based on DistilBERT for good performance
    fn default() -> Self {
        Self::colbert_small()
    }
}
