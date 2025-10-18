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

use super::registry::PoolingStrategy;

/// Configuration for a BERT-based model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Name of the model on HuggingFace Hub
    pub model_name: String,
    /// Dimensionality of the embedding vectors
    pub embedding_dim: usize,
    /// Maximum sequence length the model can handle
    pub max_seq_length: usize,
    /// Optional target dimension for Matryoshka truncation
    pub target_dimension: Option<usize>,
    /// Pooling strategy for dense models (None for multi-vector models)
    pub pooling_strategy: Option<PoolingStrategy>,
    /// Whether to normalize embeddings after pooling
    pub normalize_embeddings: bool,
}

// Model name constants

// ColBERT models (recommended)
/// ColBERT v2 model identifier
pub const COLBERT_V2: &str = "colbert-ir/colbertv2.0";
/// Jina ColBERT v2 model identifier
pub const JINA_COLBERT_V2: &str = "jinaai/jina-colbert-v2";
/// AnswerAI ColBERT Small model identifier
pub const COLBERT_SMALL: &str = "answerdotai/answerai-colbert-small-v1";

// Standard BERT models for comparison
/// DistilBERT Base Uncased model identifier
pub const DISTILBERT_BASE_UNCASED: &str = "distilbert-base-uncased";

impl ModelConfig {
    /// Creates a new model configuration.
    pub fn new(model_name: String, embedding_dim: usize, max_seq_length: usize) -> Self {
        Self {
            model_name,
            embedding_dim,
            max_seq_length,
            target_dimension: None,
            pooling_strategy: None,
            normalize_embeddings: false,
        }
    }

    /// Sets the target dimension for Matryoshka truncation.
    ///
    /// If set, the encoder will truncate embeddings to this dimension.
    /// The dimension must be supported by the model's Matryoshka configuration.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tessera::models::ModelConfig;
    ///
    /// let config = ModelConfig::from_registry("jina-colbert-v2")
    ///     .unwrap()
    ///     .with_target_dimension(128);
    /// ```
    pub fn with_target_dimension(mut self, dim: usize) -> Self {
        self.target_dimension = Some(dim);
        self
    }

    /// Sets the pooling configuration for dense models.
    ///
    /// This method configures how token-level embeddings should be pooled
    /// into a single vector and whether the result should be normalized.
    ///
    /// # Arguments
    /// * `strategy` - The pooling strategy (Cls, Mean, or Max)
    /// * `normalize` - Whether to L2-normalize the pooled embedding
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tessera::models::{ModelConfig, registry::PoolingStrategy};
    ///
    /// let config = ModelConfig::from_registry("bge-base-en-v1.5")
    ///     .unwrap()
    ///     .with_pooling(PoolingStrategy::Mean, true);
    /// ```
    pub fn with_pooling(mut self, strategy: PoolingStrategy, normalize: bool) -> Self {
        self.pooling_strategy = Some(strategy);
        self.normalize_embeddings = normalize;
        self
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

        let (pooling_strategy, normalize_embeddings) = model
            .pooling
            .map(|p| (Some(p.strategy), p.normalize))
            .unwrap_or((None, false));

        Ok(Self {
            model_name: model.huggingface_id.to_string(),
            embedding_dim: model.embedding_dim.default_dim(),
            max_seq_length: model.context_length,
            target_dimension: None,
            pooling_strategy,
            normalize_embeddings,
        })
    }

    /// Creates a configuration from the model registry with a specific dimension.
    ///
    /// For models with Matryoshka support, this sets the target dimension.
    /// The dimension must be supported by the model.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tessera::models::ModelConfig;
    ///
    /// let config = ModelConfig::from_registry_with_dimension("jina-colbert-v2", 128)
    ///     .expect("Invalid dimension");
    /// ```
    pub fn from_registry_with_dimension(id: &str, target_dim: usize) -> Result<Self> {
        let model = super::registry::get_model(id)
            .ok_or_else(|| anyhow!("Model '{}' not found in registry", id))?;

        // Validate dimension is supported
        if !model.embedding_dim.supports_dimension(target_dim) {
            return Err(anyhow!(
                "Dimension {} not supported by model '{}'. Supported: {:?}",
                target_dim,
                id,
                model.embedding_dim.supported_dimensions()
            ));
        }

        let (pooling_strategy, normalize_embeddings) = model
            .pooling
            .map(|p| (Some(p.strategy), p.normalize))
            .unwrap_or((None, false));

        Ok(Self {
            model_name: model.huggingface_id.to_string(),
            embedding_dim: target_dim,
            max_seq_length: model.context_length,
            target_dimension: Some(target_dim),
            pooling_strategy,
            normalize_embeddings,
        })
    }

    /// Creates a configuration for distilbert-base-uncased.
    ///
    /// This is recommended for prototyping with standard BERT models as it's faster than full BERT.
    ///
    /// Note: This is a multi-vector model without pooling. For dense embeddings with pooling,
    /// use models from the registry like "bge-base-en-v1.5" or "nomic-embed-v1.5".
    pub fn distilbert_base_uncased() -> Self {
        Self {
            model_name: DISTILBERT_BASE_UNCASED.to_string(),
            embedding_dim: 768,
            max_seq_length: 512,
            target_dimension: None,
            pooling_strategy: None,
            normalize_embeddings: false,
        }
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
        Self {
            model_name: COLBERT_V2.to_string(),
            embedding_dim: 128,
            max_seq_length: 512,
            target_dimension: None,
            pooling_strategy: None,
            normalize_embeddings: false,
        }
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
        Self {
            model_name: JINA_COLBERT_V2.to_string(),
            embedding_dim: 768,
            max_seq_length: 8192,
            target_dimension: None,
            pooling_strategy: None,
            normalize_embeddings: false,
        }
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
        Self {
            model_name: COLBERT_SMALL.to_string(),
            embedding_dim: 96,
            max_seq_length: 512,
            target_dimension: None,
            pooling_strategy: None,
            normalize_embeddings: false,
        }
    }

    /// Creates a configuration for a custom model.
    ///
    /// The custom configuration defaults to no pooling (multi-vector mode).
    /// Use `with_pooling()` to configure dense embeddings if needed.
    pub fn custom(
        model_name: impl Into<String>,
        embedding_dim: usize,
        max_seq_length: usize,
    ) -> Self {
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
