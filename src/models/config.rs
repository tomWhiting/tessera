//! Model configuration for BERT-based encoders.
//!
//! Provides registry-backed settings for audited model paths, including
//! `ColBERT` models optimized for late-interaction retrieval.
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

use super::registry::{ModelInfo, PoolingStrategy};

/// Configuration for a BERT-based model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Name of the model on `HuggingFace` Hub
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
/// `ColBERT` v2 model identifier
pub const COLBERT_V2: &str = "colbert-ir/colbertv2.0";
/// Jina `ColBERT` v2 model identifier
pub const JINA_COLBERT_V2: &str = "jinaai/jina-colbert-v2";
/// `AnswerAI` `ColBERT` Small model identifier
pub const COLBERT_SMALL: &str = "answerdotai/answerai-colbert-small-v1";

impl ModelConfig {
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
    /// let config = ModelConfig::from_registry("colbert-v2")
    ///     .unwrap()
    ///     .with_target_dimension(128);
    /// ```
    #[must_use]
    pub const fn with_target_dimension(mut self, dim: usize) -> Self {
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
    #[must_use]
    pub const fn with_pooling(mut self, strategy: PoolingStrategy, normalize: bool) -> Self {
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
        let model = runnable_registry_model(id)?;

        let (pooling_strategy, normalize_embeddings) = model
            .pooling
            .map_or((None, false), |p| (Some(p.strategy), p.normalize));

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
    /// let config = ModelConfig::from_registry_with_dimension("colbert-v2", 128)
    ///     .expect("Invalid dimension");
    /// ```
    pub fn from_registry_with_dimension(id: &str, target_dim: usize) -> Result<Self> {
        let model = runnable_registry_model(id)?;

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
            .map_or((None, false), |p| (Some(p.strategy), p.normalize));

        Ok(Self {
            model_name: model.huggingface_id.to_string(),
            embedding_dim: target_dim,
            max_seq_length: model.context_length,
            target_dimension: Some(target_dim),
            pooling_strategy,
            normalize_embeddings,
        })
    }

    /// Creates a configuration for `ColBERT` v2.
    ///
    /// `ColBERT` v2 is a BERT-based model specifically trained for late-interaction retrieval.
    /// This is the original `ColBERT` implementation from Stanford.
    ///
    /// Model: colbert-ir/colbertv2.0
    /// Size: ~440MB (110M parameters)
    /// Embedding dim: 128 (after projection from 768-dim BERT)
    /// Max sequence length: 512 tokens
    #[must_use]
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

    /// Attempts to create a configuration for Jina `ColBERT` v2.
    ///
    /// This entry is currently catalog-only because its rotary XLM-R checkpoint
    /// is incompatible with Tessera's `ColBERT` loader. The method returns an
    /// actionable error instead of constructing a configuration that will fail
    /// after downloading model artifacts.
    pub fn jina_colbert_v2() -> Result<Self> {
        Self::from_registry("jina-colbert-v2")
    }

    /// Creates a configuration for `ColBERT` Small.
    ///
    /// A smaller, faster `ColBERT` variant.
    /// Recommended for development and testing due to faster download and inference.
    ///
    /// Model: answerdotai/answerai-colbert-small-v1
    /// Size: ~130MB (33M parameters)
    /// Embedding dim: 96 (after projection from 384-dim BERT)
    /// Max sequence length: 512 tokens
    #[must_use]
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
}

impl Default for ModelConfig {
    /// Returns the default configuration (`ColBERT` Small).
    ///
    /// `ColBERT` Small is recommended as the default because:
    /// - It's a real `ColBERT` model optimized for retrieval
    /// - Small size (~130MB) means faster downloads
    /// - Based on a compact BERT encoder for good performance
    fn default() -> Self {
        Self::colbert_small()
    }
}

fn runnable_registry_model(id: &str) -> Result<&'static ModelInfo> {
    let model = super::registry::get_model(id)
        .ok_or_else(|| anyhow!("Model '{id}' not found in registry"))?;
    if !model.is_runnable() {
        return Err(anyhow!(
            "Model '{}' is catalog-only and cannot be configured for loading: {}",
            model.id,
            model.support_note
        ));
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    #[test]
    fn registry_config_rejects_catalog_only_models() {
        let error = ModelConfig::from_registry("jina-colbert-v2").unwrap_err();

        assert!(error.to_string().contains("catalog-only"));
        assert!(error.to_string().contains("Rotary XLM-R"));
    }

    #[test]
    fn registry_config_keeps_runnable_long_context_metadata() {
        let config = ModelConfig::from_registry("nomic-embed-v1.5").unwrap();

        assert_eq!(config.max_seq_length, 8192);
        assert_eq!(config.embedding_dim, 768);
    }
}
