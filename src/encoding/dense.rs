//! Dense single-vector encoder using Candle backend.
//!
//! Implements traditional sentence embedding approaches that pool
//! token-level BERT representations into a single fixed-size vector:
//!
//! - **CLS pooling**: Use the `[CLS]` token representation
//! - **Mean pooling**: Average all token embeddings (attention-weighted)
//! - **Max pooling**: Take element-wise maximum across tokens
//!
//! Dense encodings store one vector per text rather than one vector per token.
//! This reduces retained vector storage relative to multi-vector encodings but
//! does not preserve the same token-level scoring information as `ColBERT`.
//!
//! # Use Cases
//!
//! - Semantic search with large document collections
//! - Clustering and classification
//! - Workloads that have validated single-vector retrieval for their corpus
//!
//! # Example
//!
//! ```no_run
//! use tessera::TesseraDense;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Load a dense model from registry (e.g., BGE, Nomic)
//! let encoder = TesseraDense::new("bge-base-en-v1.5")?;
//!
//! // Encode text to single vector
//! let embedding = encoder.encode("Machine learning is a subset of AI")?;
//! assert_eq!(embedding.dim(), 768);
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use candle_core::{Device, Module, Tensor};
use serde::Deserialize;

use crate::core::{DenseEmbedding, DenseEncoder, Encoder, PoolingStrategy, Tokenizer};
use crate::models::ModelConfig;
use crate::runtime::{ModelDType, ModelResidencyPermit, ResourcePolicy, TransformerProfile};

mod inference;
mod loading;

/// Enum to hold different BERT model variants
enum BertVariant {
    Bert(candle_transformers::models::bert::BertModel),
    DistilBert(candle_transformers::models::distilbert::DistilBertModel),
    JinaBert(candle_transformers::models::jina_bert::BertModel),
    /// JinaBERT Code model variant (uses same architecture as standard JinaBERT)
    JinaBertCode(candle_transformers::models::jina_bert::BertModel),
    XlmRoberta(candle_transformers::models::xlm_roberta::XLMRobertaModel),
    ModernBert(candle_transformers::models::modernbert::ModernBert),
    NomicBert(candle_transformers::models::nomic_bert::NomicBertModel),
}

impl BertVariant {
    fn forward(&self, token_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        match self {
            Self::Bert(model) => {
                let token_type_ids = token_ids
                    .zeros_like()
                    .context("Creating BERT token type IDs")?;
                model
                    .forward(token_ids, &token_type_ids, Some(attention_mask))
                    .context("BERT forward pass")
            }
            Self::DistilBert(model) => model
                .forward(token_ids, attention_mask)
                .context("DistilBERT forward pass"),
            Self::JinaBert(model) => model.forward(token_ids).context("JinaBERT forward pass"),
            Self::JinaBertCode(model) => model
                .forward(token_ids)
                .context("JinaBERT Code forward pass"),
            Self::XlmRoberta(model) => {
                // XLM-RoBERTa uses token_type_ids (all zeros for single sequence)
                let token_type_ids = token_ids.zeros_like().context("Creating token type IDs")?;
                model
                    .forward(token_ids, attention_mask, &token_type_ids, None, None, None)
                    .context("XLM-RoBERTa forward pass")
            }
            Self::ModernBert(model) => model
                .forward(token_ids, attention_mask)
                .context("ModernBERT forward pass"),
            Self::NomicBert(model) => model
                .forward(token_ids, None, Some(attention_mask))
                .context("NomicBERT forward pass"),
        }
    }
}

/// Helper struct to detect model type from config
#[derive(Debug, Deserialize)]
struct ModelTypeDetector {
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    dim: Option<usize>,
}

/// Dense encoder using the Candle backend.
///
/// This encoder produces single-vector embeddings by applying pooling
/// to token-level BERT outputs. Supports CLS, mean, and max pooling
/// strategies with optional L2 normalization and Matryoshka truncation.
pub struct CandleDenseEncoder {
    model: BertVariant,
    tokenizer: Tokenizer,
    device: Device,
    config: ModelConfig,
    pooling_strategy: PoolingStrategy,
    normalize: bool,
    supports_padded_batch: bool,
    dtype: ModelDType,
    resource_policy: ResourcePolicy,
    transformer_profile: TransformerProfile,
    _residency: ModelResidencyPermit<'static>,
}

impl Encoder for CandleDenseEncoder {
    type Output = DenseEmbedding;

    fn encode(&self, input: &str) -> Result<Self::Output> {
        Self::encode(self, input)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Self::Output>> {
        Self::encode_batch(self, inputs)
    }
}

impl DenseEncoder for CandleDenseEncoder {
    fn embedding_dim(&self) -> usize {
        // Return target dimension if Matryoshka is configured, otherwise base dimension
        self.config
            .target_dimension
            .unwrap_or(self.config.embedding_dim)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        self.pooling_strategy
    }
}

impl CandleDenseEncoder {
    /// Parameter dtype selected when this model was loaded.
    #[must_use]
    pub const fn model_dtype(&self) -> ModelDType {
        self.dtype
    }
}

#[cfg(test)]
mod tests;
