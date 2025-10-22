//! Sparse vocabulary-space encodings (SPLADE-style).
//!
//! Implements sparse embedding models that represent text as weighted
//! vocabulary distributions rather than dense vectors:
//!
//! - Based on BERT's MLM (Masked Language Modeling) head
//! - Produces sparse vectors in vocabulary space (30k+ dimensions)
//! - Activations represent term importance/presence
//! - Interpretable: non-zero dimensions map to actual tokens
//!
//! # Architecture
//!
//! SPLADE uses:
//! 1. BERT encoder for contextualized representations
//! 2. MLM head to project to vocabulary space
//! 3. log(1 + ReLU(x)) transformation for sparsity
//! 4. Max pooling across token positions
//!
//! # Advantages
//!
//! - Interpretable: can see which terms are activated
//! - Efficient retrieval with inverted indexes
//! - Better out-of-domain generalization than dense embeddings
//! - Natural vocabulary expansion (related terms activated)
//!
//! # Example
//!
//! ```no_run
//! use tessera::encoding::sparse::CandleSparseEncoder;
//! use tessera::models::ModelConfig;
//! use tessera::core::Encoder;
//! use candle_core::Device;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Load SPLADE model
//! let config = ModelConfig::from_registry("splade-cocondenser")?;
//! let device = Device::Cpu;
//! let encoder = CandleSparseEncoder::new(config, device)?;
//!
//! // Encode text to sparse vector
//! let embedding = encoder.encode("machine learning")?;
//! println!("Sparsity: {:.2}%", embedding.sparsity() * 100.0);
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};
use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::path::Path;

use crate::core::{Encoder, SparseEmbedding, SparseEncoder, Tokenizer};
use crate::error::TesseraError;
use crate::models::ModelConfig;

/// Enum to hold different BERT model variants
enum BertVariant {
    Bert(candle_transformers::models::bert::BertModel),
    DistilBert(candle_transformers::models::distilbert::DistilBertModel),
}

impl BertVariant {
    fn forward(&self, token_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        match self {
            Self::Bert(model) => model
                .forward(token_ids, attention_mask, None)
                .context("BERT forward pass"),
            Self::DistilBert(model) => model
                .forward(token_ids, attention_mask)
                .context("DistilBERT forward pass"),
        }
    }
}

/// Helper struct to detect model type from config
#[derive(Debug, Deserialize)]
struct ModelTypeDetector {
    model_type: Option<String>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    dim: Option<usize>,
    #[serde(default)]
    vocab_size: Option<usize>,
}

/// MLM (Masked Language Modeling) head for SPLADE.
///
/// Projects BERT hidden states to vocabulary logits via:
/// 1. Dense transformation with GELU activation
/// 2. Layer normalization
/// 3. Final linear projection to vocabulary space
struct MlmHead {
    /// Transform layer: hidden → hidden with GELU
    transform_dense: Linear,
    /// Layer normalization
    transform_layer_norm: LayerNorm,
    /// Final projection: hidden → `vocab_size`
    decoder: Linear,
}

impl MlmHead {
    /// Load MLM head from weights.
    ///
    /// # Arguments
    /// * `vb` - Variable builder for loading weights
    /// * `hidden_size` - Hidden dimension size
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Initialized MLM head ready for inference
    fn load(vb: VarBuilder, hidden_size: usize, vocab_size: usize) -> Result<Self> {
        // Load transform layer (dense + activation)
        let transform_vb = vb.pp("cls").pp("predictions").pp("transform");
        let transform_dense = linear(hidden_size, hidden_size, transform_vb.pp("dense"))
            .context("Loading MLM transform dense layer")?;

        // Load layer norm
        let transform_layer_norm = layer_norm(
            hidden_size,
            1e-12, // epsilon - standard BERT value
            transform_vb.pp("LayerNorm"),
        )
        .context("Loading MLM layer normalization")?;

        // Load decoder (final projection to vocab)
        let decoder_vb = vb.pp("cls").pp("predictions");
        let decoder = linear(hidden_size, vocab_size, decoder_vb.pp("decoder"))
            .context("Loading MLM decoder layer")?;

        Ok(Self {
            transform_dense,
            transform_layer_norm,
            decoder,
        })
    }

    /// Forward pass: `hidden_states` → `vocab_logits`
    ///
    /// # Arguments
    /// * `hidden_states` - Token representations from BERT [`seq_len`, `hidden_size`]
    ///
    /// # Returns
    /// Vocabulary logits [`seq_len`, `vocab_size`]
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Transform: linear + GELU
        let transformed = self
            .transform_dense
            .forward(hidden_states)
            .context("MLM transform dense forward")?;
        let activated = transformed.gelu().context("MLM GELU activation")?;

        // Layer norm
        let normalized = self
            .transform_layer_norm
            .forward(&activated)
            .context("MLM layer norm forward")?;

        // Project to vocabulary
        let logits = self
            .decoder
            .forward(&normalized)
            .context("MLM decoder forward")?;

        Ok(logits)
    }
}

/// Candle-based sparse encoder for SPLADE models.
///
/// Implements the SPLADE architecture using BERT + MLM head with
/// log(1 + ReLU(x)) transformation and max pooling for sparse embeddings.
pub struct CandleSparseEncoder {
    /// BERT model
    model: BertVariant,
    /// MLM head
    mlm_head: MlmHead,
    /// Tokenizer
    tokenizer: Tokenizer,
    /// Device (Metal/CUDA/CPU)
    device: Device,
    /// Vocabulary size
    vocab_size: usize,
}

impl CandleSparseEncoder {
    /// Create new sparse encoder.
    ///
    /// # Arguments
    /// * `config` - Model configuration from registry
    /// * `device` - Device to run inference on
    ///
    /// # Returns
    /// Initialized sparse encoder with loaded weights
    ///
    /// # Errors
    /// Returns error if model files cannot be downloaded/loaded or
    /// if the model architecture is incompatible
    pub fn new(config: ModelConfig, device: Device) -> Result<Self> {
        let model_name = &config.model_name;

        // Download model from HuggingFace
        let api = Api::new().context("Failed to initialize HuggingFace Hub API")?;
        let repo = api.model(model_name.to_string());

        // Load config to get vocab size and architecture
        let config_path = repo
            .get("config.json")
            .with_context(|| format!("Downloading config for {model_name}"))?;

        let config_str =
            std::fs::read_to_string(&config_path).context("Reading model config file")?;

        let detector: ModelTypeDetector = serde_json::from_str(&config_str)
            .context("Parsing config to detect model type and vocab size")?;

        // Get vocabulary size
        let vocab_size = detector
            .vocab_size
            .ok_or_else(|| TesseraError::ConfigError("Missing vocab_size in config".into()))?;

        // Detect model type
        let model_type = Self::detect_model_type(&detector)
            .with_context(|| format!("Detecting model type for {model_name}"))?;

        // Get hidden size
        let hidden_size = detector
            .hidden_size
            .or(detector.dim)
            .ok_or_else(|| TesseraError::ConfigError("Missing hidden_size/dim in config".into()))?;

        // Try to load safetensors first, fall back to pytorch_model.bin
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .with_context(|| format!("Downloading model weights for {model_name}"))?;

        // Load model weights
        let vb = if weights_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, &device)
                    .context("Loading model from safetensors")?
            }
        } else {
            VarBuilder::from_pth(&weights_path, DType::F32, &device)
                .context("Loading model from pytorch_model.bin")?
        };

        // Detect model prefix by checking actual tensor names
        let has_prefix = Self::detect_model_prefix(&weights_path)
            .with_context(|| format!("Detecting model prefix for {model_name}"))?;

        // Create the appropriate model variant with correct prefix
        let model_vb = match (has_prefix, model_type.as_str()) {
            (true, "distilbert") => vb.pp("distilbert"),
            (true, _) => vb.pp("bert"),
            (false, _) => vb.clone(), // No prefix
        };

        let model = Self::load_model(&config_str, model_vb, &model_type)
            .with_context(|| format!("Loading {model_type} model"))?;

        // Load MLM head (always from root vb, regardless of model prefix)
        let mlm_head = MlmHead::load(vb, hidden_size, vocab_size)
            .context("Loading MLM head for sparse encoding")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_pretrained(model_name)
            .with_context(|| format!("Loading tokenizer for {model_name}"))?;

        Ok(Self {
            model,
            mlm_head,
            tokenizer,
            device,
            vocab_size,
        })
    }

    /// Detects the model type from config
    fn detect_model_type(detector: &ModelTypeDetector) -> Result<String> {
        // First check explicit model_type field
        if let Some(ref model_type) = detector.model_type {
            let model_type_lower = model_type.to_lowercase();
            if model_type_lower.contains("distilbert") {
                return Ok("distilbert".to_string());
            } else if model_type_lower.contains("bert") {
                return Ok("bert".to_string());
            }
        }

        // Fallback: detect by config structure
        if detector.dim.is_some() && detector.hidden_size.is_none() {
            Ok("distilbert".to_string())
        } else if detector.hidden_size.is_some() {
            Ok("bert".to_string())
        } else {
            anyhow::bail!("Could not detect model type from config")
        }
    }

    /// Detects whether the model weights use a prefix (e.g., "bert.", "distilbert.")
    fn detect_model_prefix(weights_path: &Path) -> Result<bool> {
        let extension = weights_path.extension().and_then(|s| s.to_str());

        if extension == Some("safetensors") {
            use safetensors::SafeTensors;

            let buffer = std::fs::read(weights_path)
                .context("Reading safetensors file for prefix detection")?;

            let tensors = SafeTensors::deserialize(&buffer).context("Deserializing safetensors")?;

            let tensor_names = tensors.names();

            // Check if any tensor starts with "bert." or "distilbert."
            for name in tensor_names {
                if name.starts_with("bert.embeddings.word_embeddings") {
                    return Ok(true);
                } else if name.starts_with("distilbert.embeddings.word_embeddings") {
                    return Ok(true);
                } else if name == "embeddings.word_embeddings.weight" {
                    return Ok(false);
                }
            }

            // Default: assume has prefix (safer for SPLADE models)
            Ok(true)
        } else {
            // For pytorch_model.bin
            let weights = candle_core::pickle::read_pth_tensor_info(weights_path, false, None)
                .context("Reading pytorch model info for prefix detection")?;

            for tensor_info in &weights {
                let name = &tensor_info.name;
                if name.starts_with("bert.embeddings.word_embeddings") {
                    return Ok(true);
                } else if name.starts_with("distilbert.embeddings.word_embeddings") {
                    return Ok(true);
                } else if name == "embeddings.word_embeddings.weight" {
                    return Ok(false);
                }
            }

            // Default: assume has prefix
            Ok(true)
        }
    }

    /// Loads the appropriate model variant
    fn load_model(config_str: &str, vb: VarBuilder, model_type: &str) -> Result<BertVariant> {
        if model_type == "distilbert" {
            let config: candle_transformers::models::distilbert::Config =
                serde_json::from_str(config_str).context("Parsing DistilBERT config")?;
            let model = candle_transformers::models::distilbert::DistilBertModel::load(vb, &config)
                .context("Loading DistilBERT model")?;
            Ok(BertVariant::DistilBert(model))
        } else {
            let config: candle_transformers::models::bert::Config =
                serde_json::from_str(config_str).context("Parsing BERT config")?;
            let model = candle_transformers::models::bert::BertModel::load(vb, &config)
                .context("Loading BERT model")?;
            Ok(BertVariant::Bert(model))
        }
    }

    /// Apply SPLADE transformation: log(1 + ReLU(x))
    ///
    /// This transformation encourages sparsity while maintaining smoothness.
    ///
    /// # Arguments
    /// * `logits` - Raw vocabulary logits from MLM head
    ///
    /// # Returns
    /// Transformed logits with sparsity-inducing properties
    fn splade_transform(&self, logits: &Tensor) -> Result<Tensor> {
        // ReLU: max(0, x) - zeros out negative values
        let relu = logits.relu().context("Applying ReLU")?;

        // 1 + ReLU
        let one = Tensor::ones_like(&relu).context("Creating ones tensor")?;
        let one_plus_relu = (one + relu).context("Computing 1 + ReLU")?;

        // log(1 + ReLU) - log saturation for smoother values
        let log_result = one_plus_relu.log().context("Applying log")?;

        Ok(log_result)
    }

    /// Max pool across token dimension.
    ///
    /// Takes the maximum value for each vocabulary dimension across all tokens.
    /// This aggregates token-level predictions into a document-level sparse vector.
    ///
    /// # Arguments
    /// * `tensor` - Token-level vocabulary scores [`seq_len`, `vocab_size`]
    /// * `attention_mask` - Attention mask (1=valid, 0=padding)
    ///
    /// # Returns
    /// Max-pooled sparse vector [`vocab_size`]
    fn max_pool_tokens(&self, tensor: &Tensor, attention_mask: &[u32]) -> Result<Tensor> {
        let dims = tensor.dims();
        anyhow::ensure!(
            dims.len() == 2,
            "Expected 2D tensor [seq_len, vocab_size], got shape {dims:?}"
        );

        let vocab_size = dims[1];

        anyhow::ensure!(
            vocab_size == self.vocab_size,
            "Vocabulary size mismatch: expected {}, got {}",
            self.vocab_size,
            vocab_size
        );

        // Get valid token count
        let valid_tokens: usize = attention_mask.iter().map(|&m| m as usize).sum();

        if valid_tokens == 0 {
            return Tensor::zeros((vocab_size,), DType::F32, &self.device)
                .context("Creating zero tensor for empty input");
        }

        // Convert to CPU for processing
        let tensor_cpu = tensor
            .to_device(&Device::Cpu)
            .context("Moving tensor to CPU for max pooling")?;
        let tensor_data = tensor_cpu
            .to_vec2::<f32>()
            .context("Converting tensor to Vec2 for max pooling")?;

        // Max across valid tokens only
        let mut max_values = vec![f32::NEG_INFINITY; vocab_size];

        for (token_idx, row) in tensor_data.iter().enumerate() {
            if token_idx >= attention_mask.len() || attention_mask[token_idx] == 0 {
                continue; // Skip padding tokens
            }

            for (vocab_idx, &value) in row.iter().enumerate() {
                if value > max_values[vocab_idx] {
                    max_values[vocab_idx] = value;
                }
            }
        }

        // Replace -inf with 0.0 (no valid tokens activated this dimension)
        for val in &mut max_values {
            if *val == f32::NEG_INFINITY {
                *val = 0.0;
            }
        }

        Tensor::from_vec(max_values, (vocab_size,), &self.device)
            .context("Creating max-pooled tensor")
    }

    /// Convert dense tensor to sparse representation.
    ///
    /// Filters out near-zero values to produce a sparse embedding.
    ///
    /// # Arguments
    /// * `tensor` - Dense vocabulary vector [`vocab_size`]
    /// * `text` - Original input text
    ///
    /// # Returns
    /// Sparse embedding with only non-zero values
    fn to_sparse(&self, tensor: &Tensor, text: String) -> Result<SparseEmbedding> {
        let values = tensor
            .to_vec1::<f32>()
            .context("Converting tensor to vector for sparse conversion")?;

        anyhow::ensure!(
            values.len() == self.vocab_size,
            "Vocabulary size mismatch in sparse conversion: expected {}, got {}",
            self.vocab_size,
            values.len()
        );

        // Keep only non-zero values (with small threshold to filter numerical noise)
        let threshold = 1e-6;
        let sparse_values: Vec<(usize, f32)> = values
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > threshold)
            .map(|(idx, &v)| (idx, v))
            .collect();

        Ok(SparseEmbedding::new(sparse_values, self.vocab_size, text))
    }
}

impl Encoder for CandleSparseEncoder {
    type Output = SparseEmbedding;

    fn encode(&self, input: &str) -> Result<Self::Output> {
        // Tokenize
        let (token_ids, attention_mask) = self
            .tokenizer
            .encode(input, true)
            .with_context(|| format!("Tokenizing input: {input}"))?;

        // Convert to tensors
        let token_ids_i64: Vec<i64> = token_ids.iter().map(|&x| i64::from(x)).collect();
        let token_ids_tensor = Tensor::from_vec(token_ids_i64, (1, token_ids.len()), &self.device)
            .context("Creating token IDs tensor")?;

        // Handle attention mask - DistilBERT expects inverted mask
        let attention_mask_processed: Vec<i64> = match &self.model {
            BertVariant::DistilBert(_) => {
                // Invert mask for DistilBERT: 1 -> 0, 0 -> 1
                attention_mask.iter().map(|&x| i64::from(x != 1)).collect()
            }
            _ => {
                // Standard BERT convention
                attention_mask.iter().map(|&x| i64::from(x)).collect()
            }
        };

        let attention_mask_tensor = Tensor::from_vec(
            attention_mask_processed,
            (1, attention_mask.len()),
            &self.device,
        )
        .context("Creating attention mask tensor")?;

        // BERT forward pass
        let hidden_states = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor)
            .context("BERT forward pass")?;

        // Remove batch dimension: [1, seq_len, hidden_size] -> [seq_len, hidden_size]
        let hidden_states = hidden_states
            .squeeze(0)
            .context("Squeezing batch dimension")?;

        // MLM head forward pass: [seq_len, hidden_size] -> [seq_len, vocab_size]
        let logits = self
            .mlm_head
            .forward(&hidden_states)
            .context("MLM head forward pass")?;

        // SPLADE transformation: log(1 + ReLU)
        let transformed = self
            .splade_transform(&logits)
            .context("Applying SPLADE transformation")?;

        // Max pool across tokens: [seq_len, vocab_size] -> [vocab_size]
        let pooled = self
            .max_pool_tokens(&transformed, &attention_mask)
            .context("Max pooling across tokens")?;

        // Convert to sparse representation
        self.to_sparse(&pooled, input.to_string())
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Self::Output>> {
        // For now, sequential encoding
        // TODO: Implement true batch processing for better performance
        inputs
            .iter()
            .map(|&text| self.encode(text))
            .collect::<Result<Vec<_>>>()
            .context("Batch encoding sparse embeddings")
    }
}

impl SparseEncoder for CandleSparseEncoder {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn expected_sparsity(&self) -> f32 {
        0.99 // 99% sparse for SPLADE models
    }
}

/// Legacy type for backward compatibility.
///
/// # Deprecated
/// Use `CandleSparseEncoder` instead.
#[deprecated(since = "0.2.0", note = "Use CandleSparseEncoder instead")]
pub type SparseEncoding = CandleSparseEncoder;
