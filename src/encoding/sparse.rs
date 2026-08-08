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
//! # Characteristics
//!
//! - Interpretable: can see which terms are activated
//! - Compatible with inverted-index retrieval structures
//! - Can activate vocabulary terms that are absent from the input
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
//! let config = ModelConfig::from_registry("splade-pp-en-v1")?;
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
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::core::{Encoder, SparseEmbedding, SparseEncoder, Tokenizer};
use crate::error::TesseraError;
use crate::models::loader::ModelFileResolver;
use crate::models::{registry::ModelType, ModelConfig};
use crate::runtime::{preflight_registered_model, ResourcePolicy};

mod mlm;
mod model;

use mlm::MlmHead;
use model::{detect_model_prefix, detect_model_type, load_model, BertVariant, ModelTypeDetector};

/// Max-pool raw vocabulary logits across valid token rows.
///
/// Valid rows are reduced in contiguous runs so masking never requires a
/// broadcast `[seq_len, vocab_size]` tensor. Each run is reduced on the
/// logits' device, and only vocabulary-sized tensors are merged.
fn max_pool_token_logits(
    logits: &Tensor,
    attention_mask: &[u32],
    expected_vocab_size: usize,
) -> Result<Tensor> {
    let dims = logits.dims();
    anyhow::ensure!(
        dims.len() == 2,
        "Expected 2D tensor [seq_len, vocab_size], got shape {dims:?}"
    );

    let (sequence_length, vocab_size) = (dims[0], dims[1]);
    anyhow::ensure!(
        vocab_size == expected_vocab_size,
        "Vocabulary size mismatch: expected {expected_vocab_size}, got {vocab_size}"
    );

    let is_valid = |token_index: usize| {
        attention_mask
            .get(token_index)
            .is_some_and(|&value| value != 0)
    };
    let mut pooled: Option<Tensor> = None;
    let mut token_index = 0;

    while token_index < sequence_length {
        while token_index < sequence_length && !is_valid(token_index) {
            token_index += 1;
        }
        let run_start = token_index;
        while token_index < sequence_length && is_valid(token_index) {
            token_index += 1;
        }

        if run_start == token_index {
            break;
        }

        let run_length = token_index - run_start;
        let run = logits
            .narrow(0, run_start, run_length)
            .with_context(|| format!("Selecting valid token rows {run_start}..{token_index}"))?;
        let run_max = run.max(0).context("Max pooling a valid token run")?;

        pooled = Some(match pooled {
            Some(current_max) => current_max
                .maximum(&run_max)
                .context("Combining max-pooled token runs")?,
            None => run_max,
        });
    }

    pooled.map_or_else(
        || {
            Tensor::zeros((vocab_size,), DType::F32, logits.device())
                .context("Creating zero tensor for input without valid tokens")
        },
        Ok,
    )
}

/// Apply the monotone SPLADE transform to an already pooled vocabulary vector.
fn splade_transform(pooled_logits: &Tensor) -> Result<Tensor> {
    let relu = pooled_logits.relu().context("Applying ReLU")?;
    let shifted = relu.affine(1.0, 1.0).context("Computing 1 + ReLU")?;
    shifted.log().context("Applying log")
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
        let resource_policy = ResourcePolicy::for_model_context(config.max_seq_length);
        Self::new_with_resource_policy(config, device, resource_policy)
    }

    /// Creates a sparse encoder with explicit resource limits.
    pub fn new_with_resource_policy(
        config: ModelConfig,
        device: Device,
        resource_policy: ResourcePolicy,
    ) -> Result<Self> {
        let model_name = &config.model_name;

        let model_info = preflight_registered_model(
            model_name,
            config.max_seq_length,
            ModelType::Sparse,
            &device,
            &resource_policy,
        )?;

        let files = ModelFileResolver::new(model_info)?;

        // Load config to get vocab size and architecture
        let config_path = files
            .get(model_info.config_file)
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
        let model_type = detect_model_type(&detector)
            .with_context(|| format!("Detecting model type for {model_name}"))?;

        // Get hidden size
        let hidden_size = detector
            .hidden_size
            .or(detector.dim)
            .ok_or_else(|| TesseraError::ConfigError("Missing hidden_size/dim in config".into()))?;

        // Try to load safetensors first, fall back to pytorch_model.bin
        let weights_path = files
            .weights()
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
        let has_prefix = detect_model_prefix(&weights_path)
            .with_context(|| format!("Detecting model prefix for {model_name}"))?;

        // Create the appropriate model variant with correct prefix
        let model_vb = match (has_prefix, model_type.as_str()) {
            (true, "distilbert") => vb.pp("distilbert"),
            (true, _) => vb.pp("bert"),
            (false, _) => vb.clone(), // No prefix
        };

        let model = load_model(&config_str, model_vb, &model_type)
            .with_context(|| format!("Loading {model_type} model"))?;

        // Load MLM head (always from root vb, regardless of model prefix)
        let mlm_head = MlmHead::load(vb, hidden_size, vocab_size)
            .context("Loading MLM head for sparse encoding")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_model_files_with_policy(&files, resource_policy)
            .with_context(|| format!("Loading tokenizer for {model_name}"))?;

        Ok(Self {
            model,
            mlm_head,
            tokenizer,
            device,
            vocab_size,
        })
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

        SparseEmbedding::new(sparse_values, self.vocab_size, text)
    }
}

impl Encoder for CandleSparseEncoder {
    type Output = SparseEmbedding;

    fn encode(&self, input: &str) -> Result<Self::Output> {
        // Tokenize
        let (token_ids, attention_mask) = self
            .tokenizer
            .encode(input, true)
            .with_context(|| format!("Tokenizing input ({} UTF-8 bytes)", input.len()))?;

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
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
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

        // SPLADE's transform is monotone, so max-pool the raw logits first.
        // This keeps the reduction on-device and applies elementwise transforms
        // only to the pooled vocabulary vector, not the full token matrix.
        let pooled_logits = max_pool_token_logits(&logits, &attention_mask, self.vocab_size)
            .context("Max pooling vocabulary logits across tokens")?;
        let pooled = splade_transform(&pooled_logits).context("Applying SPLADE transformation")?;

        // Convert to sparse representation
        let sparse_embedding = self.to_sparse(&pooled, input.to_string());
        drop(inference_permit);
        sparse_embedding
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

#[cfg(test)]
#[path = "sparse/tests.rs"]
mod tests;
