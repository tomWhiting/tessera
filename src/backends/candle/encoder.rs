//! BERT encoder implementation using Candle.
//!
//! The active multi-vector registry paths use BERT checkpoints such as
//! `colbert-v2` and `colbert-small`. The loader contains additional BERT-family
//! variants, but registry support metadata determines whether a checkpoint is
//! runnable.
//!
//! This path does not yet implement role-specific ColBERT tokenization: `[Q]`/
//! `[D]` marker insertion, fixed-length `[MASK]` query augmentation, or document
//! punctuation masking. Callers should treat retrieval parity as provisional.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use ndarray::Array2;

use crate::core::{Encoder, MultiVectorEncoder, TokenEmbedder, TokenEmbeddings, Tokenizer};
use crate::models::loader::ModelFileResolver;
use crate::models::{registry::ModelType, ModelConfig};
use crate::runtime::{preflight_registered_model, ResourcePolicy};

mod model;
mod tensor;

use model::{detect_model_type, load_model, BertVariant, ModelTypeDetector};

/// BERT encoder using the Candle backend.
///
/// This encoder is specifically for BERT-style models producing multi-vector
/// token embeddings. Dense and vision-language architectures are handled by
/// their own encoder modules.
pub struct CandleBertEncoder {
    model: BertVariant,
    projection: Option<Tensor>, // ColBERT projection layer: [colbert_dim, hidden_size]
    tokenizer: Tokenizer,
    device: Device,
    config: ModelConfig,
    matryoshka_strategy: Option<crate::utils::MatryoshkaStrategy>,
}

impl CandleBertEncoder {
    /// Creates a new Candle-based BERT encoder.
    ///
    /// Detects the implemented BERT-family variant from `config.json` and loads
    /// its weights. Public builders separately enforce registry support status.
    ///
    /// # Arguments
    /// * `model_config` - Configuration for the model
    /// * `device` - Device to run the model on (CPU or Metal)
    ///
    /// # Returns
    /// A new `CandleEncoder` instance with the loaded model
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer or model cannot be loaded.
    pub fn new(model_config: ModelConfig, device: Device) -> Result<Self> {
        let resource_policy = ResourcePolicy::for_model_context(model_config.max_seq_length);
        Self::new_with_resource_policy(model_config, device, resource_policy)
    }

    /// Creates a BERT encoder with explicit resource limits.
    pub fn new_with_resource_policy(
        model_config: ModelConfig,
        device: Device,
        resource_policy: ResourcePolicy,
    ) -> Result<Self> {
        let model_name = &model_config.model_name;

        let model_info = preflight_registered_model(
            model_name,
            model_config.max_seq_length,
            ModelType::Colbert,
            &device,
            &resource_policy,
        )?;

        let files = ModelFileResolver::new(model_info)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_model_files_with_policy(&files, resource_policy)
            .with_context(|| format!("Loading tokenizer for {model_name}"))?;

        // Load config to detect model type
        let config_path = files
            .get(model_info.config_file)
            .with_context(|| format!("Downloading config for {model_name}"))?;

        let config_str =
            std::fs::read_to_string(&config_path).context("Reading model config file")?;

        // Detect model type
        let detector: ModelTypeDetector =
            serde_json::from_str(&config_str).context("Parsing config to detect model type")?;

        let model_type = detect_model_type(&detector)
            .with_context(|| format!("Detecting model type for {model_name}"))?;

        // Try to load safetensors first, fall back to pytorch_model.bin
        let weights_path = files
            .weights()
            .with_context(|| format!("Downloading model weights for {model_name}"))?;

        // Load model weights
        let vb = if weights_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .context("Loading model from safetensors")?
            }
        } else {
            VarBuilder::from_pth(&weights_path, DType::F32, &device)
                .context("Loading model from pytorch_model.bin")?
        };

        // Create the appropriate model variant
        // BERT and JinaBERT models have a "bert" prefix, but DistilBERT doesn't
        let model_vb = if model_type == "distilbert" {
            vb.pp("distilbert")
        } else {
            vb.pp("bert")
        };

        let model = load_model(&config_str, model_vb, &model_type)
            .with_context(|| format!("Loading {model_type} model"))?;

        // Try to load ColBERT projection layer (linear.weight)
        // This is optional - only ColBERT models have this layer
        let hidden_size = detector.hidden_size.or(detector.dim).unwrap_or(768);
        let projection = vb
            .get((model_config.embedding_dim, hidden_size), "linear.weight")
            .ok();

        // Determine Matryoshka strategy from registry if available
        #[allow(clippy::option_if_let_else)]
        let matryoshka_strategy = if model_config.target_dimension.is_some() {
            // Try to get strategy from registry
            if let Some(model_info) =
                crate::models::registry::get_model_by_hf_id(&model_config.model_name)
            {
                model_info
                    .embedding_dim
                    .matryoshka_strategy()
                    .and_then(crate::utils::MatryoshkaStrategy::from_str)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            model,
            projection,
            tokenizer,
            device,
            config: model_config,
            matryoshka_strategy,
        })
    }
}

impl TokenEmbedder for CandleBertEncoder {
    fn encode(&self, text: &str) -> Result<TokenEmbeddings> {
        // Tokenize input
        let (token_ids, attention_mask) = self
            .tokenizer
            .encode(text, true)
            .with_context(|| format!("Tokenizing text ({} UTF-8 bytes)", text.len()))?;

        // Convert to tensors
        let token_ids_tensor = tensor::tokens_to_tensor(&token_ids, &self.device)?;

        // Handle attention mask - DistilBERT expects inverted mask (0=attend, 1=mask)
        // Standard tokenizers return 1=attend, 0=pad, so we need to invert for DistilBERT
        let attention_mask_processed = match &self.model {
            BertVariant::DistilBert(_) => {
                // Invert mask for DistilBERT: 1 -> 0, 0 -> 1
                attention_mask.iter().map(|&x| i64::from(x != 1)).collect()
            }
            _ => {
                // BERT and JinaBERT use standard mask: 1=attend, 0=pad
                attention_mask.iter().map(|&x| i64::from(x)).collect()
            }
        };

        let attention_mask_tensor = Tensor::from_vec(
            attention_mask_processed,
            (1, attention_mask.len()),
            &self.device,
        )
        .context("Creating attention mask tensor")?;

        // Run model forward pass (handles all variants)
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let mut output = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor)?;

        // Apply Matryoshka truncation based on strategy
        if let (Some(target_dim), Some(strategy)) =
            (self.config.target_dimension, self.matryoshka_strategy)
        {
            use crate::utils::MatryoshkaStrategy;

            match strategy {
                MatryoshkaStrategy::TruncateHidden => {
                    // Truncate BEFORE projection (for models like ColBERT v2 with projection)
                    output = crate::utils::apply_matryoshka(&output, target_dim, strategy)
                        .context("Applying Matryoshka truncation to hidden states")?;

                    // Then apply projection if it exists
                    if let Some(ref projection) = self.projection {
                        let projection_t = projection.t()?;
                        output = output
                            .broadcast_matmul(&projection_t)
                            .context("Applying ColBERT projection after truncation")?;
                    }
                }
                MatryoshkaStrategy::TruncateOutput => {
                    // Apply projection first (if exists), then truncate
                    if let Some(ref projection) = self.projection {
                        let projection_t = projection.t()?;
                        output = output
                            .broadcast_matmul(&projection_t)
                            .context("Applying ColBERT projection layer")?;
                    }

                    // Then truncate the output
                    output = crate::utils::apply_matryoshka(&output, target_dim, strategy)
                        .context("Applying Matryoshka truncation to output")?;
                }
                MatryoshkaStrategy::TruncatePooled => {
                    // For dense encoders - truncate after pooling (not applicable here for multi-vector)
                    // Apply projection if exists, then truncate
                    if let Some(ref projection) = self.projection {
                        let projection_t = projection.t()?;
                        output = output
                            .broadcast_matmul(&projection_t)
                            .context("Applying projection layer")?;
                    }

                    output = crate::utils::apply_matryoshka(&output, target_dim, strategy)
                        .context("Applying Matryoshka truncation")?;
                }
            }
        } else {
            // No Matryoshka truncation - just apply projection if present
            if let Some(ref projection) = self.projection {
                // Output shape: [batch_size=1, seq_len, hidden_size]
                // Projection shape: [colbert_dim, hidden_size]
                // We need to do: output @ projection.T
                // Result: [batch_size=1, seq_len, colbert_dim]

                // Projection is [colbert_dim, hidden_size], we need [hidden_size, colbert_dim]
                let projection_t = projection.t()?;

                output = output
                    .broadcast_matmul(&projection_t)
                    .context("Applying ColBERT projection layer")?;
            }
        }

        output = tensor::l2_normalize_tokens(&output)
            .context("L2-normalizing projected ColBERT token embeddings")?;

        // Extract token embeddings
        let embeddings = tensor::extract_embeddings(&output)
            .context("Extracting embeddings from model output")?;
        drop(inference_permit);
        let embeddings = tensor::filter_by_attention_mask(&embeddings, &attention_mask)
            .context("Removing masked token embeddings")?;

        // Create TokenEmbeddings
        TokenEmbeddings::new(embeddings, text.to_string()).context("Creating TokenEmbeddings")
    }
}

// ============================================================================
// Unified Encoder Trait Implementations
// ============================================================================

impl Encoder for CandleBertEncoder {
    type Output = TokenEmbeddings;

    fn encode(&self, input: &str) -> Result<Self::Output> {
        // Delegate to TokenEmbedder implementation
        <Self as TokenEmbedder>::encode(self, input)
    }

    #[allow(clippy::too_many_lines)]
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Self::Output>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Special case: single input - use regular encode
        if inputs.len() == 1 {
            return Ok(vec![<Self as TokenEmbedder>::encode(self, inputs[0])?]);
        }

        // Batch tokenization with padding
        let batch_tokenized = self
            .tokenizer
            .encode_batch(inputs, true)
            .context("Batch tokenization")?;

        // Extract batch size and max sequence length
        let batch_size = batch_tokenized.len();
        let max_seq_len = batch_tokenized[0].0.len();

        // Convert token IDs to 2D tensor: [batch_size, max_seq_len]
        let mut all_token_ids = Vec::with_capacity(batch_size * max_seq_len);
        for (token_ids, _) in &batch_tokenized {
            for &token_id in token_ids {
                all_token_ids.push(i64::from(token_id));
            }
        }

        let token_ids_tensor =
            Tensor::from_vec(all_token_ids, (batch_size, max_seq_len), &self.device)
                .context("Creating batch token IDs tensor")?;

        // Convert attention masks to 2D tensor: [batch_size, max_seq_len]
        // Handle DistilBERT's inverted mask convention
        let mut all_attention_masks = Vec::with_capacity(batch_size * max_seq_len);
        for (_, attention_mask) in &batch_tokenized {
            for &mask_val in attention_mask {
                let processed_val = match &self.model {
                    BertVariant::DistilBert(_) => {
                        // Invert mask for DistilBERT: 1 -> 0, 0 -> 1
                        i64::from(mask_val != 1)
                    }
                    _ => {
                        // BERT and JinaBERT use standard mask: 1=attend, 0=pad
                        i64::from(mask_val)
                    }
                };
                all_attention_masks.push(processed_val);
            }
        }

        let attention_mask_tensor =
            Tensor::from_vec(all_attention_masks, (batch_size, max_seq_len), &self.device)
                .context("Creating batch attention mask tensor")?;

        // Single forward pass for entire batch
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let mut batch_output = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor)
            .context("Batch forward pass")?;

        // Apply Matryoshka truncation based on strategy (if configured)
        if let (Some(target_dim), Some(strategy)) =
            (self.config.target_dimension, self.matryoshka_strategy)
        {
            use crate::utils::MatryoshkaStrategy;

            match strategy {
                MatryoshkaStrategy::TruncateHidden => {
                    // Truncate BEFORE projection
                    batch_output =
                        crate::utils::apply_matryoshka(&batch_output, target_dim, strategy)
                            .context("Applying Matryoshka truncation to batch hidden states")?;

                    // Then apply projection if it exists
                    if let Some(ref projection) = self.projection {
                        let projection_t = projection.t()?;
                        batch_output = batch_output
                            .broadcast_matmul(&projection_t)
                            .context("Applying ColBERT projection to batch after truncation")?;
                    }
                }
                MatryoshkaStrategy::TruncateOutput => {
                    // Apply projection first (if exists), then truncate
                    if let Some(ref projection) = self.projection {
                        let projection_t = projection.t()?;
                        batch_output = batch_output
                            .broadcast_matmul(&projection_t)
                            .context("Applying ColBERT projection to batch")?;
                    }

                    // Then truncate the output
                    batch_output =
                        crate::utils::apply_matryoshka(&batch_output, target_dim, strategy)
                            .context("Applying Matryoshka truncation to batch output")?;
                }
                MatryoshkaStrategy::TruncatePooled => {
                    // For dense encoders - apply projection then truncate
                    if let Some(ref projection) = self.projection {
                        let projection_t = projection.t()?;
                        batch_output = batch_output
                            .broadcast_matmul(&projection_t)
                            .context("Applying projection to batch")?;
                    }

                    batch_output =
                        crate::utils::apply_matryoshka(&batch_output, target_dim, strategy)
                            .context("Applying Matryoshka truncation to batch")?;
                }
            }
        } else {
            // No Matryoshka truncation - just apply projection if present
            if let Some(ref projection) = self.projection {
                // batch_output shape: [batch_size, seq_len, hidden_size]
                // projection shape: [colbert_dim, hidden_size]
                // Result: [batch_size, seq_len, colbert_dim]
                let projection_t = projection.t()?;
                batch_output = batch_output
                    .broadcast_matmul(&projection_t)
                    .context("Applying ColBERT projection to batch")?;
            }
        }

        batch_output = tensor::l2_normalize_tokens(&batch_output)
            .context("L2-normalizing projected ColBERT batch token embeddings")?;

        // Extract individual embeddings from batch
        // batch_output shape: [batch_size, max_seq_len, embedding_dim]
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // Extract embeddings for this sample: [max_seq_len, embedding_dim]
            let sample_output = batch_output
                .get(i)
                .context("Extracting sample from batch")?;

            // Move to CPU and convert to ndarray
            let embeddings_cpu = sample_output
                .to_dtype(DType::F32)
                .context("Converting to F32")?
                .to_device(&Device::Cpu)
                .context("Moving tensor to CPU")?;

            let shape = embeddings_cpu.dims();
            let seq_len = shape[0];
            let embedding_dim = shape[1];

            let embeddings_vec = embeddings_cpu
                .flatten_all()
                .context("Flattening tensor")?
                .to_vec1::<f32>()
                .context("Converting tensor to Vec<f32>")?;

            // Convert to ndarray
            let embeddings_array = Array2::from_shape_vec((seq_len, embedding_dim), embeddings_vec)
                .context("Converting to ndarray Array2")?;

            let (_, attention_mask) = &batch_tokenized[i];
            let embeddings_filtered =
                tensor::filter_by_attention_mask(&embeddings_array, attention_mask)
                    .context("Removing padded token embeddings from batch item")?;

            // Create TokenEmbeddings with original text
            let token_embeddings = TokenEmbeddings::new(embeddings_filtered, inputs[i].to_string())
                .context("Creating TokenEmbeddings from batch")?;

            results.push(token_embeddings);
        }
        drop(inference_permit);

        Ok(results)
    }
}

impl MultiVectorEncoder for CandleBertEncoder {
    fn num_vectors(&self, text: &str) -> Result<usize> {
        // Tokenize to count tokens
        let (token_ids, _) = self.tokenizer.encode(text, true).with_context(|| {
            format!(
                "Tokenizing text to count vectors ({} UTF-8 bytes)",
                text.len()
            )
        })?;

        Ok(token_ids.len())
    }

    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }
}
