//! BERT encoder implementation using Candle.
//!
//! Supports multiple BERT variants:
//! - BERT (bert-base-uncased, colbert-ir/colbertv2.0, etc.)
//! - `DistilBERT` (distilbert-base-uncased, answerdotai/answerai-colbert-small-v1)
//! - `JinaBERT` (jinaai/jina-colbert-v2)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use ndarray::Array2;
use serde::Deserialize;

use crate::core::{Encoder, MultiVectorEncoder, TokenEmbedder, TokenEmbeddings, Tokenizer};
use crate::models::ModelConfig;

/// Enum to hold different BERT model variants
enum BertVariant {
    Bert(candle_transformers::models::bert::BertModel),
    DistilBert(candle_transformers::models::distilbert::DistilBertModel),
    JinaBert(candle_transformers::models::jina_bert::BertModel),
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
            Self::JinaBert(model) => {
                // JinaBERT uses ALiBi position embeddings and doesn't need attention_mask
                // in its forward pass
                model.forward(token_ids).context("JinaBERT forward pass")
            }
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
}

/// BERT encoder using the Candle backend.
///
/// This encoder is specifically for BERT-style models producing multi-vector
/// token embeddings. Future variants (e.g., `CandleVisionEncoder`, `CandleTimeSeriesEncoder`)
/// will handle other model types.
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
    /// Automatically detects the model type (BERT, `DistilBERT`, `JinaBERT`) from config.json
    /// and loads the appropriate model variant.
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
        let model_name = &model_config.model_name;

        // Load tokenizer
        let tokenizer = Tokenizer::from_pretrained(model_name)
            .with_context(|| format!("Loading tokenizer for {model_name}"))?;

        // Download model files from HuggingFace Hub
        let api =
            hf_hub::api::sync::Api::new().context("Failed to initialize HuggingFace Hub API")?;
        let repo = api.model(model_name.to_string());

        // Load config to detect model type
        let config_path = repo
            .get("config.json")
            .with_context(|| format!("Downloading config for {model_name}"))?;

        let config_str =
            std::fs::read_to_string(&config_path).context("Reading model config file")?;

        // Detect model type
        let detector: ModelTypeDetector =
            serde_json::from_str(&config_str).context("Parsing config to detect model type")?;

        let model_type = Self::detect_model_type(&detector)
            .with_context(|| format!("Detecting model type for {model_name}"))?;

        // Try to load safetensors first, fall back to pytorch_model.bin
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
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

        let model = Self::load_model(&config_str, model_vb, &model_type)
            .with_context(|| format!("Loading {model_type} model"))?;

        // Try to load ColBERT projection layer (linear.weight)
        // This is optional - only ColBERT models have this layer
        let hidden_size = detector.hidden_size.or(detector.dim).unwrap_or(768);
        let projection = vb
            .get((model_config.embedding_dim, hidden_size), "linear.weight")
            .ok();

        if projection.is_some() {
            println!(
                "Loaded ColBERT projection layer: {} -> {} dimensions",
                hidden_size, model_config.embedding_dim
            );
        }

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

    /// Detects the model type from config
    fn detect_model_type(detector: &ModelTypeDetector) -> Result<String> {
        // First check explicit model_type field
        if let Some(ref model_type) = detector.model_type {
            let model_type_lower = model_type.to_lowercase();
            if model_type_lower.contains("distilbert") {
                return Ok("distilbert".to_string());
            } else if model_type_lower.contains("jina") {
                return Ok("jinabert".to_string());
            } else if model_type_lower.contains("bert") {
                return Ok("bert".to_string());
            }
        }

        // Fallback: detect by config structure
        // DistilBERT uses 'dim', while BERT/JinaBERT use 'hidden_size'
        if detector.dim.is_some() && detector.hidden_size.is_none() {
            Ok("distilbert".to_string())
        } else if detector.hidden_size.is_some() {
            // Default to BERT (JinaBERT configs are similar to BERT)
            Ok("bert".to_string())
        } else {
            anyhow::bail!("Could not detect model type from config")
        }
    }

    /// Loads the appropriate model variant
    fn load_model(config_str: &str, vb: VarBuilder, model_type: &str) -> Result<BertVariant> {
        match model_type {
            "distilbert" => {
                let config: candle_transformers::models::distilbert::Config =
                    serde_json::from_str(config_str).context("Parsing DistilBERT config")?;
                let model =
                    candle_transformers::models::distilbert::DistilBertModel::load(vb, &config)
                        .context("Loading DistilBERT model")?;
                Ok(BertVariant::DistilBert(model))
            }
            "jinabert" => {
                let config: candle_transformers::models::jina_bert::Config =
                    serde_json::from_str(config_str).context("Parsing JinaBERT config")?;
                let model = candle_transformers::models::jina_bert::BertModel::new(vb, &config)
                    .context("Loading JinaBERT model")?;
                Ok(BertVariant::JinaBert(model))
            }
            _ => {
                // Default to BERT for unknown types
                let config: candle_transformers::models::bert::Config =
                    serde_json::from_str(config_str).context("Parsing BERT config")?;
                let model = candle_transformers::models::bert::BertModel::load(vb, &config)
                    .context("Loading BERT model")?;
                Ok(BertVariant::Bert(model))
            }
        }
    }

    /// Converts token IDs to a Candle tensor.
    fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<Tensor> {
        let token_ids_as_i64: Vec<i64> = token_ids.iter().map(|&x| i64::from(x)).collect();

        Tensor::from_vec(token_ids_as_i64, (1, token_ids.len()), &self.device)
            .context("Creating token ID tensor")
    }

    /// Extracts token embeddings from BERT model output.
    #[allow(clippy::unused_self)]
    fn extract_embeddings(&self, output: &Tensor) -> Result<Array2<f32>> {
        // Output shape is (batch_size=1, seq_len, hidden_size)
        // We need to squeeze the batch dimension and convert to ndarray

        let embeddings = output.squeeze(0).context("Squeezing batch dimension")?;

        // Convert to CPU and then to Vec
        let embeddings_cpu = embeddings
            .to_dtype(DType::F32)
            .context("Converting to F32")?
            .to_device(&Device::Cpu)
            .context("Moving tensor to CPU")?;

        let shape = embeddings_cpu.dims();
        let seq_len = shape[0];
        let hidden_size = shape[1];

        let embeddings_vec = embeddings_cpu
            .flatten_all()
            .context("Flattening tensor")?
            .to_vec1::<f32>()
            .context("Converting tensor to Vec<f32>")?;

        // Convert to ndarray
        Array2::from_shape_vec((seq_len, hidden_size), embeddings_vec)
            .context("Converting to ndarray Array2")
    }
}

impl TokenEmbedder for CandleBertEncoder {
    fn encode(&self, text: &str) -> Result<TokenEmbeddings> {
        // Tokenize input
        let (token_ids, attention_mask) = self
            .tokenizer
            .encode(text, true)
            .with_context(|| format!("Tokenizing text: {text}"))?;

        // Convert to tensors
        let token_ids_tensor = self.tokens_to_tensor(&token_ids)?;

        // Handle attention mask - DistilBERT expects inverted mask (0=attend, 1=mask)
        // Standard tokenizers return 1=attend, 0=pad, so we need to invert for DistilBERT
        let attention_mask_processed = match &self.model {
            BertVariant::DistilBert(_) => {
                // Invert mask for DistilBERT: 1 -> 0, 0 -> 1
                attention_mask
                    .iter()
                    .map(|&x| i64::from(x != 1))
                    .collect()
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

        // Extract token embeddings
        let embeddings = self
            .extract_embeddings(&output)
            .context("Extracting embeddings from model output")?;

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

            // Filter out padding tokens using attention mask
            // Get the original attention mask for this sample
            let (_, attention_mask) = &batch_tokenized[i];
            let actual_length = attention_mask.iter().filter(|&&x| x == 1).count();

            // Extract only non-padded token embeddings
            let embeddings_filtered = if actual_length < seq_len {
                embeddings_array
                    .slice(ndarray::s![..actual_length, ..])
                    .to_owned()
            } else {
                embeddings_array
            };

            // Create TokenEmbeddings with original text
            let token_embeddings = TokenEmbeddings::new(embeddings_filtered, inputs[i].to_string())
                .context("Creating TokenEmbeddings from batch")?;

            results.push(token_embeddings);
        }

        Ok(results)
    }
}

impl MultiVectorEncoder for CandleBertEncoder {
    fn num_vectors(&self, text: &str) -> Result<usize> {
        // Tokenize to count tokens
        let (token_ids, _) = self
            .tokenizer
            .encode(text, true)
            .with_context(|| format!("Tokenizing text to count vectors: {text}"))?;

        Ok(token_ids.len())
    }

    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }
}
