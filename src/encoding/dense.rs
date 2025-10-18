//! Dense single-vector encoder using Candle backend.
//!
//! Implements traditional sentence embedding approaches that pool
//! token-level BERT representations into a single fixed-size vector:
//!
//! - **CLS pooling**: Use the [CLS] token representation
//! - **Mean pooling**: Average all token embeddings (attention-weighted)
//! - **Max pooling**: Take element-wise maximum across tokens
//!
//! Dense encodings are memory-efficient (one vector per text) but lose
//! fine-grained token-level information compared to `ColBERT`.
//!
//! # Use Cases
//!
//! - Semantic search with large document collections
//! - Clustering and classification
//! - When memory/speed constraints prevent multi-vector approaches
//!
//! # Example
//!
//! ```no_run
//! use tessera::encoding::dense::CandleDenseEncoder;
//! use tessera::models::ModelConfig;
//! use tessera::core::Encoder;
//! use candle_core::Device;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Load a dense model from registry (e.g., BGE, Nomic)
//! let config = ModelConfig::from_registry("bge-base-en-v1.5")?;
//! let device = Device::Cpu;
//! let encoder = CandleDenseEncoder::new(config, device)?;
//!
//! // Encode text to single vector
//! let embedding = encoder.encode("Machine learning is a subset of AI")?;
//! assert_eq!(embedding.dim(), 768);
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use ndarray::Array1;
use serde::Deserialize;

use crate::core::{DenseEmbedding, DenseEncoder, Encoder, PoolingStrategy, Tokenizer};
use crate::error::TesseraError;
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
}

impl CandleDenseEncoder {
    /// Creates a new Candle-based dense encoder.
    ///
    /// Automatically detects the model type (BERT, `DistilBERT`, `JinaBERT`) from config.json
    /// and loads the appropriate model variant.
    ///
    /// # Arguments
    /// * `model_config` - Configuration for the model (must have `pooling_strategy` set)
    /// * `device` - Device to run the model on (CPU or Metal)
    ///
    /// # Returns
    /// A new `CandleDenseEncoder` instance with the loaded model
    ///
    /// # Errors
    /// Returns an error if:
    /// - Pooling strategy is not configured (required for dense models)
    /// - Model files cannot be downloaded or loaded
    /// - Model type cannot be detected
    pub fn new(model_config: ModelConfig, device: Device) -> Result<Self> {
        let model_name = &model_config.model_name;

        // Validate that pooling strategy is configured
        let registry_pooling = model_config.pooling_strategy.ok_or_else(|| {
            TesseraError::ConfigError(format!(
                "Dense encoder requires pooling_strategy to be configured for model '{model_name}'"
            ))
        })?;

        // Convert from registry PoolingStrategy to core PoolingStrategy
        let pooling_strategy = match registry_pooling {
            crate::models::registry::PoolingStrategy::Cls => PoolingStrategy::Cls,
            crate::models::registry::PoolingStrategy::Mean => PoolingStrategy::Mean,
            crate::models::registry::PoolingStrategy::Max => PoolingStrategy::Max,
        };

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
            (false, _) => vb, // No prefix (e.g., BGE models)
        };

        let model = Self::load_model(&config_str, model_vb, &model_type)
            .with_context(|| format!("Loading {model_type} model"))?;

        let normalize = model_config.normalize_embeddings;

        Ok(Self {
            model,
            tokenizer,
            device,
            config: model_config,
            pooling_strategy,
            normalize,
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
        if detector.dim.is_some() && detector.hidden_size.is_none() {
            Ok("distilbert".to_string())
        } else if detector.hidden_size.is_some() {
            Ok("bert".to_string())
        } else {
            anyhow::bail!("Could not detect model type from config")
        }
    }

    /// Detects whether the model weights use a prefix (e.g., "bert.", "distilbert.")
    ///
    /// Some models like `ColBERT` use "bert." prefix, while others like BGE don't.
    /// We detect this by checking the actual tensor names in the weights file.
    fn detect_model_prefix(weights_path: &std::path::Path) -> Result<bool> {
        let extension = weights_path.extension().and_then(|s| s.to_str());

        if extension == Some("safetensors") {
            // Check safetensors for prefix
            use safetensors::SafeTensors;

            let buffer = std::fs::read(weights_path)
                .context("Reading safetensors file for prefix detection")?;

            let tensors = SafeTensors::deserialize(&buffer).context("Deserializing safetensors")?;

            let tensor_names = tensors.names();

            // Check if any tensor starts with "bert." or "distilbert."
            // Look for the word embeddings tensor which should always exist
            for name in tensor_names {
                if name.starts_with("bert.embeddings.word_embeddings") {
                    return Ok(true); // Has "bert." prefix
                } else if name.starts_with("distilbert.embeddings.word_embeddings") {
                    return Ok(true); // Has "distilbert." prefix
                } else if name == "embeddings.word_embeddings.weight" {
                    return Ok(false); // No prefix (e.g., BGE models)
                }
            }

            // Default: assume no prefix if we can't find the embeddings
            Ok(false)
        } else {
            // For pytorch_model.bin, we need to load it to check keys
            // This is more expensive, but necessary
            let weights = candle_core::pickle::read_pth_tensor_info(weights_path, false, None)
                .context("Reading pytorch model info for prefix detection")?;

            // Check tensor names (TensorInfo has name field)
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

            // Default: assume no prefix
            Ok(false)
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
                let config: candle_transformers::models::bert::Config =
                    serde_json::from_str(config_str).context("Parsing BERT config")?;
                let model = candle_transformers::models::bert::BertModel::load(vb, &config)
                    .context("Loading BERT model")?;
                Ok(BertVariant::Bert(model))
            }
        }
    }

    /// Converts token IDs to a Candle tensor.
    fn tokens_to_tensor(&self, token_ids: &[u32], batch_size: usize) -> Result<Tensor> {
        let token_ids_i64: Vec<i64> = token_ids.iter().map(|&x| i64::from(x)).collect();
        let seq_len = token_ids.len() / batch_size;

        Tensor::from_vec(token_ids_i64, (batch_size, seq_len), &self.device)
            .context("Creating token ID tensor")
    }

    /// Applies pooling strategy to token embeddings.
    ///
    /// # Arguments
    /// * `token_embeddings` - Token embedding matrix (`seq_len` × `hidden_dim`)
    /// * `attention_mask` - Attention mask (1 = valid token, 0 = padding)
    ///
    /// # Returns
    /// Pooled embedding vector (`hidden_dim`)
    ///
    /// # Errors
    /// Returns an error if the token embeddings cannot be reshaped (shape mismatch)
    fn apply_pooling(
        &self,
        token_embeddings: &Array1<f32>,
        attention_mask: &[i64],
    ) -> Result<Array1<f32>> {
        // Convert flattened array back to 2D for pooling functions
        let seq_len = attention_mask.len();

        anyhow::ensure!(seq_len > 0, "Attention mask cannot be empty");

        let total_elements = token_embeddings.len();
        anyhow::ensure!(
            total_elements % seq_len == 0,
            "Token embeddings length ({total_elements}) must be divisible by sequence length ({seq_len}). \
             This indicates a shape mismatch between model output and attention mask."
        );

        let hidden_dim = total_elements / seq_len;
        let embeddings_2d =
            ndarray::Array2::from_shape_vec((seq_len, hidden_dim), token_embeddings.to_vec())
                .context("Failed to reshape token embeddings: ndarray shape mismatch")?;

        let pooled = match self.pooling_strategy {
            PoolingStrategy::Cls => {
                crate::utils::pooling::cls_pooling(&embeddings_2d, attention_mask)
            }
            PoolingStrategy::Mean => {
                crate::utils::pooling::mean_pooling(&embeddings_2d, attention_mask)
            }
            PoolingStrategy::Max => {
                crate::utils::pooling::max_pooling(&embeddings_2d, attention_mask)
            }
        };

        Ok(pooled)
    }

    /// Processes output embeddings: applies Matryoshka truncation and normalization.
    ///
    /// # Arguments
    /// * `embedding` - Input embedding vector
    ///
    /// # Returns
    /// Processed embedding (truncated if configured, normalized if configured)
    ///
    /// # Errors
    /// Returns an error if target dimension is invalid
    fn process_output(&self, mut embedding: Array1<f32>) -> Result<Array1<f32>> {
        // Apply Matryoshka truncation if configured
        if let Some(target_dim) = self.config.target_dimension {
            anyhow::ensure!(
                target_dim > 0,
                "Target dimension must be greater than 0, got {target_dim}"
            );
            anyhow::ensure!(
                target_dim <= embedding.len(),
                "Target dimension ({}) cannot exceed embedding dimension ({})",
                target_dim,
                embedding.len()
            );

            embedding = embedding.slice(ndarray::s![..target_dim]).to_owned();
        }

        // Apply L2 normalization if configured
        if self.normalize {
            embedding = crate::utils::normalization::l2_normalize(&embedding);
        }

        Ok(embedding)
    }

    /// Encodes a single text input to a dense embedding.
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    ///
    /// # Returns
    /// Dense embedding for the input text
    pub fn encode(&self, text: &str) -> Result<DenseEmbedding> {
        // Tokenize input
        let (token_ids, attention_mask) = self
            .tokenizer
            .encode(text, true)
            .with_context(|| format!("Tokenizing text: {text}"))?;

        // Convert to tensors
        let token_ids_tensor = self.tokens_to_tensor(&token_ids, 1)?;

        // Handle attention mask - DistilBERT in Candle uses inverted convention
        // Standard tokenizer: 1=attend, 0=pad
        // DistilBERT model: 0=attend, 1=pad
        // See: candle_transformers::models::distilbert::DistilBertModel::forward
        let attention_mask_processed: Vec<i64> = match &self.model {
            BertVariant::DistilBert(_) => {
                // Invert mask for DistilBERT
                attention_mask
                    .iter()
                    .map(|&x| i64::from(x != 1))
                    .collect()
            }
            _ => {
                // Standard BERT convention (no inversion needed)
                attention_mask.iter().map(|&x| i64::from(x)).collect()
            }
        };

        let attention_mask_tensor = Tensor::from_vec(
            attention_mask_processed.clone(),
            (1, attention_mask.len()),
            &self.device,
        )
        .context("Creating attention mask tensor")?;

        // Run model forward pass
        let output = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor)
            .context("Model forward pass")?;

        // Output shape: [1, seq_len, hidden_dim]
        // Squeeze batch dimension
        let embeddings = output.squeeze(0).context("Squeezing batch dimension")?;

        // Convert to CPU and flatten
        let embeddings_cpu = embeddings
            .to_dtype(DType::F32)
            .context("Converting to F32")?
            .to_device(&Device::Cpu)
            .context("Moving tensor to CPU")?;

        let embeddings_vec = embeddings_cpu
            .flatten_all()
            .context("Flattening tensor")?
            .to_vec1::<f32>()
            .context("Converting tensor to Vec<f32>")?;

        let embeddings_array = Array1::from_vec(embeddings_vec);

        // Apply pooling
        let pooled = self.apply_pooling(&embeddings_array, &attention_mask_processed)?;

        // Process output (Matryoshka + normalization)
        let final_embedding = self.process_output(pooled)?;

        Ok(DenseEmbedding::new(final_embedding, text.to_string()))
    }

    /// Encodes multiple text inputs in batch.
    ///
    /// # Arguments
    /// * `texts` - Slice of text inputs to encode
    ///
    /// # Returns
    /// Vector of dense embeddings, one per input
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Special case: single input
        if texts.len() == 1 {
            return Ok(vec![self.encode(texts[0])?]);
        }

        // Batch tokenization with padding
        let batch_tokenized = self
            .tokenizer
            .encode_batch(texts, true)
            .context("Batch tokenization")?;

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

        // Convert attention masks - handle DistilBERT's inverted mask convention
        // We maintain two versions:
        // 1. all_attention_masks: For the model forward pass (inverted for DistilBERT)
        // 2. attention_masks_for_pooling: For pooling logic (always standard: 1=valid, 0=pad)
        let mut all_attention_masks = Vec::with_capacity(batch_size * max_seq_len);
        let mut attention_masks_for_pooling = Vec::with_capacity(batch_size);

        for (_, attention_mask) in &batch_tokenized {
            let mut mask_for_pooling = Vec::with_capacity(max_seq_len);

            for &mask_val in attention_mask {
                // Apply inversion for DistilBERT model input
                let processed_val = match &self.model {
                    BertVariant::DistilBert(_) => {
                        // DistilBERT expects: 0=attend, 1=pad
                        i64::from(mask_val != 1)
                    }
                    _ => {
                        // Standard BERT: 1=attend, 0=pad
                        i64::from(mask_val)
                    }
                };
                all_attention_masks.push(processed_val);

                // For pooling, we always use standard convention (1=valid, 0=padding)
                mask_for_pooling.push(i64::from(mask_val));
            }

            attention_masks_for_pooling.push(mask_for_pooling);
        }

        let attention_mask_tensor =
            Tensor::from_vec(all_attention_masks, (batch_size, max_seq_len), &self.device)
                .context("Creating batch attention mask tensor")?;

        // Single forward pass for entire batch
        let batch_output = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor)
            .context("Batch forward pass")?;

        // batch_output shape: [batch_size, max_seq_len, hidden_dim]
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // Extract embeddings for this sample
            let sample_output = batch_output
                .get(i)
                .context("Extracting sample from batch")?;

            // Move to CPU and convert
            let embeddings_cpu = sample_output
                .to_dtype(DType::F32)
                .context("Converting to F32")?
                .to_device(&Device::Cpu)
                .context("Moving tensor to CPU")?;

            let embeddings_vec = embeddings_cpu
                .flatten_all()
                .context("Flattening tensor")?
                .to_vec1::<f32>()
                .context("Converting tensor to Vec<f32>")?;

            let embeddings_array = Array1::from_vec(embeddings_vec);

            // Apply pooling using the standard attention mask
            let pooled = self.apply_pooling(&embeddings_array, &attention_masks_for_pooling[i])?;

            // Process output (Matryoshka + normalization)
            let final_embedding = self.process_output(pooled)?;

            results.push(DenseEmbedding::new(final_embedding, texts[i].to_string()));
        }

        Ok(results)
    }
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
