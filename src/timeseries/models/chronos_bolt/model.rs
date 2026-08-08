use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::t5::{Config as T5Config, T5Stack};
use candle_transformers::models::with_tracing::Embedding;

use super::ResidualMLP;
use crate::timeseries::config::ChronosBoltConfig;
use crate::timeseries::preprocessing::{create_patches, scale_by_mean};

/// Chronos Bolt time series foundation model.
///
/// This model uses a T5 encoder-decoder architecture with `ResidualMLP`
/// patch embeddings for continuous time series processing.
pub struct ChronosBolt {
    /// Input patch embedding (`patch_size` -> `d_model`)
    input_patch_embedding: ResidualMLP,

    /// Output patch embedding (`d_model` -> `prediction_length`)
    output_patch_embedding: ResidualMLP,

    /// Shared embedding table (`vocab_size=2`)
    #[allow(dead_code)]
    shared: Arc<Embedding>,

    /// T5 encoder stack
    encoder: T5Stack,

    /// T5 decoder stack
    decoder: T5Stack,

    /// Model configuration
    pub config: ChronosBoltConfig,

    /// Device for tensor operations
    device: Device,

    /// Stored scale factors for denormalization (updated during forward pass)
    scale: Option<Tensor>,
}

impl ChronosBolt {
    /// Create `ChronosBolt` model from configuration with weights from `VarBuilder`.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `vb` - Variable builder for loading weights
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Initialized `ChronosBolt` model
    ///
    /// # Errors
    /// Returns error if model construction fails
    pub fn new(config: ChronosBoltConfig, vb: VarBuilder, device: Device) -> Result<Self> {
        config
            .validate()
            .context("Invalid ChronosBolt configuration")?;

        // Load shared embedding (vocab_size=2 for PAD and EOS tokens)
        let shared = Embedding::new(config.vocab_size, config.d_model, vb.pp("shared"))
            .context("Failed to load shared embedding")?;
        let shared = Arc::new(shared);

        // Load input patch embedding (patch_size=32 -> d_model=512)
        let input_patch_embedding = ResidualMLP::new(
            config.patch_size,  // 32
            config.d_model * 4, // 2048 (hidden_layer.weight is [2048, 32])
            config.d_model,     // 512
            vb.pp("input_patch_embedding"),
        )
        .context("Failed to load input patch embedding")?;

        // Load output patch embedding (d_model=512 -> prediction_length * num_quantiles)
        // The decoder produces a single aggregated representation, which is then
        // mapped to prediction_length * num_quantiles values (64 steps × 9 quantiles = 576)
        let output_dim = config.prediction_length * config.quantiles.len();
        let output_patch_embedding = ResidualMLP::new(
            config.d_model,     // 512
            config.d_model * 4, // 2048 (hidden_layer.weight is [2048, 512])
            output_dim,         // 576 (64 prediction steps × 9 quantiles)
            vb.pp("output_patch_embedding"),
        )
        .context("Failed to load output patch embedding")?;

        // Create T5 config
        let t5_config = T5Config {
            vocab_size: config.vocab_size,
            d_model: config.d_model,
            d_kv: config.d_kv,
            d_ff: config.d_ff,
            num_heads: config.num_heads,
            num_layers: config.num_encoder_layers,
            num_decoder_layers: Some(config.num_decoder_layers),
            dropout_rate: f64::from(config.dropout),
            is_encoder_decoder: true,
            ..Default::default()
        };

        // Load T5 encoder
        let mut encoder_cfg = t5_config.clone();
        encoder_cfg.is_decoder = false;
        encoder_cfg.use_cache = false;
        encoder_cfg.is_encoder_decoder = false;
        let encoder = T5Stack::load(false, vb.pp("encoder"), &shared, &encoder_cfg)
            .context("Failed to load T5 encoder")?;

        // Load T5 decoder
        let mut decoder_cfg = t5_config;
        decoder_cfg.is_decoder = true;
        decoder_cfg.is_encoder_decoder = false;
        decoder_cfg.num_layers = config.num_decoder_layers;
        let decoder = T5Stack::load(true, vb.pp("decoder"), &shared, &decoder_cfg)
            .context("Failed to load T5 decoder")?;

        Ok(Self {
            input_patch_embedding,
            output_patch_embedding,
            shared,
            encoder,
            decoder,
            config,
            device,
            scale: None,
        })
    }

    /// Load `ChronosBolt` model from `HuggingFace` pre-trained weights.
    ///
    /// # Arguments
    /// * `model_id` - `HuggingFace` model ID (e.g., "amazon/chronos-bolt-small")
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Loaded `ChronosBolt` model with pre-trained weights
    ///
    /// # Errors
    /// Returns error if model download or loading fails
    ///
    /// # Example
    /// ```ignore
    /// let device = Device::Cpu;
    /// let model = ChronosBolt::from_pretrained("amazon/chronos-bolt-small", &device)?;
    /// ```
    pub fn from_pretrained(model_id: &str, device: &Device) -> Result<Self> {
        // Download model files from HuggingFace
        let api = hf_hub::api::sync::Api::new().context("Failed to initialize HuggingFace API")?;
        let repo = api.model(model_id.to_string());
        let weights_path = repo
            .get("model.safetensors")
            .with_context(|| format!("Failed to download weights for {model_id}"))?;

        // Determine config based on model ID
        let config = if model_id.contains("base") {
            ChronosBoltConfig::chronos_bolt_base()
        } else {
            ChronosBoltConfig::chronos_bolt_small()
        };

        // Load weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
                .context("Failed to load model weights")?
        };

        Self::new(config, vb, device.clone())
    }

    /// Forward pass through the model returning all quantile predictions.
    ///
    /// # Arguments
    /// * `input` - Time series tensor [batch, `context_length`]
    ///
    /// # Returns
    /// Quantile predictions [batch, `prediction_length`, `num_quantiles`]
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Validate input shape
        let (_batch, length) = input.dims2().context("Input must be 2D [batch, length]")?;
        if length != self.config.context_length {
            anyhow::bail!(
                "Input length {} does not match config context_length {}",
                length,
                self.config.context_length
            );
        }

        // 1. Scale by absolute mean
        let (scaled, scale) = scale_by_mean(input).context("Failed to scale time series")?;
        self.scale = Some(scale.clone());

        // 2. Create patches [batch, num_patches, patch_size]
        let patches =
            create_patches(&scaled, self.config.patch_size).context("Failed to create patches")?;
        let (batch, num_patches, patch_size) = patches.dims3()?;

        // 3. Apply input patch embedding (continuous, NO quantization!)
        // Flatten patches: [batch, num_patches, patch_size] -> [batch * num_patches, patch_size]
        let patches_flat = patches
            .reshape((batch * num_patches, patch_size))
            .context("Failed to flatten patches")?;

        // Apply ResidualMLP: [batch * num_patches, patch_size] -> [batch * num_patches, d_model]
        let embeddings_flat = self
            .input_patch_embedding
            .forward(&patches_flat)
            .context("Failed to apply input patch embedding")?;

        // Reshape back: [batch * num_patches, d_model] -> [batch, num_patches, d_model]
        let embeddings = embeddings_flat
            .reshape((batch, num_patches, self.config.d_model))
            .context("Failed to reshape embeddings")?;

        // 4. Run T5 encoder
        // Use forward_with_embeddings to bypass the embedding layer
        let encoder_output = self
            .encoder
            .forward_with_embeddings(&embeddings, None)
            .context("Failed to encode with T5")?;
        // encoder_output: [batch, num_patches, d_model]

        // 5. Create decoder input (single position that aggregates encoder context)
        // For forecasting, we use a single learned query (zeros for now)
        let decoder_input =
            Tensor::zeros((batch, 1, self.config.d_model), DType::F32, &self.device)
                .context("Failed to create decoder input")?;

        // 6. Run T5 decoder
        // Use forward_with_embeddings to bypass the embedding layer
        let decoder_output = self
            .decoder
            .forward_with_embeddings(&decoder_input, Some(&encoder_output))
            .context("Failed to decode with T5")?;
        // decoder_output: [batch, 1, d_model]

        // 7. Apply output patch embedding to map to prediction_length * num_quantiles values
        // Squeeze the sequence dimension: [batch, 1, d_model] -> [batch, d_model]
        let decoder_squeezed = decoder_output
            .squeeze(1)
            .context("Failed to squeeze decoder output")?;

        // Apply ResidualMLP: [batch, d_model] -> [batch, prediction_length * num_quantiles]
        let output_flat = self
            .output_patch_embedding
            .forward(&decoder_squeezed)
            .context("Failed to apply output patch embedding")?;

        // 8. Reshape to [batch, prediction_length, num_quantiles]
        // The output is organized as: for each prediction step, we have 9 quantiles
        let num_quantiles = self.config.quantiles.len();
        let forecast_quantiles = output_flat
            .reshape((batch, self.config.prediction_length, num_quantiles))
            .context("Failed to reshape output to quantile predictions")?;

        // 9. Denormalize using stored scale (broadcast over prediction_length and num_quantiles)
        // scale is [batch, 1], need to reshape for broadcasting: [batch, 1, 1]
        let scale_reshaped = scale
            .unsqueeze(1)
            .context("Failed to reshape scale for broadcasting")?;

        let denormalized = forecast_quantiles
            .broadcast_mul(&scale_reshaped)
            .context("Failed to denormalize output")?;

        Ok(denormalized)
    }

    /// Get all quantile predictions.
    ///
    /// This method returns the full quantile distribution for each prediction step,
    /// allowing uncertainty quantification and probabilistic forecasting.
    ///
    /// # Arguments
    /// * `input` - Time series tensor [batch, `context_length`]
    ///
    /// # Returns
    /// Quantile predictions [batch, `prediction_length`, `num_quantiles`]
    /// where quantiles are ordered as specified in config.quantiles
    /// (typically [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ///
    /// # Errors
    /// Returns error if prediction fails
    ///
    /// # Example
    /// ```ignore
    /// let quantiles = model.predict_quantiles(&input)?; // [batch, 64, 9]
    /// let q10 = quantiles.i((.., .., 0))?; // 10th percentile
    /// let q50 = quantiles.i((.., .., 4))?; // 50th percentile (median)
    /// let q90 = quantiles.i((.., .., 8))?; // 90th percentile
    /// ```
    pub fn predict_quantiles(&mut self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    /// Get median forecast (point forecast).
    ///
    /// Extracts the 0.5 quantile (median) from the full quantile predictions,
    /// providing a single point forecast for each prediction step.
    ///
    /// # Arguments
    /// * `input` - Time series tensor [batch, `context_length`]
    ///
    /// # Returns
    /// Median forecast predictions [batch, `prediction_length`]
    ///
    /// # Errors
    /// Returns error if forecasting fails
    ///
    /// # Example
    /// ```ignore
    /// let forecast = model.forecast(&input)?; // [batch, 64]
    /// ```
    pub fn forecast(&mut self, input: &Tensor) -> Result<Tensor> {
        // Get all quantile predictions [batch, prediction_length, num_quantiles]
        let quantiles = self.forward(input)?;

        // Extract median (0.5 quantile, which is at index 4 in the standard 9-quantile setup)
        // config.quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        // So quantile index 4 corresponds to 0.5 (median)
        let median_idx = self
            .config
            .quantiles
            .iter()
            .position(|&q| (q - 0.5).abs() < 1e-6)
            .unwrap_or(self.config.quantiles.len() / 2); // fallback to middle quantile

        let median = quantiles
            .i((.., .., median_idx))
            .context("Failed to extract median quantile")?;

        Ok(median)
    }
}
