//! Configuration for Chronos Bolt time series foundation model.

use serde::{Deserialize, Serialize};

/// Configuration for Chronos Bolt time series foundation model.
///
/// Chronos Bolt uses a T5 encoder-decoder architecture with custom
/// preprocessing (scaling, patching, quantization) and quantile prediction
/// heads for probabilistic forecasting.
///
/// # Architecture
/// - T5 encoder-decoder backbone (already in candle-transformers)
/// - Custom preprocessing: scaling, patching (16 timesteps), quantization (4096 bins)
/// - 9 quantile prediction heads (0.1, 0.2, ..., 0.9)
/// - Pre-trained on diverse time series datasets
///
/// # Pre-trained Models
/// - `amazon/chronos-bolt-small`: 191MB, 6 layers
/// - `amazon/chronos-bolt-base`: 821MB, more layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronosBoltConfig {
    // T5 config
    /// Vocabulary size (4096 for Chronos, not 32128 like standard T5)
    pub vocab_size: usize,

    /// Model dimension (hidden size)
    pub d_model: usize,

    /// Key/value dimension per head
    pub d_kv: usize,

    /// Feed-forward dimension
    pub d_ff: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Number of encoder layers
    pub num_encoder_layers: usize,

    /// Number of decoder layers
    pub num_decoder_layers: usize,

    /// Dropout rate
    pub dropout: f32,

    // Time series specific
    /// Context length (input sequence length)
    pub context_length: usize,

    /// Prediction length (forecast horizon)
    pub prediction_length: usize,

    /// Patch size in timesteps
    pub patch_size: usize,

    /// Number of quantization bins
    pub num_bins: usize,

    /// Quantiles to predict (e.g., [0.1, 0.2, ..., 0.9])
    pub quantiles: Vec<f32>,
}

impl ChronosBoltConfig {
    /// Create configuration for amazon/chronos-bolt-small (191MB).
    ///
    /// This is the smaller pre-trained Chronos Bolt model.
    ///
    /// # Returns
    /// Configuration for Chronos Bolt Small
    pub fn chronos_bolt_small() -> Self {
        Self {
            vocab_size: 2, // PAD and EOS only (Bolt uses continuous patches, not discrete tokens)
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_heads: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            dropout: 0.1,
            context_length: 2048,
            prediction_length: 64,
            patch_size: 32, // Based on input_patch_embedding.hidden_layer.weight shape
            num_bins: 4096,
            quantiles: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    }

    /// Create configuration for amazon/chronos-bolt-base (821MB).
    ///
    /// This is the larger pre-trained Chronos Bolt model.
    ///
    /// # Returns
    /// Configuration for Chronos Bolt Base
    pub fn chronos_bolt_base() -> Self {
        Self {
            vocab_size: 2, // PAD and EOS only
            d_model: 768,
            d_kv: 64,
            d_ff: 3072,
            num_heads: 12,
            num_encoder_layers: 12,
            num_decoder_layers: 12,
            dropout: 0.1,
            context_length: 2048,
            prediction_length: 64,
            patch_size: 32, // Based on actual Chronos Bolt architecture
            num_bins: 4096,
            quantiles: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    }

    /// Create custom Chronos Bolt configuration.
    ///
    /// # Arguments
    /// * `context_length` - Input sequence length
    /// * `prediction_length` - Forecast horizon
    ///
    /// # Returns
    /// Customized Chronos Bolt configuration
    pub fn custom(context_length: usize, prediction_length: usize) -> Self {
        Self {
            context_length,
            prediction_length,
            ..Self::chronos_bolt_small()
        }
    }

    /// Validate configuration parameters.
    ///
    /// # Returns
    /// Ok(()) if configuration is valid
    ///
    /// # Errors
    /// Returns error if configuration is invalid
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.vocab_size > 0,
            "vocab_size must be positive, got {}",
            self.vocab_size
        );
        anyhow::ensure!(
            self.d_model > 0,
            "d_model must be positive, got {}",
            self.d_model
        );
        anyhow::ensure!(
            self.num_heads > 0,
            "num_heads must be positive, got {}",
            self.num_heads
        );
        anyhow::ensure!(
            self.context_length >= self.patch_size,
            "context_length ({}) must be >= patch_size ({})",
            self.context_length,
            self.patch_size
        );
        anyhow::ensure!(
            self.context_length % self.patch_size == 0,
            "context_length ({}) must be divisible by patch_size ({})",
            self.context_length,
            self.patch_size
        );
        anyhow::ensure!(!self.quantiles.is_empty(), "quantiles must not be empty");
        anyhow::ensure!(
            self.quantiles.iter().all(|&q| q > 0.0 && q < 1.0),
            "all quantiles must be in range (0, 1)"
        );

        Ok(())
    }
}

impl Default for ChronosBoltConfig {
    fn default() -> Self {
        Self::chronos_bolt_small()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chronos_bolt_small_config() {
        let config = ChronosBoltConfig::chronos_bolt_small();
        assert_eq!(config.vocab_size, 2);
        assert_eq!(config.d_model, 512);
        assert_eq!(config.num_encoder_layers, 6);
        assert_eq!(config.quantiles.len(), 9);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_chronos_bolt_base_config() {
        let config = ChronosBoltConfig::chronos_bolt_base();
        assert_eq!(config.vocab_size, 2);
        assert_eq!(config.d_model, 768);
        assert_eq!(config.num_encoder_layers, 12);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_chronos_bolt_custom_config() {
        let config = ChronosBoltConfig::custom(4096, 128);
        assert_eq!(config.context_length, 4096);
        assert_eq!(config.prediction_length, 128);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_chronos_bolt_invalid_context() {
        let mut config = ChronosBoltConfig::default();
        config.context_length = 100; // Not divisible by patch_size (32)
        assert!(config.validate().is_err());
    }
}
