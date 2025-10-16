//! Builder pattern for configuring Tessera embedders.
//!
//! Provides a fluent interface for constructing embedders with
//! custom configuration. Handles validation and provides clear
//! error messages for invalid combinations.
//!
//! # Example
//!
//! ```ignore
//! use tessera::api::TesseraBuilder;
//!
//! let embedder = TesseraBuilder::new()
//!     .model("jina-colbert-v2")
//!     .device("metal")
//!     .dimension(96)
//!     .normalize(true)
//!     .build()?;
//! ```
//!
//! # Configuration Options
//!
//! - **model**: Model identifier (HuggingFace or local path)
//! - **device**: Target device (auto, cpu, metal, cuda)
//! - **dimension**: Output dimension (Matryoshka support)
//! - **quantization**: Quantization method (binary, int8, int4)
//! - **normalize**: Whether to L2-normalize embeddings
//! - **batch_size**: Default batch size for encoding

use crate::api::Tessera;
use crate::backends::CandleBertEncoder;
use crate::error::{Result, TesseraError};
use crate::models::{registry, ModelConfig};
use crate::quantization::BinaryQuantization;
use candle_core::Device;

/// Quantization configuration for embeddings.
///
/// Enables compression of embeddings for reduced memory footprint and
/// faster distance computation with minimal accuracy loss.
#[derive(Debug, Clone, Copy)]
pub enum QuantizationConfig {
    /// No quantization (full precision float32)
    None,
    /// Binary quantization (1-bit, 32x compression, 95%+ accuracy)
    ///
    /// Converts each dimension to a single bit (sign of the value).
    /// Provides maximum compression with acceptable accuracy for most
    /// retrieval tasks. Ideal for initial filtering + reranking workflows.
    Binary,
    /// Int8 quantization (8-bit, 4x compression) - Phase 2
    #[allow(dead_code)]
    Int8,
    /// Int4 quantization (4-bit, 8x compression) - Phase 2
    #[allow(dead_code)]
    Int4,
}

/// Builder for constructing Tessera embedders.
///
/// Provides a fluent interface with sensible defaults and
/// validation of configuration options.
pub struct TesseraBuilder {
    /// Model identifier from registry
    model_id: Option<String>,
    /// Target device (if None, auto-select)
    device: Option<Device>,
    /// Target embedding dimension for Matryoshka models
    dimension: Option<usize>,
    /// Quantization configuration
    quantization: Option<QuantizationConfig>,
}

impl TesseraBuilder {
    /// Create a new builder with default configuration.
    ///
    /// All fields are initially None, requiring at minimum a model to be set.
    pub fn new() -> Self {
        Self {
            model_id: None,
            device: None,
            dimension: None,
            quantization: None,
        }
    }

    /// Set the model identifier.
    ///
    /// This is the only required field. The model ID must exist in the registry.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from registry (e.g., "colbert-v2", "jina-colbert-v2")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = TesseraBuilder::new()
    ///     .model("colbert-v2");
    /// ```
    pub fn model(mut self, model_id: &str) -> Self {
        self.model_id = Some(model_id.to_string());
        self
    }

    /// Set the target device.
    ///
    /// If not set, the device will be auto-selected using the following priority:
    /// 1. Metal (on macOS with Apple Silicon)
    /// 2. CUDA (on systems with NVIDIA GPU)
    /// 3. CPU (fallback)
    ///
    /// # Arguments
    ///
    /// * `device` - Candle Device to use
    ///
    /// # Example
    ///
    /// ```ignore
    /// use candle_core::Device;
    ///
    /// let builder = TesseraBuilder::new()
    ///     .model("colbert-v2")
    ///     .device(Device::Cpu);
    /// ```
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the output dimension for Matryoshka models.
    ///
    /// Only applicable to models with Matryoshka support. The dimension
    /// must be in the model's supported dimension list.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Target embedding dimension
    ///
    /// # Example
    ///
    /// ```ignore
    /// // jina-colbert-v2 supports Matryoshka dimensions
    /// let builder = TesseraBuilder::new()
    ///     .model("jina-colbert-v2")
    ///     .dimension(128);  // Use 128 instead of default 768
    /// ```
    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Set the quantization configuration.
    ///
    /// Enables compression of embeddings for reduced memory usage and faster
    /// distance computation with minimal accuracy loss.
    ///
    /// # Arguments
    ///
    /// * `quant` - Quantization configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::{TesseraBuilder, QuantizationConfig};
    ///
    /// // Enable binary quantization (32x compression)
    /// let builder = TesseraBuilder::new()
    ///     .model("colbert-v2")
    ///     .quantization(QuantizationConfig::Binary);
    /// ```
    pub fn quantization(mut self, quant: QuantizationConfig) -> Self {
        self.quantization = Some(quant);
        self
    }

    /// Build the configured embedder.
    ///
    /// This method:
    /// 1. Validates that a model ID was provided
    /// 2. Looks up the model in the registry
    /// 3. Validates the dimension (if specified) against model's supported dimensions
    /// 4. Auto-selects device if not specified
    /// 5. Creates a ModelConfig from the registry information
    /// 6. Initializes the backend encoder
    /// 7. Wraps it in a Tessera instance
    ///
    /// # Returns
    ///
    /// Initialized Tessera instance ready for encoding.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No model ID was provided
    /// - Model is not found in the registry
    /// - Dimension is specified but not supported by the model
    /// - Device initialization fails
    /// - Model loading fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedder = TesseraBuilder::new()
    ///     .model("colbert-v2")
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<Tessera> {
        // Validate model ID was provided
        let model_id = self.model_id.ok_or_else(|| TesseraError::ConfigError(
            "Model ID must be specified. Use .model(\"model-id\")".to_string(),
        ))?;

        // Look up model in registry
        let model_info = registry::get_model(&model_id).ok_or_else(|| {
            TesseraError::ModelNotFound {
                model_id: model_id.clone(),
            }
        })?;

        // Validate dimension if specified
        if let Some(dim) = self.dimension {
            if !model_info.embedding_dim.supports_dimension(dim) {
                return Err(TesseraError::UnsupportedDimension {
                    model_id: model_id.clone(),
                    requested: dim,
                    supported: model_info.embedding_dim.supported_dimensions(),
                });
            }
        }

        // Get or auto-select device
        let device = if let Some(dev) = self.device {
            dev
        } else {
            crate::backends::candle::get_device().map_err(|e| {
                TesseraError::DeviceError(format!("Failed to auto-select device: {}", e))
            })?
        };

        // Create ModelConfig
        let config = if let Some(dim) = self.dimension {
            // Use specific dimension (Matryoshka)
            ModelConfig::from_registry_with_dimension(&model_id, dim).map_err(|e| {
                TesseraError::ConfigError(format!(
                    "Failed to create config for model '{}' with dimension {}: {}",
                    model_id, dim, e
                ))
            })?
        } else {
            // Use default dimension
            ModelConfig::from_registry(&model_id).map_err(|e| {
                TesseraError::ConfigError(format!(
                    "Failed to create config for model '{}': {}",
                    model_id, e
                ))
            })?
        };

        // Create encoder
        let encoder = CandleBertEncoder::new(config, device).map_err(|e| {
            TesseraError::ModelLoadError {
                model_id: model_id.clone(),
                source: e,
            }
        })?;

        // Create quantizer if requested
        let quantizer = match self.quantization.unwrap_or(QuantizationConfig::None) {
            QuantizationConfig::Binary => Some(BinaryQuantization::new()),
            QuantizationConfig::None => None,
            QuantizationConfig::Int8 | QuantizationConfig::Int4 => {
                return Err(TesseraError::QuantizationError(
                    "Int8/Int4 quantization not yet implemented (Phase 2)".to_string(),
                ))
            }
        };

        // Create Tessera instance
        Ok(Tessera::from_encoder(encoder, model_id, quantizer))
    }
}

impl Default for TesseraBuilder {
    fn default() -> Self {
        Self::new()
    }
}
