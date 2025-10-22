//! Builder pattern for configuring Tessera embedders.
//!
//! Provides a fluent interface for constructing embedders with
//! custom configuration. Handles validation and provides clear
//! error messages for invalid combinations.
//!
//! Separate builders for dense and multi-vector embedders ensure
//! type-safe configuration (e.g., quantization only for multi-vector).
//!
//! # Example
//!
//! ```ignore
//! use tessera::api::{TesseraMultiVectorBuilder, TesseraDenseBuilder};
//!
//! // Multi-vector builder (supports quantization)
//! let mv_embedder = TesseraMultiVectorBuilder::new()
//!     .model("jina-colbert-v2")
//!     .device("metal")
//!     .dimension(96)
//!     .quantization(QuantizationConfig::Binary)
//!     .build()?;
//!
//! // Dense builder (no quantization)
//! let dense_embedder = TesseraDenseBuilder::new()
//!     .model("bge-base-en-v1.5")
//!     .device("metal")
//!     .dimension(384)
//!     .build()?;
//! ```

use crate::api::{
    TesseraDense, TesseraMultiVector, TesseraSparse, TesseraTimeSeries, TesseraVision,
};
use crate::backends::CandleBertEncoder;
use crate::encoding::dense::CandleDenseEncoder;
use crate::encoding::sparse::CandleSparseEncoder;
use crate::encoding::vision::ColPaliEncoder;
use crate::error::{Result, TesseraError};
use crate::models::{registry, ModelConfig};
use crate::quantization::BinaryQuantization;
use crate::timeseries::models::ChronosBolt;
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

/// Builder for constructing multi-vector embedders.
///
/// Provides a fluent interface with sensible defaults and
/// validation of configuration options. Supports quantization
/// for multi-vector embeddings.
pub struct TesseraMultiVectorBuilder {
    /// Model identifier from registry
    model_id: Option<String>,
    /// Target device (if None, auto-select)
    device: Option<Device>,
    /// Target embedding dimension for Matryoshka models
    dimension: Option<usize>,
    /// Quantization configuration
    quantization: Option<QuantizationConfig>,
}

impl TesseraMultiVectorBuilder {
    /// Create a new builder with default configuration.
    ///
    /// All fields are initially None, requiring at minimum a model to be set.
    #[must_use]
    pub const fn new() -> Self {
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
    #[must_use]
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
    #[must_use]
    pub const fn device(mut self, device: Device) -> Self {
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
    #[must_use]
    pub const fn dimension(mut self, dimension: usize) -> Self {
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
    #[must_use]
    pub const fn quantization(mut self, quant: QuantizationConfig) -> Self {
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
    /// 5. Creates a `ModelConfig` from the registry information
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
    /// let embedder = TesseraMultiVectorBuilder::new()
    ///     .model("colbert-v2")
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<TesseraMultiVector> {
        // Validate model ID was provided
        let model_id = self.model_id.ok_or_else(|| {
            TesseraError::ConfigError(
                "Model ID must be specified. Use .model(\"model-id\")".to_string(),
            )
        })?;

        // Look up model in registry
        let model_info =
            registry::get_model(&model_id).ok_or_else(|| TesseraError::ModelNotFound {
                model_id: model_id.clone(),
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
                TesseraError::DeviceError(format!("Failed to auto-select device: {e}"))
            })?
        };

        // Create ModelConfig
        let config = if let Some(dim) = self.dimension {
            // Use specific dimension (Matryoshka)
            ModelConfig::from_registry_with_dimension(&model_id, dim).map_err(|e| {
                TesseraError::ConfigError(format!(
                    "Failed to create config for model '{model_id}' with dimension {dim}: {e}"
                ))
            })?
        } else {
            // Use default dimension
            ModelConfig::from_registry(&model_id).map_err(|e| {
                TesseraError::ConfigError(format!(
                    "Failed to create config for model '{model_id}': {e}"
                ))
            })?
        };

        // Create encoder
        let encoder =
            CandleBertEncoder::new(config, device).map_err(|e| TesseraError::ModelLoadError {
                model_id: model_id.clone(),
                source: e,
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

        // Create TesseraMultiVector instance
        Ok(TesseraMultiVector::from_encoder(
            encoder, model_id, quantizer,
        ))
    }
}

impl Default for TesseraMultiVectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Dense Builder
// ============================================================================

/// Builder for constructing dense single-vector embedders.
///
/// Provides a fluent interface with sensible defaults and
/// validation of configuration options. Dense encoders do not
/// support quantization (use multi-vector for that).
pub struct TesseraDenseBuilder {
    /// Model identifier from registry
    model_id: Option<String>,
    /// Target device (if None, auto-select)
    device: Option<Device>,
    /// Target embedding dimension for Matryoshka models
    dimension: Option<usize>,
}

impl TesseraDenseBuilder {
    /// Create a new builder with default configuration.
    ///
    /// All fields are initially None, requiring at minimum a model to be set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model_id: None,
            device: None,
            dimension: None,
        }
    }

    /// Set the model identifier.
    ///
    /// This is the only required field. The model ID must exist in the registry
    /// and must be a dense model type.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from registry (e.g., "bge-base-en-v1.5", "nomic-embed-text-v1")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = TesseraDenseBuilder::new()
    ///     .model("bge-base-en-v1.5");
    /// ```
    #[must_use]
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
    /// let builder = TesseraDenseBuilder::new()
    ///     .model("bge-base-en-v1.5")
    ///     .device(Device::Cpu);
    /// ```
    #[must_use]
    pub const fn device(mut self, device: Device) -> Self {
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
    /// // If model supports Matryoshka dimensions
    /// let builder = TesseraDenseBuilder::new()
    ///     .model("nomic-embed-text-v1.5")
    ///     .dimension(256);  // Use 256 instead of default 768
    /// ```
    #[must_use]
    pub const fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Build the configured dense embedder.
    ///
    /// This method:
    /// 1. Validates that a model ID was provided
    /// 2. Looks up the model in the registry
    /// 3. Validates the model is a dense type
    /// 4. Validates the dimension (if specified) against model's supported dimensions
    /// 5. Auto-selects device if not specified
    /// 6. Creates a `ModelConfig` from the registry information
    /// 7. Initializes the dense encoder
    /// 8. Wraps it in a `TesseraDense` instance
    ///
    /// # Returns
    ///
    /// Initialized `TesseraDense` instance ready for encoding.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No model ID was provided
    /// - Model is not found in the registry
    /// - Model is not a dense type
    /// - Dimension is specified but not supported by the model
    /// - Device initialization fails
    /// - Model loading fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedder = TesseraDenseBuilder::new()
    ///     .model("bge-base-en-v1.5")
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<TesseraDense> {
        // Validate model ID was provided
        let model_id = self.model_id.ok_or_else(|| {
            TesseraError::ConfigError(
                "Model ID must be specified. Use .model(\"model-id\")".to_string(),
            )
        })?;

        // Look up model in registry
        let model_info =
            registry::get_model(&model_id).ok_or_else(|| TesseraError::ModelNotFound {
                model_id: model_id.clone(),
            })?;

        // Validate model type is Dense
        if model_info.model_type != registry::ModelType::Dense {
            return Err(TesseraError::ConfigError(format!(
                "Model '{}' is not a dense model (type: {:?}). Use TesseraMultiVectorBuilder for multi-vector models.",
                model_id, model_info.model_type
            )));
        }

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
                TesseraError::DeviceError(format!("Failed to auto-select device: {e}"))
            })?
        };

        // Create ModelConfig
        let config = if let Some(dim) = self.dimension {
            // Use specific dimension (Matryoshka)
            ModelConfig::from_registry_with_dimension(&model_id, dim).map_err(|e| {
                TesseraError::ConfigError(format!(
                    "Failed to create config for model '{model_id}' with dimension {dim}: {e}"
                ))
            })?
        } else {
            // Use default dimension
            ModelConfig::from_registry(&model_id).map_err(|e| {
                TesseraError::ConfigError(format!(
                    "Failed to create config for model '{model_id}': {e}"
                ))
            })?
        };

        // Create dense encoder
        let encoder =
            CandleDenseEncoder::new(config, device).map_err(|e| TesseraError::ModelLoadError {
                model_id: model_id.clone(),
                source: e,
            })?;

        // Create TesseraDense instance
        Ok(TesseraDense::from_encoder(encoder, model_id))
    }
}

impl Default for TesseraDenseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Sparse Builder
// ============================================================================

/// Builder for constructing sparse embedders with advanced configuration.
///
/// Provides a fluent interface with sensible defaults and
/// validation of configuration options. Sparse encoders produce
/// vocabulary-space sparse embeddings for SPLADE-style models.
pub struct TesseraSparseBuilder {
    /// Model identifier from registry
    model_id: Option<String>,
    /// Target device (if None, auto-select)
    device: Option<Device>,
}

impl TesseraSparseBuilder {
    /// Create a new builder with default configuration.
    ///
    /// All fields are initially None, requiring at minimum a model to be set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model_id: None,
            device: None,
        }
    }

    /// Set the model identifier.
    ///
    /// This is the only required field. The model ID must exist in the registry
    /// and must be a sparse model type.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from registry (e.g., "splade-cocondenser", "splade-pp-en-v1")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = TesseraSparseBuilder::new()
    ///     .model("splade-cocondenser");
    /// ```
    #[must_use]
    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
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
    /// let builder = TesseraSparseBuilder::new()
    ///     .model("splade-cocondenser")
    ///     .device(Device::Cpu);
    /// ```
    #[must_use]
    pub const fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Build the configured sparse embedder.
    ///
    /// This method:
    /// 1. Validates that a model ID was provided
    /// 2. Looks up the model in the registry
    /// 3. Validates the model is a sparse type
    /// 4. Auto-selects device if not specified
    /// 5. Creates a `ModelConfig` from the registry information
    /// 6. Initializes the sparse encoder (BERT + MLM head)
    /// 7. Wraps it in a `TesseraSparse` instance
    ///
    /// # Returns
    ///
    /// Initialized `TesseraSparse` instance ready for encoding.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No model ID was provided
    /// - Model is not found in the registry
    /// - Model is not a sparse type
    /// - Device initialization fails
    /// - Model loading fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedder = TesseraSparseBuilder::new()
    ///     .model("splade-cocondenser")
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<TesseraSparse> {
        // Validate model ID was provided
        let model_id = self.model_id.ok_or_else(|| {
            TesseraError::ConfigError(
                "Model ID must be specified. Use .model(\"model-id\")".to_string(),
            )
        })?;

        // Look up model in registry
        let model_info =
            registry::get_model(&model_id).ok_or_else(|| TesseraError::ModelNotFound {
                model_id: model_id.clone(),
            })?;

        // Validate model type is Sparse
        if model_info.model_type != registry::ModelType::Sparse {
            return Err(TesseraError::ConfigError(format!(
                "Model '{}' is type '{:?}', not Sparse. Use TesseraDense or TesseraMultiVector for this model.",
                model_id, model_info.model_type
            )));
        }

        // Get or auto-select device
        let device = if let Some(dev) = self.device {
            dev
        } else {
            crate::backends::candle::get_device().map_err(|e| {
                TesseraError::DeviceError(format!("Failed to auto-select device: {e}"))
            })?
        };

        // Create ModelConfig
        let config = ModelConfig::from_registry(&model_id).map_err(|e| {
            TesseraError::ConfigError(format!(
                "Failed to create config for model '{model_id}': {e}"
            ))
        })?;

        // Create sparse encoder
        let encoder =
            CandleSparseEncoder::new(config, device).map_err(|e| TesseraError::ModelLoadError {
                model_id: model_id.clone(),
                source: e,
            })?;

        // Create TesseraSparse instance
        Ok(TesseraSparse::from_encoder(encoder, model_id))
    }
}

impl Default for TesseraSparseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Vision Builder
// ============================================================================

/// Builder for vision-language embedders with advanced configuration.
pub struct TesseraVisionBuilder {
    model_id: Option<String>,
    device: Option<Device>,
}

impl TesseraVisionBuilder {
    /// Create new vision embedder builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model_id: None,
            device: None,
        }
    }

    /// Set the model identifier.
    ///
    /// Must be a vision-language model from the registry (e.g., "colpali-v1.3-hf").
    #[must_use]
    pub fn model(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Set explicit device.
    ///
    /// If not set, auto-selects best available device (Metal > CUDA > CPU).
    #[must_use]
    pub const fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Build the vision embedder.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model ID not set
    /// - Model not found in registry
    /// - Model is not a vision-language type
    /// - Model loading fails
    pub fn build(self) -> Result<TesseraVision> {
        let model_id = self
            .model_id
            .ok_or_else(|| TesseraError::ConfigError("Model ID is required".into()))?;

        // Get model info from registry
        let model_info =
            registry::get_model(&model_id).ok_or_else(|| TesseraError::ModelNotFound {
                model_id: model_id.clone(),
            })?;

        // Validate it's a vision-language model
        if model_info.model_type != registry::ModelType::VisionLanguage {
            return Err(TesseraError::ConfigError(format!(
                "Model '{}' is type '{:?}', not VisionLanguage. Use TesseraDense/MultiVector/Sparse for this model.",
                model_id, model_info.model_type
            )));
        }

        // Select device
        let device = if let Some(dev) = self.device {
            dev
        } else {
            crate::backends::candle::get_device()?
        };

        // Create model config
        let config = ModelConfig::from_registry(&model_id)?;

        // Create encoder
        let encoder = ColPaliEncoder::new(config, device)?;

        Ok(TesseraVision::from_encoder(encoder, model_id))
    }
}

impl Default for TesseraVisionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Time Series Builder
// ============================================================================

/// Builder for time series forecasting embedders with advanced configuration.
pub struct TesseraTimeSeriesBuilder {
    model_id: Option<String>,
    device: Option<Device>,
}

impl TesseraTimeSeriesBuilder {
    /// Create new time series builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model_id: None,
            device: None,
        }
    }

    /// Set the model identifier.
    ///
    /// Must be a time series model from the registry (e.g., "chronos-bolt-small").
    #[must_use]
    pub fn model(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Set explicit device.
    ///
    /// If not set, auto-selects best available device (Metal > CUDA > CPU).
    #[must_use]
    pub const fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Build the time series forecaster.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model ID not set
    /// - Model not found in registry
    /// - Model is not a time series type
    /// - Model loading fails
    pub fn build(self) -> Result<TesseraTimeSeries> {
        let model_id = self
            .model_id
            .ok_or_else(|| TesseraError::ConfigError("Model ID is required".into()))?;

        // Get model info from registry
        let model_info =
            registry::get_model(&model_id).ok_or_else(|| TesseraError::ModelNotFound {
                model_id: model_id.clone(),
            })?;

        // Validate it's a time series model
        if model_info.model_type != registry::ModelType::Timeseries {
            return Err(TesseraError::ConfigError(format!(
                "Model '{}' is type '{:?}', not Timeseries. Use TesseraDense/MultiVector/Sparse/Vision for this model.",
                model_id, model_info.model_type
            )));
        }

        // Select device
        let device = if let Some(dev) = self.device {
            dev
        } else {
            crate::backends::candle::get_device()?
        };

        // Create encoder using from_pretrained
        let encoder = ChronosBolt::from_pretrained(model_info.huggingface_id, &device)?;

        Ok(TesseraTimeSeries::from_encoder(encoder, model_id))
    }
}

impl Default for TesseraTimeSeriesBuilder {
    fn default() -> Self {
        Self::new()
    }
}
