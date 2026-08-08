use super::{ensure_runnable_model, QuantizationConfig};
use crate::api::TesseraMultiVector;
use crate::backends::candle::encoder::ColbertConfig;
use crate::backends::CandleBertEncoder;
use crate::error::{Result, TesseraError};
use crate::models::{registry, ModelConfig};
use crate::quantization::BinaryQuantization;
use crate::runtime::{resolve_registry_policy_with_dtype, ModelDType, ResourcePolicy};
use candle_core::Device;

#[cfg(test)]
mod tests;

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
    /// Hard limits for model loading and text tensor allocation
    resource_policy: Option<ResourcePolicy>,
    /// Fixed query length after ColBERT mask augmentation
    query_max_length: Option<usize>,
    /// Maximum document length including ColBERT framing tokens
    document_max_length: Option<usize>,
    /// Explicit parameter dtype; F32 by default.
    dtype: ModelDType,
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
            resource_policy: None,
            query_max_length: None,
            document_max_length: None,
            dtype: ModelDType::F32,
        }
    }

    /// Set the model identifier.
    ///
    /// This is the only required field. The model ID must identify a runnable
    /// multi-vector entry in the registry.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from registry (e.g., "colbert-v2", "colbert-small")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = TesseraMultiVectorBuilder::new()
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
    /// let builder = TesseraMultiVectorBuilder::new()
    ///     .model("colbert-v2")
    ///     .device(Device::Cpu);
    /// ```
    #[must_use]
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
    /// Catalog-only models are rejected even if their metadata advertises
    /// several dimensions. At present, no runnable multi-vector registry entry
    /// advertises a selectable Matryoshka dimension.
    #[must_use]
    pub const fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Set the quantization configuration.
    ///
    /// Enables compression of output embeddings. Measure retrieval quality and
    /// query performance on the target corpus before relying on the tradeoff.
    ///
    /// # Arguments
    ///
    /// * `quant` - Quantization configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::{QuantizationConfig, TesseraMultiVectorBuilder};
    ///
    /// // Enable a packed one-bit representation
    /// let builder = TesseraMultiVectorBuilder::new()
    ///     .model("colbert-v2")
    ///     .quantization(QuantizationConfig::Binary);
    /// ```
    #[must_use]
    pub const fn quantization(mut self, quant: QuantizationConfig) -> Self {
        self.quantization = Some(quant);
        self
    }

    /// Override the conservative runtime resource limits.
    #[must_use]
    pub const fn resource_policy(mut self, resource_policy: ResourcePolicy) -> Self {
        self.resource_policy = Some(resource_policy);
        self
    }

    /// Selects the model parameter dtype.
    #[must_use]
    pub const fn dtype(mut self, dtype: ModelDType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Sets the fixed ColBERT query length, including framing and `[MASK]`
    /// augmentation. The reference default is 32 tokens.
    #[must_use]
    pub const fn query_max_length(mut self, query_max_length: usize) -> Self {
        self.query_max_length = Some(query_max_length);
        self
    }

    /// Sets the maximum ColBERT document length, including `[CLS]`, `[D]`, and
    /// `[SEP]`. The reference default is 180 tokens.
    #[must_use]
    pub const fn document_max_length(mut self, document_max_length: usize) -> Self {
        self.document_max_length = Some(document_max_length);
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
        ensure_runnable_model(model_info)?;

        if model_info.model_type != registry::ModelType::Colbert {
            return Err(TesseraError::ConfigError(format!(
                "Model '{}' is type '{:?}', not Colbert. Use the matching typed builder for this model.",
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

        let resource_policy = resolve_registry_policy_with_dtype(
            self.resource_policy,
            model_info.context_length,
            model_info.parameters,
            self.dtype,
        )
        .map_err(|error| {
            TesseraError::ConfigError(format!(
                "Invalid resource policy for model '{model_id}': {error}"
            ))
        })?;

        let colbert_config = ColbertConfig::resolve(
            self.query_max_length,
            self.document_max_length,
            &resource_policy,
        )
        .map_err(|error| {
            TesseraError::ConfigError(format!(
                "Invalid ColBERT role lengths for model '{model_id}': {error}"
            ))
        })?;

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
        let encoder = CandleBertEncoder::new_with_dtype_and_colbert_config(
            config,
            device,
            self.dtype,
            resource_policy,
            colbert_config,
        )
        .map_err(|e| TesseraError::ModelLoadError {
            model_id: model_id.clone(),
            source: e,
        })?;

        // Create quantizer if requested
        let quantizer = match self.quantization.unwrap_or(QuantizationConfig::None) {
            QuantizationConfig::Binary => Some(BinaryQuantization::new()),
            QuantizationConfig::None => None,
        };

        // Create TesseraMultiVector instance
        Ok(TesseraMultiVector::from_encoder(
            encoder,
            model_id,
            quantizer,
            resource_policy.conservative_batch_size(),
            resource_policy,
        ))
    }
}

impl Default for TesseraMultiVectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
