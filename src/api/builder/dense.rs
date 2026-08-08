use super::ensure_runnable_model;
use crate::api::TesseraDense;
use crate::encoding::dense::CandleDenseEncoder;
use crate::error::{Result, TesseraError};
use crate::models::{registry, ModelConfig};
use crate::runtime::{resolve_registry_policy_with_dtype, ModelDType, ResourcePolicy};
use candle_core::Device;
use std::num::NonZeroUsize;

#[cfg(test)]
mod tests;

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
    /// Explicit maximum batch size for encode_batch (None = policy limit)
    batch_size: Option<usize>,
    /// Milliseconds to sleep between batches (for GPU throttling)
    yield_ms: Option<u64>,
    /// Hard limits for model loading and text tensor allocation
    resource_policy: Option<ResourcePolicy>,
    /// Explicit parameter dtype; F32 by default.
    dtype: ModelDType,
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
            batch_size: None,
            yield_ms: None,
            resource_policy: None,
            dtype: ModelDType::F32,
        }
    }

    /// Set the model identifier.
    ///
    /// This is the only required field. The model ID must exist in the registry
    /// and must be a dense model type.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from registry (e.g., "bge-base-en-v1.5", "nomic-embed-v1.5")
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
    /// // nomic-embed-v1.5 exposes registered Matryoshka dimensions.
    /// let builder = TesseraDenseBuilder::new()
    ///     .model("nomic-embed-v1.5")
    ///     .dimension(256);  // Use 256 instead of default 768
    /// ```
    #[must_use]
    pub const fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Set the maximum batch size for `encode_batch`.
    ///
    /// When encoding many texts, they will be processed in chunks of this size.
    /// This helps prevent GPU memory exhaustion and allows the system to remain
    /// responsive during long encoding operations.
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum number of texts in one forward pass. Must be greater than zero.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = TesseraDenseBuilder::new()
    ///     .model("bge-base-en-v1.5")
    ///     .batch_size(8);  // Process 8 texts at a time
    /// ```
    #[must_use]
    pub const fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Set milliseconds to yield between batches.
    ///
    /// When processing multiple batches, sleep for this duration between each batch.
    /// This prevents GPU saturation and keeps the system responsive (especially
    /// important on macOS where Metal shares GPU with the display).
    ///
    /// # Arguments
    ///
    /// * `ms` - Milliseconds to sleep between batches (0 = no sleep)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = TesseraDenseBuilder::new()
    ///     .model("bge-base-en-v1.5")
    ///     .batch_size(4)
    ///     .yield_ms(50);  // 50ms pause between batches
    /// ```
    #[must_use]
    pub const fn yield_ms(mut self, ms: u64) -> Self {
        self.yield_ms = Some(ms);
        self
    }

    /// Override the conservative runtime resource limits.
    ///
    /// The sequence limit cannot exceed the selected model's registered
    /// context length. Model-size limits are checked before model download.
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
    /// - Batch size is zero
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

        // Validate before loading the model and preserve the invariant needed by slice::chunks.
        let batch_size = self
            .batch_size
            .map(|size| {
                NonZeroUsize::new(size).ok_or_else(|| {
                    TesseraError::ConfigError(
                        "Batch size must be greater than zero. Use .batch_size(1) or larger"
                            .to_string(),
                    )
                })
            })
            .transpose()?;

        // Look up model in registry
        let model_info =
            registry::get_model(&model_id).ok_or_else(|| TesseraError::ModelNotFound {
                model_id: model_id.clone(),
            })?;
        ensure_runnable_model(model_info)?;

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

        if let Some(batch_size) = batch_size {
            resource_policy
                .validate_batch(batch_size.get(), 0)
                .map_err(|error| {
                    TesseraError::ConfigError(format!(
                        "Invalid dense batch size for model '{model_id}': {error}"
                    ))
                })?;
        }
        let batch_size = batch_size.or_else(|| resource_policy.conservative_batch_size());

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
        let encoder = CandleDenseEncoder::new_with_dtype_and_resource_policy(
            config,
            device,
            self.dtype,
            resource_policy,
        )
        .map_err(|e| TesseraError::ModelLoadError {
            model_id: model_id.clone(),
            source: e,
        })?;

        // Create TesseraDense instance with batch options
        Ok(TesseraDense::from_encoder_with_options(
            encoder,
            model_id,
            batch_size,
            self.yield_ms,
            resource_policy,
        ))
    }
}

impl Default for TesseraDenseBuilder {
    fn default() -> Self {
        Self::new()
    }
}
