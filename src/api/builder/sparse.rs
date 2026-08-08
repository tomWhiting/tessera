use super::ensure_runnable_model;
use crate::api::TesseraSparse;
use crate::encoding::sparse::CandleSparseEncoder;
use crate::error::{Result, TesseraError};
use crate::models::{registry, ModelConfig};
use crate::runtime::{resolve_registry_policy_with_dtype, ModelDType, ResourcePolicy};
use candle_core::Device;

#[cfg(test)]
mod tests;

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
    /// Hard limits for model loading and text tensor allocation
    resource_policy: Option<ResourcePolicy>,
    /// Explicit parameter dtype; F32 by default.
    dtype: ModelDType,
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
            resource_policy: None,
            dtype: ModelDType::F32,
        }
    }

    /// Set the model identifier.
    ///
    /// This is the only required field. The model ID must exist in the registry
    /// and must be a sparse model type.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from registry (e.g., "splade-pp-en-v1", "splade-pp-en-v2")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = TesseraSparseBuilder::new()
    ///     .model("splade-pp-en-v1");
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
    ///     .model("splade-pp-en-v1")
    ///     .device(Device::Cpu);
    /// ```
    #[must_use]
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
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
    ///     .model("splade-pp-en-v1")
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
        ensure_runnable_model(model_info)?;

        // Validate model type is Sparse
        if model_info.model_type != registry::ModelType::Sparse {
            return Err(TesseraError::ConfigError(format!(
                "Model '{}' is type '{:?}', not Sparse. Use TesseraDense or TesseraMultiVector for this model.",
                model_id, model_info.model_type
            )));
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
        let encoder = CandleSparseEncoder::new_with_dtype_and_resource_policy(
            config,
            device,
            self.dtype,
            resource_policy,
        )
        .map_err(|e| TesseraError::ModelLoadError {
            model_id: model_id.clone(),
            source: e,
        })?;

        // Create TesseraSparse instance
        Ok(TesseraSparse::from_encoder(
            encoder,
            model_id,
            resource_policy,
        ))
    }
}

impl Default for TesseraSparseBuilder {
    fn default() -> Self {
        Self::new()
    }
}
