use super::ensure_runnable_model;
use crate::api::TesseraVision;
use crate::encoding::vision::ColPaliEncoder;
use crate::error::{Result, TesseraError};
use crate::models::{registry, ModelConfig};
use crate::runtime::{resolve_registry_policy, ResourcePolicy};
use candle_core::Device;

#[cfg(test)]
mod tests;

/// Builder for vision-language embedders with advanced configuration.
pub struct TesseraVisionBuilder {
    model_id: Option<String>,
    device: Option<Device>,
    resource_policy: Option<ResourcePolicy>,
}

impl TesseraVisionBuilder {
    /// Create new vision embedder builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model_id: None,
            device: None,
            resource_policy: None,
        }
    }

    /// Set the model identifier.
    ///
    /// Must be a runnable vision-language model from the registry (currently
    /// the experimental `colpali-v1.2` adapter).
    #[must_use]
    pub fn model(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Set explicit device.
    ///
    /// If not set, auto-selects best available device (Metal > CUDA > CPU).
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
        ensure_runnable_model(model_info)?;

        // Validate it's a vision-language model
        if model_info.model_type != registry::ModelType::VisionLanguage {
            return Err(TesseraError::ConfigError(format!(
                "Model '{}' is type '{:?}', not VisionLanguage. Use TesseraDense/MultiVector/Sparse for this model.",
                model_id, model_info.model_type
            )));
        }

        let resource_policy = resolve_registry_policy(
            self.resource_policy,
            model_info.context_length,
            model_info.parameters,
        )
        .map_err(|error| {
            TesseraError::ConfigError(format!(
                "Invalid resource policy for model '{model_id}': {error}"
            ))
        })?;

        // Select device
        let device = if let Some(dev) = self.device {
            dev
        } else {
            crate::backends::candle::get_device()?
        };
        // Create model config
        let config = ModelConfig::from_registry(&model_id)?;

        // Create encoder
        let encoder = ColPaliEncoder::new_with_resource_policy(config, device, resource_policy)?;

        Ok(TesseraVision::from_encoder(encoder, model_id))
    }
}

impl Default for TesseraVisionBuilder {
    fn default() -> Self {
        Self::new()
    }
}
