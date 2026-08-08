use crate::api::TesseraTimeSeries;
use crate::error::{Result, TesseraError};
use crate::models::registry;
use crate::timeseries::models::ChronosBolt;
use candle_core::Device;

/// Builder for time series forecasting embedders with advanced configuration.
#[cfg(feature = "timeseries")]
pub struct TesseraTimeSeriesBuilder {
    model_id: Option<String>,
    device: Option<Device>,
}

#[cfg(feature = "timeseries")]
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
    pub fn device(mut self, device: Device) -> Self {
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

#[cfg(feature = "timeseries")]
impl Default for TesseraTimeSeriesBuilder {
    fn default() -> Self {
        Self::new()
    }
}
