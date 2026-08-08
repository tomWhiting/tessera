use crate::api::TesseraTimeSeriesBuilder;
use crate::error::{Result, TesseraError};
use crate::timeseries::models::ChronosBolt;
use candle_core::Tensor;

/// Time series embedder for Chronos Bolt forecasting.
///
/// Provides probabilistic time series forecasting using Amazon's Chronos Bolt
/// T5-based foundation model. Produces quantile predictions for uncertainty
/// quantification and point forecasts (median).
///
/// Thread-safe and can be shared across threads.
#[cfg(feature = "timeseries")]
pub struct TesseraTimeSeries {
    /// Backend encoder (`ChronosBolt` model)
    encoder: ChronosBolt,
    /// Model identifier from registry
    model_id: String,
}

#[cfg(feature = "timeseries")]
impl TesseraTimeSeries {
    /// Create a new time series forecaster with default configuration.
    ///
    /// This is the simplest way to create a forecaster - it automatically:
    /// - Looks up the model in the registry
    /// - Selects the best available device (Metal > CUDA > CPU)
    /// - Downloads the model from `HuggingFace` if needed
    /// - Initializes the T5-based forecasting model
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry (e.g., "chronos-bolt-small")
    ///
    /// # Returns
    ///
    /// Initialized forecaster ready for time series predictions.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - Model is not a time series model type
    /// - Model cannot be downloaded or loaded
    /// - Device initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraTimeSeries;
    /// use candle_core::Tensor;
    ///
    /// let forecaster = TesseraTimeSeries::new("chronos-bolt-small")?;
    /// let data = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
    /// let forecast = forecaster.forecast(&data)?;
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        TesseraTimeSeriesBuilder::new().model(model_id).build()
    }

    /// Create a builder for advanced configuration.
    ///
    /// Use this for advanced use cases like:
    /// - Specifying a custom device
    /// - Setting custom context/prediction lengths
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraTimeSeries;
    /// use candle_core::Device;
    ///
    /// let forecaster = TesseraTimeSeries::builder()
    ///     .model("chronos-bolt-small")
    ///     .device(Device::Cpu)
    ///     .build()?;
    /// ```
    #[must_use]
    pub const fn builder() -> TesseraTimeSeriesBuilder {
        TesseraTimeSeriesBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) const fn from_encoder(encoder: ChronosBolt, model_id: String) -> Self {
        Self { encoder, model_id }
    }

    /// Generate point forecast (median prediction).
    ///
    /// Returns the median quantile (50th percentile) as a point forecast.
    /// For uncertainty quantification, use `forecast_quantiles()` instead.
    ///
    /// # Arguments
    ///
    /// * `context` - Historical time series data [batch, `context_length`]
    ///
    /// # Returns
    ///
    /// Tensor of forecasted values [batch, `prediction_length`]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Context tensor has wrong shape
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let data = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
    /// let forecast = forecaster.forecast(&data)?;
    /// println!("Forecast shape: {:?}", forecast.shape());  // [1, 64]
    /// ```
    pub fn forecast(&mut self, context: &Tensor) -> Result<Tensor> {
        self.encoder
            .forecast(context)
            .map_err(|e| TesseraError::EncodingError {
                context: "Failed to generate forecast".to_string(),
                source: e,
            })
    }

    /// Generate probabilistic forecast with all quantiles.
    ///
    /// Returns predictions for all 9 quantiles (0.1, 0.2, ..., 0.9),
    /// enabling uncertainty quantification and prediction intervals.
    ///
    /// # Arguments
    ///
    /// * `context` - Historical time series data [batch, `context_length`]
    ///
    /// # Returns
    ///
    /// Tensor of quantile predictions [batch, `prediction_length`, `num_quantiles`]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Context tensor has wrong shape
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let data = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
    /// let quantiles = forecaster.forecast_quantiles(&data)?;
    /// println!("Quantiles shape: {:?}", quantiles.shape());  // [1, 64, 9]
    /// ```
    pub fn forecast_quantiles(&mut self, context: &Tensor) -> Result<Tensor> {
        self.encoder
            .predict_quantiles(context)
            .map_err(|e| TesseraError::EncodingError {
                context: "Failed to generate quantile predictions".to_string(),
                source: e,
            })
    }

    /// Get the prediction horizon length.
    ///
    /// Returns the number of timesteps forecasted.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Prediction length: {}", forecaster.prediction_length());
    /// ```
    #[must_use]
    pub const fn prediction_length(&self) -> usize {
        self.encoder.config.prediction_length
    }

    /// Get the context length.
    ///
    /// Returns the required input sequence length.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Context length: {}", forecaster.context_length());
    /// ```
    #[must_use]
    pub const fn context_length(&self) -> usize {
        self.encoder.config.context_length
    }

    /// Get the quantile levels.
    ///
    /// Returns the quantiles predicted by the model (typically [0.1, 0.2, ..., 0.9]).
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Quantiles: {:?}", forecaster.quantiles());
    /// ```
    #[must_use]
    pub fn quantiles(&self) -> &[f32] {
        &self.encoder.config.quantiles
    }

    /// Get the model identifier.
    ///
    /// Returns the model ID from the registry (e.g., "chronos-bolt-small").
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Using model: {}", forecaster.model());
    /// ```
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model_id
    }
}
