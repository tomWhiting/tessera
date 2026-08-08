//! Chronos Bolt: Time series foundation model using T5 architecture.
//!
//! Chronos Bolt is Amazon's production time series forecasting model that uses
//! a T5 encoder-decoder architecture with `ResidualMLP` patch embeddings.
//!
//! # Architecture
//! - T5 encoder-decoder backbone (from candle-transformers)
//! - `ResidualMLP` input patch embeddings (`patch_size=32` -> hidden=2048 -> `d_model=512`)
//! - `ResidualMLP` output patch embeddings (`d_model=512` -> hidden=2048 -> `pred_len×quantiles=576`)
//! - Continuous embeddings fed directly to `T5Stack` (NO quantization/tokenization)
//! - Decoder uses single aggregated position to produce forecast
//! - Custom preprocessing: scaling by absolute mean, patching
//! - Quantile predictions: 9 quantiles per prediction step (0.1, 0.2, ..., 0.9)
//! - Pre-trained on diverse time series datasets
//!
//! # Current Implementation Status
//!
//! Full implementation with exposed `T5Stack` components:
//! - `input_patch_embedding`: `ResidualMLP` (32 -> 2048 -> 512)
//! - `output_patch_embedding`: `ResidualMLP` (512 -> 2048 -> 576)
//! - Quantile output: [batch, `pred_len=64`, `num_quantiles=9`]
//! - T5 encoder and decoder stacks
//! - shared embedding (`vocab_size=2`)
//!
//! # Example
//! ```ignore
//! use tessera::timeseries::{ChronosBolt, ChronosBoltConfig};
//! use candle_core::{Device, Tensor};
//!
//! // Load pre-trained model
//! let device = Device::Cpu;
//! let mut model = ChronosBolt::from_pretrained("amazon/chronos-bolt-small", &device)?;
//!
//! // Get median forecast (point prediction)
//! let input = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
//! let forecast = model.forecast(&input)?; // [1, 64] - median prediction
//!
//! // Get all quantile predictions (probabilistic forecast)
//! let quantiles = model.predict_quantiles(&input)?; // [1, 64, 9]
//! ```

mod model;
mod residual_mlp;

#[cfg(test)]
mod tests;

pub use model::ChronosBolt;
pub use residual_mlp::ResidualMLP;
