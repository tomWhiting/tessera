//! Time series embeddings and forecasting.
//!
//! This module provides time series foundation models for both forecasting
//! and embedding extraction.
//!
//! # Models
//!
//! - **ChronosBolt**: Amazon's T5-based time series foundation model
//!   - T5 encoder-decoder architecture
//!   - Quantile-based probabilistic forecasting
//!   - Pre-trained on diverse time series datasets
//!   - 191MB (small) / 821MB (base) model sizes
//!
//! # Example: Chronos Bolt
//!
//! ```ignore
//! use tessera::timeseries::{ChronosBolt, ChronosBoltConfig};
//! use candle_core::{Device, Tensor};
//!
//! // Load pre-trained model
//! let device = Device::Cpu;
//! let model = ChronosBolt::from_pretrained("amazon/chronos-bolt-small", &device)?;
//!
//! // Forecast with quantiles
//! let input = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
//! let forecast = model.forecast(&input)?; // [1, 64] (median)
//! let quantiles = model.predict_quantiles(&input)?; // [1, 64, 9]
//! ```

pub mod config;
pub mod models;
pub mod preprocessing;

// Re-export public API
pub use config::ChronosBoltConfig;
pub use models::ChronosBolt;
