//! Encoding strategies for different embedding paradigms.
//!
//! This module provides the core encoding implementations that convert
//! input data (text, images, time series) into embedding representations.
//! Each submodule handles a specific embedding paradigm:
//!
//! - [`colbert`]: Token-level multi-vector encodings with late interaction
//! - [`dense`]: Single-vector embeddings via pooling strategies
//! - [`sparse`]: Sparse vocabulary-space embeddings (SPLADE-style)
//! - [`timeseries`]: Temporal data encoding and forecasting
//! - [`vision`]: Image and visual document encoding
//!
//! The encoding layer sits between the backend (Candle/Burn) and the
//! high-level API, implementing paradigm-specific logic while remaining
//! backend-agnostic where possible.

pub mod colbert;
pub mod dense;
pub mod sparse;
pub mod timeseries;
pub mod vision;

// Re-exports for convenience
pub use colbert::ColBERTEncoding;
pub use dense::CandleDenseEncoder;
pub use sparse::CandleSparseEncoder;

#[allow(deprecated)]
pub use sparse::SparseEncoding;
