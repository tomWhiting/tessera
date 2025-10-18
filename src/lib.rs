#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::trivially_copy_pass_by_ref,
    clippy::needless_pass_by_value,
    clippy::if_same_then_else,
    clippy::needless_lifetimes,
    clippy::unused_self,
    clippy::doc_markdown,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::clone_on_copy,
    clippy::manual_range_contains,
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::missing_const_for_fn,
    clippy::unreadable_literal,
    clippy::cloned_ref_to_slice_refs,
    clippy::branches_sharing_code,
    clippy::match_wildcard_for_single_variants,
    clippy::wrong_self_convention,
    clippy::use_self,
    clippy::struct_field_names,
    clippy::should_implement_trait,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop
)]

//! Tessera: Multi-vector embeddings and geometric representations for Rust
//!
//! Production-ready embedding library featuring `ColBERT` multi-vector models,
//! time series models, and exotic geometries (hyperbolic, spherical, quaternion).
//!
//! # Features
//!
//! - Token-level `ColBERT` embeddings with `MaxSim` similarity
//! - GPU acceleration (Metal on Apple Silicon, CUDA)
//! - Build-time model registry with type-safe access
//! - Matryoshka dimension support for flexible embedding sizes
//! - Pure Rust (single binary, no Python runtime)
//!
//! # Architecture
//!
//! - **Core**: Abstract types and traits for embeddings and similarity
//! - **Backends**: Pluggable backend implementations (Candle, Burn)
//! - **Models**: Model configuration and loading utilities
//! - **Encoding**: Paradigm-specific encoding strategies (`ColBERT`, dense, sparse, etc.)
//! - **Quantization**: Compression methods (binary, int8, int4)
//! - **API**: High-level user-facing interface with builder pattern
//! - **Bindings**: Language bindings (Python, WebAssembly)
//! - **Model Registry**: Compile-time generated model metadata
//!
//! # Example
//!
//! ```no_run
//! use tessera::{
//!     backends::CandleBertEncoder,
//!     core::TokenEmbedder,
//!     models::ModelConfig,
//!     utils::similarity::max_sim,
//! };
//!
//! # fn main() -> anyhow::Result<()> {
//! // Create encoder with Candle backend
//! let config = ModelConfig::distilbert_base_uncased();
//! let device = tessera::backends::candle::get_device()?;
//! let encoder = CandleBertEncoder::new(config, device)?;
//!
//! // Encode query and document
//! let query = encoder.encode("What is machine learning?")?;
//! let document = encoder.encode("Machine learning is a subset of AI")?;
//!
//! // Compute similarity
//! let score = max_sim(&query, &document)?;
//! println!("Similarity: {}", score);
//! # Ok(())
//! # }
//! ```

pub mod api;
pub mod backends;
pub mod bindings;
pub mod core;
pub mod encoding;
pub mod error;
pub mod models;
pub mod quantization;
pub mod timeseries;
pub mod utils;
pub mod vision;

// Re-export commonly used types
pub use api::{
    QuantizationConfig, QuantizedEmbeddings, Tessera, TesseraDense, TesseraDenseBuilder,
    TesseraMultiVector, TesseraMultiVectorBuilder, TesseraSparse, TesseraSparseBuilder,
    TesseraTimeSeries, TesseraTimeSeriesBuilder, TesseraVision, TesseraVisionBuilder,
};
pub use core::{TokenEmbedder, TokenEmbeddings, Tokenizer};
pub use error::{Result, TesseraError};
pub use models::ModelConfig;
pub use quantization::{multi_vector_distance, quantize_multi, BinaryQuantization, Quantization};
pub use utils::similarity::max_sim;

/// Model registry with compile-time generated metadata
pub mod model_registry {
    pub use crate::models::registry::*;
}
