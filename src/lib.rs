//! Tessera: Multi-vector embeddings and geometric representations for Rust
//!
//! Production-ready embedding library featuring ColBERT multi-vector models,
//! time series models, and exotic geometries (hyperbolic, spherical, quaternion).
//!
//! # Features
//!
//! - Token-level ColBERT embeddings with MaxSim similarity
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
//! - **Model Registry**: Compile-time generated model metadata
//!
//! # Example
//!
//! ```no_run
//! use tessera::{
//!     backends::CandleEncoder,
//!     core::{TokenEmbedder, max_sim},
//!     models::ModelConfig,
//! };
//!
//! # fn main() -> anyhow::Result<()> {
//! // Create encoder with Candle backend
//! let config = ModelConfig::distilbert_base_uncased();
//! let device = tessera::backends::candle::get_device()?;
//! let encoder = CandleEncoder::new(config, device)?;
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

pub mod backends;
pub mod core;
pub mod models;

// Re-export commonly used types
pub use core::{max_sim, TokenEmbedder, TokenEmbeddings, Tokenizer};
pub use models::ModelConfig;

/// Model registry with compile-time generated metadata
pub mod model_registry {
    pub use crate::models::registry::*;
}
