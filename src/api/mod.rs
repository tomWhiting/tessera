//! High-level user-facing API for Tessera.
//!
//! Provides typed facades and builders for dense, multi-vector, sparse, and
//! vision-language embeddings. Registry entries without a compatible runtime
//! adapter remain discoverable but are rejected before model loading.
//!
//! # Simple Usage
//!
//! ```ignore
//! use tessera::Tessera;
//!
//! let embedder = Tessera::new("colbert-v2")?;
//! let tessera::Tessera::MultiVector(embedder) = embedder else {
//!     unreachable!("colbert-v2 is registered as a multi-vector model");
//! };
//! let embeddings = embedder.encode("What is machine learning?")?;
//! ```
//!
//! # Advanced Usage
//!
//! ```ignore
//! use tessera::{QuantizationConfig, TesseraMultiVectorBuilder};
//!
//! let embedder = TesseraMultiVectorBuilder::new()
//!     .model("colbert-v2")
//!     .quantization(QuantizationConfig::Binary)
//!     .build()?;
//! ```
//!
//! # Design Philosophy
//!
//! The API is designed around these principles:
//! - **Sensible defaults**: Common use cases require minimal code
//! - **Progressive disclosure**: Advanced features available but not required
//! - **Typed configuration**: Paradigm-specific options live on typed builders
//! - **Early validation**: Catalog-only and over-budget configurations fail before loading
//!
//! # Features
//!
//! - Automatic model downloading from `HuggingFace` Hub
//! - Device detection (CPU, Metal, CUDA)
//! - Binary quantization for multi-vector output
//! - Batch encoding for efficiency
//! - Matryoshka dimension support

pub mod builder;
pub mod embedder;

pub use builder::{
    QuantizationConfig, TesseraDenseBuilder, TesseraMultiVectorBuilder, TesseraSparseBuilder,
    TesseraVisionBuilder,
};
pub use embedder::{
    QuantizedEmbeddings, Tessera, TesseraDense, TesseraMultiVector, TesseraSparse, TesseraVision,
};
