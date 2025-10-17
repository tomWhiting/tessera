//! High-level user-facing API for Tessera.
//!
//! Provides ergonomic interfaces that hide complexity while offering
//! advanced configuration options for power users. The builder pattern
//! enables progressive disclosure: simple usage is trivial, advanced
//! usage is possible.
//!
//! # Simple Usage
//!
//! ```ignore
//! use tessera::Tessera;
//!
//! let embedder = Tessera::new("colbert-v2")?;
//! let embeddings = embedder.encode("What is machine learning?")?;
//! ```
//!
//! # Advanced Usage
//!
//! ```ignore
//! use tessera::api::TesseraBuilder;
//! use tessera::quantization::BinaryQuantization;
//!
//! let embedder = TesseraBuilder::new()
//!     .model("jina-colbert-v2")
//!     .device("metal")
//!     .dimension(96)  // Matryoshka
//!     .quantization(BinaryQuantization::new())
//!     .build()?;
//! ```
//!
//! # Design Philosophy
//!
//! The API is designed around these principles:
//! - **Sensible defaults**: Common use cases require minimal code
//! - **Progressive disclosure**: Advanced features available but not required
//! - **Type safety**: Invalid configurations caught at compile time
//! - **Clear errors**: Runtime errors include actionable messages
//!
//! # Features
//!
//! - Automatic model downloading from HuggingFace Hub
//! - Device detection (CPU, Metal, CUDA)
//! - Built-in quantization support
//! - Batch encoding for efficiency
//! - Matryoshka dimension support

pub mod builder;
pub mod embedder;

pub use builder::{
    QuantizationConfig,
    TesseraDenseBuilder,
    TesseraMultiVectorBuilder,
    TesseraSparseBuilder,
    TesseraVisionBuilder,
    TesseraTimeSeriesBuilder,
};
pub use embedder::{
    QuantizedEmbeddings,
    Tessera,
    TesseraDense,
    TesseraMultiVector,
    TesseraSparse,
    TesseraVision,
    TesseraTimeSeries,
};
