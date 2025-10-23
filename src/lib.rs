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

//! Tessera: Multi-paradigm embeddings for semantic search and representation learning
//!
//! A production-ready embedding library that combines five complementary approaches to semantic
//! representation: dense single-vector embeddings, multi-vector token embeddings via ColBERT,
//! sparse learned representations via SPLADE, vision-language embeddings via ColPali, and
//! probabilistic time series forecasting via Chronos Bolt.
//!
//! Tessera supports 23+ production models with native GPU acceleration (Metal on Apple Silicon,
//! CUDA on NVIDIA), comprehensive batch processing, and binary quantization for 32x compression
//! of multi-vector embeddings.
//!
//! # Features
//!
//! - **Five embedding paradigms** for different use cases and trade-offs
//! - **Multi-vector (ColBERT)** with `MaxSim` late interaction and binary quantization (32x compression)
//! - **Dense embeddings** with 8 state-of-the-art models from BGE, Nomic, GTE, Jina, Qwen, and Snowflake
//! - **Sparse embeddings (SPLADE)** with 99% sparsity for interpretable keyword search
//! - **Vision-language embeddings (ColPali)** for OCR-free document understanding with PDFs and images
//! - **Time series forecasting (Chronos Bolt)** with uncertainty quantification across 9 quantile levels
//! - **GPU acceleration** with automatic device selection (Metal > CUDA > CPU)
//! - **Batch processing** for 5-10x throughput improvements
//! - **Matryoshka dimensions** for variable embedding sizes without model reloading
//! - **Type-safe API** preventing mismatched embedding operations at compile time
//! - **Rust and Python support** with PyO3 bindings for seamless NumPy interoperability
//!
//! # Quick Start
//!
//! Dense embeddings for semantic similarity:
//!
//! ```no_run
//! use tessera::TesseraDense;
//!
//! # fn main() -> anyhow::Result<()> {
//! let embedder = TesseraDense::new("bge-base-en-v1.5")?;
//!
//! let query_embedding = embedder.encode("What is machine learning?")?;
//! let doc_embedding = embedder.encode("Machine learning is a subset of artificial intelligence")?;
//!
//! let similarity = embedder.similarity(
//!     "What is machine learning?",
//!     "Machine learning is a subset of artificial intelligence"
//! )?;
//! println!("Similarity score: {:.4}", similarity);
//! # Ok(())
//! # }
//! ```
//!
//! Multi-vector embeddings for precise phrase matching:
//!
//! ```no_run
//! use tessera::TesseraMultiVector;
//!
//! # fn main() -> anyhow::Result<()> {
//! let embedder = TesseraMultiVector::new("colbert-v2")?;
//!
//! let query_vectors = embedder.encode("machine learning algorithms")?;
//! let doc_vectors = embedder.encode("algorithms for machine learning applications")?;
//!
//! let similarity = embedder.similarity_multi_vectors(&query_vectors, &doc_vectors)?;
//! println!("MaxSim score: {:.4}", similarity);
//! # Ok(())
//! # }
//! ```
//!
//! # Embedding Paradigms
//!
//! ## Dense Embeddings
//!
//! Compress text into a single fixed-size vector through pooling operations. Best for broad semantic
//! meaning and efficient similarity search via cosine distance or dot product.
//!
//! **Use cases:** Semantic search, clustering, recommendation systems, topic modeling.
//!
//! **Models:** BGE Base/Large, Nomic Embed, GTE, Qwen, Jina, Snowflake Arctic.
//!
//! ## Multi-Vector Embeddings (ColBERT)
//!
//! Preserve token-level granularity with a vector per token. Similarity computed via `MaxSim`
//! (maximum similarity between any token pair). Exceptional for exact phrase matching and
//! information retrieval.
//!
//! **Use cases:** Precise search, question answering, passage retrieval, academic search.
//!
//! **Features:** Binary quantization for 32x compression, Matryoshka dimension support.
//!
//! **Models:** ColBERT v2, ColBERT Small, Jina ColBERT v2/v3, Nomic BERT MultiVector.
//!
//! ## Sparse Embeddings (SPLADE)
//!
//! Map text to vocabulary space producing 99% sparse representations. Each dimension corresponds
//! to a token with learned context-aware weights. Enables efficient inverted index search while
//! maintaining semantic expansion.
//!
//! **Use cases:** Interpretable search, hybrid retrieval, legal/medical document search, keyword expansion.
//!
//! **Models:** SPLADE CoCondenser, SPLADE++ EN v1.
//!
//! ## Vision-Language Embeddings (ColPali)
//!
//! Encode images and PDFs directly at the patch level for OCR-free document understanding. Processes
//! visual content through a vision transformer and projects into the same embedding space as text,
//! enabling late interaction search over documents containing tables, figures, and handwriting.
//!
//! **Use cases:** Document search, invoice processing, diagram retrieval, visual question answering.
//!
//! **Models:** ColPali v1.3-hf, ColPali v1.2.
//!
//! ## Time Series Forecasting (Chronos Bolt)
//!
//! Zero-shot probabilistic forecasting through continuous-time embeddings. Generates forecasts with
//! nine quantile levels for uncertainty quantification without task-specific fine-tuning.
//!
//! **Use cases:** Demand forecasting, anomaly detection, capacity planning, financial prediction.
//!
//! **Models:** Chronos Bolt Small.
//!
//! # Supported Models
//!
//! Tessera provides 23+ production models across five paradigms:
//!
//! ## Multi-Vector (9 models)
//! - ColBERT v2 (110M parameters, 128 dimensions)
//! - ColBERT Small (33M parameters, 96 dimensions)
//! - Jina ColBERT v2 (137M parameters, 768 dimensions with Matryoshka)
//! - Jina ColBERT v3 (250M parameters, 768 dimensions)
//! - Nomic BERT MultiVector (137M parameters, 768 dimensions)
//!
//! ## Dense (8 models)
//! - BGE Base/Large EN v1.5 (110M/335M parameters, 768/1024 dimensions)
//! - Nomic Embed Text v1 (137M parameters, 768 dimensions)
//! - GTE Large EN v1.5 (335M parameters, 1024 dimensions)
//! - Qwen 2.5 0.5B (100M parameters, 1024 dimensions)
//! - Qwen3 Embedding 8B/4B/0.6B (8B/4B/600M parameters, 4096 dimensions)
//! - Jina Embeddings v3 (570M parameters, 1024 dimensions)
//! - Snowflake Arctic Embed Large (735M parameters, 1024 dimensions)
//!
//! ## Sparse (4 models)
//! - SPLADE CoCondenser (110M parameters, 30522 vocabulary)
//! - SPLADE++ EN v1 (110M parameters, 30522 vocabulary)
//!
//! ## Vision-Language (2 models)
//! - ColPali v1.3-hf (3B parameters, 128 dimensions, 1024 patches)
//! - ColPali v1.2 (3B parameters, 128 dimensions, 1024 patches)
//!
//! ## Time Series (1 model)
//! - Chronos Bolt Small (48M parameters)
//!
//! # Advanced Usage
//!
//! ## Builder Pattern for Configuration
//!
//! ```no_run
//! use tessera::TesseraMultiVector;
//!
//! # fn main() -> anyhow::Result<()> {
//! let embedder = TesseraMultiVector::builder()
//!     .model("jina-colbert-v2")
//!     .dimension(96)  // Matryoshka support
//!     .batch_size(32)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Batch Processing
//!
//! ```no_run
//! use tessera::TesseraDense;
//!
//! # fn main() -> anyhow::Result<()> {
//! let embedder = TesseraDense::new("bge-base-en-v1.5")?;
//!
//! let texts = vec![
//!     "First document about machine learning",
//!     "Second document about neural networks",
//!     "Third document about deep learning",
//! ];
//!
//! let embeddings = embedder.encode_batch(&texts)?;
//! println!("Encoded {} documents", embeddings.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Binary Quantization for Multi-Vector
//!
//! ```no_run
//! use tessera::{TesseraMultiVector, QuantizationConfig};
//!
//! # fn main() -> anyhow::Result<()> {
//! let embedder = TesseraMultiVector::builder()
//!     .model("colbert-v2")
//!     .quantization(QuantizationConfig::Binary)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! # GPU Acceleration
//!
//! Tessera automatically selects the best available compute device with a fallback chain:
//! Metal (Apple Silicon) > CUDA (NVIDIA) > CPU. Models are cached after first load for
//! efficient repeated encoding.
//!
//! Enable GPU support with Cargo features:
//!
//! ```toml
//! [dependencies]
//! tessera = { version = "0.1", features = ["metal"] }  # Apple Silicon
//! tessera = { version = "0.1", features = ["cuda"] }   # NVIDIA GPUs
//! ```
//!
//! # Feature Flags
//!
//! - `metal` - Apple Silicon acceleration (macOS only)
//! - `cuda` - NVIDIA GPU acceleration
//! - `pdf` - PDF rendering for ColPali document processing (enabled by default)
//! - `python` - PyO3 bindings for Python support (includes timeseries)
//! - `timeseries` - Time series forecasting with Chronos Bolt (requires candle fork)
//! - `wasm` - WebAssembly bindings (experimental)
//!
//! # Architecture
//!
//! - **Core**: Abstract types and traits for embeddings and similarity metrics
//! - **Backends**: Pluggable backend implementations via Candle ML framework
//! - **Models**: Model configuration, loading, and caching utilities
//! - **Encoding**: Paradigm-specific encoding strategies for each embedding type
//! - **Quantization**: Compression methods (binary, int8, int4) with minimal quality loss
//! - **API**: High-level user-facing interface with builder pattern and type safety
//! - **Vision**: Image and PDF processing for vision-language models
//! - **TimeSeries**: Probabilistic forecasting utilities
//! - **Bindings**: Language bindings (Python via PyO3, WebAssembly)
//! - **Utils**: Common utilities (batching, pooling, normalization, similarity metrics)
//!
//! # Performance
//!
//! Typical performance on Apple M1 Max:
//!
//! - **Dense encoding:** 125 docs/sec (batch=1), 711 docs/sec (batch=32)
//! - **ColBERT encoding:** 83 docs/sec (batch=1), 410 docs/sec (batch=32)
//! - **Binary quantization:** 3,333 ops/sec
//!
//! # Benchmark Results
//!
//! - **BGE Base EN v1.5:** 63.55 BEIR Average, 85.29 MS MARCO MRR@10
//! - **ColBERT v2:** 65.12 BEIR Average, 87.43 MS MARCO MRR@10
//! - **SPLADE++ EN v1:** 61.23 BEIR Average, 86.15 MS MARCO MRR@10
//! - **Jina Embeddings v3:** 66.84 MTEB Average (#2 under 1B parameters)
//! - **Qwen3 Embedding 8B:** 70.58 MTEB Average (#1 multilingual model)
//!
//! # Error Handling
//!
//! The library uses [`Result<T>`] with [`TesseraError`] for comprehensive error handling:
//!
//! ```no_run
//! use tessera::{TesseraDense, Result};
//!
//! # fn main() -> Result<()> {
//! let embedder = TesseraDense::new("invalid-model")?;
//! # Ok(())
//! # }
//! ```
//!
//! # See Also
//!
//! - [`TesseraDense`] - Dense embedding API
//! - [`TesseraMultiVector`] - Multi-vector ColBERT API
//! - [`TesseraSparse`] - Sparse SPLADE API
//! - [`TesseraVision`] - Vision-language ColPali API
//! - [`TesseraTimeSeries`] - Time series forecasting API (requires `timeseries` feature)
//! - [`ModelConfig`] - Model configuration and registry
//! - [`QuantizationConfig`] - Quantization options

pub mod api;
pub mod backends;
pub mod bindings;
pub mod core;
pub mod encoding;
pub mod error;
pub mod models;
pub mod quantization;
#[cfg(feature = "timeseries")]
pub mod timeseries;
pub mod utils;
pub mod vision;

// Re-export commonly used types
pub use api::{
    QuantizationConfig, QuantizedEmbeddings, Tessera, TesseraDense, TesseraDenseBuilder,
    TesseraMultiVector, TesseraMultiVectorBuilder, TesseraSparse, TesseraSparseBuilder,
    TesseraVision, TesseraVisionBuilder,
};
#[cfg(feature = "timeseries")]
pub use api::{TesseraTimeSeries, TesseraTimeSeriesBuilder};
pub use core::{TokenEmbedder, TokenEmbeddings, Tokenizer};
pub use error::{Result, TesseraError};
pub use models::ModelConfig;
pub use quantization::{multi_vector_distance, quantize_multi, BinaryQuantization, Quantization};
pub use utils::similarity::max_sim;

/// Model registry with compile-time generated metadata
pub mod model_registry {
    pub use crate::models::registry::*;
}
