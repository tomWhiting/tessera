// Legacy pedantic/style lint debt is kept explicit while the revival replaces
// old APIs. CI promotes every lint outside this finite ratchet to an error; do
// not add entries here to make new warnings disappear.
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

//! Tessera: dense, sparse, multi-vector, and vision-language embeddings.
//!
//! Tessera is a Rust-first embedding library built on Candle. The current
//! revival focuses on retrieval representations and predictable resource use;
//! it is alpha-quality software rather than a production-ready model suite.
//!
//! # Revival status
//!
//! The generated registry contains 22 entries. Support metadata intentionally
//! distinguishes model discovery from model execution:
//!
//! - 10 entries are [`SupportTier::Experimental`]: an adapter path exists, but
//!   remote checkpoint execution and output quality still need repeatable
//!   validation.
//! - 12 entries are [`SupportTier::CatalogOnly`]: metadata is discoverable, but
//!   the current runtime has no compatible adapter.
//! - No entry is [`SupportTier::Supported`] yet.
//!
//! [`get_model`](model_registry::get_model) is catalog-complete. Use
//! [`ModelInfo::is_runnable`](model_registry::ModelInfo::is_runnable) or
//! [`runnable_models`](model_registry::runnable_models) when choosing a model
//! for execution. See the checked-in `models.json` for per-model support notes.
//!
//! # Quick start
//!
//! Creating an embedder may download model artifacts from Hugging Face. The
//! BGE path below is currently experimental, so it is suitable for smoke
//! testing rather than a stable compatibility promise.
//!
//! ```no_run
//! use tessera::TesseraDense;
//!
//! # fn main() -> tessera::Result<()> {
//! let embedder = TesseraDense::new("bge-base-en-v1.5")?;
//! let embedding = embedder.encode("A tessera is one tile in a mosaic.")?;
//! println!("{} dimensions", embedding.dim());
//! # Ok(())
//! # }
//! ```
//!
//! The public façades are separated by representation:
//!
//! - [`TesseraDense`] produces one pooled vector per text.
//! - [`TesseraSparse`] produces vocabulary-sized SPLADE-style weights.
//! - [`TesseraMultiVector`] produces token vectors for late-interaction
//!   retrieval.
//! - [`TesseraVision`] produces patch vectors from image inputs and token
//!   vectors from text queries.
//!
//! Multi-vector and vision scores use bounded-memory MaxSim. Binary
//! quantization is available as an explicit multi-vector option, but Tessera
//! does not claim a quality or throughput result without a checked benchmark.
//!
//! # Resource limits
//!
//! Tessera validates work before model input tensors are allocated. The default
//! [`ResourcePolicy`] allows at most:
//!
//! - 1 MiB of UTF-8 input per sequence, checked before tokenization;
//! - 512 tokens in one sequence, including special tokens;
//! - 16 items in one batch;
//! - 2,048 padded token cells (`items * longest sequence`);
//! - 1,048,576 attention cells (`items * longest sequence^2`); and
//! - 2 GiB of estimated F32 model parameter storage.
//!
//! These are safety limits, not model capabilities. A registered 8K model
//! remains capped at 512 tokens unless the caller opts in. A sequence limit can
//! never exceed the selected model's registered context window.
//!
//! ```no_run
//! use tessera::{ResourcePolicy, TesseraDense};
//!
//! # fn main() -> tessera::Result<()> {
//! let single_document_8k = ResourcePolicy::default()
//!     .with_max_sequence_tokens(8_192)
//!     .with_max_batch_items(1)
//!     .with_max_batch_tokens(8_192)
//!     .with_max_attention_cells(67_108_864);
//!
//! let embedder = TesseraDense::builder()
//!     .model("jina-embeddings-v2-small-en")
//!     .resource_policy(single_document_8k)
//!     .build()?;
//! # let _ = embedder;
//! # Ok(())
//! # }
//! ```
//!
//! This override only passes Tessera's request-shape preflight; it does not
//! make 8K inference safe. Full attention is quadratic. One 8,192-token item
//! permits 67,108,864 cells before multiplying by attention heads and
//! accounting for model layers, temporary tensors, allocator overhead, or
//! other process memory. It can still exhaust CPU, GPU, or Metal shared memory.
//! Measure the selected model on the target hardware before opting in.
//!
//! Raise the model-byte limit separately for a deliberate large-model load. A
//! 3B-parameter checkpoint has an approximately 12 GB F32 parameter estimate
//! before allocator overhead, so the 2 GiB default rejects it. The estimate is
//! a preflight guard, not a full peak-memory calculation.
//!
//! # CPU worker ceiling
//!
//! Before constructing the first CPU encoder, Tessera attempts to configure
//! Candle's process-global Rayon and barrier pools with a ceiling of two
//! workers (or fewer when less parallelism is available). To opt into a higher
//! ceiling, call [`configure_cpu_threads`] during single-threaded startup,
//! before constructing any Tessera builder or allowing other code to initialize
//! those pools. The first Tessera configuration call wins for the lifetime of
//! the process.
//!
//! ```no_run
//! use candle_core::Device;
//! use tessera::{configure_cpu_threads, TesseraDense};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! configure_cpu_threads(8)?;
//! let embedder = TesseraDense::builder()
//!     .model("bge-base-en-v1.5")
//!     .device(Device::Cpu)
//!     .build()?;
//! # let _ = embedder;
//! # Ok(())
//! # }
//! ```
//!
//! This setup is best-effort because an already initialized external Rayon pool
//! cannot be resized through environment variables. It is not a process-wide
//! guarantee for every thread, allocation, or accelerator operation.
//!
//! # Multi-vector example
//!
//! ```no_run
//! use tessera::TesseraMultiVector;
//!
//! # fn main() -> tessera::Result<()> {
//! let embedder = TesseraMultiVector::new("colbert-small")?;
//! let score = embedder.similarity(
//!     "late interaction retrieval",
//!     "retrieval with token-level late interaction",
//! )?;
//! println!("MaxSim score: {score:.4}");
//! # Ok(())
//! # }
//! ```
//!
//! `colbert-small` is also currently experimental.
//!
//! # Vision and PDF scope
//!
//! The high-level [`TesseraVision::encode_document`] method currently accepts
//! an image path. The default `pdf` feature provides rendering plumbing used by
//! lower layers; it is not yet a high-level PDF-document façade. ColPali is a
//! large experimental path and requires an explicit model-memory override.
//!
//! # Time-series scope
//!
//! Chronos and TimesFM remain catalog-only. The retained Chronos runtime used
//! hidden-state T5 APIs from an old Candle fork and is quarantined under stock
//! Candle 0.11. The `timeseries` Cargo feature exposes generic time-series
//! types only; it does not activate a forecasting model or Python façade. See
//! `docs/legacy/TIMESERIES.md` for the exact incompatibilities and reactivation
//! criteria.
//!
//! # Cargo features
//!
//! - `pdf` (default): PDF rendering plumbing; no public document façade yet.
//! - `metal`: Apple Metal support in Candle.
//! - `cuda`: NVIDIA CUDA support in Candle.
//! - `python`: PyO3 bindings for active embedding façades.
//! - `timeseries`: generic time-series core types only.
//! - `wasm`: reserved for experimental WebAssembly work; bindings are not yet
//!   implemented.
//!
//! # Error handling
//!
//! Public operations return [`Result<T>`] with structured [`TesseraError`]
//! values. Resource-limit errors include both the measured and allowed values.
//!
//! [`SupportTier::Experimental`]: model_registry::SupportTier::Experimental
//! [`SupportTier::CatalogOnly`]: model_registry::SupportTier::CatalogOnly
//! [`SupportTier::Supported`]: model_registry::SupportTier::Supported

pub mod api;
pub mod backends;
pub mod bindings;
pub mod core;
pub mod encoding;
pub mod error;
pub mod models;
pub mod quantization;
pub mod runtime;
pub mod utils;
pub mod vision;

// Re-export commonly used types
pub use api::{
    QuantizationConfig, QuantizedEmbeddings, Tessera, TesseraDense, TesseraDenseBuilder,
    TesseraMultiVector, TesseraMultiVectorBuilder, TesseraSparse, TesseraSparseBuilder,
    TesseraVision, TesseraVisionBuilder,
};
pub use core::{TokenEmbedder, TokenEmbeddings, Tokenizer};
pub use error::{Result, TesseraError};
pub use models::ModelConfig;
pub use quantization::{multi_vector_distance, quantize_multi, BinaryQuantization, Quantization};
pub use runtime::{
    configure_cpu_threads, CpuThreadConfig, CpuThreadConfigError, ResourcePolicy,
    ResourcePolicyError,
};
pub use utils::similarity::max_sim;

/// Model registry with compile-time generated metadata
pub mod model_registry {
    pub use crate::models::registry::*;
}
