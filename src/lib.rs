// Legacy pedantic/style lint debt is kept explicit while the revival replaces
// old APIs. The local repository gate promotes every lint outside this finite
// ratchet to an error; do not add entries merely to hide new warnings.
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
//! Tessera's high-level builders preflight registered-model and request-shape
//! limits before their main model input tensors are allocated. The default
//! [`ResourcePolicy`] allows at most:
//!
//! - 1 MiB of UTF-8 input per sequence, checked before tokenization;
//! - 512 tokens in one sequence, including special tokens;
//! - 16 items in one batch;
//! - 2,048 padded token cells (`items * longest sequence`);
//! - 1,048,576 attention cells (`items * longest sequence^2`);
//! - 1,024 inputs and 64 MiB of aggregate UTF-8 input in one logical job;
//! - 64 MiB of retained embedding values from one collecting API;
//! - 512 MiB of estimated live inference scratch space per forward pass; and
//! - 2 GiB of estimated resident model parameter storage (F32 by default).
//!
//! Batch limits apply to one tensor forward pass; job and output limits also
//! bound work accumulated across internal chunks. Activation storage is a
//! conservative estimate from the pinned transformer configuration and dtype,
//! not a measurement of total process or accelerator memory.
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
//!     .with_max_attention_cells(67_108_864)
//!     .with_max_activation_bytes(8 * 1024 * 1024 * 1024);
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
//! other process memory. The 8 GiB scratch allowance is illustrative, not a
//! certification result. It can still exhaust CPU, GPU, or Metal shared
//! memory. Measure the selected model on the target hardware before opting in.
//!
//! The model-byte limit bounds both one model and the prospective aggregate
//! estimated parameter storage retained by Tessera encoders. A 3B-parameter
//! checkpoint has an approximately 12 GB F32 parameter estimate before
//! allocator overhead, so the 2 GiB default rejects it. The estimate is an
//! admission guard, not a full peak-memory calculation.
//!
//! Active constructors reserve estimated parameter bytes before tokenizer,
//! Hub, or artifact I/O. Reservations are keyed by immutable model revision,
//! physical Candle device, and dtype. A duplicate retained key is rejected;
//! callers must reuse or drop the existing embedder because Tessera does not
//! share model tensors between instances. A distinct key is admitted only when
//! its estimate plus existing Tessera reservations fits the requesting
//! policy's model-byte limit. Reservations release automatically on constructor
//! failure and after encoder tensors are dropped.
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
//! # Inference admission
//!
//! Tessera admits one Candle forward pass at a time across its encoders. The
//! default process-wide queue permits 16 waiting callers with a 30-second wait
//! deadline. Excess callers and expired waits return structured errors. Call
//! [`configure_inference_gate`] before the first forward pass to choose
//! other immutable process-wide bounds. This gate does not govern allocations
//! made by other libraries in the process.
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
//! The high-level [`TesseraVision::encode_document`] method accepts an image
//! path. The opt-in `pdf` feature adds bounded page and whole-document methods
//! to the same façade and requires a system Poppler installation. ColPali is a
//! large experimental path and requires explicit visual-sequence, attention,
//! activation, and model-memory overrides. Passing those preflights is not
//! evidence that it fits the target machine.
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
//! - `pdf`: opt-in bounded PDF page/document encoding; requires Poppler.
//! - `metal`: Apple Metal support in Candle.
//! - `cuda`: NVIDIA CUDA support in Candle.
//! - `python`: PyO3 bindings for active embedding façades.
//! - `timeseries`: generic time-series core types only.
//!
//! # Licensing
//!
//! Tessera source code and repository documentation are Apache-2.0 licensed.
//! Model checkpoints are not bundled and are governed by their upstream
//! licenses and terms. The catalog includes permissive, non-commercial, and
//! model-specific licenses; consult the pinned model repository before use.
//! Registry license fields are discovery metadata, not a relicensing of the
//! checkpoint.
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
mod backends;
mod bindings;
pub mod core;
mod encoding;
pub mod error;
pub mod models;
pub mod quantization;
pub mod runtime;
pub mod utils;
mod vision;

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
    configure_cpu_threads, configure_inference_gate, try_acquire_inference, ContextWindowConfig,
    ContextWindowError, CpuThreadConfig, CpuThreadConfigError, InferenceGateConfig,
    InferenceGateConfigError, InferenceGateError, ModelDType, ModelDTypeError, ResourcePolicy,
    ResourcePolicyError,
};
pub use utils::similarity::max_sim;

/// Model registry with compile-time generated metadata
pub mod model_registry {
    pub use crate::models::registry::*;
}
