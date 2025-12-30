//! Common utilities for embedding operations.
//!
//! Provides reusable components used across different encoding paradigms:
//!
//! - [`matryoshka`]: Matryoshka representation learning (dimension truncation)
//! - [`pooling`]: Pooling strategies (CLS, mean, max) for dense embeddings
//! - [`similarity`]: Similarity/distance functions (cosine, dot, Euclidean, `MaxSim`)
//! - [`normalization`]: Vector normalization (L2 norm, standardization)
//! - [`batching`]: Batching utilities (padding, masking)
//!
//! These utilities enable code reuse across multi-vector, dense, and sparse
//! encoding implementations, ensuring consistent behavior and reducing duplication.

pub mod batching;
pub mod matryoshka;
pub mod normalization;
pub mod pooling;
pub mod similarity;

#[cfg(feature = "pdf")]
pub mod pdf;

// Re-export commonly used functions for convenience
pub use batching::{create_attention_mask, pad_sequences};
pub use matryoshka::{apply_matryoshka, MatryoshkaStrategy};
pub use normalization::{l2_norm, l2_normalize};
pub use pooling::{cls_pooling, last_token_pooling, max_pooling, mean_pooling};
pub use similarity::{cosine_similarity, dot_product, euclidean_distance, max_sim};

#[cfg(feature = "pdf")]
pub use pdf::PdfRenderer;
