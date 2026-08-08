//! Integration tests for sparse embeddings (Phase 2.2).
//!
//! Tests the full API surface for `TesseraSparse`, Tessera factory pattern,
//! and sparse-specific functionality including:
//! - Basic single-text encoding
//! - Batch processing
//! - Sparsity verification (99%+)
//! - Similarity computation (dot product)
//! - Factory pattern with auto-detection
//! - Builder validation and error handling
//! - Device selection
//! - Interpretability (non-zero indices)

use candle_core::Device;
use tessera::{Tessera, TesseraSparse, TesseraSparseBuilder};

include!("sparse_embeddings_test/encoding.rs");
include!("sparse_embeddings_test/api.rs");
