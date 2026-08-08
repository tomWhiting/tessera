//! Integration tests for dense embeddings (Phase 2.1).
//!
//! Tests the full API surface for `TesseraDense`, Tessera factory pattern,
//! and dense-specific functionality including:
//! - Basic single-text encoding
//! - Batch processing
//! - Cosine similarity
//! - Normalization validation
//! - Pooling strategies
//! - Matryoshka dimension truncation
//! - Factory pattern with auto-detection
//! - Builder validation and error handling
//! - Device selection

use candle_core::Device;
use tessera::{Tessera, TesseraDense, TesseraDenseBuilder};

include!("dense_embeddings_test/encoding.rs");
include!("dense_embeddings_test/api.rs");
