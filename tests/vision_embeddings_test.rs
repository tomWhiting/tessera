//! Integration tests for vision-language embeddings (Phase 3.1).
//!
//! Tests the full API surface for `TesseraVision`, Tessera factory pattern,
//! and vision-specific functionality including:
//! - Document image encoding (patch embeddings)
//! - Text query encoding
//! - Late interaction scoring (`MaxSim`)
//! - Factory pattern with auto-detection
//! - Builder validation and error handling
//! - Model info accessors

use candle_core::Device;
use tessera::{Tessera, TesseraVision, TesseraVisionBuilder};

include!("vision_embeddings_test/encoding_and_api.rs");
include!("vision_embeddings_test/behavior.rs");
