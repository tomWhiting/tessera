//! Backend implementations for token embedding generation.
//!
//! This module provides multiple backend implementations for generating
//! token-level embeddings from BERT models:
//!
//! - `candle`: Production-ready backend using the Candle framework with full BERT support
//! - `burn`: Experimental backend using the Burn framework (simplified for prototype)

pub mod candle;
pub mod burn;

pub use candle::CandleEncoder;
pub use burn::BurnEncoder;
