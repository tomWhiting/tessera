//! Candle backend for BERT-based token embeddings.
//!
//! This module provides a BERT encoder implementation using the Candle
//! deep learning framework, with optional Metal and CUDA acceleration.

pub mod device;
pub mod encoder;

pub use device::get_device;
pub use encoder::CandleBertEncoder;
