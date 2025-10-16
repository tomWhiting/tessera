//! Candle backend for BERT-based token embeddings.
//!
//! This module provides a BERT encoder implementation using the Candle
//! deep learning framework, with support for CPU and Metal acceleration.

pub mod device;
pub mod encoder;

pub use device::{cpu_device, device_description, get_device};
pub use encoder::CandleEncoder;

#[cfg(target_os = "macos")]
pub use device::metal_device;
