//! Candle backend for BERT-based token embeddings.
//!
//! This module provides a BERT encoder implementation using the Candle
//! deep learning framework, with support for CPU and Metal acceleration.

pub mod device;
pub mod encoder;

pub use device::{cpu_device, device_description, get_device};
pub use encoder::CandleBertEncoder;

/// Type alias for backward compatibility with pre-0.2.0 code.
///
/// # Deprecated
/// Use `CandleBertEncoder` instead. This alias will be removed in version 0.3.0.
#[deprecated(since = "0.2.0", note = "Use CandleBertEncoder instead")]
pub type CandleEncoder = CandleBertEncoder;

#[cfg(target_os = "macos")]
pub use device::metal_device;
