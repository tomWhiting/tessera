//! Backend implementations for model inference.
//!
//! Tessera uses the `HuggingFace` Candle framework as its primary backend,
//! providing excellent performance across multiple platforms:
//!
//! - [`candle`]: Production backend using `HuggingFace` Candle framework.
//!   Supports CPU, Metal (Apple Silicon), and CUDA acceleration.
//!   Mature, well-tested, recommended for all production use.
//!
//! # Device Support
//!
//! - **CPU**: Always available, good for development
//! - **Metal**: Apple Silicon GPU acceleration (M1/M2/M3/M4)
//! - **CUDA**: NVIDIA GPU acceleration
//!
//! # Adding New Backends
//!
//! Future backends (Burn, ONNX Runtime, Tract, etc.) can be added by:
//! 1. Implementing the `TokenEmbedder` trait
//! 2. Handling model loading and device management
//! 3. Optimizing for the target platform
//! 4. Adding feature flag to Cargo.toml

pub mod candle;

pub use candle::CandleBertEncoder;
