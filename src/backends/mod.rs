//! Backend implementations for model inference.
//!
//! Tessera currently uses the `HuggingFace` Candle framework as its only
//! inference backend:
//!
//! - [`candle`]: CPU execution plus opt-in Metal (Apple Silicon) and CUDA
//!   acceleration. Tessera's model adapters remain subject to the support tier
//!   and certification status recorded in the model registry.
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
