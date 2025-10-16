//! Backend implementations for model inference.
//!
//! Tessera supports multiple deep learning backends for flexibility
//! and performance:
//!
//! - [`candle`]: Production backend using HuggingFace Candle framework.
//!   Supports CPU, Metal (Apple Silicon), and CUDA acceleration.
//!   Mature, well-tested, recommended for production use.
//!
//! - [`burn`]: Experimental backend using Burn deep learning framework.
//!   Under development, demonstrates framework flexibility.
//!
//! The backend abstraction allows adding new inference engines
//! (ONNX Runtime, Tract, etc.) without changing higher-level code.
//!
//! # Backend Selection
//!
//! Backends are selected based on:
//! - Available hardware (CPU, GPU, NPU)
//! - Model format compatibility
//! - Performance requirements
//! - Deployment constraints
//!
//! # Device Support
//!
//! - **CPU**: Always available, good for development
//! - **Metal**: Apple Silicon GPU acceleration (M1/M2/M3)
//! - **CUDA**: NVIDIA GPU acceleration
//!
//! # Adding New Backends
//!
//! To add a new backend:
//! 1. Implement the `TokenEmbedder` trait
//! 2. Handle model loading and device management
//! 3. Optimize for the target platform
//! 4. Add feature flag to Cargo.toml

pub mod burn;
pub mod candle;

pub use burn::BurnEncoder;
pub use candle::CandleBertEncoder;
