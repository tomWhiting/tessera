//! Burn backend for BERT-based token embeddings.
//!
//! This module provides a BERT encoder implementation using the Burn
//! deep learning framework.
//!
//! Note: This is a simplified implementation for the prototype.
//! Full production implementation requires integration with pre-trained
//! BERT weights from HuggingFace Hub.

pub mod backend;
pub mod encoder;

pub use backend::{backend_description, cpu_device, CpuBackend};
pub use encoder::BurnEncoder;
