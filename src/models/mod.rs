//! Model loading and configuration utilities.
//!
//! This module provides utilities for loading BERT-based models from
//! HuggingFace Hub and configuring them for ColBERT-style encoding.

pub mod config;
pub mod loader;
pub mod registry;

pub use config::ModelConfig;
pub use loader::{download_config, download_model_file, download_tokenizer};
pub use registry::{get_model, models_by_type, ModelInfo, ModelType, MODEL_REGISTRY};
