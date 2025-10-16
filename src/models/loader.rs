//! Utilities for loading models from HuggingFace Hub.

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// Downloads and caches a model from HuggingFace Hub.
///
/// # Arguments
/// * `model_name` - Name of the model on HuggingFace Hub
/// * `filename` - Specific file to download (e.g., "model.safetensors", "pytorch_model.bin")
///
/// # Returns
/// Path to the downloaded model file in the local cache
pub fn download_model_file(model_name: &str, filename: &str) -> Result<PathBuf> {
    let api = Api::new().context("Failed to initialize HuggingFace Hub API")?;

    let repo = api.model(model_name.to_string());

    let path = repo
        .get(filename)
        .with_context(|| format!("Failed to download {} from {}", filename, model_name))?;

    Ok(path)
}

/// Downloads and caches model configuration from HuggingFace Hub.
///
/// # Arguments
/// * `model_name` - Name of the model on HuggingFace Hub
///
/// # Returns
/// Path to the config.json file in the local cache
pub fn download_config(model_name: &str) -> Result<PathBuf> {
    download_model_file(model_name, "config.json")
}

/// Downloads and caches tokenizer from HuggingFace Hub.
///
/// # Arguments
/// * `model_name` - Name of the model on HuggingFace Hub
///
/// # Returns
/// Path to the tokenizer.json file in the local cache
pub fn download_tokenizer(model_name: &str) -> Result<PathBuf> {
    download_model_file(model_name, "tokenizer.json")
}
