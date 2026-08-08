use std::path::Path;

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use serde::Deserialize;

/// Enum to hold different BERT model variants.
pub(super) enum BertVariant {
    Bert(candle_transformers::models::bert::BertModel),
    DistilBert(candle_transformers::models::distilbert::DistilBertModel),
}

impl BertVariant {
    pub(super) fn forward(&self, token_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        match self {
            Self::Bert(model) => model
                .forward(token_ids, attention_mask, None)
                .context("BERT forward pass"),
            Self::DistilBert(model) => model
                .forward(token_ids, attention_mask)
                .context("DistilBERT forward pass"),
        }
    }
}

/// Helper struct to detect model type from config.
#[derive(Debug, Deserialize)]
pub(super) struct ModelTypeDetector {
    pub(super) model_type: Option<String>,
    #[serde(default)]
    pub(super) hidden_size: Option<usize>,
    #[serde(default)]
    pub(super) dim: Option<usize>,
    #[serde(default)]
    pub(super) vocab_size: Option<usize>,
}

/// Detects the model type from config.
pub(super) fn detect_model_type(detector: &ModelTypeDetector) -> Result<String> {
    // First check explicit model_type field
    if let Some(ref model_type) = detector.model_type {
        let model_type_lower = model_type.to_lowercase();
        if model_type_lower.contains("distilbert") {
            return Ok("distilbert".to_string());
        } else if model_type_lower.contains("bert") {
            return Ok("bert".to_string());
        }
    }

    // Fallback: detect by config structure
    if detector.dim.is_some() && detector.hidden_size.is_none() {
        Ok("distilbert".to_string())
    } else if detector.hidden_size.is_some() {
        Ok("bert".to_string())
    } else {
        anyhow::bail!("Could not detect model type from config")
    }
}

/// Detects whether the model weights use a prefix (e.g., "bert.", "distilbert.").
pub(super) fn detect_model_prefix(weights_path: &Path) -> Result<bool> {
    let extension = weights_path.extension().and_then(|s| s.to_str());

    if extension == Some("safetensors") {
        let tensor_names = crate::models::weights::safetensors_tensor_names(weights_path)
            .context("Reading safetensors header for prefix detection")?;

        // Check if any tensor starts with "bert." or "distilbert."
        for name in tensor_names {
            if name.starts_with("bert.embeddings.word_embeddings") {
                return Ok(true);
            } else if name.starts_with("distilbert.embeddings.word_embeddings") {
                return Ok(true);
            } else if name == "embeddings.word_embeddings.weight" {
                return Ok(false);
            }
        }

        // Default: assume has prefix (safer for SPLADE models)
        Ok(true)
    } else {
        // For pytorch_model.bin
        let weights = candle_core::pickle::read_pth_tensor_info(weights_path, false, None)
            .context("Reading pytorch model info for prefix detection")?;

        for tensor_info in &weights {
            let name = &tensor_info.name;
            if name.starts_with("bert.embeddings.word_embeddings") {
                return Ok(true);
            } else if name.starts_with("distilbert.embeddings.word_embeddings") {
                return Ok(true);
            } else if name == "embeddings.word_embeddings.weight" {
                return Ok(false);
            }
        }

        // Default: assume has prefix
        Ok(true)
    }
}

/// Loads the appropriate model variant.
pub(super) fn load_model(
    config_str: &str,
    vb: VarBuilder,
    model_type: &str,
) -> Result<BertVariant> {
    if model_type == "distilbert" {
        let config: candle_transformers::models::distilbert::Config =
            serde_json::from_str(config_str).context("Parsing DistilBERT config")?;
        let model = candle_transformers::models::distilbert::DistilBertModel::load(vb, &config)
            .context("Loading DistilBERT model")?;
        Ok(BertVariant::DistilBert(model))
    } else {
        let config: candle_transformers::models::bert::Config =
            serde_json::from_str(config_str).context("Parsing BERT config")?;
        let model = candle_transformers::models::bert::BertModel::load(vb, &config)
            .context("Loading BERT model")?;
        Ok(BertVariant::Bert(model))
    }
}
