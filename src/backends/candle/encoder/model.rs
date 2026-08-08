use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

/// Enum to hold different BERT model variants.
pub(super) enum BertVariant {
    Bert(candle_transformers::models::bert::BertModel),
    DistilBert(candle_transformers::models::distilbert::DistilBertModel),
    JinaBert(candle_transformers::models::jina_bert::BertModel),
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
            Self::JinaBert(model) => {
                // JinaBERT uses ALiBi position embeddings and doesn't need attention_mask
                // in its forward pass
                model.forward(token_ids).context("JinaBERT forward pass")
            }
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
}

/// Detects the model type from config.
pub(super) fn detect_model_type(detector: &ModelTypeDetector) -> Result<String> {
    // First check explicit model_type field
    if let Some(ref model_type) = detector.model_type {
        let model_type_lower = model_type.to_lowercase();
        if model_type_lower.contains("distilbert") {
            return Ok("distilbert".to_string());
        } else if model_type_lower.contains("jina") {
            return Ok("jinabert".to_string());
        } else if model_type_lower.contains("bert") {
            return Ok("bert".to_string());
        }
    }

    // Fallback: detect by config structure
    // DistilBERT uses 'dim', while BERT/JinaBERT use 'hidden_size'
    if detector.dim.is_some() && detector.hidden_size.is_none() {
        Ok("distilbert".to_string())
    } else if detector.hidden_size.is_some() {
        // Default to BERT (JinaBERT configs are similar to BERT)
        Ok("bert".to_string())
    } else {
        anyhow::bail!("Could not detect model type from config")
    }
}

/// Loads the appropriate model variant.
pub(super) fn load_model(
    config_str: &str,
    vb: VarBuilder,
    model_type: &str,
) -> Result<BertVariant> {
    match model_type {
        "distilbert" => {
            let config: candle_transformers::models::distilbert::Config =
                serde_json::from_str(config_str).context("Parsing DistilBERT config")?;
            let model = candle_transformers::models::distilbert::DistilBertModel::load(vb, &config)
                .context("Loading DistilBERT model")?;
            Ok(BertVariant::DistilBert(model))
        }
        "jinabert" => {
            let config: candle_transformers::models::jina_bert::Config =
                serde_json::from_str(config_str).context("Parsing JinaBERT config")?;
            let model = candle_transformers::models::jina_bert::BertModel::new(vb, &config)
                .context("Loading JinaBERT model")?;
            Ok(BertVariant::JinaBert(model))
        }
        _ => {
            // Default to BERT for unknown types
            let config: candle_transformers::models::bert::Config =
                serde_json::from_str(config_str).context("Parsing BERT config")?;
            let model = candle_transformers::models::bert::BertModel::load(vb, &config)
                .context("Loading BERT model")?;
            Ok(BertVariant::Bert(model))
        }
    }
}
