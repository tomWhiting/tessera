use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use super::{BertVariant, CandleDenseEncoder, ModelTypeDetector};
use crate::core::{PoolingStrategy, Tokenizer};
use crate::error::TesseraError;
use crate::models::{registry::ModelType, ModelConfig};
use crate::runtime::{preflight_registered_model, ResourcePolicy};

impl CandleDenseEncoder {
    /// Creates a new Candle-based dense encoder.
    ///
    /// Automatically detects the model type (BERT, `DistilBERT`, `JinaBERT`) from config.json
    /// and loads the appropriate model variant.
    ///
    /// # Arguments
    /// * `model_config` - Configuration for the model (must have `pooling_strategy` set)
    /// * `device` - Device to run the model on (CPU or Metal)
    ///
    /// # Returns
    /// A new `CandleDenseEncoder` instance with the loaded model
    ///
    /// # Errors
    /// Returns an error if:
    /// - Pooling strategy is not configured (required for dense models)
    /// - Model files cannot be downloaded or loaded
    /// - Model type cannot be detected
    pub fn new(model_config: ModelConfig, device: Device) -> Result<Self> {
        let resource_policy = ResourcePolicy::for_model_context(model_config.max_seq_length);
        Self::new_with_resource_policy(model_config, device, resource_policy)
    }

    /// Creates a dense encoder with explicit resource limits.
    pub fn new_with_resource_policy(
        model_config: ModelConfig,
        device: Device,
        resource_policy: ResourcePolicy,
    ) -> Result<Self> {
        let model_name = &model_config.model_name;

        preflight_registered_model(
            model_name,
            model_config.max_seq_length,
            ModelType::Dense,
            &device,
            &resource_policy,
        )?;

        // Validate that pooling strategy is configured
        let registry_pooling = model_config.pooling_strategy.ok_or_else(|| {
            TesseraError::ConfigError(format!(
                "Dense encoder requires pooling_strategy to be configured for model '{model_name}'"
            ))
        })?;

        // Convert from registry PoolingStrategy to core PoolingStrategy
        let pooling_strategy = match registry_pooling {
            crate::models::registry::PoolingStrategy::Cls => PoolingStrategy::Cls,
            crate::models::registry::PoolingStrategy::Mean => PoolingStrategy::Mean,
            crate::models::registry::PoolingStrategy::Max => PoolingStrategy::Max,
            crate::models::registry::PoolingStrategy::LastToken => PoolingStrategy::LastToken,
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_pretrained_with_policy(model_name, resource_policy)
            .with_context(|| format!("Loading tokenizer for {model_name}"))?;

        // Download model files from HuggingFace Hub
        let api =
            hf_hub::api::sync::Api::new().context("Failed to initialize HuggingFace Hub API")?;
        let repo = api.model(model_name.clone());

        // Load config to detect model type
        let config_path = repo
            .get("config.json")
            .with_context(|| format!("Downloading config for {model_name}"))?;

        let config_str =
            std::fs::read_to_string(&config_path).context("Reading model config file")?;

        // Detect model type
        let detector: ModelTypeDetector =
            serde_json::from_str(&config_str).context("Parsing config to detect model type")?;

        let mut model_type = Self::detect_model_type(&detector)
            .with_context(|| format!("Detecting model type for {model_name}"))?;

        // Try to load safetensors first, fall back to pytorch_model.bin
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .with_context(|| format!("Downloading model weights for {model_name}"))?;

        // Refine JinaBERT detection: check if it's the code variant
        if model_type == "jinabert" {
            let is_code_variant = Self::is_jinabert_code_variant(&weights_path)
                .with_context(|| format!("Checking JinaBERT variant for {model_name}"))?;
            if is_code_variant {
                model_type = "jinabert-code".to_string();
            }
        }

        // Load model weights
        let vb = if weights_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, &device)
                    .context("Loading model from safetensors")?
            }
        } else {
            VarBuilder::from_pth(&weights_path, DType::F32, &device)
                .context("Loading model from pytorch_model.bin")?
        };

        // Detect model prefix by checking actual tensor names
        let has_prefix = Self::detect_model_prefix(&weights_path)
            .with_context(|| format!("Detecting model prefix for {model_name}"))?;

        // Create the appropriate model variant with correct prefix
        let model_vb = match Self::model_weight_prefix(has_prefix, &model_type) {
            Some(prefix) => vb.pp(prefix),
            None => vb,
        };

        let model = Self::load_model(&config_str, model_vb, &model_type)
            .with_context(|| format!("Loading {model_type} model"))?;

        let normalize = model_config.normalize_embeddings;

        Ok(Self {
            model,
            tokenizer,
            device,
            config: model_config,
            pooling_strategy,
            normalize,
        })
    }

    /// Detects the model type from config
    pub(super) fn detect_model_type(detector: &ModelTypeDetector) -> Result<String> {
        // First check architectures field for specific model variants
        // This catches cases where model_type is generic "bert" but architecture is specific
        for arch in &detector.architectures {
            let arch_lower = arch.to_lowercase();
            if arch_lower.contains("nomicbert") || arch_lower.contains("nomic_bert") {
                return Ok("nomic-bert".to_string());
            } else if arch_lower.contains("jinabert") {
                return Ok("jinabert".to_string());
            }
        }

        // Then check explicit model_type field
        if let Some(ref model_type) = detector.model_type {
            let model_type_lower = model_type.to_lowercase();

            // Check for specific model types (order matters - more specific first)
            if model_type_lower.contains("nomic_bert")
                || model_type_lower.contains("nomic-bert")
                || model_type_lower == "nomicbert"
            {
                return Ok("nomic-bert".to_string());
            } else if model_type_lower.contains("distilbert") {
                return Ok("distilbert".to_string());
            } else if model_type_lower.contains("jina") || model_type_lower.contains("jinabert") {
                return Ok("jinabert".to_string());
            } else if model_type_lower.contains("xlm") && model_type_lower.contains("roberta") {
                return Ok("xlm-roberta".to_string());
            } else if model_type_lower == "xlm-roberta" {
                return Ok("xlm-roberta".to_string());
            } else if model_type_lower.contains("modernbert") || model_type_lower == "modernbert" {
                return Ok("modernbert".to_string());
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

    /// Returns the prefix Tessera should add before constructing a model.
    ///
    /// NomicBERT is intentionally kept at the root: Candle's stock loader first
    /// tries the upstream prefixless layout and then its `model_type` prefix.
    pub(super) fn model_weight_prefix(has_prefix: bool, model_type: &str) -> Option<&'static str> {
        match (has_prefix, model_type) {
            (_, "nomic-bert") | (true, "modernbert") | (false, _) => None,
            (true, "distilbert") => Some("distilbert"),
            (true, "xlm-roberta") => Some("roberta"),
            (true, _) => Some("bert"),
        }
    }

    /// Detects if a JinaBERT model is the code variant (different FFN structure).
    ///
    /// JinaBERT code models use `up_gated_layer`/`down_layer` instead of
    /// `gated_layers`/`wo` in the MLP layers.
    fn is_jinabert_code_variant(weights_path: &std::path::Path) -> Result<bool> {
        let extension = weights_path.extension().and_then(|s| s.to_str());

        if extension == Some("safetensors") {
            let tensor_names = crate::models::weights::safetensors_tensor_names(weights_path)
                .context("Reading safetensors header for JinaBERT variant detection")?;

            for name in tensor_names {
                // Code variant uses up_gated_layer instead of gated_layers
                if name.contains("mlp.up_gated_layer") {
                    return Ok(true);
                } else if name.contains("mlp.gated_layers") {
                    return Ok(false);
                }
            }
        } else {
            let weights = candle_core::pickle::read_pth_tensor_info(weights_path, false, None)
                .context("Reading pytorch model for variant detection")?;

            for tensor_info in &weights {
                if tensor_info.name.contains("mlp.up_gated_layer") {
                    return Ok(true);
                } else if tensor_info.name.contains("mlp.gated_layers") {
                    return Ok(false);
                }
            }
        }

        // Default to standard JinaBERT
        Ok(false)
    }

    /// Detects whether the model weights use a prefix (e.g., "bert.", "distilbert.")
    ///
    /// Some models like `ColBERT` use "bert." prefix, while others like BGE don't.
    /// We detect this by checking the actual tensor names in the weights file.
    fn detect_model_prefix(weights_path: &std::path::Path) -> Result<bool> {
        let extension = weights_path.extension().and_then(|s| s.to_str());

        if extension == Some("safetensors") {
            let tensor_names = crate::models::weights::safetensors_tensor_names(weights_path)
                .context("Reading safetensors header for prefix detection")?;
            Ok(Self::tensor_names_have_model_prefix(
                tensor_names.iter().map(String::as_str),
            ))
        } else {
            // For pytorch_model.bin, we need to load it to check keys
            // This is more expensive, but necessary
            let weights = candle_core::pickle::read_pth_tensor_info(weights_path, false, None)
                .context("Reading pytorch model info for prefix detection")?;

            Ok(Self::tensor_names_have_model_prefix(
                weights.iter().map(|tensor_info| tensor_info.name.as_str()),
            ))
        }
    }

    pub(super) fn tensor_names_have_model_prefix<'a>(
        tensor_names: impl IntoIterator<Item = &'a str>,
    ) -> bool {
        for name in tensor_names {
            if name.starts_with("bert.embeddings.word_embeddings")
                || name.starts_with("distilbert.embeddings.word_embeddings")
                || name.starts_with("roberta.embeddings.word_embeddings")
                || name.starts_with("model.embeddings")
                || name.starts_with("nomic_bert.embeddings.word_embeddings")
            {
                return true;
            }

            if name == "embeddings.word_embeddings.weight" || name == "word_embeddings.weight" {
                return false;
            }
        }

        false
    }

    /// Loads the appropriate model variant
    fn load_model(config_str: &str, vb: VarBuilder, model_type: &str) -> Result<BertVariant> {
        match model_type {
            "distilbert" => {
                let config: candle_transformers::models::distilbert::Config =
                    serde_json::from_str(config_str).context("Parsing DistilBERT config")?;
                let model =
                    candle_transformers::models::distilbert::DistilBertModel::load(vb, &config)
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
            "jinabert-code" => {
                let config: candle_transformers::models::jina_bert::Config =
                    serde_json::from_str(config_str).context("Parsing JinaBERT Code config")?;
                let model = candle_transformers::models::jina_bert::BertModel::new(vb, &config)
                    .context("Loading JinaBERT Code model")?;
                Ok(BertVariant::JinaBertCode(model))
            }
            "xlm-roberta" => {
                let config: candle_transformers::models::xlm_roberta::Config =
                    serde_json::from_str(config_str).context("Parsing XLM-RoBERTa config")?;
                let model =
                    candle_transformers::models::xlm_roberta::XLMRobertaModel::new(&config, vb)
                        .context("Loading XLM-RoBERTa model")?;
                Ok(BertVariant::XlmRoberta(model))
            }
            "modernbert" => {
                let config: candle_transformers::models::modernbert::Config =
                    serde_json::from_str(config_str).context("Parsing ModernBERT config")?;
                let model = candle_transformers::models::modernbert::ModernBert::load(vb, &config)
                    .context("Loading ModernBERT model")?;
                Ok(BertVariant::ModernBert(model))
            }
            "nomic-bert" => {
                let config: candle_transformers::models::nomic_bert::Config =
                    serde_json::from_str(config_str).context("Parsing NomicBERT config")?;
                let model =
                    candle_transformers::models::nomic_bert::NomicBertModel::load(vb, &config)
                        .context("Loading NomicBERT model")?;
                Ok(BertVariant::NomicBert(model))
            }
            _ => {
                let config: candle_transformers::models::bert::Config =
                    serde_json::from_str(config_str).context("Parsing BERT config")?;
                let model = candle_transformers::models::bert::BertModel::load(vb, &config)
                    .context("Loading BERT model")?;
                Ok(BertVariant::Bert(model))
            }
        }
    }
}
