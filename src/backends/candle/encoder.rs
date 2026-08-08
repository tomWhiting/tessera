//! BERT encoder implementation using Candle.
//!
//! The active multi-vector registry paths use BERT checkpoints such as
//! `colbert-v2` and `colbert-small`. The loader contains additional BERT-family
//! variants, but registry support metadata determines whether a checkpoint is
//! runnable.
//!
//! Query and document paths implement role-specific ColBERT framing, query
//! augmentation, and document punctuation filtering. The trait-based encode
//! path remains a deliberately generic compatibility operation.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::core::{Encoder, MultiVectorEncoder, TokenEmbedder, TokenEmbeddings, Tokenizer};
use crate::models::loader::ModelFileResolver;
use crate::models::{registry::ModelType, ModelConfig};
use crate::runtime::{
    preflight_and_reserve_registered_model_with_dtype, ModelDType, ModelResidencyPermit,
    ResourcePolicy, TransformerProfile,
};

mod inference;
mod model;
mod projection;
mod role;
mod tensor;
mod window;

use model::{detect_model_type, load_model, BertVariant, ModelTypeDetector};
use projection::validate_projection_contract;
use role::{ColbertPreprocessor, InputRole};

pub use role::ColbertConfig;

/// BERT encoder using the Candle backend.
///
/// This encoder is specifically for BERT-style models producing multi-vector
/// token embeddings. Dense and vision-language architectures are handled by
/// their own encoder modules.
pub struct CandleBertEncoder {
    model: BertVariant,
    projection: Tensor,
    tokenizer: Tokenizer,
    preprocessor: ColbertPreprocessor,
    device: Device,
    config: ModelConfig,
    matryoshka_strategy: Option<crate::utils::MatryoshkaStrategy>,
    dtype: ModelDType,
    resource_policy: ResourcePolicy,
    transformer_profile: TransformerProfile,
    _residency: ModelResidencyPermit<'static>,
}

impl CandleBertEncoder {
    /// Creates a BERT encoder with explicit dtype, resources, and role limits.
    pub(crate) fn new_with_dtype_and_colbert_config(
        model_config: ModelConfig,
        device: Device,
        dtype: ModelDType,
        resource_policy: ResourcePolicy,
        colbert_config: ColbertConfig,
    ) -> Result<Self> {
        let model_name = &model_config.model_name;

        let (model_info, residency) = preflight_and_reserve_registered_model_with_dtype(
            model_name,
            model_config.max_seq_length,
            ModelType::Colbert,
            &device,
            dtype,
            &resource_policy,
        )?;

        let files = ModelFileResolver::new(model_info)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_model_files_with_policy(&files, resource_policy)
            .with_context(|| format!("Loading tokenizer for {model_name}"))?;
        let preprocessor = ColbertPreprocessor::from_tokenizer(&tokenizer, colbert_config)
            .with_context(|| format!("Validating ColBERT tokenizer artifacts for {model_name}"))?;

        // Load config to detect model type
        let config_path = files
            .get(model_info.config_file)
            .with_context(|| format!("Downloading config for {model_name}"))?;

        let config_str =
            std::fs::read_to_string(&config_path).context("Reading model config file")?;
        let profile = TransformerProfile::from_config_json(&config_str)
            .context("Reading ColBERT dimensions for resource estimation")?;
        resource_policy
            .validate_transformer_activations(
                profile,
                1,
                resource_policy.max_sequence_tokens(),
                dtype,
            )
            .map_err(|error| anyhow::anyhow!("ColBERT activation preflight failed: {error}"))?;

        // Detect model type
        let detector: ModelTypeDetector =
            serde_json::from_str(&config_str).context("Parsing config to detect model type")?;

        let model_type = detect_model_type(&detector)
            .with_context(|| format!("Detecting model type for {model_name}"))?;

        let hidden_size = detector
            .hidden_size
            .or(detector.dim)
            .context("ColBERT checkpoint config has no hidden dimension")?;
        validate_projection_contract(
            model_info.has_projection,
            model_info.projection_dims,
            model_config.embedding_dim,
            model_info.hidden_dim,
            hidden_size,
        )
        .with_context(|| format!("Validating ColBERT projection contract for {model_name}"))?;

        // Try to load safetensors first, fall back to pytorch_model.bin
        let weights_path = files
            .weights()
            .with_context(|| format!("Downloading model weights for {model_name}"))?;

        // Load model weights
        let vb = if weights_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], dtype.candle_dtype(), &device)
                    .context("Loading model from safetensors")?
            }
        } else {
            VarBuilder::from_pth(&weights_path, dtype.candle_dtype(), &device)
                .context("Loading model from pytorch_model.bin")?
        };

        // Create the appropriate model variant
        // BERT and JinaBERT models have a "bert" prefix, but DistilBERT doesn't
        let model_vb = if model_type == "distilbert" {
            vb.pp("distilbert")
        } else {
            vb.pp("bert")
        };

        let model = load_model(&config_str, model_vb, &model_type)
            .with_context(|| format!("Loading {model_type} model"))?;

        let projection = vb
            .get((model_config.embedding_dim, hidden_size), "linear.weight")
            .with_context(|| format!("Loading mandatory ColBERT projection for {model_name}"))?;

        // Determine Matryoshka strategy from registry if available
        #[allow(clippy::option_if_let_else)]
        let matryoshka_strategy = if model_config.target_dimension.is_some() {
            model_info
                .embedding_dim
                .matryoshka_strategy()
                .and_then(crate::utils::MatryoshkaStrategy::from_str)
        } else {
            None
        };

        Ok(Self {
            model,
            projection,
            tokenizer,
            preprocessor,
            device,
            config: model_config,
            matryoshka_strategy,
            dtype,
            resource_policy,
            transformer_profile: profile,
            _residency: residency,
        })
    }
}

impl CandleBertEncoder {
    /// Parameter dtype selected when this model was loaded.
    #[must_use]
    pub const fn model_dtype(&self) -> ModelDType {
        self.dtype
    }

    /// Encodes a retrieval query with reference ColBERT role semantics.
    pub fn encode_query(&self, text: &str) -> Result<TokenEmbeddings> {
        self.encode_role(text, InputRole::Query)
    }

    /// Encodes a retrieval document with reference ColBERT role semantics.
    pub fn encode_document(&self, text: &str) -> Result<TokenEmbeddings> {
        self.encode_role(text, InputRole::Document)
    }

    /// Encodes retrieval queries in one or more bounded forward passes.
    pub fn encode_query_batch(&self, inputs: &[&str]) -> Result<Vec<TokenEmbeddings>> {
        self.encode_role_batch(inputs, InputRole::Query)
    }

    /// Encodes retrieval documents in one or more bounded forward passes.
    pub fn encode_document_batch(&self, inputs: &[&str]) -> Result<Vec<TokenEmbeddings>> {
        self.encode_role_batch(inputs, InputRole::Document)
    }

    fn encode_role(&self, text: &str, role: InputRole) -> Result<TokenEmbeddings> {
        let prepared = self
            .preprocessor
            .prepare(&self.tokenizer, text, role)
            .with_context(|| {
                format!(
                    "Preparing {role:?} text for ColBERT ({} UTF-8 bytes)",
                    text.len()
                )
            })?;
        self.infer_one(prepared, text)
    }

    fn encode_role_batch(&self, inputs: &[&str], role: InputRole) -> Result<Vec<TokenEmbeddings>> {
        let prepared = self
            .preprocessor
            .prepare_batch(&self.tokenizer, inputs, role)
            .with_context(|| format!("Preparing {} {role:?} inputs for ColBERT", inputs.len()))?;
        self.infer_batch(&prepared, inputs)
    }
}

impl TokenEmbedder for CandleBertEncoder {
    /// Encodes with generic BERT framing and no retrieval role.
    fn encode(&self, text: &str) -> Result<TokenEmbeddings> {
        self.encode_role(text, InputRole::Generic)
    }
}

impl Encoder for CandleBertEncoder {
    type Output = TokenEmbeddings;

    /// Encodes with generic BERT framing and no retrieval role.
    fn encode(&self, input: &str) -> Result<Self::Output> {
        <Self as TokenEmbedder>::encode(self, input)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Self::Output>> {
        self.encode_role_batch(inputs, InputRole::Generic)
    }
}

impl MultiVectorEncoder for CandleBertEncoder {
    fn num_vectors(&self, text: &str) -> Result<usize> {
        self.preprocessor
            .prepare(&self.tokenizer, text, InputRole::Generic)
            .map(|input| input.token_ids.len())
            .with_context(|| {
                format!(
                    "Tokenizing generic text to count vectors ({} UTF-8 bytes)",
                    text.len()
                )
            })
    }

    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }
}
