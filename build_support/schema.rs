use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
pub struct ModelRegistry {
    pub version: String,
    pub model_categories: BTreeMap<String, ModelCategory>,
}

impl ModelRegistry {
    pub fn models(&self) -> impl Iterator<Item = &ModelMetadata> {
        self.model_categories
            .values()
            .flat_map(|category| category.models.iter())
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelCategory {
    #[serde(rename = "description")]
    pub _description: String,
    pub models: Vec<ModelMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub name: String,
    pub huggingface_id: String,
    pub revision: Option<String>,
    pub organization: String,
    pub release_date: String,
    pub support: SupportMetadata,
    pub architecture: Architecture,
    #[serde(default)]
    pub pooling: Option<PoolingConfig>,
    pub specs: Specs,
    pub files: Files,
    pub capabilities: Capabilities,
    pub license: String,
    pub description: String,
}

#[derive(Debug, Deserialize)]
pub struct SupportMetadata {
    pub tier: SupportTier,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportTier {
    Supported,
    Experimental,
    CatalogOnly,
}

impl SupportTier {
    pub const fn is_runnable(self) -> bool {
        matches!(self, Self::Supported | Self::Experimental)
    }

    pub const fn rust_variant(self) -> &'static str {
        match self {
            Self::Supported => "Supported",
            Self::Experimental => "Experimental",
            Self::CatalogOnly => "CatalogOnly",
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Architecture {
    #[serde(rename = "type")]
    pub arch_type: String,
    pub variant: String,
    pub has_projection: bool,
    pub projection_dims: Option<usize>,
    #[serde(default, rename = "matryoshka_dims")]
    pub _matryoshka_dims: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingDimSpec {
    Fixed(usize),
    Matryoshka {
        default: usize,
        matryoshka: MatryoshkaSpec,
    },
}

#[derive(Debug, Deserialize)]
pub struct MatryoshkaSpec {
    pub min: usize,
    pub max: usize,
    pub supported: Vec<usize>,
    #[serde(default)]
    pub strategy: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Specs {
    pub parameters: String,
    pub embedding_dim: EmbeddingDimSpec,
    pub hidden_dim: usize,
    pub context_length: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct Files {
    pub tokenizer: String,
    pub config: String,
    pub weights: Weights,
}

#[derive(Debug, Deserialize)]
pub struct Weights {
    pub safetensors: Option<String>,
    pub pytorch: String,
}

#[derive(Debug, Deserialize)]
pub struct Capabilities {
    pub languages: Vec<String>,
    pub modalities: Vec<String>,
    pub multi_vector: bool,
    pub quantization: Vec<String>,
    #[serde(default, rename = "matryoshka")]
    pub _matryoshka: bool,
}

#[derive(Debug, Deserialize)]
pub struct PoolingConfig {
    pub strategy: String,
    pub normalize: bool,
}
