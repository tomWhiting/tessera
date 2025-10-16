//! Build script that generates model registry code from models.json.
//!
//! This script runs at compile time and generates type-safe Rust code
//! containing all model metadata from the JSON registry.

use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ModelRegistry {
    #[allow(dead_code)]
    version: String,
    model_categories: HashMap<String, ModelCategory>,
}

#[derive(Debug, Deserialize)]
struct ModelCategory {
    #[allow(dead_code)]
    description: String,
    models: Vec<ModelMetadata>,
}

#[derive(Debug, Deserialize)]
struct ModelMetadata {
    id: String,
    #[serde(rename = "type")]
    model_type: String,
    name: String,
    huggingface_id: String,
    organization: String,
    release_date: String,
    architecture: Architecture,
    specs: Specs,
    #[allow(dead_code)]
    files: Files,
    capabilities: Capabilities,
    performance: Performance,
    license: String,
    description: String,
}

#[derive(Debug, Deserialize)]
struct Architecture {
    #[serde(rename = "type")]
    arch_type: String,
    variant: String,
    has_projection: bool,
    projection_dims: Option<usize>,
    #[allow(dead_code)]
    #[serde(default)]
    matryoshka_dims: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingDimSpec {
    Fixed(usize),
    Matryoshka {
        default: usize,
        matryoshka: MatryoshkaSpec,
    },
}

#[derive(Debug, Deserialize)]
struct MatryoshkaSpec {
    min: usize,
    max: usize,
    supported: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct Specs {
    parameters: String,
    embedding_dim: EmbeddingDimSpec,
    hidden_dim: usize,
    context_length: usize,
    max_position_embeddings: usize,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Files {
    tokenizer: String,
    config: String,
    weights: Weights,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Weights {
    safetensors: String,
    pytorch: String,
}

#[derive(Debug, Deserialize)]
struct Capabilities {
    languages: Vec<String>,
    modalities: Vec<String>,
    multi_vector: bool,
    quantization: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    matryoshka: bool,
}

#[derive(Debug, Deserialize)]
struct Performance {
    beir_avg: f64,
    ms_marco_mrr10: f64,
}

fn main() {
    println!("cargo:rerun-if-changed=models.json");

    let models_json = fs::read_to_string("models.json")
        .expect("Failed to read models.json - ensure it exists in the project root");

    let registry: ModelRegistry = serde_json::from_str(&models_json)
        .expect("Failed to parse models.json - check JSON syntax");

    let total_models = registry.model_categories.values()
        .map(|cat| cat.models.len())
        .sum::<usize>();

    validate_registry(&registry);

    let generated_code = generate_code(&registry);

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest_path = Path::new(&out_dir).join("model_registry.rs");

    fs::write(&dest_path, generated_code)
        .expect("Failed to write generated model registry code");

    println!("cargo:warning=Generated model registry with {} models across {} categories",
             total_models, registry.model_categories.len());
}

fn validate_registry(registry: &ModelRegistry) {
    let mut ids = std::collections::HashSet::new();

    for (_category, cat_data) in &registry.model_categories {
        for model in &cat_data.models {
            // Check for duplicate IDs
            if !ids.insert(&model.id) {
                panic!("Duplicate model ID found: {}", model.id);
            }

            // Validate embedding dimensions
            let embedding_dim = match &model.specs.embedding_dim {
                EmbeddingDimSpec::Fixed(dim) => {
                    if *dim == 0 {
                        panic!("Model {} has invalid embedding_dim: 0", model.id);
                    }
                    *dim
                }
                EmbeddingDimSpec::Matryoshka { default, matryoshka } => {
                    // Validate Matryoshka configuration
                    if matryoshka.min >= matryoshka.max {
                        panic!(
                            "Model {} has invalid Matryoshka range: min ({}) >= max ({})",
                            model.id, matryoshka.min, matryoshka.max
                        );
                    }
                    if *default < matryoshka.min || *default > matryoshka.max {
                        panic!(
                            "Model {} has default dimension ({}) outside Matryoshka range ({}-{})",
                            model.id, default, matryoshka.min, matryoshka.max
                        );
                    }
                    // Validate all supported dimensions are within range
                    for &dim in &matryoshka.supported {
                        if dim < matryoshka.min || dim > matryoshka.max {
                            panic!(
                                "Model {} has supported dimension {} outside Matryoshka range ({}-{})",
                                model.id, dim, matryoshka.min, matryoshka.max
                            );
                        }
                    }
                    // Validate supported dimensions are in ascending order
                    let mut sorted = matryoshka.supported.clone();
                    sorted.sort();
                    if sorted != matryoshka.supported {
                        panic!(
                            "Model {} Matryoshka supported dimensions must be in ascending order",
                            model.id
                        );
                    }
                    *default
                }
            };

            // Validate context length
            if model.specs.context_length == 0 {
                panic!("Model {} has invalid context_length: 0", model.id);
            }

            // Validate HuggingFace ID format
            if !model.huggingface_id.contains('/') {
                panic!(
                    "Model {} has invalid huggingface_id format: {}",
                    model.id, model.huggingface_id
                );
            }

            // Validate projection consistency
            if model.architecture.has_projection && model.architecture.projection_dims.is_none() {
                panic!(
                    "Model {} has has_projection=true but no projection_dims",
                    model.id
                );
            }

            if model.architecture.has_projection {
                if let Some(proj_dim) = model.architecture.projection_dims {
                    if proj_dim != embedding_dim {
                        panic!(
                            "Model {} projection_dims ({}) doesn't match embedding_dim ({})",
                            model.id, proj_dim, embedding_dim
                        );
                    }
                }
            }
        }
    }
}

fn generate_code(registry: &ModelRegistry) -> String {
    let mut code = String::from(
        r#"// Generated by build.rs from models.json
// DO NOT EDIT THIS FILE MANUALLY

"#,
    );

    // Generate EmbeddingDimension enum
    code.push_str(&generate_embedding_dimension_enum());
    code.push_str("\n\n");

    // Generate ModelType enum
    code.push_str(&generate_model_type_enum(registry));
    code.push_str("\n\n");

    // Generate ModelInfo struct
    code.push_str(&generate_model_info_struct());
    code.push_str("\n\n");

    // Generate individual model constants
    for (_category, cat_data) in &registry.model_categories {
        for model in &cat_data.models {
            code.push_str(&generate_model_constant(model));
            code.push_str("\n\n");
        }
    }

    // Generate registry array
    code.push_str(&generate_registry_array(registry));
    code.push_str("\n\n");

    // Generate accessor functions
    code.push_str(&generate_accessor_functions());

    code
}

fn generate_embedding_dimension_enum() -> String {
    r#"/// Embedding dimension specification supporting fixed and Matryoshka dimensions.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingDimension {
    /// Fixed dimension size
    Fixed(usize),
    /// Matryoshka representation with variable dimensions
    Matryoshka {
        /// Default/recommended dimension
        default: usize,
        /// Minimum supported dimension
        min: usize,
        /// Maximum supported dimension
        max: usize,
        /// Explicitly supported dimension values
        supported: &'static [usize],
    },
}

impl EmbeddingDimension {
    /// Get the default dimension size
    pub fn default_dim(&self) -> usize {
        match self {
            EmbeddingDimension::Fixed(d) => *d,
            EmbeddingDimension::Matryoshka { default, .. } => *default,
        }
    }

    /// Check if a specific dimension is supported
    pub fn supports_dimension(&self, dim: usize) -> bool {
        match self {
            EmbeddingDimension::Fixed(d) => *d == dim,
            EmbeddingDimension::Matryoshka { supported, .. } => {
                supported.contains(&dim)
            }
        }
    }

    /// Get all supported dimensions
    pub fn supported_dimensions(&self) -> Vec<usize> {
        match self {
            EmbeddingDimension::Fixed(d) => vec![*d],
            EmbeddingDimension::Matryoshka { supported, .. } => {
                supported.to_vec()
            }
        }
    }
}

impl std::fmt::Display for EmbeddingDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingDimension::Fixed(d) => write!(f, "{}", d),
            EmbeddingDimension::Matryoshka { default, min, max, .. } => {
                write!(f, "{} (Matryoshka: {}-{})", default, min, max)
            }
        }
    }
}"#
    .to_string()
}

fn generate_model_type_enum(registry: &ModelRegistry) -> String {
    let mut types = std::collections::HashSet::new();
    for (_category, cat_data) in &registry.model_categories {
        for model in &cat_data.models {
            types.insert(&model.model_type);
        }
    }

    let mut variants = types.into_iter().collect::<Vec<_>>();
    variants.sort();

    let mut code = String::from(
        r#"/// Type of embedding model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
"#,
    );

    for variant in &variants {
        let variant_name = to_pascal_case(variant);
        code.push_str(&format!("    /// {} model\n", variant_name));
        code.push_str(&format!("    {},\n", variant_name));
    }

    code.push_str("}\n");
    code
}

fn generate_model_info_struct() -> String {
    r#"/// Comprehensive metadata for a model from the registry.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Unique identifier (kebab-case)
    pub id: &'static str,
    /// Model category
    pub model_type: ModelType,
    /// Display name
    pub name: &'static str,
    /// HuggingFace Hub repository ID
    pub huggingface_id: &'static str,
    /// Organization that released the model
    pub organization: &'static str,
    /// Release year or date
    pub release_date: &'static str,
    /// Architecture type (bert, distilbert, jina-bert, etc.)
    pub architecture_type: &'static str,
    /// Architecture variant
    pub architecture_variant: &'static str,
    /// Whether model has a projection layer
    pub has_projection: bool,
    /// Projection output dimensions (if applicable)
    pub projection_dims: Option<usize>,
    /// Number of parameters (as string, e.g., "110M")
    pub parameters: &'static str,
    /// Embedding dimension specification (fixed or Matryoshka)
    pub embedding_dim: EmbeddingDimension,
    /// Hidden layer dimensions
    pub hidden_dim: usize,
    /// Maximum sequence length
    pub context_length: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Supported languages
    pub languages: &'static [&'static str],
    /// Supported modalities
    pub modalities: &'static [&'static str],
    /// Whether model supports multi-vector embeddings
    pub multi_vector: bool,
    /// Available quantization methods
    pub quantization: &'static [&'static str],
    /// BEIR average score (if available)
    pub beir_avg: f64,
    /// MS MARCO MRR@10 score (if available)
    pub ms_marco_mrr10: f64,
    /// License
    pub license: &'static str,
    /// Description
    pub description: &'static str,
}"#
    .to_string()
}

fn generate_model_constant(model: &ModelMetadata) -> String {
    let const_name = to_screaming_snake_case(&model.id);
    let model_type = to_pascal_case(&model.model_type);

    let projection_dims = match model.architecture.projection_dims {
        Some(dim) => format!("Some({})", dim),
        None => "None".to_string(),
    };

    let languages = model
        .capabilities
        .languages
        .iter()
        .map(|l| format!("\"{}\"", l))
        .collect::<Vec<_>>()
        .join(", ");

    let modalities = model
        .capabilities
        .modalities
        .iter()
        .map(|m| format!("\"{}\"", m))
        .collect::<Vec<_>>()
        .join(", ");

    let quantization = model
        .capabilities
        .quantization
        .iter()
        .map(|q| format!("\"{}\"", q))
        .collect::<Vec<_>>()
        .join(", ");

    let (embedding_dim_code, embedding_dim_display) = match &model.specs.embedding_dim {
        EmbeddingDimSpec::Fixed(dim) => (
            format!("EmbeddingDimension::Fixed({})", dim),
            dim.to_string(),
        ),
        EmbeddingDimSpec::Matryoshka { default, matryoshka } => {
            let supported = matryoshka
                .supported
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            (
                format!(
                    "EmbeddingDimension::Matryoshka {{ default: {}, min: {}, max: {}, supported: &[{}] }}",
                    default, matryoshka.min, matryoshka.max, supported
                ),
                format!("{} (Matryoshka: {}-{})", default, matryoshka.min, matryoshka.max),
            )
        }
    };

    format!(
        r#"/// {}
///
/// {}
///
/// - Organization: {}
/// - Release: {}
/// - Parameters: {}
/// - Embedding dim: {}
/// - Context length: {}
/// - Languages: {}
pub const {}: ModelInfo = ModelInfo {{
    id: "{}",
    model_type: ModelType::{},
    name: "{}",
    huggingface_id: "{}",
    organization: "{}",
    release_date: "{}",
    architecture_type: "{}",
    architecture_variant: "{}",
    has_projection: {},
    projection_dims: {},
    parameters: "{}",
    embedding_dim: {},
    hidden_dim: {},
    context_length: {},
    max_position_embeddings: {},
    vocab_size: {},
    languages: &[{}],
    modalities: &[{}],
    multi_vector: {},
    quantization: &[{}],
    beir_avg: {},
    ms_marco_mrr10: {},
    license: "{}",
    description: "{}",
}};"#,
        model.name,
        model.description,
        model.organization,
        model.release_date,
        model.specs.parameters,
        embedding_dim_display,
        model.specs.context_length,
        model.capabilities.languages.len(),
        const_name,
        model.id,
        model_type,
        model.name,
        model.huggingface_id,
        model.organization,
        model.release_date,
        model.architecture.arch_type,
        model.architecture.variant,
        model.architecture.has_projection,
        projection_dims,
        model.specs.parameters,
        embedding_dim_code,
        model.specs.hidden_dim,
        model.specs.context_length,
        model.specs.max_position_embeddings,
        model.specs.vocab_size,
        languages,
        modalities,
        model.capabilities.multi_vector,
        quantization,
        model.performance.beir_avg,
        model.performance.ms_marco_mrr10,
        model.license,
        model.description,
    )
}

fn generate_registry_array(registry: &ModelRegistry) -> String {
    let mut code = String::from(
        r#"/// Complete model registry containing all available models.
///
/// This is generated at compile time from models.json.
pub const MODEL_REGISTRY: &[ModelInfo] = &[
"#,
    );

    for (_category, cat_data) in &registry.model_categories {
        for model in &cat_data.models {
            let const_name = to_screaming_snake_case(&model.id);
            code.push_str(&format!("    {},\n", const_name));
        }
    }

    code.push_str("];\n");
    code
}

fn generate_accessor_functions() -> String {
    r#"/// Get a model by its ID.
///
/// # Example
///
/// ```
/// use tessera::model_registry::get_model;
///
/// let model = get_model("colbert-v2").expect("Model not found");
/// assert_eq!(model.embedding_dim.default_dim(), 128);
/// ```
pub fn get_model(id: &str) -> Option<&'static ModelInfo> {
    MODEL_REGISTRY.iter().find(|m| m.id == id)
}

/// Get all models of a specific type.
///
/// # Example
///
/// ```
/// use tessera::model_registry::{models_by_type, ModelType};
///
/// let colbert_models = models_by_type(ModelType::Colbert);
/// for model in colbert_models {
///     println!("{}: {} dims", model.name, model.embedding_dim.default_dim());
/// }
/// ```
pub fn models_by_type(model_type: ModelType) -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|m| m.model_type == model_type)
        .collect()
}

/// Get all models from a specific organization.
pub fn models_by_organization(organization: &str) -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|m| m.organization.eq_ignore_ascii_case(organization))
        .collect()
}

/// Get all models supporting a specific language.
pub fn models_by_language(language: &str) -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|m| m.languages.contains(&language))
        .collect()
}

/// Get all models with default embedding dimension less than or equal to the specified size.
pub fn models_by_max_embedding_dim(max_dim: usize) -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|m| m.embedding_dim.default_dim() <= max_dim)
        .collect()
}

/// Get all models supporting Matryoshka representation.
pub fn models_with_matryoshka() -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|m| matches!(m.embedding_dim, EmbeddingDimension::Matryoshka { .. }))
        .collect()
}"#
    .to_string()
}

fn to_pascal_case(s: &str) -> String {
    s.split('-')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    first.to_uppercase().collect::<String>() + chars.as_str().to_lowercase().as_str()
                }
            }
        })
        .collect()
}

fn to_screaming_snake_case(s: &str) -> String {
    s.replace('-', "_").to_uppercase()
}
