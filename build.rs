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
    #[serde(default)]
    pooling: Option<PoolingConfig>,
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
    #[serde(default)]
    strategy: Option<String>,
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
struct PoolingConfig {
    strategy: String,
    normalize: bool,
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

    let total_models = registry
        .model_categories
        .values()
        .map(|cat| cat.models.len())
        .sum::<usize>();

    validate_registry(&registry);

    let generated_code = generate_code(&registry);

    // Write to OUT_DIR (required for compilation)
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest_path = Path::new(&out_dir).join("model_registry.rs");
    fs::write(&dest_path, &generated_code).expect("Failed to write generated model registry code");

    // ALSO write to src/models/generated.rs (visible in source tree)
    let visible_path = Path::new("src/models/generated.rs");
    fs::write(visible_path, &generated_code)
        .expect("Failed to write visible generated model registry code");

    println!(
        "cargo:warning=Generated model registry with {} models across {} categories",
        total_models,
        registry.model_categories.len()
    );
}

fn validate_registry(registry: &ModelRegistry) {
    let mut ids = std::collections::HashSet::new();

    for cat_data in registry.model_categories.values() {
        for model in &cat_data.models {
            // Check for duplicate IDs
            assert!(
                ids.insert(&model.id),
                "Duplicate model ID found: {}",
                model.id
            );

            // Validate embedding dimensions
            let embedding_dim =
                match &model.specs.embedding_dim {
                    EmbeddingDimSpec::Fixed(dim) => {
                        assert!(*dim != 0, "Model {} has invalid embedding_dim: 0", model.id);
                        *dim
                    }
                    EmbeddingDimSpec::Matryoshka {
                        default,
                        matryoshka,
                    } => {
                        // Validate Matryoshka configuration
                        assert!(
                            matryoshka.min < matryoshka.max,
                            "Model {} has invalid Matryoshka range: min ({}) >= max ({})",
                            model.id,
                            matryoshka.min,
                            matryoshka.max
                        );
                        assert!(
                            *default >= matryoshka.min && *default <= matryoshka.max,
                            "Model {} has default dimension ({}) outside Matryoshka range ({}-{})",
                            model.id,
                            default,
                            matryoshka.min,
                            matryoshka.max
                        );
                        // Validate all supported dimensions are within range
                        for &dim in &matryoshka.supported {
                            assert!(
                            dim >= matryoshka.min && dim <= matryoshka.max,
                            "Model {} has supported dimension {} outside Matryoshka range ({}-{})",
                            model.id, dim, matryoshka.min, matryoshka.max
                        );
                        }
                        // Validate supported dimensions are in ascending order
                        let mut sorted = matryoshka.supported.clone();
                        sorted.sort_unstable();
                        assert!(
                            sorted == matryoshka.supported,
                            "Model {} Matryoshka supported dimensions must be in ascending order",
                            model.id
                        );
                        // Validate strategy if present
                        if let Some(ref strategy) = matryoshka.strategy {
                            let valid_strategies =
                                ["truncate_hidden", "truncate_output", "truncate_pooled"];
                            assert!(
                                valid_strategies.contains(&strategy.as_str()),
                                "Model {} has invalid Matryoshka strategy '{}'. Valid: {:?}",
                                model.id,
                                strategy,
                                valid_strategies
                            );
                        }
                        *default
                    }
                };

            // Validate context length
            assert!(
                model.specs.context_length != 0,
                "Model {} has invalid context_length: 0",
                model.id
            );

            // Validate HuggingFace ID format
            assert!(
                model.huggingface_id.contains('/'),
                "Model {} has invalid huggingface_id format: {}",
                model.id,
                model.huggingface_id
            );

            // Validate projection consistency
            assert!(
                !model.architecture.has_projection || model.architecture.projection_dims.is_some(),
                "Model {} has has_projection=true but no projection_dims",
                model.id
            );

            if model.architecture.has_projection {
                if let Some(proj_dim) = model.architecture.projection_dims {
                    assert!(
                        proj_dim == embedding_dim,
                        "Model {} projection_dims ({}) doesn't match embedding_dim ({})",
                        model.id,
                        proj_dim,
                        embedding_dim
                    );
                }
            }

            // Validate pooling configuration if present
            if let Some(ref pooling) = model.pooling {
                let valid_strategies = ["mean", "cls", "max"];
                let strategy_lower = pooling.strategy.to_lowercase();
                assert!(
                    valid_strategies.contains(&strategy_lower.as_str()),
                    "Model {} has invalid pooling strategy '{}'. Valid: {:?}",
                    model.id,
                    pooling.strategy,
                    valid_strategies
                );
            }
        }
    }
}

fn generate_code(registry: &ModelRegistry) -> String {
    let mut code = String::from(
        r"// This file is AUTO-GENERATED by build.rs from models.json
// DO NOT EDIT MANUALLY - changes will be overwritten
// To modify models, edit models.json and rebuild

",
    );

    // Generate EmbeddingDimension enum
    code.push_str(&generate_embedding_dimension_enum());
    code.push_str("\n\n");

    // Generate PoolingStrategy enum and PoolingConfig struct
    code.push_str(&generate_pooling_types());
    code.push_str("\n\n");

    // Generate ModelType enum
    code.push_str(&generate_model_type_enum(registry));
    code.push_str("\n\n");

    // Generate ModelInfo struct
    code.push_str(&generate_model_info_struct());
    code.push_str("\n\n");

    // Generate individual model constants
    for cat_data in registry.model_categories.values() {
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

fn generate_pooling_types() -> String {
    r#"/// Pooling strategy for dense encodings.
///
/// Determines how token-level embeddings are aggregated into a single vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use the [CLS] token embedding (first token).
    ///
    /// Common in BERT-style models where [CLS] is trained to represent
    /// the entire sequence.
    Cls,

    /// Average all token embeddings (weighted by attention mask).
    ///
    /// Produces a centroid representation of all tokens, ignoring padding.
    /// Most common strategy in sentence transformers.
    Mean,

    /// Element-wise maximum across all token embeddings.
    ///
    /// Captures the most salient features from any token position.
    Max,
}

/// Pooling configuration for dense embedding models.
///
/// Specifies how token-level embeddings should be pooled into a single
/// vector representation, and whether the result should be normalized.
#[derive(Debug, Clone, Copy)]
pub struct PoolingConfig {
    /// The pooling strategy to use
    pub strategy: PoolingStrategy,
    /// Whether to L2-normalize the pooled embedding
    pub normalize: bool,
}"#
    .to_string()
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
        /// Truncation strategy (when to apply Matryoshka truncation)
        strategy: Option<&'static str>,
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

    /// Get the Matryoshka truncation strategy, if applicable
    pub fn matryoshka_strategy(&self) -> Option<&'static str> {
        match self {
            EmbeddingDimension::Fixed(_) => None,
            EmbeddingDimension::Matryoshka { strategy, .. } => *strategy,
        }
    }
}

impl std::fmt::Display for EmbeddingDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingDimension::Fixed(d) => write!(f, "{}", d),
            EmbeddingDimension::Matryoshka { default, min, max, strategy, .. } => {
                let strategy_str = strategy.map(|s| format!(" [{}]", s)).unwrap_or_default();
                write!(f, "{} (Matryoshka: {}-{}{})", default, min, max, strategy_str)
            }
        }
    }
}"#
    .to_string()
}

fn generate_model_type_enum(registry: &ModelRegistry) -> String {
    let mut types = std::collections::HashSet::new();
    for cat_data in registry.model_categories.values() {
        for model in &cat_data.models {
            types.insert(&model.model_type);
        }
    }

    let mut variants = types.into_iter().collect::<Vec<_>>();
    variants.sort();

    let mut code = String::from(
        r"/// Type of embedding model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
",
    );

    for variant in &variants {
        let variant_name = to_pascal_case(variant);
        code.push_str("    /// ");
        code.push_str(&variant_name);
        code.push_str(" model\n");
        code.push_str("    ");
        code.push_str(&variant_name);
        code.push_str(",\n");
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
    /// Pooling configuration (for dense models)
    pub pooling: Option<PoolingConfig>,
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

#[allow(clippy::too_many_lines)]
fn generate_model_constant(model: &ModelMetadata) -> String {
    let const_name = to_screaming_snake_case(&model.id);
    let model_type = to_pascal_case(&model.model_type);

    let projection_dims = model
        .architecture
        .projection_dims
        .map_or_else(|| "None".to_string(), |dim| format!("Some({dim})"));

    // Generate pooling constant and reference
    let (pooling_const_def, pooling_ref) = if let Some(ref pooling_cfg) = model.pooling {
        let pooling_const_name = format!("{}_POOLING", const_name);
        let strategy_enum = pooling_strategy_to_enum(&pooling_cfg.strategy);
        let normalize = pooling_cfg.normalize;

        let pooling_def = format!(
            "/// Pooling configuration for {}.\npub const {}: PoolingConfig = PoolingConfig {{\n    strategy: {},\n    normalize: {},\n}};\n\n",
            model.name, pooling_const_name, strategy_enum, normalize
        );

        (pooling_def, format!("Some({})", pooling_const_name))
    } else {
        (String::new(), "None".to_string())
    };

    let languages = model
        .capabilities
        .languages
        .iter()
        .map(|l| format!("\"{l}\""))
        .collect::<Vec<_>>()
        .join(", ");

    let modalities = model
        .capabilities
        .modalities
        .iter()
        .map(|m| format!("\"{m}\""))
        .collect::<Vec<_>>()
        .join(", ");

    let quantization = model
        .capabilities
        .quantization
        .iter()
        .map(|q| format!("\"{q}\""))
        .collect::<Vec<_>>()
        .join(", ");

    let (embedding_dim_code, embedding_dim_display) = match &model.specs.embedding_dim {
        EmbeddingDimSpec::Fixed(dim) => {
            (format!("EmbeddingDimension::Fixed({dim})"), dim.to_string())
        }
        EmbeddingDimSpec::Matryoshka {
            default,
            matryoshka,
        } => {
            let supported = matryoshka
                .supported
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let strategy_code = matryoshka
                .strategy
                .as_ref()
                .map_or_else(|| "None".to_string(), |s| format!("Some(\"{s}\")"));
            let strategy_display = matryoshka
                .strategy
                .as_ref()
                .map_or_else(String::new, |s| format!(" [{s}]"));
            (
                format!(
                    "EmbeddingDimension::Matryoshka {{ default: {default}, min: {}, max: {}, supported: &[{supported}], strategy: {strategy_code} }}",
                    matryoshka.min, matryoshka.max
                ),
                format!("{default} (Matryoshka: {}-{}{strategy_display})", matryoshka.min, matryoshka.max),
            )
        }
    };

    format!(
        r#"{}/// {}
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
    pooling: {},
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
        pooling_const_def,
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
        pooling_ref,
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
        format_f64(model.performance.beir_avg),
        format_f64(model.performance.ms_marco_mrr10),
        model.license,
        model.description,
    )
}

fn generate_registry_array(registry: &ModelRegistry) -> String {
    let mut code = String::from(
        r"/// Complete model registry containing all available models.
///
/// This is generated at compile time from models.json.
pub const MODEL_REGISTRY: &[ModelInfo] = &[
",
    );

    for cat_data in registry.model_categories.values() {
        for model in &cat_data.models {
            let const_name = to_screaming_snake_case(&model.id);
            code.push_str("    ");
            code.push_str(&const_name);
            code.push_str(",\n");
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
}

/// Get a model by its HuggingFace Hub ID.
///
/// # Example
///
/// ```
/// use tessera::model_registry::get_model_by_hf_id;
///
/// let model = get_model_by_hf_id("jinaai/jina-colbert-v2");
/// assert!(model.is_some());
/// ```
pub fn get_model_by_hf_id(hf_id: &str) -> Option<&'static ModelInfo> {
    MODEL_REGISTRY.iter().find(|m| m.huggingface_id == hf_id)
}"#
    .to_string()
}

fn to_pascal_case(s: &str) -> String {
    s.split('-')
        .map(|word| {
            let mut chars = word.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>() + chars.as_str().to_lowercase().as_str()
            })
        })
        .collect()
}

fn to_screaming_snake_case(s: &str) -> String {
    s.replace(['-', '.'], "_").to_uppercase()
}

fn format_f64(value: f64) -> String {
    // Ensure f64 values are formatted with decimal point
    if value.fract() == 0.0 && value.abs() < 1e10 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

fn pooling_strategy_to_enum(strategy: &str) -> &'static str {
    match strategy.to_lowercase().as_str() {
        "mean" => "PoolingStrategy::Mean",
        "cls" => "PoolingStrategy::Cls",
        "max" => "PoolingStrategy::Max",
        _ => panic!("Invalid pooling strategy: {}", strategy),
    }
}
