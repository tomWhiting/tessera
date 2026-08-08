use crate::schema::{EmbeddingDimSpec, ModelMetadata};

pub fn generate_model_constant(model: &ModelMetadata) -> String {
    let constant_name = to_screaming_snake_case(&model.id);
    let model_type = to_pascal_case(&model.model_type);

    let projection_dims = model.architecture.projection_dims.map_or_else(
        || "None".to_string(),
        |dimension| format!("Some({dimension})"),
    );

    let (pooling_definition, pooling_reference) = pooling_config(model, &constant_name);

    let languages = quoted_list(&model.capabilities.languages);
    let modalities = quoted_list(&model.capabilities.modalities);
    let quantization = quoted_list(&model.capabilities.quantization);
    let (embedding_dimension, embedding_dimension_display) = embedding_dimension(model);
    let support_note = format!("{:?}", model.support.note);
    let revision = optional_string_literal(model.revision.as_deref());
    let safetensors_file = optional_string_literal(model.files.weights.safetensors.as_deref());

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
/// - Support: {}
pub const {}: ModelInfo = ModelInfo {{
    id: "{}",
    model_type: ModelType::{},
    support_tier: SupportTier::{},
    support_note: {},
    name: "{}",
    huggingface_id: "{}",
    revision: {},
    tokenizer_file: "{}",
    config_file: "{}",
    safetensors_file: {},
    pytorch_file: "{}",
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
    license: "{}",
    description: "{}",
}};"#,
        pooling_definition,
        model.name,
        model.description,
        model.organization,
        model.release_date,
        model.specs.parameters,
        embedding_dimension_display,
        model.specs.context_length,
        model.capabilities.languages.len(),
        model.support.tier.rust_variant(),
        constant_name,
        model.id,
        model_type,
        model.support.tier.rust_variant(),
        support_note,
        model.name,
        model.huggingface_id,
        revision,
        model.files.tokenizer,
        model.files.config,
        safetensors_file,
        model.files.weights.pytorch,
        model.organization,
        model.release_date,
        model.architecture.arch_type,
        model.architecture.variant,
        model.architecture.has_projection,
        projection_dims,
        pooling_reference,
        model.specs.parameters,
        embedding_dimension,
        model.specs.hidden_dim,
        model.specs.context_length,
        model.specs.max_position_embeddings,
        model.specs.vocab_size,
        languages,
        modalities,
        model.capabilities.multi_vector,
        quantization,
        model.license,
        model.description,
    )
}

fn pooling_config(model: &ModelMetadata, constant_name: &str) -> (String, String) {
    model.pooling.as_ref().map_or_else(
        || (String::new(), "None".to_string()),
        |pooling| {
            let pooling_name = format!("{constant_name}_POOLING");
            let strategy = pooling_strategy_to_enum(&pooling.strategy);
            let definition = format!(
                "/// Pooling configuration for {}.\npub const {}: PoolingConfig = PoolingConfig {{\n    strategy: {},\n    normalize: {},\n}};\n\n",
                model.name, pooling_name, strategy, pooling.normalize
            );
            (definition, format!("Some({pooling_name})"))
        },
    )
}

fn optional_string_literal(value: Option<&str>) -> String {
    value.map_or_else(|| "None".to_string(), |value| format!("Some({value:?})"))
}

pub fn to_pascal_case(value: &str) -> String {
    value
        .split('-')
        .map(|word| {
            let mut characters = word.chars();
            characters.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>()
                    + characters.as_str().to_lowercase().as_str()
            })
        })
        .collect()
}

pub fn to_screaming_snake_case(value: &str) -> String {
    value.replace(['-', '.'], "_").to_uppercase()
}

fn quoted_list(values: &[String]) -> String {
    values
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

fn embedding_dimension(model: &ModelMetadata) -> (String, String) {
    match &model.specs.embedding_dim {
        EmbeddingDimSpec::Fixed(dimension) => (
            format!("EmbeddingDimension::Fixed({dimension})"),
            dimension.to_string(),
        ),
        EmbeddingDimSpec::Matryoshka {
            default,
            matryoshka,
        } => {
            let supported = matryoshka
                .supported
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let strategy = matryoshka.strategy.as_ref().map_or_else(
                || "None".to_string(),
                |strategy| format!("Some(\"{strategy}\")"),
            );
            let display_strategy = matryoshka
                .strategy
                .as_ref()
                .map_or_else(String::new, |strategy| format!(" [\\{strategy}\\]"));

            (
                format!(
                    "EmbeddingDimension::Matryoshka {{ default: {default}, min: {}, max: {}, supported: &[{supported}], strategy: {strategy} }}",
                    matryoshka.min, matryoshka.max
                ),
                format!(
                    "{default} (Matryoshka: {}-{}{display_strategy})",
                    matryoshka.min, matryoshka.max
                ),
            )
        }
    }
}

fn pooling_strategy_to_enum(strategy: &str) -> &'static str {
    match strategy.to_lowercase().as_str() {
        "mean" => "PoolingStrategy::Mean",
        "cls" => "PoolingStrategy::Cls",
        "max" => "PoolingStrategy::Max",
        "last_token" | "lasttoken" => "PoolingStrategy::LastToken",
        _ => panic!("Invalid pooling strategy: {strategy}"),
    }
}
