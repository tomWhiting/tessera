pub fn generate_accessor_functions() -> String {
    r#"/// Get a catalog entry by its ID.
///
/// This lookup is catalog-complete and may return a model without a runtime
/// adapter. Call [`ModelInfo::is_runnable`] before selecting it for execution.
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
    MODEL_REGISTRY.iter().find(|model| model.id == id)
}

/// Get every model with a currently exposed runtime path.
///
/// This includes both [`SupportTier::Supported`] and
/// [`SupportTier::Experimental`] entries and excludes catalog-only metadata.
pub fn runnable_models() -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|model| model.is_runnable())
        .collect()
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
        .filter(|model| model.model_type == model_type)
        .collect()
}

/// Get all models from a specific organization.
pub fn models_by_organization(organization: &str) -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|model| model.organization.eq_ignore_ascii_case(organization))
        .collect()
}

/// Get all models supporting a specific language.
pub fn models_by_language(language: &str) -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|model| model.languages.contains(&language))
        .collect()
}

/// Get all models with default embedding dimension less than or equal to the specified size.
pub fn models_by_max_embedding_dim(max_dim: usize) -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|model| model.embedding_dim.default_dim() <= max_dim)
        .collect()
}

/// Get all models supporting Matryoshka representation.
pub fn models_with_matryoshka() -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY
        .iter()
        .filter(|model| matches!(model.embedding_dim, EmbeddingDimension::Matryoshka { .. }))
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
    MODEL_REGISTRY
        .iter()
        .find(|model| model.huggingface_id == hf_id)
}"#
    .to_string()
}
