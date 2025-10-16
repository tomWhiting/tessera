//! Model registry with compile-time generated model metadata.
//!
//! This module provides access to all supported models through a type-safe
//! registry generated at compile time from models.json.
//!
//! # Example
//!
//! ```no_run
//! use tessera::model_registry::{get_model, MODEL_REGISTRY, COLBERT_V2};
//!
//! // Access specific model constant
//! println!("Model: {}", COLBERT_V2.name);
//! println!("Dimensions: {}", COLBERT_V2.embedding_dim);
//!
//! // Lookup by ID
//! let model = get_model("colbert-small").expect("Model not found");
//! println!("Found: {}", model.name);
//!
//! // List all models
//! for model in MODEL_REGISTRY {
//!     println!("{}: {} dims, {}K context",
//!         model.name,
//!         model.embedding_dim,
//!         model.context_length / 1000
//!     );
//! }
//! ```

// Include the generated model registry code from visible location
include!("generated.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_not_empty() {
        assert!(
            !MODEL_REGISTRY.is_empty(),
            "Model registry should contain models"
        );
    }

    #[test]
    fn test_get_model_by_id() {
        let model = get_model("colbert-v2");
        assert!(model.is_some(), "Should find colbert-v2");

        let model = model.unwrap();
        assert_eq!(model.id, "colbert-v2");
        assert_eq!(model.embedding_dim.default_dim(), 128);
        assert_eq!(model.context_length, 512);
    }

    #[test]
    fn test_get_nonexistent_model() {
        let model = get_model("nonexistent-model");
        assert!(model.is_none(), "Should return None for nonexistent model");
    }

    #[test]
    fn test_models_by_type() {
        let colbert_models = models_by_type(ModelType::Colbert);
        assert!(!colbert_models.is_empty(), "Should have ColBERT models");

        for model in colbert_models {
            assert_eq!(model.model_type, ModelType::Colbert);
        }
    }

    #[test]
    fn test_models_by_organization() {
        let stanford_models = models_by_organization("Stanford NLP");
        assert!(!stanford_models.is_empty(), "Should have Stanford models");

        for model in stanford_models {
            assert_eq!(model.organization, "Stanford NLP");
        }
    }

    #[test]
    fn test_models_by_language() {
        let english_models = models_by_language("en");
        assert!(!english_models.is_empty(), "Should have English models");

        for model in english_models {
            assert!(model.languages.contains(&"en"));
        }
    }

    #[test]
    fn test_models_by_max_embedding_dim() {
        let compact_models = models_by_max_embedding_dim(128);
        assert!(!compact_models.is_empty(), "Should have compact models");

        for model in compact_models {
            assert!(model.embedding_dim.default_dim() <= 128);
        }
    }

    #[test]
    fn test_models_with_matryoshka() {
        let matryoshka_models = models_with_matryoshka();

        for model in matryoshka_models {
            assert!(
                matches!(model.embedding_dim, EmbeddingDimension::Matryoshka { .. }),
                "Model should have matryoshka support"
            );
            let dims = model.embedding_dim.supported_dimensions();
            assert!(!dims.is_empty(), "Should have matryoshka dimensions");
        }
    }

    #[test]
    fn test_colbert_v2_constant() {
        assert_eq!(COLBERT_V2.id, "colbert-v2");
        assert_eq!(COLBERT_V2.huggingface_id, "colbert-ir/colbertv2.0");
        assert_eq!(COLBERT_V2.embedding_dim.default_dim(), 128);
        assert_eq!(COLBERT_V2.context_length, 512);
        assert!(COLBERT_V2.has_projection);
        assert_eq!(COLBERT_V2.projection_dims, Some(128));
    }

    #[test]
    fn test_colbert_small_constant() {
        assert_eq!(COLBERT_SMALL.id, "colbert-small");
        assert_eq!(
            COLBERT_SMALL.huggingface_id,
            "answerdotai/answerai-colbert-small-v1"
        );
        assert_eq!(COLBERT_SMALL.embedding_dim.default_dim(), 96);
        assert_eq!(COLBERT_SMALL.context_length, 512);
    }

    #[test]
    fn test_jina_colbert_v2_constant() {
        assert_eq!(JINA_COLBERT_V2.id, "jina-colbert-v2");
        assert_eq!(JINA_COLBERT_V2.huggingface_id, "jinaai/jina-colbert-v2");
        assert_eq!(JINA_COLBERT_V2.embedding_dim.default_dim(), 768);
        assert_eq!(JINA_COLBERT_V2.context_length, 8192);
        assert!(!JINA_COLBERT_V2.has_projection);
        // Test Matryoshka support
        assert!(JINA_COLBERT_V2.embedding_dim.supports_dimension(64));
        assert!(JINA_COLBERT_V2.embedding_dim.supports_dimension(768));
    }

    #[test]
    fn test_all_models_have_valid_metadata() {
        for model in MODEL_REGISTRY {
            assert!(!model.id.is_empty(), "Model ID should not be empty");
            assert!(!model.name.is_empty(), "Model name should not be empty");
            assert!(
                !model.huggingface_id.is_empty(),
                "HuggingFace ID should not be empty"
            );
            assert!(
                model.embedding_dim.default_dim() > 0,
                "Embedding dim should be positive"
            );
            assert!(
                model.context_length > 0,
                "Context length should be positive"
            );
            // Only text/vision models need languages; timeseries models don't
            if model.modalities.contains(&"text") || model.modalities.contains(&"vision") {
                assert!(
                    !model.languages.is_empty(),
                    "Text/vision model {} should have at least one language",
                    model.id
                );
            }
        }
    }
}
