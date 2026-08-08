use crate::schema::{EmbeddingDimSpec, ModelMetadata, ModelRegistry, SupportTier};
use std::collections::HashSet;

pub fn validate_registry(registry: &ModelRegistry) {
    let mut ids = HashSet::new();

    for model in registry.models() {
        assert!(
            ids.insert(&model.id),
            "Duplicate model ID found: {}",
            model.id
        );

        let embedding_dim = validate_embedding_dimension(model);

        assert!(
            model.specs.context_length != 0,
            "Model {} has invalid context_length: 0",
            model.id
        );
        assert!(
            model.huggingface_id.contains('/'),
            "Model {} has invalid huggingface_id format: {}",
            model.id,
            model.huggingface_id
        );
        assert!(
            !model.architecture.has_projection || model.architecture.projection_dims.is_some(),
            "Model {} has has_projection=true but no projection_dims",
            model.id
        );

        validate_support(model);

        if let Some(projection_dim) = model.architecture.projection_dims {
            if model.architecture.has_projection {
                assert_eq!(
                    projection_dim, embedding_dim,
                    "Model {} projection_dims doesn't match embedding_dim",
                    model.id
                );
            }
        }

        validate_pooling(model);
    }
}

fn validate_support(model: &ModelMetadata) {
    let note = model.support.note.trim();
    assert!(
        !note.is_empty(),
        "Model {} must have a nonempty support note",
        model.id
    );
    assert!(
        !note.contains(['\n', '\r']),
        "Model {} support note must be a single line",
        model.id
    );
    assert!(
        note.chars().count() <= 160,
        "Model {} support note must be concise (160 characters or fewer)",
        model.id
    );
    if model.support.tier == SupportTier::CatalogOnly {
        assert!(
            !model.support.tier.is_runnable(),
            "Catalog-only model {} cannot be runnable",
            model.id
        );
    }
}

fn validate_embedding_dimension(model: &ModelMetadata) -> usize {
    match &model.specs.embedding_dim {
        EmbeddingDimSpec::Fixed(dimension) => {
            assert_ne!(
                *dimension, 0,
                "Model {} has invalid embedding_dim: 0",
                model.id
            );
            *dimension
        }
        EmbeddingDimSpec::Matryoshka {
            default,
            matryoshka,
        } => {
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

            for &dimension in &matryoshka.supported {
                assert!(
                    dimension >= matryoshka.min && dimension <= matryoshka.max,
                    "Model {} has supported dimension {} outside Matryoshka range ({}-{})",
                    model.id,
                    dimension,
                    matryoshka.min,
                    matryoshka.max
                );
            }

            let mut sorted = matryoshka.supported.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted, matryoshka.supported,
                "Model {} Matryoshka supported dimensions must be in ascending order",
                model.id
            );

            if let Some(strategy) = &matryoshka.strategy {
                let valid = ["truncate_hidden", "truncate_output", "truncate_pooled"];
                assert!(
                    valid.contains(&strategy.as_str()),
                    "Model {} has invalid Matryoshka strategy '{}'. Valid: {:?}",
                    model.id,
                    strategy,
                    valid
                );
            }

            *default
        }
    }
}

fn validate_pooling(model: &ModelMetadata) {
    let Some(pooling) = &model.pooling else {
        return;
    };

    let valid = ["mean", "cls", "max", "last_token"];
    let strategy = pooling.strategy.to_lowercase();
    assert!(
        valid.contains(&strategy.as_str()),
        "Model {} has invalid pooling strategy '{}'. Valid: {:?}",
        model.id,
        pooling.strategy,
        valid
    );
}

#[cfg(test)]
#[path = "validation_tests.rs"]
mod tests;
