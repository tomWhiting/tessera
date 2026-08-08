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
    let model = get_model("colbert-v2").expect("registered ColBERT v2");
    assert_eq!(model.id, "colbert-v2");
    assert_eq!(model.huggingface_id, "colbert-ir/colbertv2.0");
    assert_eq!(
        model.revision,
        Some("c1e84128e85ef755c096a95bdb06b47793b13acf")
    );
    assert_eq!(model.embedding_dim.default_dim(), 128);
    assert_eq!(model.context_length, 512);
    assert!(model.has_projection);
    assert_eq!(model.projection_dims, Some(128));
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
    let model = get_model("jina-colbert-v2").expect("registered Jina ColBERT v2");
    assert_eq!(model.id, "jina-colbert-v2");
    assert_eq!(model.huggingface_id, "jinaai/jina-colbert-v2");
    assert_eq!(model.embedding_dim.default_dim(), 128);
    assert_eq!(model.context_length, 8192);
    assert_eq!(model.max_position_embeddings, 8194);
    assert_eq!(model.hidden_dim, 1024);
    assert_eq!(model.vocab_size, 250_004);
    assert_eq!(model.architecture_type, "xlm-roberta");
    assert!(model.has_projection);
    assert_eq!(model.projection_dims, Some(128));
    assert!(model.embedding_dim.supports_dimension(128));
    assert!(!model.embedding_dim.supports_dimension(64));
    assert_eq!(model.license, "CC-BY-NC-4.0");
    assert_eq!(model.support_tier, SupportTier::CatalogOnly);
}

#[test]
fn support_tier_runnability_is_exhaustive() {
    assert!(SupportTier::Supported.is_runnable());
    assert!(SupportTier::Experimental.is_runnable());
    assert!(!SupportTier::CatalogOnly.is_runnable());
}

#[test]
fn support_contract_matches_the_audited_catalog() {
    let mut actual = MODEL_REGISTRY
        .iter()
        .map(|model| (model.id, model.support_tier))
        .collect::<Vec<_>>();
    actual.sort_unstable_by(|left, right| left.0.cmp(right.0));

    let expected = [
        ("bge-base-en-v1.5", SupportTier::Experimental),
        ("bge-m3-multi", SupportTier::CatalogOnly),
        ("chronos-bolt-small", SupportTier::CatalogOnly),
        ("colbert-small", SupportTier::Experimental),
        ("colbert-v2", SupportTier::Experimental),
        ("colpali-v1.2", SupportTier::Experimental),
        ("colpali-v1.3-hf", SupportTier::CatalogOnly),
        ("gte-modern-colbert", SupportTier::CatalogOnly),
        ("jina-colbert-v2", SupportTier::CatalogOnly),
        ("jina-colbert-v2-64", SupportTier::CatalogOnly),
        ("jina-colbert-v2-96", SupportTier::CatalogOnly),
        ("jina-embeddings-v2-base-code", SupportTier::CatalogOnly),
        ("jina-embeddings-v2-base-en", SupportTier::Experimental),
        ("jina-embeddings-v2-small-en", SupportTier::Experimental),
        ("jina-embeddings-v3", SupportTier::CatalogOnly),
        ("minicoil-v1", SupportTier::CatalogOnly),
        ("nomic-embed-v1.5", SupportTier::Experimental),
        ("snowflake-arctic-l", SupportTier::Experimental),
        ("splade-pp-en-v1", SupportTier::Experimental),
        ("splade-pp-en-v2", SupportTier::Experimental),
        ("splade-v3", SupportTier::CatalogOnly),
        ("timesfm-1.0-200m", SupportTier::CatalogOnly),
    ];

    assert_eq!(actual.as_slice(), expected);
    assert!(MODEL_REGISTRY
        .iter()
        .all(|model| model.support_tier != SupportTier::Supported));
    assert!(MODEL_REGISTRY
        .iter()
        .all(|model| !model.support_note.trim().is_empty()));
}

#[test]
fn runnable_models_excludes_catalog_only_entries() {
    let actual = runnable_models();
    let actual_ids = actual.iter().map(|model| model.id).collect::<Vec<_>>();
    let expected_ids = [
        "bge-base-en-v1.5",
        "jina-embeddings-v2-small-en",
        "jina-embeddings-v2-base-en",
        "nomic-embed-v1.5",
        "snowflake-arctic-l",
        "colbert-small",
        "colbert-v2",
        "colpali-v1.2",
        "splade-pp-en-v1",
        "splade-pp-en-v2",
    ];

    assert_eq!(actual_ids, expected_ids);
    assert!(actual.iter().all(|model| model.is_runnable()));
    let filtered_ids = MODEL_REGISTRY
        .iter()
        .filter(|model| model.is_runnable())
        .map(|model| model.id)
        .collect::<Vec<_>>();
    assert_eq!(actual_ids, filtered_ids);
    assert!(!get_model("bge-m3-multi")
        .expect("catalog entry should remain discoverable")
        .is_runnable());
}

#[test]
fn catalog_descriptions_are_claim_neutral() {
    const UNSOURCED_CLAIM_MARKERS: &[&str] = &[
        "benchmark",
        "beir",
        "ms marco",
        "ms-marco",
        "mrr",
        "ndcg",
        "leaderboard",
        "latency",
        "faster",
        "fast inference",
        "compression",
        "competitive",
        "strong",
        "excellent",
        "frontier",
        "high-performance",
        "quality",
        "efficient",
        "improved",
        "recommended",
        "suitable",
    ];

    for model in MODEL_REGISTRY {
        assert!(
            !model.description.trim().is_empty(),
            "{} description",
            model.id
        );
        let description = model.description.to_ascii_lowercase();
        for marker in UNSOURCED_CLAIM_MARKERS {
            assert!(
                !description.contains(marker),
                "{} description contains unsourced claim marker {marker:?}",
                model.id
            );
        }
    }
}

#[test]
fn corrected_checkpoint_metadata_is_exposed() {
    assert_eq!(COLBERT_SMALL.architecture_type, "bert");
    assert_eq!(COLBERT_SMALL.hidden_dim, 384);
    assert_eq!(COLBERT_SMALL.projection_dims, Some(96));

    let gte = get_model("gte-modern-colbert").expect("registered GTE ModernColBERT");
    assert!(gte.has_projection);
    assert_eq!(gte.projection_dims, Some(128));
    assert_eq!(gte.embedding_dim.default_dim(), 128);

    let bge = get_model("bge-base-en-v1.5").expect("registered BGE base");
    assert_eq!(
        bge.pooling.expect("BGE pooling metadata").strategy,
        PoolingStrategy::Cls
    );

    let snowflake = get_model("snowflake-arctic-l").expect("registered Snowflake model");
    assert_eq!(snowflake.parameters, "568M");
    assert_eq!(snowflake.architecture_type, "xlm-roberta");
    assert_eq!(snowflake.context_length, 8192);
    assert_eq!(snowflake.max_position_embeddings, 8194);
    assert_eq!(snowflake.vocab_size, 250_002);
    assert_eq!(
        snowflake
            .pooling
            .expect("Snowflake pooling metadata")
            .strategy,
        PoolingStrategy::Cls
    );

    for id in ["splade-pp-en-v1", "splade-pp-en-v2"] {
        let splade = get_model(id).expect("registered SPLADE model");
        assert_eq!(splade.safetensors_file, None);
        assert_eq!(splade.pytorch_file, "pytorch_model.bin");
    }

    let colpali = get_model("colpali-v1.2").expect("registered ColPali model");
    assert_eq!(
        colpali.safetensors_file,
        Some("model.safetensors.index.json")
    );
    assert_eq!(colpali.context_length, 8192);
    assert_eq!(colpali.max_position_embeddings, 8192);

    for (id, huggingface_id, dimension) in [
        ("jina-colbert-v2-64", "jinaai/jina-colbert-v2-64", 64),
        ("jina-colbert-v2-96", "jinaai/jina-colbert-v2-96", 96),
        ("jina-colbert-v2", "jinaai/jina-colbert-v2", 128),
    ] {
        let model = get_model(id).expect("registered Jina ColBERT variant");
        assert_eq!(model.huggingface_id, huggingface_id);
        assert_eq!(model.embedding_dim.default_dim(), dimension);
        assert_eq!(model.projection_dims, Some(dimension));
        assert_eq!(model.license, "CC-BY-NC-4.0");
    }
}

#[test]
fn test_all_models_have_valid_metadata() {
    let mut models_without_revision = Vec::new();
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
        if let Some(revision) = model.revision {
            assert_eq!(revision.len(), 40, "{} revision length", model.id);
            assert!(
                revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                "{} must use a lowercase commit SHA",
                model.id
            );
        } else {
            models_without_revision.push(model.id);
        }
        if model.is_runnable() {
            assert!(
                model.revision.is_some(),
                "runnable model {} must have a pinned revision",
                model.id
            );
        }
        // Only text/vision models need languages; timeseries models don't
        if model.modalities.contains(&"text") || model.modalities.contains(&"vision") {
            assert!(
                !model.languages.is_empty(),
                "Text/vision model {} should have at least one language",
                model.id
            );
        }
    }

    assert_eq!(models_without_revision, ["jina-colbert-v2-96"]);
}
