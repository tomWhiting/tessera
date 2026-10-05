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
        ("bge-base-en-v1.5", SupportTier::Supported),
        ("bge-large-en-v1.5", SupportTier::Experimental),
        ("bge-m3-multi", SupportTier::CatalogOnly),
        ("bge-small-en-v1.5", SupportTier::Experimental),
        ("chronos-bolt-small", SupportTier::CatalogOnly),
        ("colbert-small", SupportTier::Experimental),
        ("colbert-v2", SupportTier::Experimental),
        ("colpali-v1.2", SupportTier::Experimental),
        ("colpali-v1.3-hf", SupportTier::CatalogOnly),
        ("gte-modern-colbert", SupportTier::CatalogOnly),
        ("gte-modernbert-base", SupportTier::Experimental),
        ("jina-colbert-v2", SupportTier::CatalogOnly),
        ("jina-colbert-v2-64", SupportTier::CatalogOnly),
        ("jina-colbert-v2-96", SupportTier::CatalogOnly),
        ("jina-embeddings-v2-base-code", SupportTier::CatalogOnly),
        ("jina-embeddings-v2-base-en", SupportTier::Experimental),
        ("jina-embeddings-v2-small-en", SupportTier::Experimental),
        ("jina-embeddings-v3", SupportTier::CatalogOnly),
        ("minicoil-v1", SupportTier::Experimental),
        ("multilingual-e5-base", SupportTier::Experimental),
        ("multilingual-e5-large", SupportTier::Experimental),
        ("multilingual-e5-small", SupportTier::Experimental),
        ("mxbai-embed-large-v1", SupportTier::Experimental),
        ("nomic-embed-v1.5", SupportTier::Experimental),
        ("snowflake-arctic-l", SupportTier::Experimental),
        ("splade-pp-en-v1", SupportTier::Experimental),
        ("splade-pp-en-v2", SupportTier::Experimental),
        ("splade-v3", SupportTier::CatalogOnly),
        ("timesfm-1.0-200m", SupportTier::CatalogOnly),
    ];

    assert_eq!(actual.as_slice(), expected);
    // A Supported tier is earned, never declared: every Supported entry must pin a
    // checked official reference in its certification specification.
    let supported = MODEL_REGISTRY
        .iter()
        .filter(|model| model.support_tier == SupportTier::Supported)
        .map(|model| model.id)
        .collect::<Vec<_>>();
    assert_eq!(supported, ["bge-base-en-v1.5"]);
    for id in supported {
        let spec_path = format!(
            "{}/certification/specs/{id}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let spec = std::fs::read_to_string(&spec_path)
            .unwrap_or_else(|error| panic!("Supported model {id} needs {spec_path}: {error}"));
        assert!(
            spec.contains("\"official_reference\": {"),
            "Supported model {id} must pin a checked official reference in {spec_path}"
        );
    }
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
        "multilingual-e5-small",
        "bge-small-en-v1.5",
        "bge-large-en-v1.5",
        "mxbai-embed-large-v1",
        "multilingual-e5-base",
        "multilingual-e5-large",
        "gte-modernbert-base",
        "colbert-small",
        "colbert-v2",
        "colpali-v1.2",
        "minicoil-v1",
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
    assert_eq!(snowflake.parameters, "567754752");
    assert_eq!(snowflake.architecture_type, "xlm-roberta");
    assert_eq!(snowflake.context_length, 2048);
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
        assert_eq!(splade.pytorch_file, Some("pytorch_model.bin"));
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

const BGE_QUERY: &str = "Represent this sentence for searching relevant passages: ";
const JINA_COS_SIM: &str = "cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))";

/// Each dense entry's prompts, distance and card words, copied from its card.
const DENSE_RETRIEVAL: &[(&str, &str, &str, Distance, &str, &str)] = &[
    (
        "bge-base-en-v1.5",
        BGE_QUERY,
        "",
        Distance::Cosine,
        "similarity = embeddings_1 @ embeddings_2.T",
        "https://huggingface.co/BAAI/bge-base-en-v1.5/blob/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/README.md#L2772",
    ),
    (
        "jina-embeddings-v2-small-en",
        "",
        "",
        Distance::Cosine,
        JINA_COS_SIM,
        "https://huggingface.co/jinaai/jina-embeddings-v2-small-en/blob/44e7d1d6caec8c883c2d4b207588504d519788d0/README.md#L2696",
    ),
    (
        "jina-embeddings-v2-base-en",
        "",
        "",
        Distance::Cosine,
        JINA_COS_SIM,
        "https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/README.md#L2695",
    ),
    (
        "jina-embeddings-v2-base-code",
        "",
        "",
        Distance::Cosine,
        JINA_COS_SIM,
        "https://huggingface.co/jinaai/jina-embeddings-v2-base-code/blob/516f4baf13dec4ddddda8631e019b5737c8bc250/README.md#L145",
    ),
    (
        "jina-embeddings-v3",
        "Represent the query for retrieving evidence documents: ",
        "Represent the document for retrieval: ",
        Distance::Cosine,
        "print(embeddings[0] @ embeddings[1].T)",
        "https://huggingface.co/jinaai/jina-embeddings-v3/blob/ab036b023d30b4d1138c4c3bfa9f0c445ab455d6/README.md#L25179",
    ),
    (
        "nomic-embed-v1.5",
        "search_query: ",
        "search_document: ",
        Distance::Cosine,
        "embeddings = F.normalize(embeddings, p=2, dim=1)",
        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/README.md#L2699",
    ),
    (
        "snowflake-arctic-l",
        "query: ",
        "",
        Distance::Cosine,
        "# Compute cosine similarity scores",
        "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0/blob/ac6544c8a46e00af67e330e85a9028c66b8cfd9a/README.md#L9126",
    ),
    (
        "multilingual-e5-small",
        "query: ",
        "passage: ",
        Distance::Cosine,
        "scores = (embeddings[:2] @ embeddings[2:].T) * 100",
        "https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md#L18355",
    ),
    (
        "bge-small-en-v1.5",
        BGE_QUERY,
        "",
        Distance::Cosine,
        "similarity = embeddings_1 @ embeddings_2.T",
        "https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md#L2771",
    ),
    (
        "bge-large-en-v1.5",
        BGE_QUERY,
        "",
        Distance::Cosine,
        "similarity = embeddings_1 @ embeddings_2.T",
        "https://huggingface.co/BAAI/bge-large-en-v1.5/blob/d4aa6901d3a41ba39fb536a557fa166f842b0e09/README.md#L2770",
    ),
    (
        "mxbai-embed-large-v1",
        BGE_QUERY,
        "",
        Distance::Cosine,
        "similarities = cos_sim(query_embedding, docs_embeddings)",
        "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/blob/b33106f585b9ce46904ad7443a3b52b7a63e231c/README.md#L2671",
    ),
    (
        "multilingual-e5-base",
        "query: ",
        "passage: ",
        Distance::Cosine,
        "scores = (embeddings[:2] @ embeddings[2:].T) * 100",
        "https://huggingface.co/intfloat/multilingual-e5-base/blob/d128750597153bb5987e10b1c3493a34e5a4502a/README.md#L6821",
    ),
    (
        "multilingual-e5-large",
        "query: ",
        "passage: ",
        Distance::Cosine,
        "scores = (embeddings[:2] @ embeddings[2:].T) * 100",
        "https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/README.md#L5994",
    ),
    (
        "gte-modernbert-base",
        "",
        "",
        Distance::Cosine,
        "scores = (embeddings[:1] @ embeddings[1:].T) * 100",
        "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/blob/e7f32e3c00f91d699e8c43b53106206bcc72bb22/README.md#L77",
    ),
];

#[test]
fn every_dense_entry_has_its_cards_prompts_and_distance() {
    let dense: Vec<_> = models_by_type(ModelType::Dense)
        .into_iter()
        .map(|model| model.id)
        .collect();
    let table: Vec<_> = DENSE_RETRIEVAL.iter().map(|row| row.0).collect();
    assert_eq!(
        dense, table,
        "the table must list every dense entry in order"
    );
    for &(id, query, document, distance, words, url) in DENSE_RETRIEVAL {
        let model = get_model(id).unwrap();
        let prompts = model
            .prompts
            .unwrap_or_else(|| panic!("{id} has no prompts"));
        assert_eq!((prompts.query, prompts.document), (query, document), "{id}");
        assert_eq!(model.distance, Some(distance), "{id}");
        let card = model
            .card_comparison
            .unwrap_or_else(|| panic!("{id} has no card comparison"));
        assert_eq!((card.words, card.url), (words, url), "{id}");
    }
}

#[test]
fn distance_words_are_the_registry_vocabulary() {
    assert_eq!(
        [Distance::Cosine, Distance::Dot, Distance::Euclidean].map(Distance::as_str),
        ["cosine", "dot", "euclidean"]
    );
}
