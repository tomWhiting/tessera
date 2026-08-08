// ============================================================================
// Test 1: Basic Document Image Encoding
// ============================================================================
#[test]
#[ignore = "requires approximately 5.88 GB of remote model artifacts"]
fn test_vision_encode_document() {
    let _embedder =
        TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    // TODO: Encode a test document image
    // let embedding = embedder.encode_document("test_data/sample_document.png")
    //     .expect("Failed to encode image");

    // TODO: Verify patch count (448x448 / 14x14 = 1024 patches)
    // assert_eq!(embedding.num_patches, 1024, "Expected 1024 patches");

    // TODO: Verify embedding dimension
    // assert_eq!(embedding.embedding_dim, 128, "Expected 128-dim embeddings");

    // TODO: Verify embeddings shape
    // assert_eq!(embedding.embeddings.len(), 1024, "Should have 1024 patch embeddings");
    // assert_eq!(embedding.embeddings[0].len(), 128, "Each patch should be 128-dim");
}

// ============================================================================
// Test 2: Text Query Encoding
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_vision_encode_query() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    let query = "What is the total amount?";
    let embedding = embedder
        .encode_query(query)
        .expect("Failed to encode query");

    // Verify token count (should be > 0, variable based on query)
    assert!(embedding.num_tokens > 0, "Should have at least one token");

    // Verify embedding dimension
    assert_eq!(embedding.embedding_dim, 128, "Expected 128-dim embeddings");

    // Verify embeddings shape
    assert_eq!(
        embedding.embeddings.ncols(),
        128,
        "Each token should be 128-dim"
    );
    assert_eq!(
        embedding.embeddings.nrows(),
        embedding.num_tokens,
        "Rows should match token count"
    );
}

// ============================================================================
// Test 3: Late Interaction Scoring
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts and image fixtures"]
fn test_vision_search() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    // TODO: Encode document
    // let doc_emb = embedder.encode_document("test_data/invoice.png")
    //     .expect("Failed to encode document");

    // Encode relevant query
    let _query_relevant = embedder
        .encode_query("total amount")
        .expect("Failed to encode relevant query");

    // Encode irrelevant query
    let _query_irrelevant = embedder
        .encode_query("weather forecast")
        .expect("Failed to encode irrelevant query");

    // TODO: Compute scores
    // let score_relevant = embedder.search(&query_relevant, &doc_emb)
    //     .expect("Failed to compute relevant score");
    // let score_irrelevant = embedder.search(&query_irrelevant, &doc_emb)
    //     .expect("Failed to compute irrelevant score");

    // TODO: Relevant query should score higher
    // assert!(
    //     score_relevant > score_irrelevant,
    //     "Relevant query should score higher: {} vs {}",
    //     score_relevant, score_irrelevant
    // );
}

#[test]
#[ignore = "requires remote model artifacts and image fixtures"]
fn test_vision_search_document_convenience() {
    let _embedder =
        TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    // TODO: Test convenience method
    // let score = embedder.search_document(
    //     "What is the total?",
    //     "test_data/invoice.png"
    // ).expect("Failed to search document");

    // assert!(score > 0.0, "Should produce positive score");
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_vision_maxsim_scoring() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    // Test MaxSim scoring properties
    let query = embedder
        .encode_query("machine learning")
        .expect("Failed to encode query");

    // Verify query structure
    assert!(query.num_tokens > 0, "Query should have tokens");
    assert_eq!(query.embedding_dim, 128, "Query tokens should be 128-dim");

    // TODO: Test actual MaxSim computation when document encoding is available
    // For each query token, find max similarity with document patches
    // Sum across all query tokens
}

// ============================================================================
// Test 4: Factory Pattern (Tessera Enum)
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_factory_vision_model() {
    // Create embedder using factory - should return Vision variant
    let embedder = Tessera::new("colpali-v1.3-hf").expect("Failed to create embedder via factory");

    // Pattern match to verify it's the Vision variant
    match embedder {
        Tessera::Vision(vision) => {
            // Verify it works
            let query_emb = vision
                .encode_query("Test factory pattern")
                .expect("Failed to encode with vision embedder");
            assert!(query_emb.num_tokens > 0);
            assert_eq!(vision.model(), "colpali-v1.3-hf");
            assert_eq!(vision.embedding_dim(), 128);

            // TODO: Test document encoding when available
            // let doc_emb = vision.encode_document("test_data/sample.png").unwrap();
            // assert_eq!(doc_emb.num_patches, 1024);
        }
        _ => panic!("Factory should have returned Vision variant for ColPali model"),
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_factory_all_four_variants() {
    // Test that we can create and use all four variants
    let dense = Tessera::new("bge-base-en-v1.5").expect("Failed to create dense embedder");
    let mv = Tessera::new("colbert-v2").expect("Failed to create multi-vector embedder");
    let sparse = Tessera::new("splade-pp-en-v1").expect("Failed to create sparse embedder");
    let vision = Tessera::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    assert!(
        matches!(dense, Tessera::Dense(_)),
        "Should be Dense variant"
    );
    assert!(
        matches!(mv, Tessera::MultiVector(_)),
        "Should be MultiVector variant"
    );
    assert!(
        matches!(sparse, Tessera::Sparse(_)),
        "Should be Sparse variant"
    );
    assert!(
        matches!(vision, Tessera::Vision(_)),
        "Should be Vision variant"
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_factory_variant_usage() {
    // Test that we can use each variant through the factory
    let text = "Test all variants";

    // Dense variant
    if let Tessera::Dense(d) = Tessera::new("bge-base-en-v1.5").unwrap() {
        let emb = d.encode(text).unwrap();
        assert_eq!(emb.dim(), 768);
    } else {
        panic!("Expected Dense variant");
    }

    // MultiVector variant
    if let Tessera::MultiVector(m) = Tessera::new("colbert-v2").unwrap() {
        let emb = m.encode(text).unwrap();
        assert_eq!(emb.embedding_dim, 128);
    } else {
        panic!("Expected MultiVector variant");
    }

    // Sparse variant
    if let Tessera::Sparse(s) = Tessera::new("splade-pp-en-v1").unwrap() {
        let emb = s.encode(text).unwrap();
        assert_eq!(emb.vocab_size, 30522);
    } else {
        panic!("Expected Sparse variant");
    }

    // Vision variant
    if let Tessera::Vision(v) = Tessera::new("colpali-v1.3-hf").unwrap() {
        let query_emb = v.encode_query(text).unwrap();
        assert_eq!(query_emb.embedding_dim, 128);
        // TODO: Test document encoding when available
    } else {
        panic!("Expected Vision variant");
    }
}

// ============================================================================
// Test 5: Builder Pattern
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_vision_builder_basic() {
    let embedder = TesseraVisionBuilder::new()
        .model("colpali-v1.3-hf")
        .device(Device::Cpu)
        .build()
        .expect("Failed to build vision embedder");

    // Verify it works
    let query = embedder
        .encode_query("test builder pattern")
        .expect("Failed to encode query");
    assert!(query.num_tokens > 0);
    assert_eq!(query.embedding_dim, 128);

    // TODO: Test document encoding when available
    // let doc = embedder.encode_document("test_data/sample.png").unwrap();
    // assert_eq!(doc.num_patches, 1024);
}

#[test]
fn test_builder_requires_model() {
    // Building without model ID should error
    let result = TesseraVisionBuilder::new().build();

    assert!(result.is_err(), "Should error when model ID not provided");
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Model ID") || err_msg.contains("model"),
            "Error should mention missing model ID: {err_msg}"
        );
    }
}

#[test]
fn test_builder_wrong_model_type() {
    // Try to use dense model with vision builder
    let result = TesseraVisionBuilder::new()
        .model("bge-base-en-v1.5") // Dense model
        .build();

    assert!(result.is_err(), "Should error with non-vision model");
    if let Err(err) = result {
        let error_msg = format!("{err:?}");
        assert!(
            error_msg.contains("not VisionLanguage")
                || error_msg.contains("not a vision")
                || error_msg.contains("type"),
            "Error should mention model type mismatch: {error_msg}"
        );
    }
}

#[test]
fn test_builder_invalid_model() {
    // Building with invalid model ID should error
    let result = TesseraVision::new("nonexistent-vision-model");

    assert!(result.is_err(), "Should error for invalid model ID");
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("not found") || err_msg.contains("nonexistent"),
            "Error should mention model not found: {err_msg}"
        );
    }
}

// ============================================================================
// Test 6: Model Info Accessors
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_vision_model_info() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    // Test model info methods
    assert_eq!(
        embedder.model(),
        "colpali-v1.3-hf",
        "Should return correct model ID"
    );
    assert_eq!(
        embedder.embedding_dim(),
        128,
        "Should return correct embedding dimension"
    );
    assert_eq!(
        embedder.num_patches(),
        1024,
        "Should return correct number of patches"
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_vision_patch_configuration() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    // Verify patch configuration matches spec
    // 448x448 image, 14x14 patch size = 32x32 grid = 1024 patches
    let num_patches = embedder.num_patches();
    assert_eq!(num_patches, 1024, "ColPali should have 1024 patches");

    // Verify embedding dimension matches ColBERT compatibility
    let emb_dim = embedder.embedding_dim();
    assert_eq!(
        emb_dim, 128,
        "ColPali should use 128 dimensions for ColBERT compatibility"
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_vision_model_metadata() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    // Verify model metadata
    assert_eq!(embedder.model(), "colpali-v1.3-hf");

    // Verify architecture specs
    let emb_dim = embedder.embedding_dim();
    let num_patches = embedder.num_patches();

    assert_eq!(emb_dim, 128, "Embedding dimension should match spec");
    assert_eq!(num_patches, 1024, "Patch count should match spec");
}

// ============================================================================
