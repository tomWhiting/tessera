// Test 7: Error Handling
// ============================================================================

#[test]
fn test_error_invalid_model_id() {
    let result = TesseraVision::new("this-vision-model-does-not-exist");

    assert!(result.is_err(), "Should error for invalid model ID");
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("not found") || err_msg.contains("this-vision-model-does-not-exist"),
            "Error should mention model not found: {err_msg}"
        );
    }
}

#[test]
fn test_error_messages_are_clear() {
    // Test that error messages provide helpful context

    // Missing model ID
    if let Err(err1) = TesseraVisionBuilder::new().build() {
        let msg = err1.to_string();
        assert!(
            msg.contains("Model ID") || msg.contains("model"),
            "Should mention Model ID: {msg}"
        );
    } else {
        panic!("Expected error for missing model ID");
    }

    // Invalid model
    if let Err(err2) = TesseraVision::new("invalid-vision-model") {
        let msg = err2.to_string();
        assert!(
            msg.contains("invalid") || msg.contains("not found"),
            "Should mention the invalid model ID or that it wasn't found: {msg}"
        );
    } else {
        panic!("Expected error for invalid model");
    }
}

#[test]
fn test_error_wrong_model_type_clear_message() {
    // Try to use ColBERT model with vision builder
    let result = TesseraVisionBuilder::new()
        .model("colbert-v2") // Multi-vector text model
        .build();

    assert!(
        result.is_err(),
        "Should error when using text model with vision builder"
    );
    if let Err(err) = result {
        let error_msg = format!("{err:?}");
        assert!(
            error_msg.contains("VisionLanguage")
                || error_msg.contains("vision")
                || error_msg.contains("type"),
            "Error should clearly indicate model type mismatch: {error_msg}"
        );
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_encode_invalid_image_path() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    // Try to encode non-existent image
    let result = embedder.encode_document("nonexistent/image/path.png");

    assert!(result.is_err(), "Should error for invalid image path");
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Failed to encode")
                || err_msg.contains("image")
                || err_msg.contains("path"),
            "Error should mention image encoding failure: {err_msg}"
        );
    }
}

// ============================================================================
// Test 8: Device Selection
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_device_auto_selection() {
    // Create embedder with auto device selection (default)
    let embedder = TesseraVision::new("colpali-v1.3-hf")
        .expect("Failed to create embedder with auto device selection");

    // Verify it works
    let query = embedder
        .encode_query("Test auto device selection")
        .expect("Failed to encode with auto-selected device");

    assert!(query.num_tokens > 0);
    assert_eq!(query.embedding_dim, 128);
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_device_explicit_cpu() {
    // Force CPU device
    let embedder = TesseraVisionBuilder::new()
        .model("colpali-v1.3-hf")
        .device(Device::Cpu)
        .build()
        .expect("Failed to create embedder with CPU device");

    // Verify it works on CPU
    let query = embedder
        .encode_query("Test CPU device")
        .expect("Failed to encode on CPU");

    assert!(query.num_tokens > 0);
    assert_eq!(query.embedding_dim, 128);
}

#[test]
#[ignore = "requires remote model artifacts and Metal hardware"]
#[cfg(target_os = "macos")]
fn test_device_metal_on_macos() {
    // Try to use Metal on macOS
    let device = Device::new_metal(0);

    if let Ok(metal_device) = device {
        let embedder = TesseraVisionBuilder::new()
            .model("colpali-v1.3-hf")
            .device(metal_device)
            .build()
            .expect("Failed to create embedder with Metal device");

        // Verify it works on Metal
        let query = embedder
            .encode_query("Test Metal device")
            .expect("Failed to encode on Metal");

        assert!(query.num_tokens > 0);
        assert_eq!(query.embedding_dim, 128);
    } else {
        // Metal not available, skip test
        println!("Metal device not available, skipping test");
    }
}

// ============================================================================
// Test 9: Query Encoding Properties
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_query_encoding_varies_by_length() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    let short_query = "invoice";
    let long_query = "What is the total amount shown in the invoice for the third quarter?";

    let emb_short = embedder
        .encode_query(short_query)
        .expect("Failed to encode short query");
    let emb_long = embedder
        .encode_query(long_query)
        .expect("Failed to encode long query");

    // Both should have valid structure
    assert!(emb_short.num_tokens > 0, "Short query should have tokens");
    assert!(emb_long.num_tokens > 0, "Long query should have tokens");

    // Longer query should have more tokens
    assert!(
        emb_long.num_tokens > emb_short.num_tokens,
        "Long query should have more tokens: {} vs {}",
        emb_long.num_tokens,
        emb_short.num_tokens
    );

    // Both should have same embedding dimension
    assert_eq!(
        emb_short.embedding_dim, emb_long.embedding_dim,
        "Both queries should have same embedding dimension"
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_query_encoding_consistency() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    let query = "What is the total amount?";

    // Encode same query twice
    let emb1 = embedder.encode_query(query).unwrap();
    let emb2 = embedder.encode_query(query).unwrap();

    // Should produce identical results
    assert_eq!(
        emb1.num_tokens, emb2.num_tokens,
        "Token count should be consistent"
    );
    assert_eq!(
        emb1.embedding_dim, emb2.embedding_dim,
        "Embedding dim should be consistent"
    );

    // Check embedding values are identical
    for i in 0..emb1.num_tokens {
        for j in 0..emb1.embedding_dim {
            let diff = (emb1.embeddings[[i, j]] - emb2.embeddings[[i, j]]).abs();
            assert!(
                diff < 1e-6,
                "Embeddings should be identical at position [{i}, {j}], got diff: {diff}"
            );
        }
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_encode_empty_query() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    // Empty query should still produce embedding (likely just special tokens)
    let result = embedder.encode_query("");

    // This might error or produce minimal embedding - either is acceptable
    match result {
        Ok(embedding) => {
            assert!(
                embedding.num_tokens > 0,
                "Should have at least special tokens"
            );
            assert_eq!(
                embedding.embedding_dim, 128,
                "Should have correct dimension"
            );
        }
        Err(e) => {
            println!("Empty query encoding errored (acceptable): {e}");
        }
    }
}

// ============================================================================
// Test 10: Multiple Model Variants
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_colpali_v1_2_variant() {
    let embedder =
        TesseraVision::new("colpali-v1.2").expect("Failed to create ColPali v1.2 embedder");

    // Verify model info
    assert_eq!(embedder.model(), "colpali-v1.2");
    assert_eq!(embedder.embedding_dim(), 128);
    assert_eq!(embedder.num_patches(), 1024);

    // Verify it works
    let query = embedder
        .encode_query("test query")
        .expect("Failed to encode with v1.2");
    assert!(query.num_tokens > 0);
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_colpali_v1_3_variant() {
    let embedder =
        TesseraVision::new("colpali-v1.3-hf").expect("Failed to create ColPali v1.3 embedder");

    // Verify model info
    assert_eq!(embedder.model(), "colpali-v1.3-hf");
    assert_eq!(embedder.embedding_dim(), 128);
    assert_eq!(embedder.num_patches(), 1024);

    // Verify it works
    let query = embedder
        .encode_query("test query")
        .expect("Failed to encode with v1.3");
    assert!(query.num_tokens > 0);
}

// ============================================================================
// Test 11: Batch Processing (Future Enhancement)
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts and batch implementation"]
fn test_vision_batch_query_encoding() {
    let embedder = TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");

    let queries = vec![
        "What is the total amount?",
        "When is the due date?",
        "Who is the vendor?",
    ];

    // TODO: Implement batch encoding for queries
    // let batch_embeddings = embedder.encode_batch_queries(&queries).unwrap();
    // assert_eq!(batch_embeddings.len(), queries.len());

    // For now, test sequential encoding
    for query in &queries {
        let emb = embedder.encode_query(query).unwrap();
        assert!(emb.num_tokens > 0);
        assert_eq!(emb.embedding_dim, 128);
    }
}

// ============================================================================
// Test 12: Integration with Other Variants
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_vision_vs_text_multivector() {
    // Compare vision multi-vector with text multi-vector (ColBERT)
    let vision_embedder =
        TesseraVision::new("colpali-v1.3-hf").expect("Failed to create vision embedder");
    let text_embedder = Tessera::new("colbert-v2").expect("Failed to create text embedder");

    let query = "What is machine learning?";

    // Vision query encoding
    let vision_query = vision_embedder
        .encode_query(query)
        .expect("Failed to encode vision query");

    // Text multi-vector encoding
    if let Tessera::MultiVector(mv) = text_embedder {
        let text_query = mv.encode(query).expect("Failed to encode text query");

        // Both should use 128 dimensions for compatibility
        assert_eq!(
            vision_query.embedding_dim, text_query.embedding_dim,
            "Vision and text should use same embedding dimension"
        );

        // Token counts may differ due to different tokenizers
        assert!(vision_query.num_tokens > 0);
        assert!(text_query.num_tokens > 0);
    } else {
        panic!("Expected MultiVector variant");
    }
}
