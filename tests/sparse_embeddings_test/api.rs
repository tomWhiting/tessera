// Test 6: Factory Pattern (Tessera Enum)
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_factory_sparse_model() {
    // Create embedder using factory - should return Sparse variant
    let embedder = Tessera::new("splade-pp-en-v1").expect("Failed to create embedder via factory");

    // Pattern match to verify it's the Sparse variant
    match embedder {
        Tessera::Sparse(sparse) => {
            // Verify it works
            let embedding = sparse
                .encode("Test factory pattern")
                .expect("Failed to encode with sparse embedder");
            assert_eq!(embedding.vocab_size, 30522);
            assert_eq!(sparse.model(), "splade-pp-en-v1");
            assert_eq!(sparse.vocab_size(), 30522);
            assert!(embedding.sparsity() > 0.99);
        }
        _ => panic!("Factory should have returned Sparse variant for SPLADE model"),
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_factory_all_variants() {
    // Test that we can create and use all three variants
    let dense = Tessera::new("bge-base-en-v1.5").unwrap();
    let mv = Tessera::new("colbert-v2").unwrap();
    let sparse = Tessera::new("splade-pp-en-v1").unwrap();

    let text = "Test all variants";

    // Use dense variant
    if let Tessera::Dense(d) = dense {
        let emb = d.encode(text).unwrap();
        assert_eq!(emb.dim(), 768);
    } else {
        panic!("Expected Dense variant");
    }

    // Use multi-vector variant
    if let Tessera::MultiVector(m) = mv {
        let emb = m.encode(text).unwrap();
        assert_eq!(emb.embedding_dim, 128);
    } else {
        panic!("Expected MultiVector variant");
    }

    // Use sparse variant
    if let Tessera::Sparse(s) = sparse {
        let emb = s.encode(text).unwrap();
        assert_eq!(emb.vocab_size, 30522);
        assert!(emb.sparsity() > 0.99);
    } else {
        panic!("Expected Sparse variant");
    }
}

// ============================================================================
// Test 7: Builder Pattern
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_builder_basic() {
    let embedder = TesseraSparseBuilder::new()
        .model("splade-pp-en-v1")
        .device(Device::Cpu)
        .build()
        .expect("Failed to build sparse embedder");

    let emb = embedder.encode("test builder pattern").unwrap();
    assert!(emb.sparsity() > 0.99);
    assert_eq!(emb.vocab_size, 30522);
}

#[test]
fn test_builder_requires_model() {
    // Building without model ID should error
    let result = TesseraSparseBuilder::new().build();

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
    // Try to use dense model with sparse builder
    let result = TesseraSparseBuilder::new()
        .model("bge-base-en-v1.5") // Dense model
        .build();

    assert!(result.is_err(), "Should error with non-sparse model");
    if let Err(err) = result {
        let error_msg = format!("{err:?}");
        assert!(
            error_msg.contains("not Sparse") || error_msg.contains("not a sparse"),
            "Error should mention model type mismatch: {error_msg}"
        );
    }
}

#[test]
fn test_builder_invalid_model() {
    // Building with invalid model ID should error
    let result = TesseraSparse::new("nonexistent-sparse-model-xyz");

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
// Test 8: Model Info Methods
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_vocab_size_accessor() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    assert_eq!(
        embedder.vocab_size(),
        30522,
        "Should return correct vocab size"
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_model_accessor() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    assert_eq!(
        embedder.model(),
        "splade-pp-en-v1",
        "Should return correct model ID"
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_metadata_preservation() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let text = "Testing metadata preservation in sparse embeddings";
    let embedding = embedder.encode(text).unwrap();

    assert_eq!(embedding.text, text, "Original text should be preserved");
    assert_eq!(embedding.vocab_size, 30522, "Vocab size should be correct");
    assert!(embedding.nnz() > 0, "Should have non-zero values");
}

// ============================================================================
// Test 9: Device Selection
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_device_auto_selection() {
    // Create embedder with auto device selection (default)
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create embedder with auto device selection");

    // Verify it works
    let embedding = embedder
        .encode("Test auto device selection")
        .expect("Failed to encode with auto-selected device");

    assert_eq!(embedding.vocab_size, 30522);
    assert!(embedding.sparsity() > 0.99);
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_device_explicit_cpu() {
    // Force CPU device
    let embedder = TesseraSparseBuilder::new()
        .model("splade-pp-en-v1")
        .device(Device::Cpu)
        .build()
        .expect("Failed to create embedder with CPU device");

    // Verify it works on CPU
    let embedding = embedder
        .encode("Test CPU device")
        .expect("Failed to encode on CPU");

    assert_eq!(embedding.vocab_size, 30522);
    assert!(embedding.sparsity() > 0.99);
}

#[test]
#[ignore = "requires remote model artifacts"]
#[cfg(target_os = "macos")]
fn test_device_metal_on_macos() {
    // Try to use Metal on macOS
    let device = Device::new_metal(0);

    if let Ok(metal_device) = device {
        let embedder = TesseraSparseBuilder::new()
            .model("splade-pp-en-v1")
            .device(metal_device)
            .build()
            .expect("Failed to create embedder with Metal device");

        // Verify it works on Metal
        let embedding = embedder
            .encode("Test Metal device")
            .expect("Failed to encode on Metal");

        assert_eq!(embedding.vocab_size, 30522);
        assert!(embedding.sparsity() > 0.99);
    } else {
        // Metal not available, skip test
        println!("Metal device not available, skipping test");
    }
}

// ============================================================================
// Test 10: Error Handling
// ============================================================================

#[test]
fn test_error_invalid_model_id() {
    let result = TesseraSparse::new("this-sparse-model-does-not-exist");

    assert!(result.is_err());
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("not found") || err_msg.contains("this-sparse-model-does-not-exist"),
            "Error should mention model not found: {err_msg}"
        );
    }
}

#[test]
fn test_error_messages_are_clear() {
    // Test that error messages provide helpful context

    // Missing model ID
    if let Err(err1) = TesseraSparseBuilder::new().build() {
        let msg = err1.to_string();
        assert!(
            msg.contains("Model ID") || msg.contains("model"),
            "Should mention Model ID: {msg}"
        );
    } else {
        panic!("Expected error for missing model ID");
    }

    // Invalid model
    if let Err(err2) = TesseraSparse::new("invalid-sparse-model") {
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
#[ignore = "requires remote model artifacts"]
fn test_encode_empty_string() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create embedder");

    // Empty string should still produce embedding (likely minimal activations)
    let result = embedder.encode("");

    // This might error or produce minimal embedding - either is acceptable
    match result {
        Ok(embedding) => {
            assert_eq!(
                embedding.vocab_size, 30522,
                "Should have correct vocab size"
            );
            println!(
                "Empty string encoded with {} non-zero values",
                embedding.nnz()
            );
        }
        Err(e) => {
            println!("Empty string encoding errored (acceptable): {e}");
        }
    }
}

// ============================================================================
// Test 11: Dot Product Similarity Implementation
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_dot_product_manual() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let text1 = "machine learning";
    let text2 = "artificial intelligence";

    let emb1 = embedder.encode(text1).unwrap();
    let emb2 = embedder.encode(text2).unwrap();

    // Manual dot product
    let mut manual_score = 0.0;
    for (idx1, weight1) in &emb1.weights {
        if let Some(&(_, weight2)) = emb2.weights.iter().find(|(idx2, _)| idx2 == idx1) {
            manual_score += weight1 * weight2;
        }
    }

    // Use convenience method
    let api_score = embedder.similarity(text1, text2).unwrap();

    // Should match
    assert!(
        (manual_score - api_score).abs() < 1e-6,
        "Manual dot product ({manual_score}) should match API similarity ({api_score})"
    );
}

// ============================================================================
// Test 12: Additional Quality Tests
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_not_dense() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let text = "Sparse embeddings should have very few non-zero dimensions";
    let embedding = embedder.encode(text).unwrap();

    // Count density
    let density = 1.0 - embedding.sparsity();

    // Sparse embeddings should have <1% density
    assert!(
        density < 0.01,
        "Sparse embedding should have <1% density (got {:.4}%)",
        density * 100.0
    );

    println!(
        "Sparsity: {:.2}%, Non-zero: {}/{}",
        embedding.sparsity() * 100.0,
        embedding.nnz(),
        embedding.vocab_size
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_weights_magnitude() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let text = "machine learning algorithm optimization";
    let embedding = embedder.encode(text).unwrap();

    // Find min and max weights
    let mut min_weight = f32::MAX;
    let mut max_weight = f32::MIN;

    for (_, weight) in &embedding.weights {
        min_weight = min_weight.min(*weight);
        max_weight = max_weight.max(*weight);
    }

    // Weights should be positive and reasonable magnitude
    assert!(
        min_weight > 0.0,
        "Min weight should be positive, got {min_weight}"
    );
    assert!(max_weight > min_weight, "Should have weight variation");

    println!("Weight range: [{min_weight:.4}, {max_weight:.4}]");
}
