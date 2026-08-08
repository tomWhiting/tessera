// ============================================================================
// Test 7: Factory Pattern (Tessera Enum)
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_factory_dense_model() {
    // Create embedder using factory - should return Dense variant
    let embedder = Tessera::new("bge-base-en-v1.5").expect("Failed to create embedder via factory");

    // Pattern match to verify it's the Dense variant
    match embedder {
        Tessera::Dense(dense) => {
            // Verify it works
            let embedding = dense
                .encode("Test factory pattern")
                .expect("Failed to encode with dense embedder");
            assert_eq!(embedding.dim(), 768);
            assert_eq!(dense.model(), "bge-base-en-v1.5");
            assert_eq!(dense.dimension(), 768);
        }
        Tessera::MultiVector(_) | Tessera::Sparse(_) | Tessera::Vision(_) => {
            panic!("Factory should have returned Dense variant for dense model");
        }
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_factory_multivector_model() {
    // Create embedder using factory with multi-vector model
    let embedder = Tessera::new("colbert-v2").expect("Failed to create embedder via factory");

    // Pattern match to verify it's the MultiVector variant
    match embedder {
        Tessera::MultiVector(mv) => {
            // Verify it works
            let embeddings = mv
                .encode("Test factory pattern")
                .expect("Failed to encode with multi-vector embedder");
            assert_eq!(embeddings.embedding_dim, 128);
            assert_eq!(mv.model(), "colbert-v2");
            assert_eq!(mv.dimension(), 128);
        }
        Tessera::Dense(_) | Tessera::Sparse(_) | Tessera::Vision(_) => {
            panic!("Factory should have returned MultiVector variant for ColBERT model");
        }
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_factory_both_variants() {
    // Test that we can create and use both variants
    let dense = Tessera::new("bge-base-en-v1.5").unwrap();
    let mv = Tessera::new("colbert-v2").unwrap();

    let text = "Test both variants";

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
}

// ============================================================================
// Test 8: Builder Validation
// ============================================================================

#[test]
fn test_builder_requires_model() {
    // Building without model ID should error
    let result = TesseraDenseBuilder::new().build();

    assert!(result.is_err(), "Should error when model ID not provided");
    if let Err(err) = result {
        assert!(
            err.to_string().contains("Model ID must be specified"),
            "Error should mention missing model ID: {err}"
        );
    }
}

#[test]
fn test_builder_invalid_model() {
    // Building with invalid model ID should error
    let result = TesseraDense::new("nonexistent-model-xyz");

    assert!(result.is_err(), "Should error for invalid model ID");
    if let Err(err) = result {
        assert!(
            err.to_string().contains("not found") || err.to_string().contains("nonexistent"),
            "Error should mention model not found: {err}"
        );
    }
}

#[test]
fn test_builder_wrong_model_type() {
    // Try to create dense embedder with multi-vector model
    let result = TesseraDense::new("colbert-v2");

    assert!(
        result.is_err(),
        "Should error when using multi-vector model with TesseraDense"
    );
    if let Err(err) = result {
        assert!(
            err.to_string().contains("not a dense model")
                || err.to_string().contains("multi-vector"),
            "Error should mention model type mismatch: {err}"
        );
    }
}

#[test]
fn test_builder_unsupported_dimension() {
    // Try to use dimension that's not in Matryoshka supported list
    let result = TesseraDenseBuilder::new()
        .model("nomic-embed-v1.5")
        .dimension(999) // Not in [64, 128, 256, 512, 768]
        .build();

    assert!(result.is_err(), "Should error for unsupported dimension");
    if let Err(err) = result {
        assert!(
            err.to_string().contains("dimension") || err.to_string().contains("999"),
            "Error should mention unsupported dimension: {err}"
        );
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_builder_dimension_on_fixed_model() {
    // Try to use dimension on model without Matryoshka support
    let result = TesseraDenseBuilder::new()
        .model("bge-base-en-v1.5")
        .dimension(384) // BGE is fixed at 768
        .build();

    assert!(
        result.is_err(),
        "Should error when setting dimension on fixed-dimension model"
    );
    if let Err(err) = result {
        assert!(
            err.to_string().contains("dimension") || err.to_string().contains("supported"),
            "Error should mention dimension not supported: {err}"
        );
    }
}

// ============================================================================
// Test 9: Device Selection
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_device_auto_selection() {
    // Create embedder with auto device selection (default)
    let embedder = TesseraDense::new("bge-base-en-v1.5")
        .expect("Failed to create embedder with auto device selection");

    // Verify it works
    let embedding = embedder
        .encode("Test auto device selection")
        .expect("Failed to encode with auto-selected device");

    assert_eq!(embedding.dim(), 768);
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_device_explicit_cpu() {
    // Force CPU device
    let embedder = TesseraDenseBuilder::new()
        .model("bge-base-en-v1.5")
        .device(Device::Cpu)
        .build()
        .expect("Failed to create embedder with CPU device");

    // Verify it works on CPU
    let embedding = embedder
        .encode("Test CPU device")
        .expect("Failed to encode on CPU");

    assert_eq!(embedding.dim(), 768);
}

#[test]
#[ignore = "requires remote model artifacts"]
#[cfg(target_os = "macos")]
fn test_device_metal_on_macos() {
    // Try to use Metal on macOS
    let device = Device::new_metal(0);

    if let Ok(metal_device) = device {
        let embedder = TesseraDenseBuilder::new()
            .model("bge-base-en-v1.5")
            .device(metal_device)
            .build()
            .expect("Failed to create embedder with Metal device");

        // Verify it works on Metal
        let embedding = embedder
            .encode("Test Metal device")
            .expect("Failed to encode on Metal");

        assert_eq!(embedding.dim(), 768);
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
    let result = TesseraDense::new("this-model-does-not-exist");

    assert!(result.is_err());
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("not found") || err_msg.contains("this-model-does-not-exist"),
            "Error should mention model not found: {err_msg}"
        );
    }
}

#[test]
fn test_error_messages_are_clear() {
    // Test that error messages provide helpful context

    // Missing model ID
    if let Err(err1) = TesseraDenseBuilder::new().build() {
        assert!(
            err1.to_string().contains("Model ID"),
            "Should mention Model ID"
        );
        assert!(
            err1.to_string().contains(".model("),
            "Should suggest how to fix"
        );
    } else {
        panic!("Expected error for missing model ID");
    }

    // Invalid model
    if let Err(err2) = TesseraDense::new("invalid") {
        assert!(
            err2.to_string().contains("invalid") || err2.to_string().contains("not found"),
            "Should mention the invalid model ID or that it wasn't found"
        );
    } else {
        panic!("Expected error for invalid model");
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_encode_empty_string() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    // Empty string should still produce embedding (likely just special tokens)
    let result = embedder.encode("");

    // This might error or produce minimal embedding - either is acceptable
    match result {
        Ok(embedding) => {
            assert!(
                embedding.dim() > 0,
                "Should produce embedding with some dimension"
            );
        }
        Err(e) => {
            println!("Empty string encoding errored (acceptable): {e}");
        }
    }
}

// ============================================================================
// Additional Quality Tests
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_dense_metadata_preservation() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let text = "Testing metadata preservation";
    let embedding = embedder.encode(text).unwrap();

    assert_eq!(embedding.text, text, "Original text should be preserved");
    assert_eq!(embedding.dim(), 768, "Dimension should be correct");
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_dense_model_info_methods() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    // Test model info methods
    assert_eq!(embedder.model(), "bge-base-en-v1.5");
    assert_eq!(embedder.dimension(), 768);
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_dense_embedding_not_sparse() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let text = "Dense embeddings should have most dimensions non-zero";
    let embedding = embedder.encode(text).unwrap();

    // Count non-zero dimensions
    let non_zero_count = embedding
        .embedding
        .iter()
        .filter(|&&x| x.abs() > 1e-6)
        .count();

    // Dense embeddings should have most dimensions non-zero (>90%)
    let non_zero_count =
        u32::try_from(non_zero_count).expect("embedding non-zero count should fit in u32");
    let embedding_dim =
        u32::try_from(embedding.dim()).expect("embedding dimension should fit in u32");
    let density = f64::from(non_zero_count) / f64::from(embedding_dim);
    assert!(
        density > 0.9,
        "Dense embedding should have >90% non-zero dimensions (got {:.1}%)",
        density * 100.0
    );
}
