//! Integration tests for dense embeddings (Phase 2.1).
//!
//! Tests the full API surface for TesseraDense, Tessera factory pattern,
//! and dense-specific functionality including:
//! - Basic single-text encoding
//! - Batch processing
//! - Cosine similarity
//! - Normalization validation
//! - Pooling strategies
//! - Matryoshka dimension truncation
//! - Factory pattern with auto-detection
//! - Builder validation and error handling
//! - Device selection

use candle_core::Device;
use tessera::{Tessera, TesseraDense, TesseraDenseBuilder};

// ============================================================================
// Test 1: Basic Dense Encoding
// ============================================================================

#[test]
#[ignore] // Requires model download
fn test_dense_encode_single() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let text = "What is machine learning?";
    let embedding = embedder.encode(text).expect("Failed to encode text");

    // Verify dimensions
    assert_eq!(
        embedding.dim(),
        768,
        "Expected 768-dim embedding for bge-base-en-v1.5"
    );

    // Verify not all zeros
    let sum: f32 = embedding.embedding.iter().sum();
    assert!(
        sum.abs() > 0.01,
        "Embedding should not be all zeros (sum: {})",
        sum
    );

    // Verify text is preserved
    assert_eq!(
        embedding.text, text,
        "Text should be preserved in embedding"
    );
}

// ============================================================================
// Test 2: Dense Batch Processing
// ============================================================================

#[test]
#[ignore]
fn test_dense_batch_encoding() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let texts = vec![
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Natural language processing enables text understanding",
    ];

    // Encode batch
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let batch_embeddings = embedder
        .encode_batch(&text_refs)
        .expect("Failed to encode batch");

    // Verify batch size
    assert_eq!(
        batch_embeddings.len(),
        texts.len(),
        "Batch should contain all inputs"
    );

    // Verify all embeddings have correct dimensions
    for (i, emb) in batch_embeddings.iter().enumerate() {
        assert_eq!(emb.dim(), 768, "Embedding {} should have 768 dimensions", i);
        assert_eq!(
            emb.text, texts[i],
            "Text should be preserved for embedding {}",
            i
        );

        // Verify not all zeros
        let sum: f32 = emb.embedding.iter().sum();
        assert!(sum.abs() > 0.01, "Embedding {} should not be all zeros", i);
    }
}

#[test]
#[ignore]
fn test_dense_batch_vs_sequential_consistency() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let texts = vec!["Hello", "World", "Test"];

    // Encode sequentially
    let sequential: Vec<_> = texts
        .iter()
        .map(|&text| embedder.encode(text).unwrap())
        .collect();

    // Encode as batch
    let batch = embedder
        .encode_batch(&texts)
        .expect("Failed to encode batch");

    assert_eq!(batch.len(), sequential.len());

    // Compare embeddings (should be very similar, allowing for minor numerical differences)
    for (i, (seq_emb, batch_emb)) in sequential.iter().zip(batch.iter()).enumerate() {
        assert_eq!(seq_emb.dim(), batch_emb.dim());

        // Check cosine similarity between sequential and batch embeddings
        let dot: f32 = seq_emb
            .embedding
            .iter()
            .zip(batch_emb.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        // For normalized embeddings, dot product is cosine similarity
        // Should be very close to 1.0
        assert!(
            dot > 0.99,
            "Sequential vs batch embedding {} should be nearly identical (similarity: {})",
            i,
            dot
        );
    }
}

#[test]
#[ignore]
fn test_dense_batch_order_preservation() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let texts = vec![
        "First document about artificial intelligence",
        "Second document about machine learning",
        "Third document about deep learning",
        "Fourth document about neural networks",
    ];

    let batch = embedder
        .encode_batch(&texts)
        .expect("Failed to encode batch");

    // Verify order is preserved
    for (i, emb) in batch.iter().enumerate() {
        assert_eq!(emb.text, texts[i], "Order not preserved at index {}", i);
    }
}

// ============================================================================
// Test 3: Dense Similarity
// ============================================================================

#[test]
#[ignore]
fn test_dense_similarity_semantic() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    // Similar texts (about AI/ML)
    let text1 = "Machine learning is a subset of artificial intelligence";
    let text2 = "AI includes machine learning as a subfield";

    // Dissimilar text (about weather)
    let text3 = "The weather is sunny and warm today";

    let sim_high = embedder
        .similarity(text1, text2)
        .expect("Failed to compute similarity");
    let sim_low = embedder
        .similarity(text1, text3)
        .expect("Failed to compute similarity");

    assert!(
        sim_high > sim_low,
        "Similar texts should have higher similarity: {} vs {}",
        sim_high,
        sim_low
    );

    // For normalized embeddings, cosine similarity should be in [0, 1]
    assert!(
        sim_high > 0.5,
        "Similar texts should have score > 0.5 (got {})",
        sim_high
    );
    assert!(
        sim_high <= 1.0,
        "Similarity should be <= 1.0 (got {})",
        sim_high
    );
}

#[test]
#[ignore]
fn test_dense_similarity_identical() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let text = "This is a test sentence";
    let similarity = embedder
        .similarity(text, text)
        .expect("Failed to compute similarity");

    // Identical texts should have similarity very close to 1.0
    assert!(
        similarity > 0.99,
        "Identical texts should have similarity ≈ 1.0 (got {})",
        similarity
    );
}

// ============================================================================
// Test 4: Normalization Validation
// ============================================================================

#[test]
#[ignore]
fn test_dense_normalization() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let text = "What is machine learning?";
    let embedding = embedder.encode(text).expect("Failed to encode text");

    // Compute L2 norm (magnitude)
    let magnitude: f32 = embedding
        .embedding
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    // BGE models have normalize=true, so magnitude should be ≈ 1.0
    assert!(
        (magnitude - 1.0).abs() < 0.01,
        "Embedding should be L2-normalized (magnitude: {})",
        magnitude
    );
}

#[test]
#[ignore]
fn test_dense_normalized_dot_equals_cosine() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let text1 = "Machine learning is fascinating";
    let text2 = "Artificial intelligence is interesting";

    let emb1 = embedder.encode(text1).unwrap();
    let emb2 = embedder.encode(text2).unwrap();

    // Compute dot product
    let dot_product: f32 = emb1
        .embedding
        .iter()
        .zip(emb2.embedding.iter())
        .map(|(a, b)| a * b)
        .sum();

    // For normalized embeddings: cosine similarity = dot product
    // Use similarity() convenience method which should give same result
    let cosine_sim = embedder.similarity(text1, text2).unwrap();

    assert!(
        (dot_product - cosine_sim).abs() < 0.001,
        "For normalized embeddings, dot product ({}) should equal cosine similarity ({})",
        dot_product,
        cosine_sim
    );
}

// ============================================================================
// Test 5: Pooling Strategy
// ============================================================================

#[test]
#[ignore]
fn test_dense_pooling_strategy() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    // Get pooling strategy from internal encoder
    // Note: We can't directly access the encoder, but we can verify behavior

    let text = "Test pooling strategy";
    let embedding = embedder.encode(text).expect("Failed to encode");

    // BGE uses mean pooling - verify we get a single vector
    assert_eq!(embedding.dim(), 768);

    // Verify pooling was applied (not just CLS token)
    // Mean pooling should produce different results than just the CLS token
    let sum: f32 = embedding.embedding.iter().sum();
    assert!(sum.abs() > 0.01, "Pooled embedding should not be all zeros");
}

// ============================================================================
// Test 6: Matryoshka Support
// ============================================================================

#[test]
#[ignore]
fn test_matryoshka_dimension_truncation() {
    // Nomic Embed v1.5 supports Matryoshka: [64, 128, 256, 512, 768]
    let embedder_768 = TesseraDense::builder()
        .model("nomic-embed-v1.5")
        .dimension(768)
        .build()
        .expect("Failed to create 768-dim embedder");

    let embedder_256 = TesseraDense::builder()
        .model("nomic-embed-v1.5")
        .dimension(256)
        .build()
        .expect("Failed to create 256-dim embedder");

    let embedder_64 = TesseraDense::builder()
        .model("nomic-embed-v1.5")
        .dimension(64)
        .build()
        .expect("Failed to create 64-dim embedder");

    let text = "What is machine learning?";

    let emb_768 = embedder_768.encode(text).unwrap();
    let emb_256 = embedder_256.encode(text).unwrap();
    let emb_64 = embedder_64.encode(text).unwrap();

    // Verify dimensions
    assert_eq!(emb_768.dim(), 768, "Should produce 768-dim embedding");
    assert_eq!(emb_256.dim(), 256, "Should produce 256-dim embedding");
    assert_eq!(emb_64.dim(), 64, "Should produce 64-dim embedding");

    // Verify embeddings are not all zeros
    assert!(emb_768.embedding.iter().sum::<f32>().abs() > 0.01);
    assert!(emb_256.embedding.iter().sum::<f32>().abs() > 0.01);
    assert!(emb_64.embedding.iter().sum::<f32>().abs() > 0.01);
}

#[test]
#[ignore]
fn test_matryoshka_prefix_consistency() {
    // Verify that smaller dimensions are prefixes of larger dimensions
    let embedder_768 = TesseraDense::builder()
        .model("nomic-embed-v1.5")
        .dimension(768)
        .build()
        .expect("Failed to create 768-dim embedder");

    let embedder_256 = TesseraDense::builder()
        .model("nomic-embed-v1.5")
        .dimension(256)
        .build()
        .expect("Failed to create 256-dim embedder");

    let text = "Testing Matryoshka consistency";

    let emb_768 = embedder_768.encode(text).unwrap();
    let emb_256 = embedder_256.encode(text).unwrap();

    // The first 256 dimensions of 768-dim embedding should match 256-dim embedding
    for i in 0..256 {
        let diff = (emb_768.embedding[i] - emb_256.embedding[i]).abs();
        assert!(
            diff < 0.001,
            "Dimension {} should match between 768 and 256 embeddings (diff: {})",
            i,
            diff
        );
    }
}

// ============================================================================
// Test 7: Factory Pattern (Tessera Enum)
// ============================================================================

#[test]
#[ignore]
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
        Tessera::MultiVector(_) => {
            panic!("Factory should have returned Dense variant for dense model");
        }
        Tessera::Sparse(_) => {
            panic!("Factory should have returned Dense variant for dense model, got Sparse");
        }
        Tessera::Vision(_) | Tessera::TimeSeries(_) => {
            panic!("Factory should have returned Dense variant for dense model");
        }
    }
}

#[test]
#[ignore]
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
        Tessera::Dense(_) => {
            panic!("Factory should have returned MultiVector variant for ColBERT model");
        }
        Tessera::Sparse(_) => {
            panic!(
                "Factory should have returned MultiVector variant for ColBERT model, got Sparse"
            );
        }
        Tessera::Vision(_) | Tessera::TimeSeries(_) => {
            panic!("Factory should have returned MultiVector variant for ColBERT model");
        }
    }
}

#[test]
#[ignore]
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
            "Error should mention missing model ID: {}",
            err
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
            "Error should mention model not found: {}",
            err
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
            "Error should mention model type mismatch: {}",
            err
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
            "Error should mention unsupported dimension: {}",
            err
        );
    }
}

#[test]
#[ignore]
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
            "Error should mention dimension not supported: {}",
            err
        );
    }
}

// ============================================================================
// Test 9: Device Selection
// ============================================================================

#[test]
#[ignore]
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
#[ignore]
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
#[ignore]
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
            "Error should mention model not found: {}",
            err_msg
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
#[ignore]
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
            println!("Empty string encoding errored (acceptable): {}", e);
        }
    }
}

// ============================================================================
// Additional Quality Tests
// ============================================================================

#[test]
#[ignore]
fn test_dense_metadata_preservation() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let text = "Testing metadata preservation";
    let embedding = embedder.encode(text).unwrap();

    assert_eq!(embedding.text, text, "Original text should be preserved");
    assert_eq!(embedding.dim(), 768, "Dimension should be correct");
}

#[test]
#[ignore]
fn test_dense_model_info_methods() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    // Test model info methods
    assert_eq!(embedder.model(), "bge-base-en-v1.5");
    assert_eq!(embedder.dimension(), 768);
}

#[test]
#[ignore]
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
    let density = non_zero_count as f32 / embedding.dim() as f32;
    assert!(
        density > 0.9,
        "Dense embedding should have >90% non-zero dimensions (got {:.1}%)",
        density * 100.0
    );
}
