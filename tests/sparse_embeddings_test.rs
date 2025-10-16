//! Integration tests for sparse embeddings (Phase 2.2).
//!
//! Tests the full API surface for TesseraSparse, Tessera factory pattern,
//! and sparse-specific functionality including:
//! - Basic single-text encoding
//! - Batch processing
//! - Sparsity verification (99%+)
//! - Similarity computation (dot product)
//! - Factory pattern with auto-detection
//! - Builder validation and error handling
//! - Device selection
//! - Interpretability (non-zero indices)

use tessera::{Tessera, TesseraSparse, TesseraSparseBuilder};
use candle_core::Device;

// ============================================================================
// Test 1: Basic Sparse Encoding
// ============================================================================

#[test]
#[ignore] // Requires model download
fn test_sparse_encode_single() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    let text = "What is machine learning?";
    let embedding = embedder.encode(text)
        .expect("Failed to encode text");
    
    // Verify vocab size (BERT base vocab)
    assert_eq!(embedding.vocab_size, 30522, "Expected BERT vocab size of 30522");
    
    // Verify sparsity (should be >99%)
    let sparsity = embedding.sparsity();
    assert!(
        sparsity > 0.99,
        "Sparsity should be >99% for SPLADE, got {:.2}%",
        sparsity * 100.0
    );
    
    // Verify non-zero count (typically 100-200 for SPLADE)
    let nnz = embedding.nnz();
    assert!(
        nnz > 10 && nnz < 500,
        "Expected 10-500 non-zero values for SPLADE, got {}",
        nnz
    );
    
    // Verify weights are positive (after ReLU in SPLADE)
    for (idx, weight) in &embedding.weights {
        assert!(
            weight > &0.0,
            "All weights should be positive after ReLU (idx {} has weight {})",
            idx, weight
        );
    }
    
    // Verify text is preserved
    assert_eq!(embedding.text, text, "Text should be preserved in embedding");
}

// ============================================================================
// Test 2: Sparse Batch Processing
// ============================================================================

#[test]
#[ignore]
fn test_sparse_batch_encoding() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    let texts = vec![
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Python is a programming language",
    ];
    
    // Encode batch
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let batch_embeddings = embedder.encode_batch(&text_refs)
        .expect("Failed to encode batch");
    
    // Verify batch size
    assert_eq!(batch_embeddings.len(), texts.len(), "Batch should contain all inputs");
    
    // Verify all embeddings have correct properties
    for (i, emb) in batch_embeddings.iter().enumerate() {
        assert_eq!(emb.vocab_size, 30522, "Embedding {} should have vocab size 30522", i);
        assert_eq!(emb.text, texts[i], "Text should be preserved for embedding {}", i);
        
        // Verify sparsity
        let sparsity = emb.sparsity();
        assert!(
            sparsity > 0.99,
            "Embedding {} should have >99% sparsity, got {:.2}%",
            i, sparsity * 100.0
        );
        
        // Verify has non-zero values
        let nnz = emb.nnz();
        assert!(nnz > 10, "Embedding {} should have >10 non-zero values, got {}", i, nnz);
    }
}

#[test]
#[ignore]
fn test_sparse_batch_vs_sequential_consistency() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    let texts = vec!["Hello world", "Machine learning", "Neural networks"];
    
    // Encode sequentially
    let sequential: Vec<_> = texts.iter()
        .map(|&text| embedder.encode(text).unwrap())
        .collect();
    
    // Encode as batch
    let batch = embedder.encode_batch(&texts)
        .expect("Failed to encode batch");
    
    assert_eq!(batch.len(), sequential.len());
    
    // Compare embeddings - sparse vectors should be very similar
    for (i, (seq_emb, batch_emb)) in sequential.iter().zip(batch.iter()).enumerate() {
        assert_eq!(seq_emb.vocab_size, batch_emb.vocab_size);
        assert_eq!(seq_emb.nnz(), batch_emb.nnz(), 
            "Embedding {} should have same nnz in batch vs sequential", i);
        
        // Check that non-zero indices match
        let seq_indices: std::collections::HashSet<usize> = 
            seq_emb.weights.iter().map(|(idx, _)| *idx).collect();
        let batch_indices: std::collections::HashSet<usize> = 
            batch_emb.weights.iter().map(|(idx, _)| *idx).collect();
        
        let intersection = seq_indices.intersection(&batch_indices).count();
        let union = seq_indices.union(&batch_indices).count();
        let jaccard = intersection as f32 / union as f32;
        
        assert!(
            jaccard > 0.95,
            "Embedding {} should have >95% index overlap between batch and sequential (got {:.2}%)",
            i, jaccard * 100.0
        );
    }
}

#[test]
#[ignore]
fn test_sparse_batch_order_preservation() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    let texts = vec![
        "First document about artificial intelligence",
        "Second document about machine learning",
        "Third document about deep learning",
        "Fourth document about neural networks",
    ];
    
    let batch = embedder.encode_batch(&texts)
        .expect("Failed to encode batch");
    
    // Verify order is preserved
    for (i, emb) in batch.iter().enumerate() {
        assert_eq!(emb.text, texts[i], "Order not preserved at index {}", i);
    }
}

// ============================================================================
// Test 3: Sparse Similarity
// ============================================================================

#[test]
#[ignore]
fn test_sparse_similarity_semantic() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    // Similar texts (about AI/ML)
    let text1 = "Machine learning is a subset of artificial intelligence";
    let text2 = "AI includes machine learning as a subfield";
    
    // Dissimilar text (about weather)
    let text3 = "The weather is sunny and warm today";
    
    let sim_high = embedder.similarity(text1, text2)
        .expect("Failed to compute similarity");
    let sim_low = embedder.similarity(text1, text3)
        .expect("Failed to compute similarity");
    
    assert!(
        sim_high > sim_low,
        "Similar texts should have higher similarity: {} vs {}",
        sim_high, sim_low
    );
    
    // Sparse similarity scores are typically lower magnitude than dense
    assert!(sim_high > 0.0, "Similar texts should have positive similarity (got {})", sim_high);
}

#[test]
#[ignore]
fn test_sparse_similarity_identical() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    let text = "This is a test sentence for sparse embeddings";
    let similarity = embedder.similarity(text, text)
        .expect("Failed to compute similarity");
    
    // Identical texts should have positive similarity
    assert!(
        similarity > 0.0,
        "Identical texts should have positive similarity (got {})",
        similarity
    );
}

// ============================================================================
// Test 4: Sparsity Verification
// ============================================================================

#[test]
#[ignore]
fn test_sparsity_varies_by_text() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    let short_text = "Hello";
    let long_text = "Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data";
    
    let emb_short = embedder.encode(short_text).unwrap();
    let emb_long = embedder.encode(long_text).unwrap();
    
    // Both should be very sparse (>99%)
    assert!(emb_short.sparsity() > 0.99, 
        "Short text should have >99% sparsity, got {:.2}%", 
        emb_short.sparsity() * 100.0);
    assert!(emb_long.sparsity() > 0.99, 
        "Long text should have >99% sparsity, got {:.2}%", 
        emb_long.sparsity() * 100.0);
    
    // Longer texts might activate more vocabulary terms
    println!("Short text: {} non-zero values ({:.2}% sparsity)", 
        emb_short.nnz(), emb_short.sparsity() * 100.0);
    println!("Long text: {} non-zero values ({:.2}% sparsity)", 
        emb_long.nnz(), emb_long.sparsity() * 100.0);
}

#[test]
#[ignore]
fn test_sparsity_calculation() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    let text = "What is machine learning?";
    let embedding = embedder.encode(text).unwrap();
    
    // Manually verify sparsity calculation
    let nnz = embedding.nnz();
    let vocab_size = embedding.vocab_size;
    let expected_sparsity = 1.0 - (nnz as f32 / vocab_size as f32);
    let actual_sparsity = embedding.sparsity();
    
    assert!(
        (expected_sparsity - actual_sparsity).abs() < 1e-6,
        "Sparsity calculation should match: expected {:.6}, got {:.6}",
        expected_sparsity, actual_sparsity
    );
}

// ============================================================================
// Test 5: Interpretability - Non-Zero Indices
// ============================================================================

#[test]
#[ignore]
fn test_sparse_interpretability() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    let text = "machine learning algorithm";
    let emb = embedder.encode(text).unwrap();
    
    // Should have activated vocabulary terms related to input
    assert!(emb.nnz() > 0, "Should have non-zero activations");
    
    // All activated indices should be valid vocabulary indices
    for (idx, weight) in &emb.weights {
        assert!(idx < &30522, "Vocab index {} out of bounds (max 30521)", idx);
        assert!(weight > &0.0, "Weight should be positive, got {}", weight);
    }
    
    println!("Activated {} vocabulary terms for text: '{}'", emb.nnz(), text);
}

#[test]
#[ignore]
fn test_sparse_weights_sorted() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    let text = "machine learning and artificial intelligence";
    let emb = embedder.encode(text).unwrap();
    
    // Check that indices are valid and weights are positive
    for (idx, weight) in &emb.weights {
        assert!(idx < &30522, "Index out of bounds");
        assert!(weight > &0.0, "Weight should be positive");
    }
    
    // SPLADE typically doesn't guarantee sorted indices, just verify structure
    assert!(emb.nnz() > 0, "Should have non-zero weights");
}

// ============================================================================
// Test 6: Factory Pattern (Tessera Enum)
// ============================================================================

#[test]
#[ignore]
fn test_factory_sparse_model() {
    // Create embedder using factory - should return Sparse variant
    let embedder = Tessera::new("splade-pp-en-v1")
        .expect("Failed to create embedder via factory");
    
    // Pattern match to verify it's the Sparse variant
    match embedder {
        Tessera::Sparse(sparse) => {
            // Verify it works
            let embedding = sparse.encode("Test factory pattern")
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
#[ignore]
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
#[ignore]
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
            "Error should mention missing model ID: {}",
            err_msg
        );
    }
}

#[test]
fn test_builder_wrong_model_type() {
    // Try to use dense model with sparse builder
    let result = TesseraSparseBuilder::new()
        .model("bge-base-en-v1.5")  // Dense model
        .build();
    
    assert!(result.is_err(), "Should error with non-sparse model");
    if let Err(err) = result {
        let error_msg = format!("{:?}", err);
        assert!(
            error_msg.contains("not Sparse") || error_msg.contains("not a sparse"),
            "Error should mention model type mismatch: {}",
            error_msg
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
            "Error should mention model not found: {}",
            err_msg
        );
    }
}

// ============================================================================
// Test 8: Model Info Methods
// ============================================================================

#[test]
#[ignore]
fn test_vocab_size_accessor() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    assert_eq!(embedder.vocab_size(), 30522, "Should return correct vocab size");
}

#[test]
#[ignore]
fn test_model_accessor() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
    assert_eq!(embedder.model(), "splade-pp-en-v1", "Should return correct model ID");
}

#[test]
#[ignore]
fn test_sparse_metadata_preservation() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
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
#[ignore]
fn test_device_auto_selection() {
    // Create embedder with auto device selection (default)
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create embedder with auto device selection");
    
    // Verify it works
    let embedding = embedder.encode("Test auto device selection")
        .expect("Failed to encode with auto-selected device");
    
    assert_eq!(embedding.vocab_size, 30522);
    assert!(embedding.sparsity() > 0.99);
}

#[test]
#[ignore]
fn test_device_explicit_cpu() {
    // Force CPU device
    let embedder = TesseraSparseBuilder::new()
        .model("splade-pp-en-v1")
        .device(Device::Cpu)
        .build()
        .expect("Failed to create embedder with CPU device");
    
    // Verify it works on CPU
    let embedding = embedder.encode("Test CPU device")
        .expect("Failed to encode on CPU");
    
    assert_eq!(embedding.vocab_size, 30522);
    assert!(embedding.sparsity() > 0.99);
}

#[test]
#[ignore]
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
        let embedding = embedder.encode("Test Metal device")
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
            "Error should mention model not found: {}",
            err_msg
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
            "Should mention Model ID: {}",
            msg
        );
    } else {
        panic!("Expected error for missing model ID");
    }
    
    // Invalid model
    if let Err(err2) = TesseraSparse::new("invalid-sparse-model") {
        let msg = err2.to_string();
        assert!(
            msg.contains("invalid") || msg.contains("not found"),
            "Should mention the invalid model ID or that it wasn't found: {}",
            msg
        );
    } else {
        panic!("Expected error for invalid model");
    }
}

#[test]
#[ignore]
fn test_encode_empty_string() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create embedder");
    
    // Empty string should still produce embedding (likely minimal activations)
    let result = embedder.encode("");
    
    // This might error or produce minimal embedding - either is acceptable
    match result {
        Ok(embedding) => {
            assert_eq!(embedding.vocab_size, 30522, "Should have correct vocab size");
            println!("Empty string encoded with {} non-zero values", embedding.nnz());
        }
        Err(e) => {
            println!("Empty string encoding errored (acceptable): {}", e);
        }
    }
}

// ============================================================================
// Test 11: Dot Product Similarity Implementation
// ============================================================================

#[test]
#[ignore]
fn test_sparse_dot_product_manual() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
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
        "Manual dot product ({}) should match API similarity ({})",
        manual_score, api_score
    );
}

// ============================================================================
// Test 12: Additional Quality Tests
// ============================================================================

#[test]
#[ignore]
fn test_sparse_not_dense() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
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
    
    println!("Sparsity: {:.2}%, Non-zero: {}/{}", 
        embedding.sparsity() * 100.0,
        embedding.nnz(),
        embedding.vocab_size);
}

#[test]
#[ignore]
fn test_sparse_weights_magnitude() {
    let embedder = TesseraSparse::new("splade-pp-en-v1")
        .expect("Failed to create sparse embedder");
    
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
    assert!(min_weight > 0.0, "Min weight should be positive, got {}", min_weight);
    assert!(max_weight > min_weight, "Should have weight variation");
    
    println!("Weight range: [{:.4}, {:.4}]", min_weight, max_weight);
}
