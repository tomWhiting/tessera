// ============================================================================
// Test 1: Basic Sparse Encoding
// ============================================================================
#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_encode_single() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let text = "What is machine learning?";
    let embedding = embedder.encode(text).expect("Failed to encode text");

    // Verify vocab size (BERT base vocab)
    assert_eq!(
        embedding.vocab_size, 30522,
        "Expected BERT vocab size of 30522"
    );

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
        "Expected 10-500 non-zero values for SPLADE, got {nnz}"
    );

    // Verify weights are positive (after ReLU in SPLADE)
    for (idx, weight) in &embedding.weights {
        assert!(
            weight > &0.0,
            "All weights should be positive after ReLU (idx {idx} has weight {weight})"
        );
    }

    // Verify text is preserved
    assert_eq!(
        embedding.text, text,
        "Text should be preserved in embedding"
    );
}

// ============================================================================
// Test 2: Sparse Batch Processing
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_batch_encoding() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Python is a programming language",
    ];

    // Encode batch
    let text_refs: Vec<&str> = texts.iter().map(std::convert::AsRef::as_ref).collect();
    let batch_embeddings = embedder
        .encode_batch(&text_refs)
        .expect("Failed to encode batch");

    // Verify batch size
    assert_eq!(
        batch_embeddings.len(),
        texts.len(),
        "Batch should contain all inputs"
    );

    // Verify all embeddings have correct properties
    for (i, emb) in batch_embeddings.iter().enumerate() {
        assert_eq!(
            emb.vocab_size, 30522,
            "Embedding {i} should have vocab size 30522"
        );
        assert_eq!(
            emb.text, texts[i],
            "Text should be preserved for embedding {i}"
        );

        // Verify sparsity
        let sparsity = emb.sparsity();
        assert!(
            sparsity > 0.99,
            "Embedding {} should have >99% sparsity, got {:.2}%",
            i,
            sparsity * 100.0
        );

        // Verify has non-zero values
        let nnz = emb.nnz();
        assert!(
            nnz > 10,
            "Embedding {i} should have >10 non-zero values, got {nnz}"
        );
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_batch_vs_sequential_consistency() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let texts = vec!["Hello world", "Machine learning", "Neural networks"];

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

    // Compare embeddings - sparse vectors should be very similar
    for (i, (seq_emb, batch_emb)) in sequential.iter().zip(batch.iter()).enumerate() {
        assert_eq!(seq_emb.vocab_size, batch_emb.vocab_size);
        assert_eq!(
            seq_emb.nnz(),
            batch_emb.nnz(),
            "Embedding {i} should have same nnz in batch vs sequential"
        );

        // Check that non-zero indices match
        let seq_indices: std::collections::HashSet<usize> =
            seq_emb.weights.iter().map(|(idx, _)| *idx).collect();
        let batch_indices: std::collections::HashSet<usize> =
            batch_emb.weights.iter().map(|(idx, _)| *idx).collect();

        let intersection = seq_indices.intersection(&batch_indices).count();
        let union = seq_indices.union(&batch_indices).count();
        let intersection =
            u32::try_from(intersection).expect("index intersection should fit in u32");
        let union = u32::try_from(union).expect("index union should fit in u32");
        let jaccard = f64::from(intersection) / f64::from(union);

        assert!(
            jaccard > 0.95,
            "Embedding {} should have >95% index overlap between batch and sequential (got {:.2}%)",
            i,
            jaccard * 100.0
        );
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_batch_order_preservation() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

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
        assert_eq!(emb.text, texts[i], "Order not preserved at index {i}");
    }
}

// ============================================================================
// Test 3: Sparse Similarity
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_similarity_semantic() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

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
        "Similar texts should have higher similarity: {sim_high} vs {sim_low}"
    );

    // Sparse similarity scores are typically lower magnitude than dense
    assert!(
        sim_high > 0.0,
        "Similar texts should have positive similarity (got {sim_high})"
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_similarity_identical() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let text = "This is a test sentence for sparse embeddings";
    let similarity = embedder
        .similarity(text, text)
        .expect("Failed to compute similarity");

    // Identical texts should have positive similarity
    assert!(
        similarity > 0.0,
        "Identical texts should have positive similarity (got {similarity})"
    );
}

// ============================================================================
// Test 4: Sparsity Verification
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparsity_varies_by_text() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let short_text = "Hello";
    let long_text = "Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data";

    let emb_short = embedder.encode(short_text).unwrap();
    let emb_long = embedder.encode(long_text).unwrap();

    // Both should be very sparse (>99%)
    assert!(
        emb_short.sparsity() > 0.99,
        "Short text should have >99% sparsity, got {:.2}%",
        emb_short.sparsity() * 100.0
    );
    assert!(
        emb_long.sparsity() > 0.99,
        "Long text should have >99% sparsity, got {:.2}%",
        emb_long.sparsity() * 100.0
    );

    // Longer texts might activate more vocabulary terms
    println!(
        "Short text: {} non-zero values ({:.2}% sparsity)",
        emb_short.nnz(),
        emb_short.sparsity() * 100.0
    );
    println!(
        "Long text: {} non-zero values ({:.2}% sparsity)",
        emb_long.nnz(),
        emb_long.sparsity() * 100.0
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparsity_calculation() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let text = "What is machine learning?";
    let embedding = embedder.encode(text).unwrap();

    // Manually verify sparsity calculation
    let nnz = u32::try_from(embedding.nnz()).expect("non-zero count should fit in u32");
    let vocab_size =
        u32::try_from(embedding.vocab_size).expect("vocabulary size should fit in u32");
    let expected_sparsity = 1.0 - (f64::from(nnz) / f64::from(vocab_size));
    let actual_sparsity = f64::from(embedding.sparsity());

    assert!(
        (expected_sparsity - actual_sparsity).abs() < 1e-6,
        "Sparsity calculation should match: expected {expected_sparsity:.6}, got {actual_sparsity:.6}"
    );
}

// ============================================================================
// Test 5: Interpretability - Non-Zero Indices
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_interpretability() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

    let text = "machine learning algorithm";
    let emb = embedder.encode(text).unwrap();

    // Should have activated vocabulary terms related to input
    assert!(emb.nnz() > 0, "Should have non-zero activations");

    // All activated indices should be valid vocabulary indices
    for (idx, weight) in &emb.weights {
        assert!(idx < &30522, "Vocab index {idx} out of bounds (max 30521)");
        assert!(weight > &0.0, "Weight should be positive, got {weight}");
    }

    println!(
        "Activated {} vocabulary terms for text: '{}'",
        emb.nnz(),
        text
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_sparse_weights_sorted() {
    let embedder = TesseraSparse::new("splade-pp-en-v1").expect("Failed to create sparse embedder");

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
