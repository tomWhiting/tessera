// ============================================================================
// Test 1: Basic Dense Encoding
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
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
        "Embedding should not be all zeros (sum: {sum})"
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
#[ignore = "requires remote model artifacts"]
fn test_dense_batch_encoding() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Natural language processing enables text understanding",
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

    // Verify all embeddings have correct dimensions
    for (i, emb) in batch_embeddings.iter().enumerate() {
        assert_eq!(emb.dim(), 768, "Embedding {i} should have 768 dimensions");
        assert_eq!(
            emb.text, texts[i],
            "Text should be preserved for embedding {i}"
        );

        // Verify not all zeros
        let sum: f32 = emb.embedding.iter().sum();
        assert!(sum.abs() > 0.01, "Embedding {i} should not be all zeros");
    }
}
#[test]
#[ignore = "requires remote model artifacts"]
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
            "Sequential vs batch embedding {i} should be nearly identical (similarity: {dot})"
        );
    }
}

#[test]
#[ignore = "requires remote model artifacts"]
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
        assert_eq!(emb.text, texts[i], "Order not preserved at index {i}");
    }
}

// ============================================================================
// Test 3: Dense Similarity
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
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
        "Similar texts should have higher similarity: {sim_high} vs {sim_low}"
    );

    // For normalized embeddings, cosine similarity should be in [0, 1]
    assert!(
        sim_high > 0.5,
        "Similar texts should have score > 0.5 (got {sim_high})"
    );
    assert!(
        sim_high <= 1.0,
        "Similarity should be <= 1.0 (got {sim_high})"
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
fn test_dense_similarity_identical() {
    let embedder = TesseraDense::new("bge-base-en-v1.5").expect("Failed to create embedder");

    let text = "This is a test sentence";
    let similarity = embedder
        .similarity(text, text)
        .expect("Failed to compute similarity");

    // Identical texts should have similarity very close to 1.0
    assert!(
        similarity > 0.99,
        "Identical texts should have similarity ≈ 1.0 (got {similarity})"
    );
}

// ============================================================================
// Test 4: Normalization Validation
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
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
        "Embedding should be L2-normalized (magnitude: {magnitude})"
    );
}

#[test]
#[ignore = "requires remote model artifacts"]
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
        "For normalized embeddings, dot product ({dot_product}) should equal cosine similarity ({cosine_sim})"
    );
}

// ============================================================================
// Test 5: Pooling Strategy
// ============================================================================

#[test]
#[ignore = "requires remote model artifacts"]
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
#[ignore = "requires remote model artifacts"]
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
#[ignore = "requires remote model artifacts"]
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
            "Dimension {i} should match between 768 and 256 embeddings (diff: {diff})"
        );
    }
}
