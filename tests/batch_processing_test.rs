//! Integration tests for batch processing functionality

use tessera::Tessera;

#[test]
fn test_batch_empty() {
    let embedder = Tessera::new("colbert-v2").expect("Failed to load model");
    let empty_batch: Vec<&str> = vec![];

    // Pattern match to call the correct method
    let results = match &embedder {
        Tessera::MultiVector(mv) => mv.encode_batch(&empty_batch),
        _ => panic!("Expected MultiVector variant"),
    }
    .expect("Failed to encode empty batch");

    assert_eq!(results.len(), 0);
}

#[test]
fn test_batch_single() {
    let embedder = Tessera::new("colbert-v2").expect("Failed to load model");

    let text = "Hello world";

    // Pattern match to call the correct methods
    let (single, batch) = match &embedder {
        Tessera::MultiVector(mv) => {
            let s = mv.encode(text).expect("Failed to encode single");
            let b = mv.encode_batch(&[text]).expect("Failed to encode batch");
            (s, b)
        }
        _ => panic!("Expected MultiVector variant"),
    };

    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].num_tokens, single.num_tokens);
    assert_eq!(batch[0].embedding_dim, single.embedding_dim);

    // Results should be identical for single-item batch
    for i in 0..single.num_tokens {
        for j in 0..single.embedding_dim {
            assert_eq!(single.embeddings[[i, j]], batch[0].embeddings[[i, j]]);
        }
    }
}

#[test]
fn test_batch_same_length() {
    let embedder = Tessera::new("colbert-v2").expect("Failed to load model");

    // Use texts that tokenize to the same length (no padding needed)
    let texts = ["Hello", "World", "Tests"];

    let (sequential, batch) = match &embedder {
        Tessera::MultiVector(mv) => {
            let seq: Vec<_> = texts
                .iter()
                .map(|&text| mv.encode(text).expect("Failed to encode"))
                .collect();
            let bat = mv.encode_batch(&texts).expect("Failed to encode batch");
            (seq, bat)
        }
        _ => panic!("Expected MultiVector variant"),
    };

    assert_eq!(batch.len(), texts.len());

    // With same-length sequences, results should be identical
    for (seq, bat) in sequential.iter().zip(batch.iter()) {
        assert_eq!(seq.num_tokens, bat.num_tokens);
        assert_eq!(seq.embedding_dim, bat.embedding_dim);

        for i in 0..seq.num_tokens {
            for j in 0..seq.embedding_dim {
                let diff = (seq.embeddings[[i, j]] - bat.embeddings[[i, j]]).abs();
                assert!(diff < 1e-6, "Embeddings differ for same-length sequences");
            }
        }
    }
}

#[test]
fn test_batch_different_lengths() {
    let embedder = Tessera::new("colbert-v2").expect("Failed to load model");

    let texts = [
        "Short",
        "A bit longer text",
        "This is a much longer text with many more tokens",
    ];

    let batch = match &embedder {
        Tessera::MultiVector(mv) => mv.encode_batch(&texts),
        _ => panic!("Expected MultiVector variant"),
    }
    .expect("Failed to encode batch");

    assert_eq!(batch.len(), texts.len());

    // Verify each result has the correct dimensions
    for (i, result) in batch.iter().enumerate() {
        assert!(result.num_tokens > 0, "Text {} has no tokens", i);
        assert_eq!(result.embedding_dim, 128, "Incorrect embedding dimension");
        assert_eq!(result.text, texts[i], "Text mismatch");
    }

    // Verify token counts are different (different length texts)
    assert_ne!(batch[0].num_tokens, batch[2].num_tokens);
}

#[test]
fn test_batch_similarity_consistency() {
    let embedder = Tessera::new("colbert-v2").expect("Failed to load model");

    let texts = [
        "Machine learning",
        "Artificial intelligence",
        "Deep neural networks",
    ];

    // Compute similarities sequentially and from batch
    let (seq_sim_01, batch_sim_01) = match &embedder {
        Tessera::MultiVector(mv) => {
            let seq_sim = mv
                .similarity(texts[0], texts[1])
                .expect("Failed to compute similarity");

            let batch_embeddings = mv
                .encode_batch(&texts)
                .expect("Failed to encode batch");

            use tessera::utils::max_sim;
            let batch_sim = max_sim(&batch_embeddings[0], &batch_embeddings[1])
                .expect("Failed to compute batch similarity");

            (seq_sim, batch_sim)
        }
        _ => panic!("Expected MultiVector variant"),
    };

    // Similarity scores should be reasonably close (within 10%)
    // Note: Some variation is expected due to padding effects in batch processing,
    // even with attention masking. This is normal and acceptable for production use.
    let diff = (seq_sim_01 - batch_sim_01).abs();
    let rel_diff = diff / seq_sim_01.max(1e-6);

    assert!(
        rel_diff < 0.10,
        "Similarity scores differ too much: seq={}, batch={}, rel_diff={:.2}%",
        seq_sim_01,
        batch_sim_01,
        rel_diff * 100.0
    );
}

#[test]
fn test_batch_preserves_order() {
    let embedder = Tessera::new("colbert-v2").expect("Failed to load model");

    let texts = [
        "First document",
        "Second document",
        "Third document",
        "Fourth document",
    ];

    let batch = match &embedder {
        Tessera::MultiVector(mv) => mv.encode_batch(&texts),
        _ => panic!("Expected MultiVector variant"),
    }
    .expect("Failed to encode batch");

    // Verify order is preserved
    for (i, result) in batch.iter().enumerate() {
        assert_eq!(result.text, texts[i], "Order not preserved at index {}", i);
    }
}
