//! Integration tests for quantization API.

use tessera::{QuantizationConfig, Result, TesseraMultiVectorBuilder};

#[test]
fn test_quantization_workflow() -> Result<()> {
    // Create embedder with binary quantization
    let embedder = TesseraMultiVectorBuilder::new()
        .model("colbert-v2")
        .quantization(QuantizationConfig::Binary)
        .build()?;

    // Encode text
    let text = "What is machine learning?";
    let embeddings = embedder.encode(text)?;

    // Quantize embeddings
    let quantized = embedder.quantize(&embeddings)?;

    // Verify metadata
    assert_eq!(quantized.num_tokens, embeddings.num_tokens);
    assert_eq!(quantized.original_dim, embeddings.embedding_dim);

    // Verify compression ratio (should be ~32x for binary)
    let ratio = quantized.compression_ratio();
    assert!(
        ratio > 30.0,
        "Expected compression ratio > 30x, got {:.1}x",
        ratio
    );
    assert!(
        ratio < 34.0,
        "Expected compression ratio < 34x, got {:.1}x",
        ratio
    );

    Ok(())
}

#[test]
fn test_encode_quantized_convenience() -> Result<()> {
    let embedder = TesseraMultiVectorBuilder::new()
        .model("colbert-v2")
        .quantization(QuantizationConfig::Binary)
        .build()?;

    // Test convenience method
    let quantized = embedder.encode_quantized("Test text")?;

    assert!(quantized.num_tokens > 0);
    assert!(quantized.original_dim > 0);
    assert!(quantized.compression_ratio() > 30.0);

    Ok(())
}

#[test]
fn test_similarity_quantized() -> Result<()> {
    let embedder = TesseraMultiVectorBuilder::new()
        .model("colbert-v2")
        .quantization(QuantizationConfig::Binary)
        .build()?;

    let query = embedder.encode_quantized("What is AI?")?;
    let doc1 = embedder.encode_quantized("Artificial intelligence is machine learning")?;
    let doc2 = embedder.encode_quantized("The weather is sunny")?;

    let score1 = embedder.similarity_quantized(&query, &doc1)?;
    let score2 = embedder.similarity_quantized(&query, &doc2)?;

    // Relevant document should score higher than irrelevant
    assert!(
        score1 > score2,
        "Expected AI-related doc to score higher: {} > {}",
        score1,
        score2
    );

    Ok(())
}

#[test]
fn test_quantization_error_without_config() -> Result<()> {
    // Create embedder WITHOUT quantization
    let embedder = TesseraMultiVectorBuilder::new()
        .model("colbert-v2")
        .build()?;

    let embeddings = embedder.encode("Test")?;

    // Should error when trying to quantize
    let result = embedder.quantize(&embeddings);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.to_string().contains("No quantizer configured"));

    Ok(())
}

#[test]
fn test_quantization_memory_savings() -> Result<()> {
    let embedder = TesseraMultiVectorBuilder::new()
        .model("colbert-v2")
        .quantization(QuantizationConfig::Binary)
        .build()?;

    let text = "This is a longer text with more tokens to demonstrate memory savings from binary quantization";
    let embeddings = embedder.encode(text)?;
    let quantized = embedder.quantize(&embeddings)?;

    // Calculate original size
    let float_bytes = embeddings.num_tokens * embeddings.embedding_dim * 4;
    let binary_bytes = quantized.memory_bytes();

    println!("Float32: {} bytes", float_bytes);
    println!("Binary: {} bytes", binary_bytes);
    println!("Ratio: {:.1}x", quantized.compression_ratio());

    // Binary should be much smaller
    assert!(binary_bytes < float_bytes / 30);

    Ok(())
}

#[test]
fn test_ranking_preservation() -> Result<()> {
    let embedder = TesseraMultiVectorBuilder::new()
        .model("colbert-v2")
        .quantization(QuantizationConfig::Binary)
        .build()?;

    let query = "machine learning algorithms";
    let docs = vec![
        "Machine learning uses statistical algorithms", // High relevance
        "Deep learning is a subset of machine learning", // High relevance
        "The weather forecast predicts rain",           // Low relevance
        "I like pizza and pasta",                       // Low relevance
    ];

    // Get quantized scores
    let query_quant = embedder.encode_quantized(query)?;
    let quant_scores: Vec<_> = docs
        .iter()
        .map(|doc| {
            let doc_quant = embedder.encode_quantized(doc).unwrap();
            embedder
                .similarity_quantized(&query_quant, &doc_quant)
                .unwrap()
        })
        .collect();

    // Get full precision scores
    let full_scores: Vec<_> = docs
        .iter()
        .map(|doc| embedder.similarity(query, doc).unwrap())
        .collect();

    // Create ranking indices
    let mut quant_ranking: Vec<_> = (0..docs.len()).collect();
    let mut full_ranking: Vec<_> = (0..docs.len()).collect();

    quant_ranking.sort_by(|&i, &j| quant_scores[j].partial_cmp(&quant_scores[i]).unwrap());
    full_ranking.sort_by(|&i, &j| full_scores[j].partial_cmp(&full_scores[i]).unwrap());

    println!("Quantized ranking: {:?}", quant_ranking);
    println!("Full ranking: {:?}", full_ranking);

    // Top 2 should be the same (high relevance docs)
    let mut quant_top2 = quant_ranking[0..2].to_vec();
    let mut full_top2 = full_ranking[0..2].to_vec();
    quant_top2.sort();
    full_top2.sort();

    assert_eq!(
        quant_top2, full_top2,
        "Top 2 rankings should match between quantized and full precision"
    );

    Ok(())
}

#[test]
fn test_no_quantization_config_default() -> Result<()> {
    // Default should be no quantization
    let embedder = TesseraMultiVectorBuilder::new()
        .model("colbert-v2")
        .build()?;

    let embeddings = embedder.encode("Test")?;
    let result = embedder.quantize(&embeddings);

    // Should error since no quantization was configured
    assert!(result.is_err());

    Ok(())
}
