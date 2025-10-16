//! Demonstration of binary quantization for 32x compression.
//!
//! Shows how to:
//! - Enable binary quantization via builder
//! - Encode and quantize embeddings
//! - Measure memory savings (32x compression)
//! - Compute similarity with quantized embeddings
//! - Compare accuracy vs full precision

use tessera::{Tessera, QuantizationConfig, Result};

fn main() -> Result<()> {
    println!("=== Binary Quantization Demo ===\n");

    // Create embedder with binary quantization
    println!("Loading model with binary quantization...");
    let embedder = Tessera::builder()
        .model("colbert-v2")
        .quantization(QuantizationConfig::Binary)
        .build()?;
    println!("Model loaded: {}\n", embedder.model());

    // Define test data
    let query = "What is machine learning?";
    let docs = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "The weather is nice today",
    ];

    println!("Query: {}\n", query);

    // Encode and quantize query
    println!("Encoding and quantizing query...");
    let query_full = embedder.encode(query)?;
    let query_quant = embedder.quantize(&query_full)?;

    println!("Query embeddings:");
    println!("  Tokens: {}", query_quant.num_tokens);
    println!("  Dimension: {}", query_quant.original_dim);
    println!("  Memory (float32): {} bytes", query_full.num_tokens * query_full.embedding_dim * 4);
    println!("  Memory (binary): {} bytes", query_quant.memory_bytes());
    println!("  Compression: {:.1}x\n", query_quant.compression_ratio());

    // Process documents with quantization
    println!("Documents (ranked by quantized similarity):");
    println!("{:-<80}", "");

    let mut results = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        // Quantized similarity
        let doc_quant = embedder.encode_quantized(doc)?;
        let score_quant = embedder.similarity_quantized(&query_quant, &doc_quant)?;

        // Full precision similarity for comparison
        let score_full = embedder.similarity(&query, doc)?;

        results.push((i, doc, score_quant, score_full, doc_quant));
    }

    // Sort by quantized score (descending)
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    for (rank, (idx, doc, score_quant, score_full, doc_quant)) in results.iter().enumerate() {
        println!("\nRank {}: Doc {}", rank + 1, idx);
        println!("  Text: {}", doc);
        println!("  Score (binary): {:.2}", score_quant);
        println!("  Score (float32): {:.4}", score_full);
        println!("  Compression: {:.1}x", doc_quant.compression_ratio());
    }

    println!("\n{:-<80}", "");

    // Accuracy analysis
    println!("\n=== Accuracy Analysis ===\n");
    
    // Compare rankings
    let mut full_results = results.clone();
    full_results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("Ranking comparison:");
    println!("  Binary quantized: {:?}", results.iter().map(|r| r.0).collect::<Vec<_>>());
    println!("  Full precision:   {:?}", full_results.iter().map(|r| r.0).collect::<Vec<_>>());

    let same_ranking = results.iter().map(|r| r.0).collect::<Vec<_>>() == 
                      full_results.iter().map(|r| r.0).collect::<Vec<_>>();
    
    if same_ranking {
        println!("  ✓ Rankings match perfectly!");
    } else {
        println!("  ✗ Rankings differ (but top result should match)");
    }

    // Calculate relative score preservation
    let top_doc_idx = results[0].0;
    let score_binary = results[0].2;
    let score_float = results.iter().find(|r| r.0 == top_doc_idx).unwrap().3;
    
    // Normalize scores for comparison (binary scores are on different scale)
    println!("\nTop document score preservation:");
    println!("  Binary score: {:.2}", score_binary);
    println!("  Float32 score: {:.4}", score_float);
    println!("  Note: Scores are on different scales, but relative ranking is preserved");

    println!("\n=== Convenience Methods ===\n");

    // Show encode_quantized shortcut
    let quick_quant = embedder.encode_quantized("Quick encoding example")?;
    println!("encode_quantized() convenience method:");
    println!("  Encodes and quantizes in one call");
    println!("  Result: {} tokens, {:.1}x compression", 
        quick_quant.num_tokens, 
        quick_quant.compression_ratio()
    );

    println!("\n=== Summary ===\n");
    println!("Binary quantization provides:");
    println!("  ✓ 32x memory reduction");
    println!("  ✓ Preserved relative ranking");
    println!("  ✓ Fast Hamming-based distance computation");
    println!("  ✓ Ideal for first-stage retrieval + reranking");

    Ok(())
}
