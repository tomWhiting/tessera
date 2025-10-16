//! Sparse vs Dense embeddings comparison.
//!
//! Demonstrates the key differences between sparse (SPLADE) and dense (BERT)
//! embeddings for semantic search. Shows complementary strengths and use cases.
//!
//! Run with:
//! ```bash
//! cargo run --release --example sparse_vs_dense
//! ```

use tessera::{TesseraSparse, TesseraDense};

fn main() -> tessera::Result<()> {
    println!("=== Sparse vs Dense Embeddings Comparison ===\n");

    // Load both embedding models
    println!("Loading models...");
    let sparse = TesseraSparse::new("splade-pp-en-v1")?;
    let dense = TesseraDense::new("bge-base-en-v1.5")?;
    println!("Both models loaded successfully!\n");

    // Test texts for comparison
    let text1 = "Machine learning is a subset of artificial intelligence";
    let text2 = "Artificial intelligence includes machine learning and deep learning";
    let text3 = "The weather forecast predicts sunny skies tomorrow";

    println!("=== SPARSE EMBEDDINGS (SPLADE) ===\n");

    let sparse_emb1 = sparse.encode(text1)?;
    println!("Text 1: \"{}\"", text1);
    println!("  Total dimensions: {}", sparse_emb1.vocab_size);
    println!("  Non-zero dimensions: {}", sparse_emb1.nnz());
    println!("  Sparsity: {:.2}%\n", sparse_emb1.sparsity() * 100.0);

    // Compute similarities
    let sim_sparse_similar = sparse.similarity(text1, text2)?;
    let sim_sparse_different = sparse.similarity(text1, text3)?;

    println!("Similarity Scores:");
    println!("  Similar texts (AI/ML):     {:.4}", sim_sparse_similar);
    println!("  Different texts (AI/weather): {:.4}", sim_sparse_different);
    println!("  Discrimination ratio:      {:.2}x\n", sim_sparse_similar / sim_sparse_different.max(0.001));

    println!("Storage efficiency:");
    println!("  Full vector would use: {} floats", sparse_emb1.vocab_size);
    println!("  Sparse storage uses:   {} floats", sparse_emb1.nnz());
    println!("  Compression ratio:     {:.1}x\n", sparse_emb1.vocab_size as f32 / sparse_emb1.nnz() as f32);

    println!("=== DENSE EMBEDDINGS (BERT) ===\n");

    let dense_emb1 = dense.encode(text1)?;
    println!("Text 1: \"{}\"", text1);
    println!("  Total dimensions: {}", dense_emb1.dim());
    println!("  All dimensions used (not sparse)\n");

    // Compute similarities
    let sim_dense_similar = dense.similarity(text1, text2)?;
    let sim_dense_different = dense.similarity(text1, text3)?;

    println!("Similarity Scores:");
    println!("  Similar texts (AI/ML):     {:.4}", sim_dense_similar);
    println!("  Different texts (AI/weather): {:.4}", sim_dense_different);
    println!("  Discrimination ratio:      {:.2}x\n", sim_dense_similar / sim_dense_different.max(0.001));

    println!("Storage efficiency:");
    println!("  Fixed storage:         {} floats", dense_emb1.dim());
    println!("  No compression available\n");

    // Batch processing comparison
    println!("=== BATCH PROCESSING COMPARISON ===\n");

    let documents = vec![
        "Artificial intelligence transforms technology",
        "Machine learning learns from data patterns",
        "Deep neural networks mimic brain structure",
        "Natural language processing analyzes text",
        "Computer vision enables image recognition",
    ];

    let docs_ref: Vec<&str> = documents.iter().map(|s| s.as_ref()).collect();

    println!("Processing {} documents in batch...\n", documents.len());

    // Sparse batch processing
    let sparse_batch = sparse.encode_batch(&docs_ref)?;
    let avg_sparse_nnz: f32 = sparse_batch.iter().map(|e| e.nnz() as f32).sum::<f32>() / sparse_batch.len() as f32;
    let avg_sparse_sparsity: f32 = sparse_batch.iter().map(|e| e.sparsity()).sum::<f32>() / sparse_batch.len() as f32;

    println!("Sparse Results:");
    println!("  Documents encoded: {}", sparse_batch.len());
    println!("  Avg non-zero dims: {:.1}", avg_sparse_nnz);
    println!("  Avg sparsity:      {:.2}%", avg_sparse_sparsity * 100.0);

    // Dense batch processing
    let dense_batch = dense.encode_batch(&docs_ref)?;

    println!("\nDense Results:");
    println!("  Documents encoded: {}", dense_batch.len());
    println!("  Fixed dimensions:  {}", dense_batch[0].dim());

    // Comparative analysis
    println!("\n=== COMPARATIVE ANALYSIS ===\n");

    println!("SPARSE (SPLADE):");
    println!("  Advantages:");
    println!("    + Interpretable (can see which terms matter)");
    println!("    + Integrates with inverted indexes");
    println!("    + Efficient storage for typical documents");
    println!("    + Natural term weighting");
    println!("  Trade-offs:");
    println!("    - High dimensionality (30K+ vocab size)");
    println!("    - Requires special indexing for large scale");

    println!("\nDENSE (BERT):");
    println!("  Advantages:");
    println!("    + Compact representation (768 dims)");
    println!("    + Fixed size regardless of content");
    println!("    + Well-studied in research");
    println!("    + Works with standard vector databases");
    println!("  Trade-offs:");
    println!("    - Black box (less interpretable)");
    println!("    - No natural term importance scores");

    println!("\n=== USE CASE RECOMMENDATIONS ===\n");

    println!("Use SPARSE when:");
    println!("  - Interpretability is important");
    println!("  - Integrating with search systems");
    println!("  - Need term-level explanations");
    println!("  - Working with inverted indexes");

    println!("\nUse DENSE when:");
    println!("  - Compact storage is priority");
    println!("  - Using vector databases");
    println!("  - Don't need interpretability");
    println!("  - Semantic similarity is primary goal");

    println!("\n=== Comparison Complete ===");
    Ok(())
}
