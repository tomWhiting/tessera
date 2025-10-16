//! Example showing how to use the model registry for similarity scoring.
//!
//! This example demonstrates:
//! - Loading a model from the registry
//! - Computing embeddings
//! - Computing MaxSim scores

use anyhow::Result;
use tessera::{
    backends::candle::{get_device, CandleBertEncoder},
    core::TokenEmbedder,
    utils::similarity::max_sim,
    ModelConfig,
};

fn main() -> Result<()> {
    println!("Model Registry Similarity Example");
    println!("=================================\n");

    // Load model from registry by ID
    println!("Loading model from registry...");
    let config = ModelConfig::from_registry("colbert-small")?;
    println!("Model: {}", config.model_name);
    println!("Embedding dimensions: {}", config.embedding_dim);
    println!("Max sequence length: {}\n", config.max_seq_length);

    // Create encoder
    let device = get_device()?;
    let encoder = CandleBertEncoder::new(config, device)?;

    // Test queries and documents
    let queries = vec![
        "What is machine learning?",
        "How do neural networks work?",
        "What is the capital of France?",
    ];

    let documents = vec![
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
    ];

    println!("Computing similarities...\n");

    // Encode all queries
    let query_embeddings: Vec<_> = queries
        .iter()
        .map(|q| encoder.encode(q))
        .collect::<Result<Vec<_>>>()?;

    // Encode all documents
    let doc_embeddings: Vec<_> = documents
        .iter()
        .map(|d| encoder.encode(d))
        .collect::<Result<Vec<_>>>()?;

    // Compute similarity matrix
    println!("Similarity Matrix:");
    println!("------------------");

    for (qi, query) in queries.iter().enumerate() {
        println!("\nQuery: \"{}\"", query);

        let mut scores: Vec<(usize, f32)> = doc_embeddings
            .iter()
            .enumerate()
            .map(|(di, doc_emb)| {
                let score = max_sim(&query_embeddings[qi], doc_emb).unwrap();
                (di, score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (rank, (di, score)) in scores.iter().enumerate() {
            println!("  {}. [{:.4}] {}", rank + 1, score, documents[*di]);
        }
    }

    println!("\nDone!");
    Ok(())
}
