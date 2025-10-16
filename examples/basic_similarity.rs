//! Basic similarity example demonstrating ColBERT-style MaxSim scoring.
//!
//! This example shows how to:
//! 1. Load a ColBERT model using the Candle backend
//! 2. Encode query and document text into token embeddings
//! 3. Compute MaxSim similarity score
//!
//! This uses REAL ColBERT inference with the projection layer,
//! outputting 128-dimensional embeddings for ColBERT v2 or
//! 96-dimensional embeddings for ColBERT Small.

use anyhow::{Context, Result};

use tessera::{
    backends::candle::{get_device, CandleBertEncoder},
    core::TokenEmbedder,
    models::ModelConfig,
    utils::similarity::max_sim,
};

fn main() -> Result<()> {
    println!("ColBERT MaxSim Similarity Example");
    println!("==================================\n");

    // Use ColBERT v2 - the original Stanford ColBERT model
    // This outputs 128-dimensional embeddings (768 BERT dims -> 128 via projection layer)
    let config = ModelConfig::colbert_v2();

    // Example query and document
    let query = "What is machine learning?";
    let document = "Machine learning is a subset of artificial intelligence";

    println!("Query:    {}", query);
    println!("Document: {}\n", document);

    // Demonstrate Candle backend with real ColBERT model
    demonstrate_candle(query, document, config)?;

    Ok(())
}

fn demonstrate_candle(query: &str, document: &str, config: ModelConfig) -> Result<()> {
    // Get the best available device (Metal on macOS, CPU otherwise)
    let device = get_device().context("Getting compute device")?;
    println!(
        "Using device: {}",
        tessera::backends::candle::device_description(&device)
    );

    // Create encoder
    println!("Loading {}...", config.model_name);
    let encoder = CandleBertEncoder::new(config, device).context("Creating Candle encoder")?;
    println!("Model loaded successfully\n");

    // Encode query
    println!("Encoding query...");
    let query_embeddings = encoder.encode(query).context("Encoding query text")?;
    println!(
        "Query: {} tokens, {} dimensions",
        query_embeddings.num_tokens, query_embeddings.embedding_dim
    );

    // Encode document
    println!("Encoding document...");
    let doc_embeddings = encoder.encode(document).context("Encoding document text")?;
    println!(
        "Document: {} tokens, {} dimensions",
        doc_embeddings.num_tokens, doc_embeddings.embedding_dim
    );

    // Compute MaxSim similarity
    println!("\nComputing MaxSim similarity...");
    let similarity =
        max_sim(&query_embeddings, &doc_embeddings).context("Computing MaxSim score")?;

    println!("MaxSim Score: {:.4}", similarity);
    println!("(Higher scores indicate greater similarity)");

    Ok(())
}
