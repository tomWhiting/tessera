//! Example: Dense single-vector embeddings with BGE models
//!
//! Demonstrates using CandleDenseEncoder for traditional sentence embeddings
//! with pooling strategies (CLS, mean, max).
//!
//! Run with:
//! ```bash
//! cargo run --example dense_encoder --features metal
//! ```

#![allow(unused_imports)]

use anyhow::Result;
use candle_core::Device;
use tessera::core::{DenseEncoder, Encoder};
use tessera::encoding::dense::CandleDenseEncoder;
use tessera::models::ModelConfig;

fn main() -> Result<()> {
    println!("=== Dense Encoder Example ===\n");

    // Get device (CPU or Metal)
    let device = if cfg!(feature = "metal") {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}\n", device);

    // Load BGE-small model (384 dimensions, mean pooling, normalized)
    println!("Loading BGE-small model...");
    let config = ModelConfig::from_registry("bge-small-en-v1.5")?;
    let encoder = CandleDenseEncoder::new(config, device)?;

    println!("Model loaded:");
    println!("  - Embedding dimension: {}", encoder.embedding_dim());
    println!("  - Pooling strategy: {:?}\n", encoder.pooling_strategy());

    // Example texts
    let query = "What is machine learning?";
    let doc1 = "Machine learning is a subset of artificial intelligence";
    let doc2 = "The weather today is sunny and warm";

    // Encode query
    println!("Encoding query: \"{}\"", query);
    let query_emb = encoder.encode(query)?;
    println!("  - Dimensions: {}", query_emb.dim());

    // Encode documents in batch
    println!("\nEncoding documents in batch...");
    let docs = vec![doc1, doc2];
    let doc_embs = encoder.encode_batch(&docs)?;

    // Compute cosine similarities (embeddings are already normalized)
    println!("\nSimilarity scores (cosine):");
    for (i, doc_emb) in doc_embs.iter().enumerate() {
        let similarity = query_emb
            .embedding
            .iter()
            .zip(doc_emb.embedding.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();

        println!("  Query <-> Doc{}: {:.4}", i + 1, similarity);
    }

    println!("\n✓ Dense encoding complete!");

    Ok(())
}
