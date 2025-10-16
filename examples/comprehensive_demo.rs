//! Comprehensive ColBERT demonstration with:
//! 1. Longer text (150-200 words)
//! 2. Actual embedding visualization
//! 3. GPU acceleration (Metal on macOS)
//!
//! Run with GPU support:
//! cargo run --example comprehensive_demo --features metal

use anyhow::{Context, Result};
use tessera::{
    backends::candle::{CandleEncoder, get_device},
    core::{max_sim, TokenEmbedder},
    models::ModelConfig,
};

fn main() -> Result<()> {
    let sep = "=".repeat(80);
    println!("{}", sep);
    println!("ColBERT Comprehensive Demonstration");
    println!("{}", sep);
    println!();

    // Use ColBERT v2 for 128-dimensional embeddings
    let config = ModelConfig::colbert_v2();

    // Longer query (approximately 180 words)
    let query = "Machine learning is a subset of artificial intelligence that focuses on \
        enabling computers to learn from data and improve their performance over time without \
        being explicitly programmed. The core idea behind machine learning is to develop \
        algorithms that can identify patterns in data and use these patterns to make predictions \
        or decisions. There are several types of machine learning approaches, including supervised \
        learning where models are trained on labeled data, unsupervised learning which finds \
        hidden patterns in unlabeled data, and reinforcement learning where agents learn through \
        trial and error by receiving rewards or penalties. Deep learning, a specialized branch \
        of machine learning, uses artificial neural networks with multiple layers to process \
        complex patterns in large datasets. Common applications of machine learning include \
        image recognition, natural language processing, recommendation systems, autonomous vehicles, \
        fraud detection, and medical diagnosis. The field has experienced rapid growth due to \
        increased computational power, availability of large datasets, and advances in algorithmic \
        techniques.";

    // Shorter document for comparison
    let document = "Artificial intelligence and machine learning are transforming how computers \
        process information and make decisions. These technologies enable systems to learn from \
        experience and adapt to new situations without explicit programming.";

    println!("QUERY ({} words):", query.split_whitespace().count());
    println!("{}\n", query);
    println!("DOCUMENT ({} words):", document.split_whitespace().count());
    println!("{}\n", document);

    // Get device (Metal GPU on macOS if available)
    let device = get_device().context("Getting compute device")?;
    let device_desc = tessera::backends::candle::device_description(&device);

    println!("Device: {}", device_desc);
    println!();

    // Create encoder
    println!("Loading model: {}", config.model_name);
    let encoder = CandleEncoder::new(config.clone(), device)
        .context("Creating ColBERT encoder")?;
    println!("Model loaded successfully");
    println!();

    // Encode query
    let sep = "=".repeat(80);
    println!("{}", sep);
    println!("ENCODING QUERY");
    println!("{}", sep);

    let query_embeddings = encoder.encode(query)
        .context("Encoding query")?;

    println!("Tokens: {}", query_embeddings.num_tokens);
    println!("Embedding dimensions: {}", query_embeddings.embedding_dim);
    println!();

    // Show actual embedding values for first few tokens
    println!("First 3 token embeddings (showing first 10 dimensions):");
    let dash = "-".repeat(80);
    println!("{}", dash);

    for token_idx in 0..3.min(query_embeddings.num_tokens) {
        print!("Token {}: [", token_idx);
        for dim in 0..10.min(query_embeddings.embedding_dim) {
            print!("{:7.4}", query_embeddings.embeddings[[token_idx, dim]]);
            if dim < 9 && dim < query_embeddings.embedding_dim - 1 {
                print!(", ");
            }
        }
        println!(", ...]");
    }
    println!();

    // Show embedding statistics
    let all_values: Vec<f32> = query_embeddings.embeddings.iter().copied().collect();
    let min_val = all_values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = all_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_val = all_values.iter().sum::<f32>() / all_values.len() as f32;

    println!("Embedding statistics:");
    println!("  Min value: {:.6}", min_val);
    println!("  Max value: {:.6}", max_val);
    println!("  Mean value: {:.6}", mean_val);
    println!("  Total elements: {}", all_values.len());
    println!();

    // Encode document
    println!("{}", sep);
    println!("ENCODING DOCUMENT");
    println!("{}", sep);

    let doc_embeddings = encoder.encode(document)
        .context("Encoding document")?;

    println!("Tokens: {}", doc_embeddings.num_tokens);
    println!("Embedding dimensions: {}", doc_embeddings.embedding_dim);
    println!();

    // Show document embedding samples
    println!("First 3 token embeddings (showing first 10 dimensions):");
    println!("{}", dash);

    for token_idx in 0..3.min(doc_embeddings.num_tokens) {
        print!("Token {}: [", token_idx);
        for dim in 0..10.min(doc_embeddings.embedding_dim) {
            print!("{:7.4}", doc_embeddings.embeddings[[token_idx, dim]]);
            if dim < 9 && dim < doc_embeddings.embedding_dim - 1 {
                print!(", ");
            }
        }
        println!(", ...]");
    }
    println!();

    // Compute MaxSim similarity
    println!("{}", sep);
    println!("COMPUTING MAXSIM SIMILARITY");
    println!("{}", sep);

    let similarity = max_sim(&query_embeddings, &doc_embeddings)
        .context("Computing MaxSim similarity")?;

    println!("MaxSim Score: {:.4}", similarity);
    println!();
    println!("This score represents the sum of maximum similarities between each");
    println!("query token and all document tokens. Higher scores indicate greater");
    println!("semantic similarity for retrieval tasks.");
    println!();

    // Summary
    println!("{}", sep);
    println!("SUMMARY");
    println!("{}", sep);
    println!("Query: {} tokens × {} dimensions = {} embeddings",
        query_embeddings.num_tokens,
        query_embeddings.embedding_dim,
        query_embeddings.num_tokens * query_embeddings.embedding_dim
    );
    println!("Document: {} tokens × {} dimensions = {} embeddings",
        doc_embeddings.num_tokens,
        doc_embeddings.embedding_dim,
        doc_embeddings.num_tokens * doc_embeddings.embedding_dim
    );
    println!();
    println!("ColBERT successfully generated token-level embeddings!");
    println!("These embeddings can be used for:");
    println!("  - Late-interaction retrieval (MaxSim scoring)");
    println!("  - Building search indices");
    println!("  - Document ranking");
    println!("  - Semantic search systems");
    println!();

    Ok(())
}
