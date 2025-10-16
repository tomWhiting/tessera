//! Matryoshka embeddings with dense models example.
//!
//! Demonstrates how to use Matryoshka embeddings (multi-scale embeddings)
//! with different dimensions for speed/accuracy trade-offs. Matryoshka
//! embeddings allow truncating to smaller dimensions with minimal quality loss.
//!
//! Run with:
//! ```bash
//! cargo run --release --example dense_matryoshka
//! ```

use tessera::TesseraDense;

fn main() -> tessera::Result<()> {
    println!("=== Matryoshka Dense Embeddings Example ===\n");

    // Test documents for semantic evaluation
    let text_pair_similar = (
        "Machine learning is a subset of artificial intelligence",
        "Artificial intelligence includes machine learning"
    );

    let text_pair_different = (
        "Machine learning is a subset of artificial intelligence",
        "The weather forecast predicts rain tomorrow"
    );

    // Test different dimension configurations
    let dimension_tests = vec![
        ("bge-base-en-v1.5", 768, "Full dimension (baseline)"),
    ];

    println!("Testing semantic similarity preservation across dimensions\n");

    for (model_id, max_dim, description) in dimension_tests {
        println!("Model: {} - {}", model_id, description);
        println!("Max dimension: {}\n", max_dim);

        let embedder = TesseraDense::new(model_id)?;
        println!("Loaded model successfully\n");

        // Test different truncation levels
        let dimensions = vec![64, 128, 256, 512, 768];

        println!("Dimension | Similar Sim | Different Sim | Ratio | Info Retained");
        println!("{}", "-".repeat(70));

        for dim in dimensions {
            if dim > max_dim {
                continue;
            }

            // Encode texts (full dimension)
            let emb1_full = embedder.encode(text_pair_similar.0)?;
            let emb2_similar = embedder.encode(text_pair_similar.1)?;
            let emb3_different = embedder.encode(text_pair_different.1)?;

            // Truncate to target dimension for similarity computation
            let truncated_len = dim.min(emb1_full.dim());

            // Compute similarities on truncated embeddings
            let mut sim_similar = 0.0f32;
            let mut sim_different = 0.0f32;

            for i in 0..truncated_len {
                sim_similar += emb1_full.embedding[i] * emb2_similar.embedding[i];
                sim_different += emb1_full.embedding[i] * emb3_different.embedding[i];
            }

            // Normalize by dimension (approximately)
            let full_sim_similar: f32 = emb1_full.embedding.iter()
                .zip(emb2_similar.embedding.iter())
                .map(|(a, b)| a * b)
                .sum();

            let full_sim_different: f32 = emb1_full.embedding.iter()
                .zip(emb3_different.embedding.iter())
                .map(|(a, b)| a * b)
                .sum();

            let ratio_truncated = if sim_different.abs() > 1e-6 {
                sim_similar / sim_different
            } else {
                0.0
            };

            let ratio_full = if full_sim_different.abs() > 1e-6 {
                full_sim_similar / full_sim_different
            } else {
                0.0
            };

            // Information retained as a percentage
            let info_retained = if ratio_full.abs() > 1e-6 {
                (ratio_truncated / ratio_full * 100.0).min(100.0)
            } else {
                100.0
            };

            println!(
                "{:^9} | {:^11.4} | {:^13.4} | {:^5.2} | {:>14.1}%",
                dim, sim_similar, sim_different, ratio_truncated, info_retained
            );
        }

        println!();
    }

    // Practical use case: dimensionality trade-offs
    println!("--- Practical Use Case: Memory vs Accuracy Trade-off ---\n");

    let documents = vec![
        "Natural language processing enables understanding of human language",
        "Deep learning neural networks process information through layers",
        "Python is widely used in data science and machine learning",
    ];

    let query = "How does machine learning work?";

    println!("Searching {} documents with query: \"{}\"\n", documents.len(), query);

    let embedder = TesseraDense::new("bge-base-en-v1.5")?;
    let query_embedding = embedder.encode(query)?;
    let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_ref()).collect();
    let doc_embeddings = embedder.encode_batch(&doc_refs)?;

    // Compare results at different truncation levels
    let dimensions_to_test = vec![128, 256, 384, 768];

    for target_dim in dimensions_to_test {
        print!("Dim {:>3}: ", target_dim);

        let truncated_len = target_dim.min(query_embedding.dim());
        let mut results: Vec<(usize, f32)> = Vec::new();

        for (idx, doc_emb) in doc_embeddings.iter().enumerate() {
            let mut similarity = 0.0f32;
            for i in 0..truncated_len {
                similarity += query_embedding.embedding[i] * doc_emb.embedding[i];
            }
            results.push((idx, similarity));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, (idx, score)) in results.iter().take(3).enumerate() {
            let prefix = if rank > 0 { " | " } else { "" };
            print!("{}{}: {:.3}", prefix, idx + 1, score);
        }
        println!();
    }

    // Memory savings analysis
    println!("\n--- Memory Savings Analysis ---");
    println!("For a collection of {} documents with {}MB base embeddings:\n", 
        100_000, 
        (100_000 * 768 * 4) / (1024 * 1024));

    let scenarios = vec![
        (768, "Full dimension (baseline)"),
        (384, "Half dimension"),
        (256, "1/3 dimension"),
        (128, "1/6 dimension"),
    ];

    for (dim, description) in scenarios {
        let memory_mb = (100_000 * dim * 4) as f64 / (1024.0 * 1024.0);
        let saved_percent = ((768 - dim) as f64 / 768.0) * 100.0;
        println!("  {:>3} dims: {:>6.1} MB - {} (saves {:.0}%)", dim, memory_mb, description, saved_percent);
    }

    println!("\nNote: Matryoshka embeddings allow graceful degradation:");
    println!("  - Similarity ranking is preserved at reduced dimensions");
    println!("  - 128 dimensions often retains 95%+ ranking accuracy");
    println!("  - Ideal for memory-constrained deployments");
    println!("  - Can dynamically adjust dimension based on latency requirements");

    println!("\n=== Matryoshka Example Complete ===");
    Ok(())
}
