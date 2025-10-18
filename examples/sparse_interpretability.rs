//! Sparse embedding interpretability example.
//!
//! Demonstrates the interpretability of SPLADE sparse embeddings by showing
//! which vocabulary dimensions are activated for different queries. This is
//! a unique advantage of sparse embeddings over dense black-box representations.
//!
//! Run with:
//! ```bash
//! cargo run --release --example sparse_interpretability
//! ```

use tessera::TesseraSparse;

fn main() -> tessera::Result<()> {
    println!("=== Sparse Embedding Interpretability Demo ===\n");

    println!("Loading model: splade-pp-en-v1...");
    let embedder = TesseraSparse::new("splade-pp-en-v1")?;
    println!("Model loaded!\n");

    // Different queries showcasing term activation
    let queries = vec![
        "machine learning algorithm",
        "deep neural network architecture",
        "python programming language",
        "natural language processing",
        "computer vision image recognition",
    ];

    println!("Analyzing query term activations...\n");
    println!("Each query activates specific vocabulary dimensions with varying weights.");
    println!("Higher weights indicate stronger relevance to the query.\n");

    for (i, query) in queries.iter().enumerate() {
        println!("{}. Query: \"{}\"", i + 1, query);

        let embedding = embedder.encode(query)?;

        println!("   Vocabulary size: {}", embedding.vocab_size);
        println!("   Non-zero dimensions: {}", embedding.nnz());
        println!("   Sparsity: {:.2}%", embedding.sparsity() * 100.0);

        // Sort weights by magnitude to show top activated terms
        let mut weights_sorted: Vec<(usize, f32)> = embedding.weights.clone();
        weights_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("   Top 10 activated dimensions:");
        for (rank, (idx, weight)) in weights_sorted.iter().take(10).enumerate() {
            println!(
                "      {}. Vocab index {} -> weight: {:.4}",
                rank + 1,
                idx,
                weight
            );
        }

        println!();
    }

    // Compare activation patterns between similar and dissimilar queries
    println!("--- Comparing Activation Patterns ---\n");

    let query1 = "artificial intelligence";
    let query2 = "machine learning algorithms";
    let query3 = "weather forecast rain";

    let emb1 = embedder.encode(query1)?;
    let emb2 = embedder.encode(query2)?;
    let emb3 = embedder.encode(query3)?;

    // Count overlapping dimensions
    let indices1: std::collections::HashSet<usize> =
        emb1.weights.iter().map(|(idx, _)| *idx).collect();
    let indices2: std::collections::HashSet<usize> =
        emb2.weights.iter().map(|(idx, _)| *idx).collect();
    let indices3: std::collections::HashSet<usize> =
        emb3.weights.iter().map(|(idx, _)| *idx).collect();

    let overlap_1_2 = indices1.intersection(&indices2).count();
    let overlap_1_3 = indices1.intersection(&indices3).count();

    println!("Query 1: \"{}\"", query1);
    println!("  Activated dimensions: {}", emb1.nnz());
    println!();

    println!("Query 2: \"{}\"", query2);
    println!("  Activated dimensions: {}", emb2.nnz());
    println!(
        "  Overlap with Query 1: {} dimensions ({:.1}%)",
        overlap_1_2,
        100.0 * overlap_1_2 as f32 / emb1.nnz().min(emb2.nnz()) as f32
    );
    println!();

    println!("Query 3: \"{}\"", query3);
    println!("  Activated dimensions: {}", emb3.nnz());
    println!(
        "  Overlap with Query 1: {} dimensions ({:.1}%)",
        overlap_1_3,
        100.0 * overlap_1_3 as f32 / emb1.nnz().min(emb3.nnz()) as f32
    );
    println!();

    println!("=== Key Insights ===\n");
    println!("1. Similar queries (AI/ML) share more activated dimensions");
    println!("2. Dissimilar queries (AI vs weather) have minimal overlap");
    println!("3. Each dimension maps to vocabulary tokens - fully interpretable!");
    println!("4. Weights indicate importance - higher = more relevant to query");
    println!("5. Sparse vectors integrate naturally with inverted indexes");

    println!("\n=== Interpretability Demo Complete ===");
    Ok(())
}
