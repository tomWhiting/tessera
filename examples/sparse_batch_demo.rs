//! Sparse embedding batch processing example.
//!
//! Demonstrates efficient batch encoding with SPLADE sparse embeddings,
//! including similarity matrix computation and performance considerations.
//!
//! Run with:
//! ```bash
//! cargo run --release --example sparse_batch_demo
//! ```

use tessera::TesseraSparse;

fn main() -> tessera::Result<()> {
    println!("=== Sparse Batch Processing Demo ===\n");

    println!("Loading model: splade-pp-en-v1...");
    let embedder = TesseraSparse::new("splade-pp-en-v1")?;
    println!("Model loaded successfully!\n");

    // Collection of AI/ML related documents
    let documents = vec![
        "Artificial intelligence transforms modern technology and society",
        "Machine learning algorithms learn patterns from large datasets",
        "Deep neural networks consist of multiple interconnected layers",
        "Natural language processing enables computers to understand text",
        "Computer vision systems can recognize objects in images",
        "Reinforcement learning agents improve through trial and error",
        "Transfer learning leverages pre-trained models for new tasks",
        "Unsupervised learning discovers hidden patterns in data",
    ];

    println!("Processing {} documents in batch...\n", documents.len());

    // Convert to references for batch encoding
    let docs_ref: Vec<&str> = documents.iter().map(|s| s.as_ref()).collect();

    // Batch encode all documents
    let embeddings = embedder.encode_batch(&docs_ref)?;

    // Display individual embedding statistics
    println!("Individual Embedding Statistics:");
    for (i, emb) in embeddings.iter().enumerate() {
        println!("  Doc {}: {} non-zero dims ({:.2}% sparse)",
            i + 1,
            emb.nnz(),
            emb.sparsity() * 100.0
        );
    }

    // Compute aggregate statistics
    let total_nnz: usize = embeddings.iter().map(|e| e.nnz()).sum();
    let avg_nnz = total_nnz as f32 / embeddings.len() as f32;
    let min_nnz = embeddings.iter().map(|e| e.nnz()).min().unwrap();
    let max_nnz = embeddings.iter().map(|e| e.nnz()).max().unwrap();

    println!("\nAggregate Statistics:");
    println!("  Average non-zero: {:.1}", avg_nnz);
    println!("  Min non-zero:     {}", min_nnz);
    println!("  Max non-zero:     {}", max_nnz);
    println!("  Vocab size:       {}", embeddings[0].vocab_size);

    // Compute similarity matrix
    println!("\n=== Document Similarity Matrix ===\n");

    println!("Computing pairwise similarities for {} documents...", documents.len());

    // Create similarity matrix
    let mut sim_matrix: Vec<Vec<f32>> = vec![vec![0.0; embeddings.len()]; embeddings.len()];

    for i in 0..embeddings.len() {
        for j in i..embeddings.len() {
            // Compute sparse dot product
            let mut score = 0.0;
            for (idx_i, weight_i) in &embeddings[i].weights {
                if let Some(&(_, weight_j)) = embeddings[j].weights.iter().find(|(idx_j, _)| idx_j == idx_i) {
                    score += weight_i * weight_j;
                }
            }
            sim_matrix[i][j] = score;
            sim_matrix[j][i] = score; // Symmetric
        }
    }

    // Display similarity matrix
    println!("\nSimilarity Matrix:");
    print!("       ");
    for i in 1..=embeddings.len() {
        print!(" Doc{}", i);
    }
    println!();

    for (i, row) in sim_matrix.iter().enumerate() {
        print!("Doc{}  ", i + 1);
        for score in row {
            print!(" {:.2}", score);
        }
        println!();
    }

    // Find most similar document pairs
    println!("\n=== Top 3 Most Similar Document Pairs ===\n");

    let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            pairs.push((i, j, sim_matrix[i][j]));
        }
    }

    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (i, j, score)) in pairs.iter().take(3).enumerate() {
        println!("{}. Score: {:.4}", rank + 1, score);
        println!("   Doc {}: {}", i + 1, documents[*i]);
        println!("   Doc {}: {}", j + 1, documents[*j]);
        println!();
    }

    // Find least similar document pairs
    println!("=== Top 3 Least Similar Document Pairs ===\n");

    pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (i, j, score)) in pairs.iter().take(3).enumerate() {
        println!("{}. Score: {:.4}", rank + 1, score);
        println!("   Doc {}: {}", i + 1, documents[*i]);
        println!("   Doc {}: {}", j + 1, documents[*j]);
        println!();
    }

    // Memory efficiency analysis
    println!("=== Memory Efficiency Analysis ===\n");

    let sparse_storage: usize = embeddings.iter().map(|e| e.nnz() * 12).sum(); // (usize, f32) = 8 + 4 = 12 bytes
    let dense_storage = embeddings.len() * embeddings[0].vocab_size * 4; // All floats

    println!("Storage comparison:");
    println!("  Sparse storage:    {} bytes ({:.2} KB)", sparse_storage, sparse_storage as f32 / 1024.0);
    println!("  Dense equivalent:  {} bytes ({:.2} KB)", dense_storage, dense_storage as f32 / 1024.0);
    println!("  Compression ratio: {:.1}x", dense_storage as f32 / sparse_storage as f32);

    println!("\n=== Key Insights ===\n");
    println!("1. Batch processing is efficient - encodes all documents together");
    println!("2. Sparse dot product is fast - only compute overlapping dimensions");
    println!("3. Similar documents share more activated vocabulary terms");
    println!("4. Memory usage scales with actual content, not vocabulary size");
    println!("5. Ideal for large-scale retrieval with inverted indexes");

    println!("\n=== Batch Processing Complete ===");
    Ok(())
}
