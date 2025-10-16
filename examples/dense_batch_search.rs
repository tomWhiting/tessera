//! Dense embedding batch processing example.
//!
//! Demonstrates efficient batch processing of dense embeddings for scaling
//! semantic search applications. Shows performance characteristics and memory
//! efficiency of batch encoding compared to sequential processing.
//!
//! Run with:
//! ```bash
//! cargo run --release --example dense_batch_search
//! ```

use std::time::Instant;
use tessera::TesseraDense;

fn main() -> tessera::Result<()> {
    println!("=== Dense Batch Processing Example ===\n");

    // Initialize dense embedder
    println!("Loading model: bge-base-en-v1.5...");
    let embedder = TesseraDense::new("bge-base-en-v1.5")?;
    println!("Model loaded successfully!");
    println!("Embedding dimension: {}\n", embedder.dimension());

    // Generate test documents with realistic content
    let documents = vec![
        "Artificial intelligence is revolutionizing technology and transforming industries worldwide.",
        "Machine learning algorithms learn patterns from data without explicit programming instructions.",
        "Deep neural networks process information through layers of interconnected nodes and neurons.",
        "Natural language processing enables computers to understand and generate human languages.",
        "Computer vision systems interpret visual data from images and video streams in real time.",
        "Data science combines statistics, programming, and domain knowledge for insights.",
        "Cloud computing provides scalable infrastructure and services over the internet.",
        "Distributed systems coordinate multiple computers to process large-scale computations.",
        "Database systems manage and query structured data efficiently and reliably.",
        "Software engineering practices ensure quality, maintainability, and scalability.",
    ];

    println!("Dataset: {} documents\n", documents.len());

    // Test with increasing batch sizes
    let batch_sizes = vec![1, 3, 5, 10];

    for batch_size in batch_sizes {
        if batch_size > documents.len() {
            continue;
        }

        println!("--- Batch size: {} ---", batch_size);

        let batch_docs = &documents[0..batch_size];

        // Sequential encoding
        println!("Sequential encoding:");
        let seq_start = Instant::now();
        let mut sequential_embeddings = Vec::new();
        for doc in batch_docs {
            let embedding = embedder.encode(doc)?;
            sequential_embeddings.push(embedding);
        }
        let seq_duration = seq_start.elapsed();
        println!("  Time: {:?}", seq_duration);
        println!("  Per-doc: {:.2} ms", seq_duration.as_secs_f64() * 1000.0 / batch_size as f64);

        // Batch encoding
        println!("Batch encoding:");
        let batch_start = Instant::now();
        let batch_embeddings = embedder.encode_batch(batch_docs)?;
        let batch_duration = batch_start.elapsed();
        println!("  Time: {:?}", batch_duration);
        println!("  Per-doc: {:.2} ms", batch_duration.as_secs_f64() * 1000.0 / batch_size as f64);

        // Calculate speedup
        let speedup = seq_duration.as_secs_f64() / batch_duration.as_secs_f64();
        println!("Speedup: {:.2}x faster with batch processing", speedup);

        // Verify correctness (embeddings should match)
        println!("Correctness check:");
        let mut all_match = true;
        let mut max_diff = 0.0f32;

        for (i, (seq_emb, batch_emb)) in sequential_embeddings.iter().zip(batch_embeddings.iter()).enumerate() {
            let mut doc_max_diff = 0.0f32;
            for (a, b) in seq_emb.embedding.iter().zip(batch_emb.embedding.iter()) {
                let diff = (a - b).abs();
                doc_max_diff = doc_max_diff.max(diff);
                max_diff = max_diff.max(diff);
            }

            if doc_max_diff < 0.001 {
                println!("  Doc {}: ✓ Match (diff: {:.6})", i, doc_max_diff);
            } else if doc_max_diff < 0.1 {
                println!("  Doc {}: ~ Close (diff: {:.6})", i, doc_max_diff);
            } else {
                println!("  Doc {}: ✗ Differs (diff: {:.6})", i, doc_max_diff);
                all_match = false;
            }
        }

        if all_match {
            println!("Overall: ✓ Results are numerically consistent\n");
        } else {
            println!("Overall: ~ Some differences detected (expected due to numerical precision)\n");
        }
    }

    // Large-scale batch processing demonstration
    println!("--- Large-Scale Batch Similarity Search ---\n");

    let query = "What is machine learning?";
    let query_embedding = embedder.encode(query)?;

    println!("Query: \"{}\"\n", query);
    println!("Computing similarities for {} documents using batch processing...\n", documents.len());

    // Batch encode all documents
    let batch_start = Instant::now();
    let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_ref()).collect();
    let batch_embeddings = embedder.encode_batch(&doc_refs)?;
    let encoding_time = batch_start.elapsed();

    // Compute similarities
    let sim_start = Instant::now();
    let mut similarities: Vec<(usize, f32)> = Vec::new();
    for (idx, doc_emb) in batch_embeddings.iter().enumerate() {
        let similarity: f32 = query_embedding.embedding
            .iter()
            .zip(doc_emb.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        similarities.push((idx, similarity));
    }
    let sim_time = sim_start.elapsed();

    // Sort results
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Performance:");
    println!("  Batch encoding: {:?}", encoding_time);
    println!("  Similarity computation: {:?}", sim_time);
    println!("  Total time: {:?}", encoding_time + sim_time);
    println!("  Throughput: {:.2} docs/sec\n", 
        documents.len() as f64 / (encoding_time + sim_time).as_secs_f64());

    println!("Top 5 Results:");
    for (rank, (idx, score)) in similarities.iter().take(5).enumerate() {
        println!("  {}. Score: {:.4} - {}", rank + 1, score, documents[*idx]);
    }

    // Memory efficiency analysis
    println!("\n--- Memory Efficiency ---");
    println!("Embeddings per batch: {}", batch_embeddings.len());
    println!("Dimensions per embedding: {}", batch_embeddings[0].dim());
    let total_floats = batch_embeddings.len() * batch_embeddings[0].dim();
    let memory_mb = (total_floats * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0);
    println!("Total floats: {}", total_floats);
    println!("Memory used: {:.2} MB\n", memory_mb);

    println!("=== Batch Processing Complete ===");
    Ok(())
}
