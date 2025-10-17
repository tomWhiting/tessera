//! Unified Tessera API demonstration.
//!
//! Shows how to use the Tessera enum factory for automatic detection and
//! creation of the appropriate embedder variant (Dense or MultiVector) based
//! on model type. Demonstrates the unified API pattern for seamless integration.
//!
//! Run with:
//! ```bash
//! cargo run --release --example unified_api_demo
//! ```

use tessera::Tessera;

fn main() -> tessera::Result<()> {
    println!("=== Unified Tessera API Demo ===\n");

    // Example 1: Auto-detecting and using a dense model
    println!("1. Auto-detecting Dense Model");
    println!("   Creating embedder for 'bge-base-en-v1.5'...\n");

    let dense_embedder = Tessera::new("bge-base-en-v1.5")?;

    match dense_embedder {
        Tessera::Dense(d) => {
            println!("   ✓ Detected as Dense embedder");
            println!("   Model: {}", d.model());
            println!("   Embedding dimension: {}", d.dimension());

            // Encode sample text
            let text = "Machine learning is transforming the world";
            let embedding = d.encode(text)?;
            println!("   Encoded text to {} dimensions", embedding.dim());
            let sample_len = 5.min(embedding.dim());
            let sample: Vec<f32> = embedding.embedding.iter().take(sample_len).copied().collect();
            println!("   Sample values (first 5): {:.4?}\n", sample);
        }
        Tessera::MultiVector(_) => {
            println!("   ✗ Error: Expected Dense but got MultiVector\n");
        }
        Tessera::Sparse(_) => {
            println!("   ✗ Error: Expected Dense but got Sparse\n");
        }
        Tessera::Vision(_) => {
            println!("   ✗ Error: Expected Dense but got Vision\n");
        }
    }

    // Example 2: Auto-detecting and using a multi-vector model
    println!("2. Auto-detecting MultiVector Model");
    println!("   Creating embedder for 'colbert-v2'...\n");

    let mv_embedder = Tessera::new("colbert-v2")?;

    match mv_embedder {
        Tessera::MultiVector(m) => {
            println!("   ✓ Detected as MultiVector embedder");
            println!("   Model: {}", m.model());
            println!("   Embedding dimension: {} per token", m.dimension());

            // Encode sample text
            let text = "What is machine learning?";
            let embeddings = m.encode(text)?;
            println!("   Encoded text to {} token vectors", embeddings.num_tokens);
            println!("   Each vector has {} dimensions", embeddings.embedding_dim);
            println!("   Shape: ({} x {})\n", embeddings.num_tokens, embeddings.embedding_dim);
        }
        Tessera::Dense(_) => {
            println!("   ✗ Error: Expected MultiVector but got Dense\n");
        }
        Tessera::Sparse(_) => {
            println!("   ✗ Error: Expected MultiVector but got Sparse\n");
        }
        Tessera::Vision(_) => {
            println!("   ✗ Error: Expected MultiVector but got Vision\n");
        }
    }

    // Example 3: Pattern matching for model-agnostic operations
    println!("3. Pattern Matching for Type-Specific Operations\n");

    let models_to_test = vec![
        ("bge-base-en-v1.5", "Dense model"),
        ("colbert-v2", "MultiVector model"),
    ];

    for (model_id, description) in models_to_test {
        println!("   Model: {} ({})", model_id, description);

        let embedder = Tessera::new(model_id)?;

        match embedder {
            Tessera::Dense(d) => {
                println!("     Type: Dense single-vector embedding");
                println!("     - Suitable for: Semantic search, classification, clustering");
                println!("     - Output: Single {} dimensional vector per text", d.dimension());
                println!("     - Similarity: Cosine distance");
                println!("     - Use case: Fast, memory-efficient semantic search");

                // Demonstrate dense usage
                let query = "What is AI?";
                let doc = "Artificial intelligence enables machines to think";
                let similarity = d.similarity(query, doc)?;
                println!("     - Example similarity score: {:.4}\n", similarity);
            }
            Tessera::MultiVector(m) => {
                println!("     Type: MultiVector token-level embedding");
                println!("     - Suitable for: Dense retrieval, ranking, fine-grained matching");
                println!("     - Output: {} dimensional vector per token", m.dimension());
                println!("     - Similarity: MaxSim (late interaction)");
                println!("     - Use case: Powerful retrieval with low latency");

                // Demonstrate multi-vector usage
                let query = "What is AI?";
                let doc = "Artificial intelligence enables machines to think";
                let similarity = m.similarity(query, doc)?;
                println!("     - Example similarity score: {:.4}\n", similarity);
            }
            Tessera::Sparse(s) => {
                println!("     Type: Sparse vocabulary-level embedding");
                println!("     - Suitable for: Interpretable search, inverted indexes");
                println!("     - Output: Sparse {} dimensional vector", s.vocab_size());
                println!("     - Similarity: Dot product");
                println!("     - Use case: Interpretable retrieval with term weighting");

                // Demonstrate sparse usage
                let query = "What is AI?";
                let doc = "Artificial intelligence enables machines to think";
                let similarity = s.similarity(query, doc)?;
                println!("     - Example similarity score: {:.4}\n", similarity);
            }
            Tessera::Vision(v) => {
                println!("     Type: Vision-language multi-vector embedding");
                println!("     - Suitable for: Document retrieval, image search");
                println!("     - Output: {} dimensional vector per patch", v.embedding_dim());
                println!("     - Similarity: MaxSim (late interaction)");
                println!("     - Use case: OCR-free document search");
                println!("     - Note: Requires image files, not text\n");
            }
        }
    }

    // Example 4: Comparison of encoding APIs
    println!("4. API Comparison: Dense vs MultiVector\n");

    println!("   Dense Encoding:");
    let dense = Tessera::new("bge-base-en-v1.5")?;
    if let Tessera::Dense(d) = dense {
        let texts = vec![
            "First document text",
            "Second document text",
            "Third document text",
        ];

        // Single text encoding
        let single = d.encode(texts[0])?;
        println!("     Single encode: {} -> {} dims", texts[0], single.dim());

        // Batch encoding
        let batch = d.encode_batch(&texts)?;
        println!("     Batch encode: {} texts -> {} embeddings", texts.len(), batch.len());
        println!("     Each embedding: {} dims\n", batch[0].dim());
    }

    println!("   MultiVector Encoding:");
    let multi_vec = Tessera::new("colbert-v2")?;
    if let Tessera::MultiVector(m) = multi_vec {
        let texts = vec![
            "First document text",
            "Second document text",
            "Third document text",
        ];

        // Single text encoding
        let single = m.encode(texts[0])?;
        println!("     Single encode: {} -> {} tokens", texts[0], single.num_tokens);

        // Batch encoding
        let batch = m.encode_batch(&texts)?;
        println!("     Batch encode: {} texts -> {} embeddings", texts.len(), batch.len());
        println!("     First embedding: {} tokens × {} dims\n", 
            batch[0].num_tokens, batch[0].embedding_dim);
    }

    // Example 5: Error handling for unsupported models
    println!("5. Error Handling for Unsupported Models\n");

    let invalid_model = "splade-v3";
    println!("   Attempting to load '{}' (unsupported type)...", invalid_model);

    match Tessera::new(invalid_model) {
        Ok(_) => {
            println!("   ✗ Unexpected success!\n");
        }
        Err(e) => {
            println!("   ✓ Caught expected error:");
            println!("     {}\n", e);
        }
    }

    // Example 6: Practical use case - content-agnostic search
    println!("6. Practical Use Case: Content-Agnostic Search\n");

    println!("   Searching with automatic model selection:");

    let documents = vec![
        "Python is a popular programming language",
        "JavaScript enables interactive web experiences",
        "Rust provides memory safety without garbage collection",
    ];

    let query = "programming languages";

    // Create embedder and let Tessera decide what type to use
    let embedder = Tessera::new("bge-base-en-v1.5")?;

    // Handle all possible variants
    match embedder {
        Tessera::Dense(d) => {
            println!("   Using Dense model for fast search");
            println!("   Query: \"{}\"\n", query);

            let query_emb = d.encode(query)?;
            let doc_embs = d.encode_batch(&documents)?;

            let mut results: Vec<(usize, f32)> = Vec::new();
            for (idx, doc_emb) in doc_embs.iter().enumerate() {
                let sim: f32 = query_emb.embedding.iter()
                    .zip(doc_emb.embedding.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                results.push((idx, sim));
            }

            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            println!("   Results:");
            for (rank, (idx, score)) in results.iter().enumerate() {
                println!("     {}. Score: {:.4} - {}", rank + 1, score, documents[*idx]);
            }
        }
        Tessera::MultiVector(_) => {
            // Would use multi-vector similarity here
            println!("   Using MultiVector model for fine-grained search");
        }
        Tessera::Sparse(_) => {
            // Would use sparse similarity here
            println!("   Using Sparse model for interpretable search");
        }
        Tessera::Vision(_) => {
            // Would use vision-language similarity here
            println!("   Using Vision model for document image search");
        }
    }

    println!("\n=== Unified API Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("  - Tessera::new() automatically detects model type");
    println!("  - Pattern matching handles Dense and MultiVector variants");
    println!("  - Same ergonomic API for encode(), encode_batch(), similarity()");
    println!("  - Choose model based on accuracy vs speed requirements");
    println!("  - Supports automatic device selection (Metal/CUDA/CPU)");

    Ok(())
}
