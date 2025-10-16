//! Sparse embedding semantic search example.
//!
//! Demonstrates how to use TesseraSparse for semantic search with SPLADE
//! embeddings. Shows practical retrieval with interpretability benefits
//! of sparse representations.
//!
//! Run with:
//! ```bash
//! cargo run --release --example sparse_semantic_search
//! ```

use tessera::TesseraSparse;

fn main() -> tessera::Result<()> {
    println!("=== Sparse Semantic Search Example (SPLADE) ===\n");

    // Initialize sparse embedder
    println!("Loading model: splade-pp-en-v1...");
    let embedder = TesseraSparse::new("splade-pp-en-v1")?;
    println!("Model loaded successfully!");
    println!("Vocabulary size: {}\n", embedder.vocab_size());

    // Collection of documents covering different domains
    let documents = vec![
        ("Machine learning fundamentals", "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed."),
        ("Deep learning networks", "Deep learning uses neural networks with many layers to automatically learn hierarchical representations of data."),
        ("Python for data science", "Python is a popular programming language for data science, offering libraries like NumPy, Pandas, and Scikit-learn."),
        ("Web development basics", "Web development involves creating websites and web applications using HTML, CSS, JavaScript, and backend technologies."),
        ("Natural language processing", "Natural language processing enables computers to understand, interpret, and generate human language in a meaningful way."),
        ("Computer vision applications", "Computer vision allows machines to interpret and analyze visual information from images and videos."),
        ("Cloud computing platforms", "Cloud computing provides on-demand access to computing resources over the internet, including servers, storage, and databases."),
        ("Database design", "Database design involves structuring data efficiently using relational models, indexes, and normalization techniques."),
    ];

    // Semantic search queries
    let queries = vec![
        "What is AI and machine learning?",
        "How do I learn Python for data science?",
        "Tell me about image recognition",
    ];

    println!("Document collection: {} documents\n", documents.len());
    for (idx, (title, _)) in documents.iter().enumerate() {
        println!("  [{}] {}", idx + 1, title);
    }
    println!();

    // Perform semantic search for each query
    for query in &queries {
        println!("Query: \"{}\"\n", query);

        // Encode query and show sparsity
        let query_embedding = embedder.encode(query)?;
        println!("Query embedding:");
        println!("  Total dimensions: {}", query_embedding.vocab_size);
        println!("  Non-zero dimensions: {}", query_embedding.nnz());
        println!("  Sparsity: {:.2}%\n", query_embedding.sparsity() * 100.0);

        // Compute similarity with all documents
        let mut results: Vec<(usize, &str, f32)> = Vec::new();

        for (idx, (title, doc_text)) in documents.iter().enumerate() {
            let similarity = embedder.similarity(query, doc_text)?;
            results.push((idx, title, similarity));
        }

        // Sort by similarity (descending) and show top results
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        println!("Top 3 Results:");
        for (rank, (idx, title, score)) in results.iter().take(3).enumerate() {
            println!("  {}. [Doc {}] Score: {:.4} - {}", rank + 1, idx + 1, score, title);
        }
        println!();
    }

    // Show sparsity statistics across document collection
    println!("--- Sparsity Statistics ---\n");

    let doc_texts: Vec<&str> = documents.iter().map(|(_, text)| *text).collect();

    // Encode all documents
    let doc_embeddings = embedder.encode_batch(&doc_texts)?;

    let total_nnz: usize = doc_embeddings.iter().map(|e| e.nnz()).sum();
    let avg_nnz = total_nnz as f32 / doc_embeddings.len() as f32;
    let avg_sparsity: f32 = doc_embeddings.iter().map(|e| e.sparsity()).sum::<f32>() / doc_embeddings.len() as f32;

    println!("Processed {} documents:", documents.len());
    println!("  Average non-zero dimensions: {:.1}", avg_nnz);
    println!("  Average sparsity: {:.2}%", avg_sparsity * 100.0);
    println!("  Vocabulary size: {}", embedder.vocab_size());
    println!();

    println!("Sparse embeddings use ~{}x less storage than dense vectors!", 
        (embedder.vocab_size() as f32 / avg_nnz).round() as usize);

    println!("\n=== Search Complete ===");
    Ok(())
}
