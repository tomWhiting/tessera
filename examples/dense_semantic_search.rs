//! Dense embedding semantic search example.
//!
//! Demonstrates how to use TesseraDense for semantic similarity search
//! across a collection of documents. Shows practical application of dense
//! embeddings for information retrieval and relevance ranking.
//!
//! Run with:
//! ```bash
//! cargo run --release --example dense_semantic_search
//! ```

use tessera::TesseraDense;

fn main() -> tessera::Result<()> {
    println!("=== Dense Semantic Search Example ===\n");

    // Initialize embedder with BGE model
    println!("Loading model: bge-base-en-v1.5...");
    let embedder = TesseraDense::new("bge-base-en-v1.5")?;
    println!("Model loaded successfully!");
    println!("Embedding dimension: {}\n", embedder.dimension());

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

        // Encode query
        let query_embedding = embedder.encode(query)?;
        println!("Query embedding computed ({} dimensions)\n", query_embedding.dim());

        // Compute similarity with all documents
        let mut results: Vec<(usize, &str, f32)> = Vec::new();

        for (idx, (title, doc_text)) in documents.iter().enumerate() {
            // For better search accuracy, we encode the document text rather than just title
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

    // Demonstrate batch similarity computation
    println!("--- Batch Processing Optimization ---\n");

    let test_query = "What is artificial intelligence?";
    let doc_texts: Vec<&str> = documents.iter().map(|(_, text)| *text).collect();

    println!("Computing similarities for {} documents...", documents.len());

    // Encode all documents at once using batch processing
    let doc_embeddings = embedder.encode_batch(&doc_texts)?;

    // Encode query once
    let query_emb = embedder.encode(test_query)?;

    // Compute all similarities
    let mut batch_results: Vec<(usize, f32)> = Vec::new();
    for (idx, doc_emb) in doc_embeddings.iter().enumerate() {
        // Cosine similarity for normalized embeddings (dot product)
        let similarity: f32 = query_emb.embedding
            .iter()
            .zip(doc_emb.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        batch_results.push((idx, similarity));
    }

    // Sort and display results
    batch_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop matches for query: \"{}\"", test_query);
    for (rank, (idx, score)) in batch_results.iter().take(5).enumerate() {
        println!("  {}. Score: {:.4} - {}", rank + 1, score, documents[*idx].0);
    }

    println!("\n=== Search Complete ===");
    Ok(())
}
