//! ColPali document search example.
//!
//! Demonstrates vision-language retrieval with ColPali for OCR-free
//! document search using page images. Shows practical application of
//! multi-vector vision embeddings for document retrieval.
//!
//! Run with:
//! ```bash
//! cargo run --release --example colpali_document_search
//! ```

use std::path::Path;
use tessera::TesseraVision;

fn main() -> tessera::Result<()> {
    println!("=== ColPali Document Search Example ===\n");

    // Create vision-language embedder
    println!("Loading ColPali model (this may take a while on first run)...");
    let embedder = TesseraVision::new("colpali-v1.3-hf")?;
    println!("Model loaded: {}", embedder.model());
    println!("Embedding dimension: {}", embedder.embedding_dim());
    println!("Patches per image: {}\n", embedder.num_patches());

    // Document collection (real paths - will check existence)
    let test_docs = vec![
        ("Financial Report", "test_data/invoice_001.png"),
        ("Legal Contract", "test_data/contract_001.png"),
        ("Research Paper", "test_data/report_001.png"),
    ];

    // Filter to only existing documents
    let documents: Vec<(&str, &str)> = test_docs
        .iter()
        .filter(|(_, path)| Path::new(path).exists())
        .map(|(title, path)| (*title, *path))
        .collect();

    if documents.is_empty() {
        println!("No test documents found in test_data/ directory.");
        println!("\nTo run this example with real documents:");
        println!("  1. Create a test_data/ directory");
        println!("  2. Add document images:");
        println!("     - invoice_001.png (invoice or receipt)");
        println!("     - contract_001.png (contract or legal document)");
        println!("     - report_001.png (report or research paper)");
        println!("\nColPali works with:");
        println!("  - Scanned PDFs converted to images");
        println!("  - Screenshots of documents");
        println!("  - Photos of printed documents");
        println!("  - Any document with text, tables, or charts\n");

        // Demonstrate API even without test data
        println!("=== API Demo (without real documents) ===\n");

        println!("Example query encoding:");
        let query_text = "What is the total amount due?";
        let query_emb = embedder.encode_query(query_text)?;
        println!("  Query: \"{}\"", query_text);
        println!("  Tokens: {}", query_emb.num_tokens);
        println!("  Embedding dim: {}\n", query_emb.embedding_dim);

        println!("Document encoding would use:");
        println!("  let doc_emb = embedder.encode_document(\"path/to/image.png\")?;");
        println!("  println!(\"Patches: {{}}\", doc_emb.num_patches);");
        println!("\nSearch with MaxSim scoring:");
        println!("  let score = embedder.search(&query_emb, &doc_emb)?;");

        return Ok(());
    }

    println!("Found {} documents:\n", documents.len());
    for (idx, (title, path)) in documents.iter().enumerate() {
        println!("  [{}] {} ({})", idx + 1, title, path);
    }
    println!();

    // Encode all documents
    println!("Encoding document images...\n");
    let mut doc_embeddings = Vec::new();
    for (title, doc_path) in &documents {
        let emb = embedder.encode_document(doc_path)?;
        println!(
            "  Encoded '{}': {} patches × {} dimensions",
            title, emb.num_patches, emb.embedding_dim
        );
        doc_embeddings.push((title, emb));
    }
    println!();

    // Search queries
    let queries = vec![
        "What is the total amount?",
        "What is the contract term?",
        "What are the key findings?",
    ];

    // Perform search for each query
    for query in &queries {
        println!("Query: \"{}\"\n", query);

        // Encode query
        let query_emb = embedder.encode_query(query)?;
        println!("Query encoded: {} tokens\n", query_emb.num_tokens);

        // Search across all documents
        let mut results: Vec<(&str, f32)> = Vec::new();
        for (title, doc_emb) in &doc_embeddings {
            let score = embedder.search(&query_emb, doc_emb)?;
            results.push((title, score));
        }

        // Sort by score (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Display top results
        println!("Search Results:");
        for (rank, (title, score)) in results.iter().enumerate() {
            println!("  {}. {} - Score: {:.4}", rank + 1, title, score);
        }
        println!();
    }

    println!("=== Key Features ===\n");
    println!("ColPali performs OCR-free document search:");
    println!("  - No text extraction needed");
    println!("  - Preserves layout and visual structure");
    println!("  - Handles tables, charts, and mixed content");
    println!("  - Robust to scan quality and image noise");
    println!("  - Multi-lingual without language-specific OCR\n");

    println!("How it works:");
    println!(
        "  1. Image split into {} patches (14×14 pixels each)",
        embedder.num_patches()
    );
    println!(
        "  2. Each patch encoded to {}-dim embedding",
        embedder.embedding_dim()
    );
    println!("  3. Query text encoded to token embeddings");
    println!("  4. Late interaction (MaxSim) scoring for relevance");

    println!("\n=== Search Complete ===");
    Ok(())
}
