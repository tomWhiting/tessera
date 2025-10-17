//! ColPali basic demonstration.
//!
//! Shows basic usage of ColPali vision-language embeddings for
//! document retrieval. Demonstrates both direct instantiation and
//! factory pattern usage.
//!
//! Run with:
//! ```bash
//! cargo run --release --example colpali_demo
//! ```

use tessera::{TesseraVision, Tessera};
use std::path::Path;

fn main() -> tessera::Result<()> {
    println!("=== ColPali Vision-Language Embeddings Demo ===\n");

    // Method 1: Direct instantiation
    println!("--- Method 1: Direct Instantiation ---\n");
    println!("Loading ColPali v1.3 (PaliGemma-3B base)...");
    let vision = TesseraVision::new("colpali-v1.3-hf")?;
    println!("Model loaded successfully!\n");

    println!("Model specifications:");
    println!("  Model: {}", vision.model());
    println!("  Embedding dim: {}", vision.embedding_dim());
    println!("  Patches per image: {}", vision.num_patches());
    println!("  Architecture: PaliGemma-3B (vision-language)");
    println!("  Image size: 448×448 pixels");
    println!("  Patch size: 14×14 pixels\n");

    // Method 2: Factory pattern
    println!("--- Method 2: Factory Pattern ---\n");
    println!("Creating embedder via Tessera factory...");
    let embedder = Tessera::new("colpali-v1.3-hf")?;

    match embedder {
        Tessera::Vision(v) => {
            println!("Factory auto-detected Vision model");
            println!("  Model: {}\n", v.model());
        }
        _ => println!("Unexpected variant\n"),
    }

    // API demonstration
    println!("--- API Demonstration ---\n");

    // Encode a sample query
    println!("Encoding text query:");
    let query_text = "machine learning algorithms";
    let query_emb = vision.encode_query(query_text)?;
    println!("  Query: \"{}\"", query_text);
    println!("  Tokens: {}", query_emb.num_tokens);
    println!("  Embedding shape: {} tokens × {} dimensions\n",
        query_emb.num_tokens, query_emb.embedding_dim);

    // Check for test image
    let test_image = "test_data/sample.png";

    if Path::new(test_image).exists() {
        println!("Encoding document image:");
        let doc_emb = vision.encode_document(test_image)?;
        println!("  Image: {}", test_image);
        println!("  Patches: {}", doc_emb.num_patches);
        println!("  Embedding shape: {} patches × {} dimensions\n",
            doc_emb.num_patches, doc_emb.embedding_dim);

        println!("Computing similarity:");
        let score = vision.search(&query_emb, &doc_emb)?;
        println!("  MaxSim score: {:.4}", score);
        println!("  (Higher score = more relevant)\n");

        // Alternative convenience method
        println!("Alternative search method:");
        let score2 = vision.search_document(query_text, test_image)?;
        println!("  vision.search_document(query, image) = {:.4}", score2);
        println!("  (Encodes both query and document internally)\n");

    } else {
        println!("No test image found at {}", test_image);
        println!("\nTo test document encoding:");
        println!("  1. Create test_data/ directory");
        println!("  2. Add a sample document image (PNG, JPEG, etc.)");
        println!("  3. Name it sample.png");
        println!("\nSupported formats:");
        println!("  - PDF pages (convert to images first)");
        println!("  - Scanned documents");
        println!("  - Screenshots");
        println!("  - Digital documents exported as images\n");
    }

    println!("--- Use Cases ---\n");
    println!("ColPali is ideal for:");
    println!("  - Scanned document retrieval");
    println!("  - Invoice and receipt search");
    println!("  - Academic paper retrieval");
    println!("  - Legal document search");
    println!("  - Presentation and slide search");
    println!("  - Technical diagram retrieval");
    println!("  - Mixed text/visual content\n");

    println!("Advantages over text-only search:");
    println!("  - No OCR preprocessing required");
    println!("  - Preserves visual layout information");
    println!("  - Handles tables and charts natively");
    println!("  - Works with handwritten text");
    println!("  - Robust to scan quality issues");
    println!("  - Multi-lingual without language-specific setup\n");

    println!("=== Demo Complete ===");
    Ok(())
}
