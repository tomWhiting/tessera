//! ColPali vs text-only comparison.
//!
//! Demonstrates advantages of vision-language retrieval over text-only
//! methods that require OCR. Compares ColPali (vision-language) with
//! dense embeddings (text-only) approaches for document retrieval.
//!
//! Run with:
//! ```bash
//! cargo run --release --example colpali_vs_text
//! ```

use tessera::{TesseraDense, TesseraVision};

fn main() -> tessera::Result<()> {
    println!("=== Vision-Language vs Text-Only Retrieval ===\n");

    // Load both model types for comparison
    println!("Loading models for comparison...\n");

    println!("1. Loading ColPali (vision-language)...");
    let colpali = TesseraVision::new("colpali-v1.3-hf")?;
    println!("   Loaded: {}", colpali.model());
    println!("   Parameters: 3B");
    println!("   Modality: Vision + Text\n");

    println!("2. Loading BGE (text-only dense)...");
    let bert = TesseraDense::new("bge-base-en-v1.5")?;
    println!("   Loaded: {}", bert.model());
    println!("   Parameters: 109M");
    println!("   Modality: Text only\n");

    println!("=== APPROACH 1: VISION-LANGUAGE (ColPali) ===\n");

    println!("How it works:");
    println!("  Input: Document image (PNG, JPEG, etc.)");
    println!("  Processing:");
    println!(
        "    1. Image split into {} patches (14×14 pixels)",
        colpali.num_patches()
    );
    println!(
        "    2. Each patch encoded to {}-dim embedding",
        colpali.embedding_dim()
    );
    println!("    3. Query text encoded to token embeddings");
    println!("    4. Late interaction (MaxSim) scoring");
    println!("  Output: Relevance score\n");

    println!("Advantages:");
    println!("  - No OCR required (processes image directly)");
    println!("  - Handles tables, charts, and diagrams");
    println!("  - Preserves visual layout and structure");
    println!("  - Multi-modal understanding (text + visual)");
    println!("  - Robust to OCR errors and scan quality");
    println!("  - Works with handwritten text");
    println!("  - Multi-lingual without language-specific setup\n");

    println!("Best for:");
    println!("  - Scanned PDFs with complex layouts");
    println!("  - Invoices and receipts (mixed formats)");
    println!("  - Academic papers with equations/figures");
    println!("  - Technical documents with diagrams");
    println!("  - Presentations and slides");
    println!("  - Forms and structured documents");
    println!("  - Historical documents (variable quality)\n");

    println!("Considerations:");
    println!("  - Larger model size (3B parameters)");
    println!("  - Higher memory requirements");
    println!("  - Slower inference (vision processing)");
    println!("  - Requires image format (not plain text)\n");

    println!("Example usage:");
    println!("  let embedder = TesseraVision::new(\"colpali-v1.3-hf\")?;");
    println!("  let query_emb = embedder.encode_query(\"total amount\")?;");
    println!("  let doc_emb = embedder.encode_document(\"invoice.png\")?;");
    println!("  let score = embedder.search(&query_emb, &doc_emb)?;\n");

    println!("{}\n", "=".repeat(60));

    println!("=== APPROACH 2: TEXT-ONLY (Dense Embeddings) ===\n");

    println!("How it works:");
    println!("  Input: Plain text (requires OCR for documents)");
    println!("  Processing:");
    println!("    1. Text tokenization (WordPiece/BPE)");
    println!("    2. BERT encoding");
    println!(
        "    3. Mean pooling to single {}-dim vector",
        bert.dimension()
    );
    println!("    4. Cosine similarity for search");
    println!("  Output: Similarity score\n");

    println!("Advantages:");
    println!("  - Smaller model size (109M parameters)");
    println!("  - Faster encoding");
    println!("  - Lower memory requirements");
    println!("  - Works directly with text");
    println!("  - Standard vector database compatible");
    println!("  - Efficient at scale\n");

    println!("Best for:");
    println!("  - Plain text documents");
    println!("  - Clean digital text (no OCR needed)");
    println!("  - Email and messages");
    println!("  - Articles and blogs");
    println!("  - FAQs and documentation");
    println!("  - Large-scale text retrieval");
    println!("  - Resource-constrained environments\n");

    println!("Limitations:");
    println!("  - Requires OCR preprocessing for documents");
    println!("  - Loses visual information (layout, formatting)");
    println!("  - OCR errors propagate to retrieval");
    println!("  - Cannot handle images, charts, diagrams");
    println!("  - Table structure often lost");
    println!("  - Language-specific OCR may be needed\n");

    println!("Example usage:");
    println!("  let embedder = TesseraDense::new(\"bge-base-en-v1.5\")?;");
    println!("  let query_emb = embedder.encode(\"total amount\")?;");
    println!("  let doc_emb = embedder.encode(extracted_text)?;");
    println!("  let score = embedder.similarity(query, doc)?;\n");

    println!("{}\n", "=".repeat(60));

    println!("=== COMPARISON SUMMARY ===\n");

    println!("Choose ColPali (Vision-Language) when:");
    println!("  - Working with scanned documents");
    println!("  - Visual layout is important");
    println!("  - Documents contain tables/charts");
    println!("  - OCR quality is unreliable");
    println!("  - Multi-modal content (text + images)");
    println!("  - Preservation of structure matters\n");

    println!("Choose Dense Embeddings (Text-Only) when:");
    println!("  - Working with clean digital text");
    println!("  - Large-scale text search (millions of docs)");
    println!("  - Resource constraints (memory/compute)");
    println!("  - Fast inference is critical");
    println!("  - Integration with vector databases");
    println!("  - No visual information needed\n");

    println!("Hybrid Approach:");
    println!("  - Use ColPali for document images");
    println!("  - Use Dense for extracted/digital text");
    println!("  - Combine for comprehensive retrieval\n");

    println!("=== PRACTICAL EXAMPLE ===\n");

    println!("Scenario: Invoice search");
    println!("\nWith ColPali (Vision-Language):");
    println!("  1. Take invoice image → encode directly");
    println!("  2. Query: 'total amount' → encode text");
    println!("  3. MaxSim scoring → find relevant invoices");
    println!("  Result: Finds total even in complex layouts\n");

    println!("With Dense Embeddings (Text-Only):");
    println!("  1. Invoice image → OCR → extracted text");
    println!("  2. OCR errors: 'T0tal Am0unt: $1,Z00' (O→0, Z→2)");
    println!("  3. Query: 'total amount' → encode text");
    println!("  4. Compare with corrupted OCR text");
    println!("  Result: May miss due to OCR errors\n");

    println!("Conclusion:");
    println!("  ColPali excels at document images and visual content");
    println!("  Dense embeddings excel at clean text at scale");
    println!("  Choose based on your data and requirements\n");

    println!("=== Comparison Complete ===");
    Ok(())
}
