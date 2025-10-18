//! ColPali multi-modal search example.
//!
//! Demonstrates searching across different document types with ColPali:
//! invoices, charts, tables, contracts, etc. Shows how vision-language
//! embeddings handle diverse visual content without specialized processing.
//!
//! Run with:
//! ```bash
//! cargo run --release --example colpali_multimodal
//! ```

use std::path::Path;
use tessera::TesseraVision;

fn main() -> tessera::Result<()> {
    println!("=== ColPali Multi-Modal Document Search ===\n");

    println!("Loading ColPali model...");
    let embedder = TesseraVision::new("colpali-v1.3-hf")?;
    println!("Model loaded: {}\n", embedder.model());

    // Different document types with realistic paths
    let all_documents = vec![
        ("Financial", "invoice", "test_data/invoice.png"),
        ("Visual", "chart", "test_data/chart.png"),
        ("Tabular", "table", "test_data/table.png"),
        ("Legal", "contract", "test_data/contract.png"),
        ("Scientific", "paper", "test_data/research_paper.png"),
        ("Technical", "diagram", "test_data/technical_diagram.png"),
    ];

    // Filter to only existing documents
    let documents: Vec<(&str, &str, &str)> = all_documents
        .iter()
        .filter(|(_, _, path)| Path::new(path).exists())
        .map(|(cat, doc_type, path)| (*cat, *doc_type, *path))
        .collect();

    if documents.is_empty() {
        println!("No test documents found in test_data/ directory.\n");

        println!("To demonstrate multi-modal search, create test_data/ with:");
        println!("  - invoice.png: Invoice or receipt");
        println!("  - chart.png: Chart or graph");
        println!("  - table.png: Spreadsheet or table");
        println!("  - contract.png: Contract or legal document");
        println!("  - research_paper.png: Academic paper page");
        println!("  - technical_diagram.png: Technical diagram or blueprint\n");

        println!("=== Multi-Modal Capabilities Demo ===\n");

        println!("ColPali handles diverse content types:");
        println!("\n1. FINANCIAL DOCUMENTS");
        println!("   - Invoices with mixed layouts");
        println!("   - Receipts with varying formats");
        println!("   - Bank statements with tables");
        println!("   Example query: 'What is the total amount?'");

        println!("\n2. VISUAL CONTENT");
        println!("   - Charts and graphs");
        println!("   - Plots and visualizations");
        println!("   - Infographics");
        println!("   Example query: 'What trend does this show?'");

        println!("\n3. TABULAR DATA");
        println!("   - Spreadsheets");
        println!("   - Data tables");
        println!("   - Comparison matrices");
        println!("   Example query: 'How many items are listed?'");

        println!("\n4. LEGAL DOCUMENTS");
        println!("   - Contracts");
        println!("   - Agreements");
        println!("   - Legal forms");
        println!("   Example query: 'What is the contract duration?'");

        println!("\n5. SCIENTIFIC PAPERS");
        println!("   - Research papers");
        println!("   - Academic publications");
        println!("   - Technical reports");
        println!("   Example query: 'What methodology was used?'");

        println!("\n6. TECHNICAL DIAGRAMS");
        println!("   - Architecture diagrams");
        println!("   - Circuit diagrams");
        println!("   - Flowcharts");
        println!("   Example query: 'What are the main components?'\n");

        println!("Key advantage: Single model handles all types!");
        println!("No need for specialized OCR or layout analysis.\n");

        return Ok(());
    }

    println!("=== Multi-Modal Search Demo ===\n");
    println!("Found {} document types:", documents.len());
    for (category, doc_type, path) in &documents {
        println!("  [{}] {} ({})", category, doc_type, path);
    }
    println!();

    // Encode all documents
    println!("Encoding documents...\n");
    let mut doc_embeddings = Vec::new();
    for (category, doc_type, path) in &documents {
        let emb = embedder.encode_document(path)?;
        println!(
            "  Encoded [{}] {}: {} patches",
            category, doc_type, emb.num_patches
        );
        doc_embeddings.push((category, doc_type, emb));
    }
    println!();

    // Different query types targeting different document types
    let queries = vec![
        ("Financial", "What is the total amount?"),
        ("Visual", "What trend does the chart show?"),
        ("Tabular", "How many items in the table?"),
        ("Legal", "What is the contract term?"),
        ("Scientific", "What is the research methodology?"),
        ("Technical", "What are the system components?"),
    ];

    println!("=== Cross-Modal Search ===\n");
    println!(
        "Testing {} queries across {} document types\n",
        queries.len(),
        documents.len()
    );

    for (query_type, query_text) in &queries {
        println!("Query [{}]: \"{}\"", query_type, query_text);

        // Encode query
        let query_emb = embedder.encode_query(query_text)?;

        // Search across all document types
        let mut scores = Vec::new();
        for (doc_category, doc_type, doc_emb) in &doc_embeddings {
            let score = embedder.search(&query_emb, doc_emb)?;
            scores.push((doc_category, doc_type, score));
        }

        // Sort by score (descending)
        scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Show top match
        let (top_category, top_type, top_score) = scores[0];
        println!(
            "  Top match: [{}] {} (score: {:.4})",
            top_category, top_type, top_score
        );

        // Show if query matched expected document type
        if *top_category == query_type {
            println!("  Correctly matched expected document type!");
        } else {
            println!("  Note: Best match was different document type");
        }
        println!();
    }

    println!("=== Multi-Modal Advantages ===\n");
    println!("ColPali handles all document types with:");
    println!("  - Single unified model (no specialized preprocessing)");
    println!("  - Native understanding of layout and structure");
    println!("  - Combined text and visual comprehension");
    println!("  - Robust to formatting variations");
    println!("  - No need for document type classification\n");

    println!("Comparison to traditional approaches:");
    println!("  Traditional: OCR → Text extraction → Text search");
    println!("  - Loses visual information");
    println!("  - Fails on charts/diagrams");
    println!("  - Sensitive to OCR errors");
    println!("  - Requires format-specific handling\n");

    println!("  ColPali: Image → Vision-language encoding → Search");
    println!("  - Preserves all visual information");
    println!("  - Handles any visual content");
    println!("  - No OCR errors");
    println!("  - Single unified approach\n");

    println!("=== Search Complete ===");
    Ok(())
}
