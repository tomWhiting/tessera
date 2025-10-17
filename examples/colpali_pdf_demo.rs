//! ColPali PDF document processing example.
//!
//! Demonstrates encoding PDF documents page-by-page with ColPali.
//!
//! Run with:
//! ```bash
//! cargo run --release --example colpali_pdf_demo --features pdf,metal
//! ```

use tessera::encoding::vision::ColPaliEncoder;
use tessera::backends::candle::get_device;
use tessera::models::ModelConfig;
use anyhow::Context;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("=== ColPali PDF Document Processing ===\n");

    let device = get_device().context("Getting compute device")?;

    println!("Loading ColPali model...");
    let config = ModelConfig::from_registry("colpali-v1.2")
        .context("Failed to load ColPali model config")?;
    let encoder = ColPaliEncoder::new(config, device)?;
    println!("Model loaded: colpali-v1.2\n");

    // Test with the attention paper
    let pdf_path = Path::new("examples/fixtures/attention_is_all_you_need.pdf");

    if !pdf_path.exists() {
        println!("PDF not found at {:?}", pdf_path);
        println!("Please ensure the PDF file exists");
        return Ok(());
    }

    println!("Processing PDF: {:?}", pdf_path);
    println!("This will encode all pages (may take a few minutes)...\n");

    let page_embeddings = encoder.encode_pdf_document(pdf_path)?;

    println!("\nResults:");
    println!("  Total pages: {}", page_embeddings.len());
    println!("  Embedding dim: {}", page_embeddings[0].embedding_dim);
    println!("  Patches per page: {}", page_embeddings[0].num_patches);

    // Example: Search within the document
    println!("\n--- Document Search Example ---");
    let query = "transformer architecture";
    println!("Query: \"{}\"", query);

    println!("\nEncoding query...");
    let query_embedding = encoder.encode_text(query)?;
    println!("Query encoded: {} tokens", query_embedding.embeddings.nrows());

    // Compute similarity with each page
    println!("\nComputing similarity with each page...");
    use tessera::utils::similarity::max_sim;
    use tessera::core::TokenEmbeddings;
    use ndarray::Array2;

    let mut page_scores: Vec<(usize, f32)> = Vec::new();
    for (page_idx, page_emb) in page_embeddings.iter().enumerate() {
        // Convert VisionEmbedding (Vec<Vec<f32>>) to Array2 for max_sim
        let flat: Vec<f32> = page_emb.embeddings.iter().flatten().copied().collect();
        let page_array = Array2::from_shape_vec(
            (page_emb.num_patches, page_emb.embedding_dim),
            flat,
        )?;

        // Create TokenEmbeddings from the page array
        let page_token_emb = TokenEmbeddings::new(
            page_array,
            format!("Page {}", page_idx + 1),
        )?;

        let score = max_sim(&query_embedding, &page_token_emb)?;
        page_scores.push((page_idx + 1, score));
    }

    // Sort by score descending
    page_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 most relevant pages:");
    for (rank, (page_num, score)) in page_scores.iter().take(5).enumerate() {
        println!("  {}. Page {} - Score: {:.4}", rank + 1, page_num, score);
    }

    println!("\nColPali successfully processed the PDF document!");
    println!("Each page is encoded as {} patch embeddings.", page_embeddings[0].num_patches);

    Ok(())
}
