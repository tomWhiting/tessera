//! ColPali PDF document processing example.
//!
//! Demonstrates encoding PDF documents page-by-page with ColPali.
//!
//! Run with:
//! ```bash
//! cargo run --release --example colpali_pdf_demo --features pdf,metal
//! ```

use anyhow::Context;
use std::path::Path;
use tessera::backends::candle::get_device;
use tessera::encoding::vision::ColPaliEncoder;
use tessera::models::ModelConfig;

fn main() -> anyhow::Result<()> {
    println!("=== ColPali PDF Document Processing ===\n");

    let device = get_device().context("Getting compute device")?;

    println!("Loading ColPali model...");
    let config = ModelConfig::from_registry("colpali-v1.2")
        .context("Failed to load ColPali model config")?;
    let encoder = ColPaliEncoder::new(config, device)?;
    println!("Model loaded: colpali-v1.2\n");

    // Test with the attention paper
    let _pdf_path = Path::new("examples/fixtures/attention_is_all_you_need.pdf");

    println!("Note: PDF document encoding (encode_pdf_document) is not yet implemented");
    println!("This example will be completed when that functionality is available\n");

    // Example: Test text query encoding which is available
    println!("--- Text Query Encoding Example ---");
    let query = "transformer architecture";
    println!("Query: \"{}\"", query);

    println!("\nEncoding query...");
    let query_embedding = encoder.encode_text(query)?;
    println!(
        "Query encoded: {} tokens, {} dimensions",
        query_embedding.num_tokens, query_embedding.embedding_dim
    );

    println!("\nTODO: PDF document encoding will enable:");
    println!("  - Page-by-page embedding extraction");
    println!("  - Document search with text queries");
    println!("  - Similarity scoring across pages");

    Ok(())
}
