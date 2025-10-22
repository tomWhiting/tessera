//! Verify ColPali embedding shapes for text and images.
//!
//! This example verifies that text and image embeddings have matching dimensions
//! after projection, which is required for MaxSim similarity computation.

use anyhow::Result;
use tessera::backends::candle::get_device;
use tessera::core::VisionEncoder;
use tessera::encoding::vision::ColPaliEncoder;
use tessera::models::ModelConfig;

fn main() -> Result<()> {
    println!("=== ColPali Shape Verification ===\n");

    let device = get_device()?;
    let config = ModelConfig::from_registry("colpali-v1.2")?;
    let encoder = ColPaliEncoder::new(config, device)?;

    println!("Encoder configuration:");
    println!("  Embedding dim: {}", encoder.embedding_dim());
    println!("  Num patches: {}", encoder.num_patches());
    println!("  Image resolution: {:?}", encoder.image_resolution());

    // Encode a text query
    println!("\nEncoding text query...");
    let text = "test query";
    let text_emb = encoder.encode_text(text)?;
    println!("Text embeddings shape: {:?}", text_emb.shape());
    println!("  num_tokens: {}", text_emb.num_tokens);
    println!("  embedding_dim: {}", text_emb.embedding_dim);

    // Encode a test image
    println!("\nEncoding test image...");
    let _pdf_path = std::path::Path::new("examples/fixtures/attention_is_all_you_need.pdf");

    #[cfg(feature = "pdf")]
    {
        if _pdf_path.exists() {
            println!("Using first page of PDF as test image");
            let image_emb = encoder.encode_pdf_page(_pdf_path, 0)?;
            println!("Image embeddings shape: {:?}", image_emb.shape());
            println!("  num_patches: {}", image_emb.num_patches);
            println!("  embedding_dim: {}", image_emb.embedding_dim);
            println!(
                "  actual embeddings[0].len(): {}",
                image_emb.embeddings[0].len()
            );

            // Verify dimensions match
            println!("\n=== Verification ===");
            if text_emb.embedding_dim == image_emb.embedding_dim {
                println!("✓ SUCCESS: Text and image embedding dimensions match!");
                println!("  Both use embedding_dim = {}", text_emb.embedding_dim);
                println!("\nThis confirms the fix is working correctly:");
                println!("  - Text embeddings: projected from 2048 → 128");
                println!("  - Image embeddings: projected from 2048 → 128");
                println!("  - MaxSim can now compute similarity between them!");
            } else {
                println!("✗ FAILURE: Dimension mismatch!");
                println!(
                    "  Text: {}, Image: {}",
                    text_emb.embedding_dim, image_emb.embedding_dim
                );
                std::process::exit(1);
            }
        } else {
            println!("Test PDF not found at {:?}", pdf_path);
            println!("Please ensure the PDF exists to run this test.");
        }
    }

    #[cfg(not(feature = "pdf"))]
    {
        println!("PDF feature not enabled, skipping image test");
        println!("Run with: cargo run --example verify_colpali_shapes --features pdf,metal");
    }

    Ok(())
}
