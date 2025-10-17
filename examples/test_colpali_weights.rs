//! Quick test to verify ColPali v1.2-merged weight loading
//!
//! Run with:
//! cargo run --release --features metal test_colpali_weights.rs

use tessera::encoding::vision::ColPaliEncoder;
use tessera::backends::candle::get_device;
use tessera::models::ModelConfig;
use tessera::core::VisionEncoder;
use anyhow::Context;

fn main() -> anyhow::Result<()> {
    println!("=== ColPali v1.2-merged Weight Loading Test ===\n");

    let device = get_device().context("Getting compute device")?;
    println!("Device: {:?}\n", device);

    println!("Loading ColPali v1.2-merged model...");
    println!("Expected weight structure:");
    println!("  - custom_text_proj.weight (root level)");
    println!("  - model.vision_tower.vision_model.embeddings.patch_embedding.weight");
    println!("  - model.language_model...\n");

    let config = ModelConfig::from_registry("colpali-v1.2")
        .context("Failed to load ColPali model config")?;

    println!("Model config:");
    println!("  Model: {}", config.model_name);
    println!("  Embedding dim: {}", config.embedding_dim);
    println!("  Max seq len: {}\n", config.max_seq_length);

    println!("Initializing encoder...");
    let encoder = ColPaliEncoder::new(config, device)?;

    println!("\n✓ SUCCESS! Model loaded without errors.");
    println!("\nModel details:");
    println!("  Embedding dimension: {}", encoder.embedding_dim());
    println!("  Number of patches: {}", encoder.num_patches());
    println!("  Image resolution: {:?}", encoder.image_resolution());

    Ok(())
}
