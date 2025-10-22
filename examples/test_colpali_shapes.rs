//! Test ColPali embedding shapes.
//!
//! This example tests the ColPali encoder by encoding text queries
//! and verifying the output shapes.

use tessera::backends::candle::get_device;
use tessera::encoding::vision::ColPaliEncoder;
use tessera::models::ModelConfig;

fn main() -> anyhow::Result<()> {
    let device = get_device()?;
    let config = ModelConfig::from_registry("colpali-v1.2")?;
    let encoder = ColPaliEncoder::new(config, device)?;

    // Test text encoding
    let query = encoder.encode_text("transformer architecture")?;
    println!("Query shape: {:?}", query.embeddings.dim());
    println!("Query embedding_dim: {}", query.embedding_dim);
    println!("Query num_tokens: {}", query.num_tokens);

    Ok(())
}
