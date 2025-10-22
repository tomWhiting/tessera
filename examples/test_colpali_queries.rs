//! ColPali query testing example.
//!
//! This example demonstrates encoding text queries and would demonstrate
//! PDF document encoding when that functionality is implemented.

use tessera::backends::candle::get_device;
use tessera::encoding::vision::ColPaliEncoder;
use tessera::models::ModelConfig;

fn main() -> anyhow::Result<()> {
    println!("=== ColPali Query Testing ===\n");

    let device = get_device()?;
    let config = ModelConfig::from_registry("colpali-v1.2")?;
    let encoder = ColPaliEncoder::new(config, device)?;

    let _pdf_path = std::path::Path::new("examples/fixtures/attention_is_all_you_need.pdf");

    println!("Note: PDF document encoding not yet fully implemented");
    println!("This example demonstrates text query encoding:\n");

    let queries = vec![
        "transformer architecture",
        "encoder decoder",
        "self attention mechanism",
        "multi-head attention",
    ];

    for query in queries {
        println!("\n--- Query: \"{}\" ---", query);
        let query_emb = encoder.encode_text(query)?;
        println!("  Encoded to {} token vectors", query_emb.num_tokens);
        println!("  Embedding dimension: {}", query_emb.embedding_dim);
    }

    println!("\nTODO: PDF document encoding will be added when encode_pdf_document is implemented");

    Ok(())
}
