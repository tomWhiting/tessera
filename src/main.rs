//! CLI demo for Tessera ColBERT inference.
//!
//! This demonstrates both Candle and Burn backends for computing
//! similarity between query and document text.

use anyhow::{Context, Result};
use clap::Parser;

use tessera::{
    backends::{candle::CandleEncoder, burn::BurnEncoder, candle::get_device},
    core::{max_sim, TokenEmbedder},
    models::ModelConfig,
};

#[derive(Parser, Debug)]
#[command(name = "tessera")]
#[command(about = "ColBERT-style similarity scoring demo", long_about = None)]
struct Args {
    /// Query text
    #[arg(short, long)]
    query: String,

    /// Document text
    #[arg(short, long)]
    document: String,

    /// Model to use: colbert-small, colbert-v2, jina-colbert-v2, or distilbert
    #[arg(short, long, default_value = "colbert-small")]
    model: String,

    /// Backend to use (candle, burn, or both)
    #[arg(short, long, default_value = "candle")]
    backend: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Tessera ColBERT Similarity Demo");
    println!("================================\n");
    println!("Query:    {}", args.query);
    println!("Document: {}\n", args.document);

    // Create model config - support both short names and full model paths
    let config = match args.model.as_str() {
        // ColBERT models (recommended)
        "colbert-small" => ModelConfig::colbert_small(),
        "colbert-v2" => ModelConfig::colbert_v2(),
        "jina-colbert-v2" => ModelConfig::jina_colbert_v2(),

        // Standard BERT model
        "distilbert" | "distilbert-base-uncased" => ModelConfig::distilbert_base_uncased(),

        // Custom model path
        other => {
            println!("Using custom model: {}", other);
            ModelConfig::custom(other, 768, 512)
        }
    };

    // Run with selected backend(s)
    match args.backend.as_str() {
        "candle" => run_candle(&args.query, &args.document, config)?,
        "burn" => run_burn(&args.query, &args.document, config)?,
        "both" => {
            run_candle(&args.query, &args.document, config.clone())?;
            println!();
            run_burn(&args.query, &args.document, config)?;
        }
        _ => anyhow::bail!("Unknown backend: {}. Use 'candle', 'burn', or 'both'", args.backend),
    }

    Ok(())
}

fn run_candle(query: &str, document: &str, config: ModelConfig) -> Result<()> {
    println!("Backend: Candle");
    println!("---------------");

    // Get device
    let device = get_device().context("Getting compute device")?;
    println!("Device: {}", tessera::backends::candle::device_description(&device));

    // Create encoder
    println!("Loading model: {}...", config.model_name);
    let encoder = CandleEncoder::new(config, device)
        .context("Creating Candle encoder")?;

    // Encode query
    println!("Encoding query...");
    let query_emb = encoder.encode(query).context("Encoding query")?;
    println!("Query tokens: {}, dims: {}", query_emb.num_tokens, query_emb.embedding_dim);

    // Encode document
    println!("Encoding document...");
    let doc_emb = encoder.encode(document).context("Encoding document")?;
    println!("Document tokens: {}, dims: {}", doc_emb.num_tokens, doc_emb.embedding_dim);

    // Compute similarity
    println!("Computing MaxSim similarity...");
    let score = max_sim(&query_emb, &doc_emb).context("Computing similarity")?;

    println!("\nSimilarity Score: {:.4}", score);

    Ok(())
}

fn run_burn(query: &str, document: &str, config: ModelConfig) -> Result<()> {
    println!("Backend: Burn");
    println!("-------------");
    println!("Device: {}", tessera::backends::burn::backend_description());

    // Create encoder
    println!("Loading model: {}...", config.model_name);
    let encoder = BurnEncoder::new(config)
        .context("Creating Burn encoder")?;

    println!("Note: Burn backend uses simplified implementation for prototype");

    // Encode query
    println!("Encoding query...");
    let query_emb = encoder.encode(query).context("Encoding query")?;
    println!("Query tokens: {}, dims: {}", query_emb.num_tokens, query_emb.embedding_dim);

    // Encode document
    println!("Encoding document...");
    let doc_emb = encoder.encode(document).context("Encoding document")?;
    println!("Document tokens: {}, dims: {}", doc_emb.num_tokens, doc_emb.embedding_dim);

    // Compute similarity
    println!("Computing MaxSim similarity...");
    let score = max_sim(&query_emb, &doc_emb).context("Computing similarity")?;

    println!("\nSimilarity Score: {:.4}", score);
    println!("(Note: Using placeholder embeddings - not pre-trained BERT weights)");

    Ok(())
}
