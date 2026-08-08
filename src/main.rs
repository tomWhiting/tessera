//! CLI demo for Tessera `ColBERT` inference.
//!
//! This demonstrates Candle backend for computing similarity between
//! query and document text using multi-vector embeddings.

use anyhow::{Context, Result};
use clap::Parser;

use tessera::TesseraMultiVector;

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

    /// Model to use: colbert-small or colbert-v2
    #[arg(short, long, default_value = "colbert-small")]
    model: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Tessera ColBERT Similarity Demo");
    println!("================================\n");
    println!("Query:    {}", args.query);
    println!("Document: {}\n", args.document);

    anyhow::ensure!(
        matches!(args.model.as_str(), "colbert-small" | "colbert-v2"),
        "Unknown or unaudited CLI model '{}'. Choose colbert-small or colbert-v2",
        args.model
    );
    run_colbert(&args.query, &args.document, &args.model)?;

    Ok(())
}

fn run_colbert(query: &str, document: &str, model: &str) -> Result<()> {
    println!("Loading model: {model}...");
    let embedder = TesseraMultiVector::new(model).context("Creating ColBERT embedder")?;
    println!("Parameter dtype: {:?}", embedder.model_dtype());

    // Encode query
    println!("Encoding query...");
    let query_emb = embedder.encode_query(query).context("Encoding query")?;
    println!(
        "Query tokens: {}, dims: {}",
        query_emb.num_tokens(),
        query_emb.embedding_dim()
    );

    // Encode document
    println!("Encoding document...");
    let doc_emb = embedder
        .encode_document(document)
        .context("Encoding document")?;
    println!(
        "Document tokens: {}, dims: {}",
        doc_emb.num_tokens(),
        doc_emb.embedding_dim()
    );

    // Compute similarity
    println!("Computing MaxSim similarity...");
    let score = embedder
        .search(&query_emb, &doc_emb)
        .context("Computing similarity")?;

    println!("\nSimilarity Score: {score:.4}");

    Ok(())
}
