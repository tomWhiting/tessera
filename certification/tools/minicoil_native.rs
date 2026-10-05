//! Verify an accepted sparse fixture through the public local constructor.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use tessera::{MinicoilEmbedder, MinicoilRole as Role};

/// Runs one accepted fixture through the public local constructor.
#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    encoder_dir: PathBuf,
    #[arg(long)]
    tables_dir: PathBuf,
    #[arg(long)]
    fixture: PathBuf,
}

#[derive(Deserialize)]
struct Fixture {
    text: String,
    role: String,
    token_ids: Vec<u32>,
    sparse: Expected,
}

#[derive(Deserialize)]
struct Expected {
    indices: Vec<u32>,
    values: Vec<f32>,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let fixture: Fixture = serde_json::from_slice(
        &std::fs::read(&arguments.fixture).context("Reading qualification fixture")?,
    )
    .context("Decoding qualification fixture")?;
    let role = match fixture.role.as_str() {
        "document" => Role::Document,
        "question" => Role::Question,
        other => anyhow::bail!("Unknown qualification role {other}"),
    };
    anyhow::ensure!(
        fixture.sparse.indices.len() == fixture.sparse.values.len()
            && fixture.sparse.values.iter().all(|value| value.is_finite()),
        "Invalid qualification sparse reference"
    );
    let tokenizer = tokenizers::Tokenizer::from_file(arguments.encoder_dir.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!("Loading qualification tokenizer: {error}"))?;
    let input_tokens = tokenizer
        .encode(fixture.text.as_str(), true)
        .map_err(|error| anyhow::anyhow!("Tokenizing qualification text: {error}"))?;
    anyhow::ensure!(
        input_tokens.get_ids() == fixture.token_ids.as_slice() && input_tokens.len() <= 512,
        "Qualification token ids differ from the accepted fixture"
    );
    let started = Instant::now();
    let embedder =
        MinicoilEmbedder::from_model_dirs(&arguments.encoder_dir, &arguments.tables_dir)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let output = embedder.encode(&fixture.text, role)?;
    let encode_seconds = started.elapsed().as_secs_f64();
    let indices_equal = output.indices == fixture.sparse.indices;
    let lengths_equal = output.values.len() == fixture.sparse.values.len();
    let maximum_difference = output
        .values
        .iter()
        .zip(&fixture.sparse.values)
        .map(|(&actual, &expected)| (f64::from(actual) - f64::from(expected)).abs())
        .fold(0.0_f64, f64::max);
    let finite = output.values.iter().all(|value| value.is_finite());
    let passed = indices_equal && lengths_equal && finite && maximum_difference <= 0.000_01;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "passed": passed,
            "fixture": arguments.fixture.file_name().and_then(|name| name.to_str()),
            "role": fixture.role,
            "token_count": input_tokens.len(),
            "indices_equal": indices_equal,
            "values_length_equal": lengths_equal,
            "maximum_absolute_difference": maximum_difference,
            "tolerance": 0.000_01,
            "load_seconds": load_seconds,
            "encode_seconds": encode_seconds,
            "cpu": true,
            "dtype": "f32",
            "batch": 1,
            "window": 512,
            "indices": output.indices,
            "values": output.values
        }))?
    );
    anyhow::ensure!(
        passed,
        "miniCOIL native fixture differs from the accepted reference"
    );
    Ok(())
}
