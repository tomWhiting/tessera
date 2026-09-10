//! End-to-end smoke run for the experimental `bge-base-en-v1.5` dense path.
//!
//! The example downloads the pinned Hugging Face artifacts on first use, encodes
//! three sentences, and checks two properties that any working dense encoder must
//! satisfy:
//!
//! 1. the produced vector has the dimension the registry declares for the model;
//! 2. a semantically related sentence pair scores above an unrelated pair.
//!
//! Run it on the CPU with:
//!
//! ```text
//! cargo run --release --example dense_bge_smoke -- cpu
//! ```
//!
//! and on Apple Metal with:
//!
//! ```text
//! cargo run --release --features metal --example dense_bge_smoke -- metal
//! ```
//!
//! `auto` asks Tessera's own device selection, which prefers Metal when the
//! `metal` feature is compiled in and falls back to the CPU otherwise.

use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use ndarray::Array1;
use tessera::model_registry::get_model;
use tessera::TesseraDenseBuilder;

/// Registry identifier of the model under test.
const MODEL_ID: &str = "bge-base-en-v1.5";

/// Reference sentence for both comparisons.
const ANCHOR: &str = "A cat is sleeping on the warm windowsill.";

/// Sentence that restates the anchor with different words.
const SIMILAR: &str = "The kitten is napping in the sunny window.";

/// Sentence from an unrelated domain.
const DISSIMILAR: &str = "Quarterly revenue rose sharply after the merger closed.";

/// Selects the Candle device named on the command line.
fn select_device(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "metal" => metal_device(),
        "auto" => {
            #[cfg(feature = "metal")]
            {
                if let Ok(device) = tessera::metal_device() {
                    return Ok(device);
                }
            }
            Ok(Device::Cpu)
        }
        other => bail!("unknown device '{other}'; expected one of: cpu, metal, auto"),
    }
}

/// Builds a Metal device, or explains why this build cannot.
#[cfg(feature = "metal")]
fn metal_device() -> Result<Device> {
    tessera::metal_device().context("failed to open Metal device 0")
}

/// Reports that Metal was not compiled into this build.
#[cfg(not(feature = "metal"))]
fn metal_device() -> Result<Device> {
    bail!("this binary was built without the `metal` feature; rebuild with --features metal")
}

/// Cosine similarity between two equal-length vectors.
fn cosine(left: &Array1<f32>, right: &Array1<f32>) -> Result<f32> {
    if left.len() != right.len() {
        bail!(
            "cannot compare vectors of different lengths: {} and {}",
            left.len(),
            right.len()
        );
    }
    let dot: f32 = left.iter().zip(right.iter()).map(|(a, b)| a * b).sum();
    let left_norm: f32 = left.iter().map(|a| a * a).sum::<f32>().sqrt();
    let right_norm: f32 = right.iter().map(|b| b * b).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        bail!("cannot compare a zero-length embedding vector");
    }
    Ok(dot / (left_norm * right_norm))
}

fn main() -> Result<()> {
    let device_name = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_string());
    let device = select_device(&device_name)?;

    let model = get_model(MODEL_ID)
        .with_context(|| format!("model '{MODEL_ID}' is absent from the generated registry"))?;
    let expected_dim = model.embedding_dim.default_dim();

    println!("model:            {MODEL_ID} ({})", model.huggingface_id);
    println!("support tier:     {:?}", model.support_tier);
    println!("requested device: {device_name} -> {device:?}");
    println!("registry dim:     {expected_dim}");

    let load_started = Instant::now();
    let embedder = TesseraDenseBuilder::new()
        .model(MODEL_ID)
        .device(device)
        .build()
        .with_context(|| format!("failed to build the dense embedder for '{MODEL_ID}'"))?;
    let load_elapsed = load_started.elapsed();
    println!("load:             {:.3} s", load_elapsed.as_secs_f64());

    let encode_started = Instant::now();
    let anchor = embedder
        .encode(ANCHOR)
        .context("failed to encode the anchor")?;
    let similar = embedder
        .encode(SIMILAR)
        .context("failed to encode the similar sentence")?;
    let dissimilar = embedder
        .encode(DISSIMILAR)
        .context("failed to encode the dissimilar sentence")?;
    let encode_elapsed = encode_started.elapsed();
    println!(
        "encode (3 texts): {:.3} s ({:.3} s per text)",
        encode_elapsed.as_secs_f64(),
        encode_elapsed.as_secs_f64() / 3.0
    );

    for embedding in [&anchor, &similar, &dissimilar] {
        if embedding.dim() != expected_dim {
            bail!(
                "encoded '{}' to {} dimensions, but the registry declares {expected_dim}",
                embedding.text(),
                embedding.dim()
            );
        }
    }
    println!("dimension:        {} (matches registry)", anchor.dim());

    let related = cosine(anchor.values(), similar.values())?;
    let unrelated = cosine(anchor.values(), dissimilar.values())?;
    let control = cosine(similar.values(), dissimilar.values())?;
    println!("cos(anchor, similar):      {related:.4}");
    println!("cos(anchor, dissimilar):   {unrelated:.4}");
    println!("cos(similar, dissimilar):  {control:.4}");

    if related <= unrelated {
        bail!(
            "the related pair scored {related:.4}, which is not above the unrelated pair at \
             {unrelated:.4}: the dense path is not discriminating"
        );
    }
    println!(
        "PASS: related pair leads the unrelated pair by {:.4}",
        related - unrelated
    );
    Ok(())
}
