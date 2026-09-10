//! Cross-device agreement check for one dense model.
//!
//! Loads the same dense checkpoint twice — once on an explicitly named
//! accelerator, once on CPU — encodes the same sentences on each, and compares
//! the two embeddings per sentence with cosine similarity. Wall times for each
//! encode pass are printed so a run also reports whether the accelerator is
//! doing anything.
//!
//! The accelerator device is constructed by name rather than through
//! `tessera::get_device`, because that selector falls back to CPU when the
//! accelerator cannot be created. A parity check that silently compared CPU to
//! CPU would pass while proving nothing.
//!
//! The two models are never resident at the same time: the accelerator embedder
//! is dropped before the CPU embedder is built, so the process-wide residency
//! ledger only ever accounts for one copy of the parameters.
//!
//! Exits non-zero when any sentence disagrees beyond the tolerance.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use tessera::{Device, TesseraDense};

/// Sentences encoded on both devices. Short, ASCII, and deterministic.
const SENTENCES: [&str; 3] = [
    "The quick brown fox jumps over the lazy dog.",
    "Durable execution replays a workflow from its recorded history.",
    "Sparse retrieval activates vocabulary terms that never appear in the text.",
];

#[derive(Parser, Debug)]
#[command(name = "device_parity")]
#[command(about = "Compare one dense model's embeddings across two devices", long_about = None)]
struct Args {
    /// Registry model identifier.
    #[arg(long, default_value = "bge-base-en-v1.5")]
    model: String,

    /// Accelerator to compare against CPU.
    #[arg(long, default_value = "cuda", value_parser = ["cuda", "metal", "cpu"])]
    device: String,

    /// Accelerator ordinal.
    #[arg(long, default_value_t = 0)]
    ordinal: usize,

    /// Minimum cosine similarity required between the two devices.
    #[arg(long, default_value_t = 0.999_f32)]
    tolerance: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.tolerance > 0.0 && args.tolerance <= 1.0,
        "tolerance must be in (0, 1], got {}",
        args.tolerance
    );

    let accelerator = build_device(&args.device, args.ordinal)?;
    println!("model:       {}", args.model);
    println!("accelerator: {accelerator:?}");
    println!("reference:   Cpu");
    println!("tolerance:   {:.6}", args.tolerance);
    println!();

    let (accelerator_vectors, accelerator_elapsed) = encode_on(&args.model, &accelerator)?;
    println!("accelerator encode: {accelerator_elapsed:?}");
    let (cpu_vectors, cpu_elapsed) = encode_on(&args.model, &Device::Cpu)?;
    println!("cpu encode:         {cpu_elapsed:?}");
    println!();

    report_speedup(accelerator_elapsed, cpu_elapsed);
    compare(&accelerator_vectors, &cpu_vectors, args.tolerance)
}

/// Builds the named device, erroring instead of falling back to CPU.
fn build_device(name: &str, ordinal: usize) -> Result<Device> {
    match name {
        "cuda" => Device::new_cuda(ordinal)
            .with_context(|| format!("Creating CUDA device at ordinal {ordinal}")),
        "metal" => metal_device_at(ordinal)
            .with_context(|| format!("Creating Metal device at ordinal {ordinal}")),
        "cpu" => Ok(Device::Cpu),
        other => anyhow::bail!("unknown device '{other}'"),
    }
}

/// Opens a Metal device through the library's ordinal guard.
#[cfg(feature = "metal")]
fn metal_device_at(ordinal: usize) -> Result<Device> {
    tessera::metal_device_at(ordinal)
}

/// Reports that Metal was not compiled into this build.
#[cfg(not(feature = "metal"))]
fn metal_device_at(_ordinal: usize) -> Result<Device> {
    anyhow::bail!(
        "this binary was built without the `metal` feature; rebuild with --features metal"
    )
}

/// Loads the model on `device`, encodes [`SENTENCES`], and drops the model.
fn encode_on(model: &str, device: &Device) -> Result<(Vec<Vec<f32>>, Duration)> {
    let embedder = TesseraDense::builder()
        .model(model)
        .device(device.clone())
        .build()
        .with_context(|| format!("Building dense embedder for {model} on {device:?}"))?;

    let started = Instant::now();
    let embeddings = embedder
        .encode_batch(&SENTENCES)
        .with_context(|| format!("Encoding on {device:?}"))?;
    let elapsed = started.elapsed();

    let vectors = embeddings
        .iter()
        .map(|embedding| embedding.values().iter().copied().collect::<Vec<f32>>())
        .collect::<Vec<_>>();
    drop(embedder);
    Ok((vectors, elapsed))
}

fn report_speedup(accelerator: Duration, cpu: Duration) {
    let accelerator_secs = accelerator.as_secs_f64();
    if accelerator_secs > 0.0 {
        println!(
            "cpu / accelerator wall-time ratio: {:.2}x",
            cpu.as_secs_f64() / accelerator_secs
        );
    } else {
        println!("cpu / accelerator wall-time ratio: not reportable (accelerator pass was 0s)");
    }
    println!();
}

/// Fails with the worst offender when any sentence falls under the tolerance.
fn compare(accelerator: &[Vec<f32>], cpu: &[Vec<f32>], tolerance: f32) -> Result<()> {
    anyhow::ensure!(
        accelerator.len() == cpu.len() && accelerator.len() == SENTENCES.len(),
        "expected {} embeddings per device, got {} and {}",
        SENTENCES.len(),
        accelerator.len(),
        cpu.len()
    );

    let mut failures = 0_usize;
    for (index, (left, right)) in accelerator.iter().zip(cpu.iter()).enumerate() {
        anyhow::ensure!(
            left.len() == right.len(),
            "sentence {index}: dimension mismatch, accelerator {} vs cpu {}",
            left.len(),
            right.len()
        );
        let similarity = cosine(left, right)?;
        let verdict = if similarity >= f64::from(tolerance) {
            "ok"
        } else {
            failures += 1;
            "FAIL"
        };
        println!(
            "sentence {index}: cosine {similarity:.8} dim {} [{verdict}]",
            left.len()
        );
    }

    anyhow::ensure!(
        failures == 0,
        "{failures} of {} sentences disagreed across devices beyond tolerance {tolerance}",
        SENTENCES.len()
    );
    println!("\nall {} sentences agree within tolerance", SENTENCES.len());
    Ok(())
}

/// Cosine similarity in f64 so the comparison itself adds no rounding error.
fn cosine(left: &[f32], right: &[f32]) -> Result<f64> {
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (a, b) in left.iter().zip(right.iter()) {
        anyhow::ensure!(
            a.is_finite() && b.is_finite(),
            "non-finite embedding component encountered"
        );
        dot = f64::from(*a).mul_add(f64::from(*b), dot);
        left_norm = f64::from(*a).mul_add(f64::from(*a), left_norm);
        right_norm = f64::from(*b).mul_add(f64::from(*b), right_norm);
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    anyhow::ensure!(
        denominator > 0.0,
        "cannot compare a zero-magnitude embedding"
    );
    Ok(dot / denominator)
}
