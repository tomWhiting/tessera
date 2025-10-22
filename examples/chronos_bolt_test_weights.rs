//! Test Chronos Bolt with real pre-trained weights from HuggingFace.
//!
//! This example downloads and tests pre-trained weights for time series forecasting.

use anyhow::{Context, Result};
use candle_core::{IndexOp, Tensor};
use tessera::backends::candle::get_device;
///
/// This example:
/// 1. Downloads pre-trained weights from amazon/chronos-bolt-small
/// 2. Loads them into ChronosBolt model
/// 3. Runs inference on synthetic time series
/// 4. Displays full quantile predictions
///
/// Run with: cargo run --example chronos_bolt_test_weights
use tessera::timeseries::models::ChronosBolt;

fn main() -> Result<()> {
    println!("\n{}\n", "=".repeat(80));
    println!("Chronos Bolt: Testing with Pre-trained Weights");
    println!("\n{}\n", "=".repeat(80));

    // Device selection
    let device = get_device().context("Getting compute device")?;
    println!("[Device] Using device: {:?}\n", device);

    // 1. Create synthetic time series
    println!("[Data] Creating synthetic time series...");
    let context_len = 2048; // Chronos Bolt context length
    let batch_size = 2;

    let mut data = Vec::new();
    for batch_idx in 0..batch_size {
        let mut series = Vec::new();
        for t in 0..context_len {
            let trend = (batch_idx + 1) as f32 * 0.02 * t as f32;
            let seasonal = 2.0 * (2.0 * std::f32::consts::PI * t as f32 / 50.0).sin();
            let noise = (t as f32 * 0.13).sin() * 0.5;
            series.push(trend + seasonal + noise + 10.0);
        }
        data.extend(series);
    }

    let input = Tensor::from_vec(data, (batch_size, context_len), &device)?;
    println!("[OK] Input shape: {:?}", input.shape());

    // Show first few values
    let first_10: Vec<f32> = input.i((0, ..10))?.to_vec1()?;
    println!("\n   First 10 values of series #1:");
    println!(
        "   {:?}",
        first_10
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    // 2. Load pre-trained model from HuggingFace
    println!("\n[Loading] Downloading Chronos Bolt from HuggingFace...");
    println!("   Model: amazon/chronos-bolt-small (191 MB)");
    println!("   This may take a few minutes on first run...");

    let mut model = ChronosBolt::from_pretrained("amazon/chronos-bolt-small", &device)?;

    println!("[OK] Model loaded successfully!");
    println!("   Parameters: 48M");
    println!("   d_model: {}", model.config.d_model);
    println!("   Context length: {}", model.config.context_length);
    println!("   Prediction length: {}", model.config.prediction_length);
    println!("   Quantiles: {:?}", model.config.quantiles);

    // 3. Run forecast
    println!("\n[Run] Running forecast...");
    let forecast = model.forecast(&input)?;

    println!("[OK] Forecast complete!");
    println!("   Output shape: {:?}", forecast.shape());
    println!(
        "   Expected: [batch={}, pred_len={}]",
        batch_size, model.config.prediction_length
    );

    // 4. Display full forecast for both series
    println!("\n[Output] FULL FORECAST OUTPUT");
    println!("{}", "-".repeat(80));

    let forecast_series1: Vec<f32> = forecast.i((0, ..))?.to_vec1()?;

    println!("   Full 64-step forecast for Series #1:\n");
    for (i, chunk) in forecast_series1.chunks(8).enumerate() {
        let values = chunk
            .iter()
            .map(|v| format!("{:>7.2}", v))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "   [t={:>2}-{:>2}]: {}",
            i * 8,
            (i * 8 + chunk.len() - 1).min(63),
            values
        );
    }

    // 5. Statistics
    let mean = forecast_series1.iter().sum::<f32>() / forecast_series1.len() as f32;
    let min = forecast_series1
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let max = forecast_series1
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    println!("\n   Statistics:");
    println!("   Mean:  {:.2}", mean);
    println!("   Min:   {:.2}", min);
    println!("   Max:   {:.2}", max);
    println!("   Range: {:.2}", max - min);

    // 6. Compare both series
    println!("\n[Compare] Series #1 vs Series #2 Forecasts:");
    println!("{}", "-".repeat(80));

    for series_idx in 0..batch_size {
        let series_forecast: Vec<f32> = forecast.i((series_idx, ..))?.to_vec1()?;
        let mean = series_forecast.iter().sum::<f32>() / series_forecast.len() as f32;
        let first = series_forecast[0];
        let last = series_forecast[series_forecast.len() - 1];

        println!(
            "   Series #{}: mean={:.2}, start={:.2}, end={:.2}, change={:+.2}",
            series_idx + 1,
            mean,
            first,
            last,
            last - first
        );
    }

    println!("\n{}", "=".repeat(80));
    println!("[Complete] Test complete!");
    println!("{}\n", "=".repeat(80));

    println!("[Note] Implementation Status:");
    println!("   - Full ResidualMLP architecture with exposed T5Stack");
    println!("   - Input patch embedding: ResidualMLP (32 -> 2048 -> 512)");
    println!("   - Output patch embedding: ResidualMLP (512 -> 2048 -> 576)");
    println!("   - Output structure: 64 prediction steps × 9 quantiles = 576 dims");
    println!("   - T5 encoder/decoder stacks with continuous embeddings");
    println!("   - NO quantization - uses continuous patch embeddings directly");
    println!("   - forecast() returns median (0.5 quantile) predictions");
    println!("   - predict_quantiles() returns all 9 quantiles for uncertainty");
    println!("   - All weights loaded from amazon/chronos-bolt-small");
    println!();

    Ok(())
}
