/// Comprehensive Tessera TimeSeries Integration Demo
///
/// This example demonstrates the integrated Chronos Bolt time series forecasting
/// through the main Tessera API, showcasing:
/// 1. Factory pattern: Tessera::new() with automatic model detection
/// 2. Builder pattern: TesseraTimeSeries::builder() for custom configuration
/// 3. Point forecasting (median prediction)
/// 4. Probabilistic forecasting (full quantile predictions)
/// 5. Model introspection (context/prediction lengths, quantiles)
///
/// Run with: cargo run --example timeseries_integration
use anyhow::{Context, Result};
use candle_core::{IndexOp, Tensor};
use tessera::backends::candle::get_device;
use tessera::{Tessera, TesseraTimeSeries};

fn main() -> Result<()> {
    println!("\n{}\n", "=".repeat(80));
    println!("Tessera Time Series Integration Demo");
    println!("Chronos Bolt: Production Time Series Forecasting");
    println!("\n{}\n", "=".repeat(80));

    // Get device once and use consistently throughout
    let device = get_device()?;

    // ========================================================================
    // Part 1: Factory Pattern - Automatic Model Detection
    // ========================================================================

    println!("\n[Part 1] Factory Pattern: Tessera::new()");
    println!("{}", "-".repeat(80));

    println!("\n[Creating] Using factory pattern with auto-detection...");
    let embedder =
        Tessera::new("chronos-bolt-small").context("Failed to create Tessera instance")?;

    // Pattern match to get the TimeSeries variant
    let mut forecaster = match embedder {
        Tessera::TimeSeries(ts) => {
            println!("[✓] Successfully created TimeSeries variant!");
            println!("    Model: {}", ts.model());
            println!("    Context length: {}", ts.context_length());
            println!("    Prediction length: {}", ts.prediction_length());
            ts
        }
        _ => {
            anyhow::bail!("Expected TimeSeries variant, got different type");
        }
    };

    // ========================================================================
    // Part 2: Builder Pattern - Custom Configuration
    // ========================================================================

    println!("\n[Part 2] Builder Pattern: TesseraTimeSeries::builder()");
    println!("{}", "-".repeat(80));

    println!("\n[Creating] Using builder pattern with explicit device...");
    let forecaster_explicit = TesseraTimeSeries::builder()
        .model("chronos-bolt-small")
        .device(device.clone())
        .build()
        .context("Failed to build TimeSeries forecaster")?;

    println!("[✓] Successfully created forecaster with explicit device!");
    println!("    Quantiles: {:?}", forecaster_explicit.quantiles());

    // ========================================================================
    // Part 3: Generate Synthetic Time Series Data
    // ========================================================================

    println!("\n[Part 3] Synthetic Time Series Generation");
    println!("{}", "-".repeat(80));

    let context_len = 2048;
    let batch_size = 1;

    println!("\n[Data] Creating synthetic time series with trend and seasonality...");
    let mut data = Vec::new();
    for t in 0..context_len {
        let trend = 0.05 * t as f32;
        let seasonal = 3.0 * (2.0 * std::f32::consts::PI * t as f32 / 50.0).sin();
        let noise = (t as f32 * 0.13).sin() * 0.5;
        data.push(trend + seasonal + noise + 10.0);
    }

    let input = Tensor::from_vec(data, (batch_size, context_len), &device)?;

    println!("[✓] Input shape: {:?}", input.shape());

    // Show last 5 values
    let last_5: Vec<f32> = input.i((0, (context_len - 5)..))?.to_vec1()?;
    println!("\n   Last 5 context values:");
    println!(
        "   {:?}",
        last_5
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    // ========================================================================
    // Part 4: Point Forecasting (Median Prediction)
    // ========================================================================

    println!("\n[Part 4] Point Forecasting - Median Prediction");
    println!("{}", "-".repeat(80));

    println!("\n[Forecast] Generating point forecast (median)...");
    let point_forecast = forecaster
        .forecast(&input)
        .context("Failed to generate point forecast")?;

    println!("[✓] Forecast shape: {:?}", point_forecast.shape());

    let forecast_vals: Vec<f32> = point_forecast.i(0)?.to_vec1()?;
    println!("\n   First 10 forecasted values:");
    println!(
        "   {:?}",
        forecast_vals
            .iter()
            .take(10)
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    let mean_forecast: f32 = forecast_vals.iter().sum::<f32>() / forecast_vals.len() as f32;
    println!("\n   Average forecast value: {:.2}", mean_forecast);

    // ========================================================================
    // Part 5: Probabilistic Forecasting (Full Quantiles)
    // ========================================================================

    println!("\n[Part 5] Probabilistic Forecasting - Quantile Predictions");
    println!("{}", "-".repeat(80));

    println!("\n[Forecast] Generating all quantile predictions...");
    let quantiles = forecaster
        .forecast_quantiles(&input)
        .context("Failed to generate quantile predictions")?;

    println!("[✓] Quantiles shape: {:?}", quantiles.shape());
    println!("    Expected: [batch=1, pred_len=64, quantiles=9]");

    // Extract specific quantiles
    let q10 = quantiles.i((0, .., 0))?.to_vec1::<f32>()?; // 10th percentile
    let q50 = quantiles.i((0, .., 4))?.to_vec1::<f32>()?; // Median
    let q90 = quantiles.i((0, .., 8))?.to_vec1::<f32>()?; // 90th percentile

    println!("\n   Prediction intervals for first 8 timesteps:\n");
    println!("   Step | Q10 (10%) | Q50 (Median) | Q90 (90%) | 80% Interval");
    println!("   {}", "-".repeat(65));
    for i in 0..8 {
        let interval = q90[i] - q10[i];
        println!(
            "   {:>4} | {:>9.2} | {:>12.2} | {:>9.2} | {:>12.2}",
            i, q10[i], q50[i], q90[i], interval
        );
    }

    // Calculate uncertainty metrics
    let mean_interval: f32 = q90
        .iter()
        .zip(&q10)
        .map(|(high, low)| high - low)
        .sum::<f32>()
        / q90.len() as f32;

    println!("\n   Average 80% prediction interval: {:.2}", mean_interval);

    // ========================================================================
    // Part 6: Model Introspection
    // ========================================================================

    println!("\n[Part 6] Model Introspection & Metadata");
    println!("{}", "-".repeat(80));

    println!("\n   Model ID: {}", forecaster.model());
    println!(
        "   Context Length: {} timesteps",
        forecaster.context_length()
    );
    println!(
        "   Prediction Length: {} timesteps",
        forecaster.prediction_length()
    );
    println!("   Quantile Levels: {:?}", forecaster.quantiles());

    println!(
        "\n   Forecast horizon: {} steps ahead",
        forecaster.prediction_length()
    );
    println!(
        "   Required input: {} historical observations",
        forecaster.context_length()
    );

    // ========================================================================
    // Part 7: Verify Point Forecast Matches Median Quantile
    // ========================================================================

    println!("\n[Part 7] Verification - Point Forecast vs Median Quantile");
    println!("{}", "-".repeat(80));

    let max_diff = forecast_vals
        .iter()
        .zip(&q50)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("\n   Maximum difference: {:.6}", max_diff);
    if max_diff < 0.001 {
        println!("   [✓] Point forecast correctly matches median quantile!");
    } else {
        println!("   [!] Warning: Point forecast differs from median");
    }

    // ========================================================================
    // Summary
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("[Complete] Integration Demonstration Complete!");
    println!("{}", "=".repeat(80));

    println!("\n[Summary] Key Integration Points:");
    println!("   ✓ Factory pattern: Tessera::new() auto-detects time series models");
    println!("   ✓ Builder pattern: TesseraTimeSeries::builder() for custom config");
    println!("   ✓ Point forecasting: forecast() returns median prediction");
    println!("   ✓ Probabilistic: forecast_quantiles() returns all 9 quantiles");
    println!("   ✓ Introspection: Access to model metadata and configuration");
    println!("   ✓ Consistent API: Same patterns as Dense/Sparse/Vision embedders");

    println!("\n[Usage Examples]:");
    println!("   // Simple factory pattern (auto-detects device)");
    println!("   let forecaster = Tessera::new(\"chronos-bolt-small\")?;");
    println!();
    println!("   // Builder with explicit device (use get_device() for consistency)");
    println!("   let device = get_device()?;");
    println!("   let forecaster = TesseraTimeSeries::builder()");
    println!("       .model(\"chronos-bolt-small\")");
    println!("       .device(device.clone())");
    println!("       .build()?;");
    println!();

    Ok(())
}
