use anyhow::{Context, Result};
use candle_core::{IndexOp, Tensor};
use tessera::backends::candle::get_device;
/// Basic time series forecasting with Chronos Bolt
///
/// This example demonstrates:
/// 1. Creating synthetic time series data
/// 2. Loading pre-trained Chronos Bolt model
/// 3. Running forecasts with real pre-trained weights
/// 4. Visualizing full output tensors
///
/// Run with: cargo run --example timeseries_basic_forecasting
use tessera::timeseries::models::ChronosBolt;

fn main() -> Result<()> {
    println!("\n{}\n", "=".repeat(80));
    println!("Chronos Bolt: Basic Time Series Forecasting");
    println!("\n{}\n", "=".repeat(80));

    // Device selection
    let device = get_device().context("Getting compute device")?;
    println!("[Device] Device: {:?}\n", device);

    // 1. Create synthetic time series data
    println!("[Data] Creating synthetic time series...");
    let context_len = 2048; // Chronos Bolt context length
    let batch_size = 4;

    // Generate synthetic seasonal + trend data
    let mut data = Vec::new();
    for batch_idx in 0..batch_size {
        let mut series = Vec::new();
        for t in 0..context_len {
            let trend = (batch_idx + 1) as f32 * 0.01 * t as f32;
            let seasonal = 2.0 * (2.0 * std::f32::consts::PI * t as f32 / 50.0).sin();
            let noise = (t as f32 * 0.1).sin() * 0.3;
            series.push(trend + seasonal + noise);
        }
        data.extend(series);
    }

    // Create tensor: [batch, timesteps]
    let input_tensor = Tensor::from_vec(data.clone(), (batch_size, context_len), &device)?;

    println!("[OK] Input shape: {:?}", input_tensor.shape());
    println!("   Batch size: {}", batch_size);
    println!("   Context length: {}", context_len);

    // Show first few values of first series
    let first_series: Vec<f32> = input_tensor.i((0, ..10))?.to_vec1()?;
    println!("\n   First 10 values of series #1:");
    println!(
        "   {:?}",
        first_series
            .iter()
            .map(|v| format!("{:.3}", v))
            .collect::<Vec<_>>()
    );

    // 2. Load pre-trained Chronos Bolt model
    println!("\n[Loading] Downloading Chronos Bolt from HuggingFace...");
    println!("   Model: amazon/chronos-bolt-small (191 MB)");
    println!("   This may take a few minutes on first run...");

    let mut model = ChronosBolt::from_pretrained("amazon/chronos-bolt-small", &device)?;

    println!("[OK] Model loaded successfully!");
    println!("   Parameters: 48M");
    println!("   Model dimension: {}", model.config.d_model);
    println!("   Context length: {}", model.config.context_length);
    println!("   Prediction length: {}", model.config.prediction_length);

    println!("\n[Note] Using real pre-trained weights from Amazon");
    println!("   (Not random initialization - actual production model)");

    // 3. Run forecast
    println!("\n[Run] Running forecast...");
    let forecast = model.forecast(&input_tensor)?;

    println!("[OK] Forecast complete!");
    println!("   Output shape: {:?}", forecast.shape());
    println!(
        "   Expected: [batch={}, pred_len={}]",
        batch_size, model.config.prediction_length
    );

    // 4. Display full forecast outputs
    println!("\n[Output] FULL FORECAST OUTPUTS (First series):");
    println!("{}", "-".repeat(80));

    let forecast_series_1: Vec<f32> = forecast.i((0, ..))?.to_vec1()?;

    println!("   Forecast for Series #1 (64 timesteps):");
    for (i, chunk) in forecast_series_1.chunks(8).enumerate() {
        let values = chunk
            .iter()
            .map(|v| format!("{:>7.3}", v))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "   [t={:>2}-{:>2}]: [{}]",
            i * 8,
            i * 8 + chunk.len() - 1,
            values
        );
    }

    // Show statistics
    let mean = forecast_series_1.iter().sum::<f32>() / forecast_series_1.len() as f32;
    let min = forecast_series_1
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let max = forecast_series_1
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    println!("\n   Statistics:");
    println!("   • Mean:  {:.3}", mean);
    println!("   • Min:   {:.3}", min);
    println!("   • Max:   {:.3}", max);
    println!("   • Range: {:.3}", max - min);

    // 5. Show all batch forecasts (summary)
    println!("\n[Data] All Batch Forecasts (Summary):");
    println!("{}", "-".repeat(80));

    for batch_idx in 0..batch_size {
        let series: Vec<f32> = forecast.i((batch_idx, ..))?.to_vec1()?;

        let mean = series.iter().sum::<f32>() / series.len() as f32;
        let first_5 = &series[0..5];
        let last_5 = &series[59..64];

        println!(
            "   Series #{}: mean={:.3}, first_5={:?}, last_5={:?}",
            batch_idx + 1,
            mean,
            first_5
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<_>>(),
            last_5
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<_>>(),
        );
    }

    // 6. Get quantile predictions for uncertainty
    println!("\n[Quantiles] Getting quantile predictions for uncertainty...");
    let quantiles = model.predict_quantiles(&input_tensor)?;

    println!("[OK] Quantiles extracted!");
    println!("   Shape: {:?}", quantiles.shape());
    println!(
        "   Expected: [batch={}, pred_len=64, quantiles=9]",
        batch_size
    );

    // Extract 10th, 50th, 90th percentiles for first series
    let q10 = quantiles.i((0, .., 0))?.to_vec1::<f32>()?; // 0.1 quantile
    let q50 = quantiles.i((0, .., 4))?.to_vec1::<f32>()?; // 0.5 quantile (median)
    let q90 = quantiles.i((0, .., 8))?.to_vec1::<f32>()?; // 0.9 quantile

    println!("\n   Uncertainty bands (first 8 steps):");
    println!("   Step | Q10 (10%) | Q50 (Median) | Q90 (90%) | Range");
    println!("   {}", "-".repeat(60));
    for i in 0..8 {
        let range = q90[i] - q10[i];
        println!(
            "   {:>4} | {:>9.2} | {:>12.2} | {:>9.2} | {:>5.2}",
            i, q10[i], q50[i], q90[i], range
        );
    }

    println!("\n{}", "=".repeat(80));
    println!("[Complete] Example complete!");
    println!("{}\n", "=".repeat(80));

    Ok(())
}
