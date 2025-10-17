/// Test Chronos Bolt quantile predictions with pre-trained weights
///
/// This example demonstrates:
/// 1. Loading pre-trained Chronos Bolt model
/// 2. Getting full quantile predictions (all 9 quantiles)
/// 3. Extracting specific quantiles (10th, 50th, 90th percentile)
/// 4. Visualizing uncertainty through quantile ranges
///
/// Run with: cargo run --example chronos_bolt_quantiles

use tessera::timeseries::models::ChronosBolt;
use tessera::backends::candle::get_device;
use candle_core::{Tensor, IndexOp};
use anyhow::{Result, Context};

fn main() -> Result<()> {
    println!("\n{}\n", "=".repeat(80));
    println!("Chronos Bolt: Quantile Predictions Demo");
    println!("\n{}\n", "=".repeat(80));

    // Device selection
    let device = get_device().context("Getting compute device")?;
    println!("[Device] Using device: {:?}\n", device);

    // 1. Create synthetic time series with trend
    println!("[Data] Creating synthetic time series with upward trend...");
    let context_len = 2048;
    let batch_size = 1;

    let mut data = Vec::new();
    for t in 0..context_len {
        let trend = 0.05 * t as f32;
        let seasonal = 3.0 * (2.0 * std::f32::consts::PI * t as f32 / 50.0).sin();
        let noise = (t as f32 * 0.13).sin() * 0.5;
        data.push(trend + seasonal + noise + 10.0);
    }

    let input = Tensor::from_vec(data, (batch_size, context_len), &device)?;
    println!("[OK] Input shape: {:?}", input.shape());

    // Show context statistics
    let last_10: Vec<f32> = input.i((0, (context_len-10)..))?.to_vec1()?;
    println!("\n   Last 10 values of context:");
    println!("   {:?}", last_10.iter()
        .map(|v| format!("{:.2}", v))
        .collect::<Vec<_>>());

    // 2. Load pre-trained model
    println!("\n[Loading] Downloading Chronos Bolt from HuggingFace...");
    println!("   Model: amazon/chronos-bolt-small (191 MB)");

    let mut model = ChronosBolt::from_pretrained("amazon/chronos-bolt-small", &device)?;

    println!("[OK] Model loaded successfully!");
    println!("   Quantiles: {:?}", model.config.quantiles);
    println!("   Prediction length: {}", model.config.prediction_length);

    // 3. Get full quantile predictions
    println!("\n[Predict] Running quantile predictions...");
    let quantiles = model.predict_quantiles(&input)?;

    println!("[OK] Quantile predictions complete!");
    println!("   Output shape: {:?}", quantiles.shape());
    println!("   Expected: [batch=1, pred_len=64, quantiles=9]");

    // 4. Extract specific quantiles
    println!("\n[Extract] Extracting specific quantiles:");
    println!("{}", "-".repeat(80));

    let q10 = quantiles.i((0, .., 0))?.to_vec1::<f32>()?;  // 0.1 quantile (10th percentile)
    let q50 = quantiles.i((0, .., 4))?.to_vec1::<f32>()?;  // 0.5 quantile (median)
    let q90 = quantiles.i((0, .., 8))?.to_vec1::<f32>()?;  // 0.9 quantile (90th percentile)

    println!("\n   Quantile predictions for first 8 steps:\n");
    println!("   Step | Q10 (10%) | Q50 (Median) | Q90 (90%) | Range");
    println!("   {}", "-".repeat(60));
    for i in 0..8 {
        let range = q90[i] - q10[i];
        println!("   {:>4} | {:>9.2} | {:>12.2} | {:>9.2} | {:>5.2}",
            i, q10[i], q50[i], q90[i], range);
    }

    // 5. Calculate uncertainty metrics
    println!("\n[Uncertainty] Forecast Uncertainty Metrics:");
    println!("{}", "-".repeat(80));

    let mean_range: f32 = q90.iter().zip(&q10)
        .map(|(high, low)| high - low)
        .sum::<f32>() / q90.len() as f32;

    let mean_q50: f32 = q50.iter().sum::<f32>() / q50.len() as f32;
    let mean_q10: f32 = q10.iter().sum::<f32>() / q10.len() as f32;
    let mean_q90: f32 = q90.iter().sum::<f32>() / q90.len() as f32;

    println!("\n   Average 80% prediction interval: {:.2}", mean_range);
    println!("   Average median forecast: {:.2}", mean_q50);
    println!("   Average 10th percentile: {:.2}", mean_q10);
    println!("   Average 90th percentile: {:.2}", mean_q90);

    // 6. Compare with point forecast
    println!("\n[Compare] Point Forecast vs Full Quantiles:");
    println!("{}", "-".repeat(80));

    let point_forecast = model.forecast(&input)?;
    let point_vals: Vec<f32> = point_forecast.i(0)?.to_vec1()?;

    println!("\n   Point forecast (median) for first 8 steps:");
    println!("   {:?}", point_vals.iter().take(8)
        .map(|v| format!("{:.2}", v))
        .collect::<Vec<_>>());

    // Verify point forecast matches median from quantiles
    println!("\n   Verifying point forecast matches median quantile:");
    let max_diff = point_vals.iter().zip(&q50)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, |a, b| a.max(b));
    println!("   Max difference: {:.6}", max_diff);
    if max_diff < 0.001 {
        println!("   [✓] Point forecast matches median quantile!");
    } else {
        println!("   [!] Warning: Point forecast differs from median");
    }

    // 7. All quantiles summary
    println!("\n[Summary] All Quantile Levels at t=32:");
    println!("{}", "-".repeat(80));

    let t = 32;
    println!("\n   Quantile | Value");
    println!("   {}", "-".repeat(25));
    for (idx, &q_level) in model.config.quantiles.iter().enumerate() {
        let q_val: f32 = quantiles.i((0, t, idx))?.to_scalar()?;
        println!("   {:>8.1} | {:>8.2}", q_level * 100.0, q_val);
    }

    println!("\n{}", "=".repeat(80));
    println!("[Complete] Quantile predictions demonstration complete!");
    println!("{}\n", "=".repeat(80));

    println!("[Note] Key Features:");
    println!("   - Full quantile predictions: [batch, 64, 9]");
    println!("   - Point forecast extracts median (0.5 quantile)");
    println!("   - Uncertainty quantification via prediction intervals");
    println!("   - All 9 quantiles: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9");
    println!();

    Ok(())
}
