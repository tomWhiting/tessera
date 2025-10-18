use anyhow::{Context, Result};
use candle_core::{IndexOp, Tensor};
use std::f32::consts::PI;
use tessera::backends::candle::get_device;
/// Multivariate time series forecasting with Chronos Bolt
///
/// This example demonstrates:
/// 1. Creating multivariate time series (3 channels: price, volume, sentiment)
/// 2. Loading pre-trained Chronos Bolt model
/// 3. Forecasting each channel independently
/// 4. Visualizing full tensor outputs for all channels
///
/// Note: Chronos Bolt processes univariate series, so we forecast each channel separately
///
/// Run with: cargo run --example timeseries_multivariate
use tessera::timeseries::models::ChronosBolt;

/// Generate realistic multivariate financial time series
/// Returns: (tensor, channel_names)
fn generate_financial_data(
    batch_size: usize,
    context_len: usize,
    device: &candle_core::Device,
) -> Result<(Tensor, Vec<String>)> {
    let num_channels = 3;
    let mut all_data = Vec::new();

    for batch_idx in 0..batch_size {
        // Channel 1: Stock Price (trending with cycles)
        for t in 0..context_len {
            let trend = 100.0 + (batch_idx + 1) as f32 * 10.0 + t as f32 * 0.05;
            let cycle = 5.0 * (2.0 * PI * t as f32 / 50.0).sin();
            let noise = (t as f32 * 0.17).sin() * 2.0;
            let price = trend + cycle + noise;
            all_data.push(price);
        }

        // Channel 2: Trading Volume (spiky with trend)
        for t in 0..context_len {
            let base_vol = 1000.0 + (batch_idx as f32 + 1.0) * 200.0;
            let spike = if t % 20 < 5 { 500.0 } else { 0.0 };
            let trend = t as f32 * 0.5;
            let noise = (t as f32 * 0.23).sin() * 100.0;
            let volume = base_vol + spike + trend + noise;
            all_data.push(volume);
        }

        // Channel 3: Market Sentiment (-1 to 1, leading indicator)
        for t in 0..context_len {
            let cycle = (2.0 * PI * t as f32 / 40.0).sin();
            let noise = (t as f32 * 0.31).sin() * 0.2;
            let sentiment = cycle + noise;
            all_data.push(sentiment.clamp(-1.0, 1.0));
        }
    }

    let channel_names = vec![
        "Stock Price ($)".to_string(),
        "Trading Volume (1000s)".to_string(),
        "Market Sentiment (-1 to 1)".to_string(),
    ];

    let tensor = Tensor::from_vec(all_data, (batch_size, num_channels, context_len), device)?;

    Ok((tensor, channel_names))
}

fn main() -> Result<()> {
    println!("\n{}\n", "=".repeat(80));
    println!("Chronos Bolt: Multivariate Time Series Forecasting");
    println!("\n{}\n", "=".repeat(80));

    // Setup
    let device = get_device().context("Getting compute device")?;
    println!("[Device] Device: {:?}\n", device);

    let batch_size = 2; // 2 different stocks
    let context_len = 2048; // Chronos Bolt context length
    let num_channels = 3;

    // 1. Generate multivariate financial data
    println!("[Data] Generating multivariate financial time series...\n");
    let (input_tensor, channel_names) = generate_financial_data(batch_size, context_len, &device)?;

    println!(
        "[OK] Generated {} stocks with {} channels each",
        batch_size, num_channels
    );
    println!("   Context length: {}\n", context_len);

    println!("   Channels:");
    for (i, name) in channel_names.iter().enumerate() {
        println!("   {} - {}", i + 1, name);
    }

    // Show sample values from first stock
    println!("\n   Sample values from Stock #1 (first 10 timesteps):\n");
    for (ch_idx, ch_name) in channel_names.iter().enumerate() {
        let values: Vec<f32> = input_tensor.i((0, ch_idx, ..10))?.to_vec1()?;
        let formatted = values
            .iter()
            .map(|v| format!("{:>7.2}", v))
            .collect::<Vec<_>>()
            .join(", ");
        println!("   {:<25}: [{}]", ch_name, formatted);
    }

    // 2. Load pre-trained Chronos Bolt model
    println!("\n[Loading] Downloading Chronos Bolt from HuggingFace...");
    println!("   Model: amazon/chronos-bolt-small (191 MB)");

    let mut model = ChronosBolt::from_pretrained("amazon/chronos-bolt-small", &device)?;

    println!("[OK] Model loaded successfully!");
    println!("   Parameters: 48M");
    println!("   Prediction length: {}\n", model.config.prediction_length);

    println!("[Note] Chronos Bolt processes univariate series");
    println!("   Forecasting each channel independently...\n");

    // 3. Forecast each channel independently
    println!("[Run] Running multivariate forecast...");

    // Reshape to process all channels as separate series
    // From [batch=2, channels=3, time=2048] to [batch=6, time=2048]
    let reshaped = input_tensor.reshape((batch_size * num_channels, context_len))?;

    let forecast = model.forecast(&reshaped)?;

    // Reshape back to [batch=2, channels=3, pred_len=64]
    let forecast = forecast.reshape((batch_size, num_channels, model.config.prediction_length))?;

    println!("[OK] Forecast complete!");
    println!("   Output shape: {:?}", forecast.shape());
    println!(
        "   [batch={}, channels={}, pred_len={}]\n",
        batch_size, num_channels, model.config.prediction_length
    );

    // 4. Display FULL forecast for Stock #1, all channels
    println!(
        "[Output] FULL FORECAST for Stock #1 (all {} channels)",
        num_channels
    );
    println!("{}", "-".repeat(80));
    println!();

    for (ch_idx, ch_name) in channel_names.iter().enumerate() {
        println!("   Channel: {}", ch_name);
        println!("   {}", "-".repeat(78));

        let channel_forecast: Vec<f32> = forecast.i((0, ch_idx, ..))?.to_vec1()?;

        // Display in chunks of 8 for readability
        for (i, chunk) in channel_forecast.chunks(8).enumerate() {
            let values = chunk
                .iter()
                .map(|v| format!("{:>8.2}", v))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "   [t={:>2}-{:>2}]: {}",
                i * 8,
                i * 8 + chunk.len() - 1,
                values
            );
        }

        // Statistics
        let mean = channel_forecast.iter().sum::<f32>() / channel_forecast.len() as f32;
        let min = channel_forecast
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b));
        let max = channel_forecast
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        println!("\n   Statistics:");
        println!("   • Mean:  {:>8.2}", mean);
        println!("   • Min:   {:>8.2}", min);
        println!("   • Max:   {:>8.2}", max);
        println!("   • Range: {:>8.2}\n", max - min);
    }

    // 5. Compare both stocks (summary)
    println!("[Data] Comparison: Stock #1 vs Stock #2");
    println!("{}", "-".repeat(80));
    println!();

    for (ch_idx, ch_name) in channel_names.iter().enumerate() {
        println!("   {}:", ch_name);

        for stock_idx in 0..batch_size {
            let forecast_data: Vec<f32> = forecast.i((stock_idx, ch_idx, ..))?.to_vec1()?;

            let mean = forecast_data.iter().sum::<f32>() / forecast_data.len() as f32;
            let first_val = forecast_data[0];
            let last_val = forecast_data[forecast_data.len() - 1];
            let change = last_val - first_val;

            println!(
                "      Stock #{}: mean={:>7.2}, start={:>7.2}, end={:>7.2}, change={:>+7.2}",
                stock_idx + 1,
                mean,
                first_val,
                last_val,
                change
            );
        }
        println!();
    }

    println!("{}", "=".repeat(80));
    println!("[Complete] Example complete!");
    println!("{}", "=".repeat(80));
    println!();

    println!("[Insight] Key Features of Multivariate Forecasting:");
    println!("   • Chronos Bolt forecasts each channel independently");
    println!("   • Each channel gets its own forecast (price, volume, sentiment)");
    println!("   • Pre-trained weights work across different variable types");
    println!("   • Useful for: financial data, IoT sensors, weather, medical vitals");
    println!();

    println!("[Data] Use Cases:");
    println!("   • Stock trading: price + volume + sentiment");
    println!("   • IoT sensors: temperature + humidity + pressure");
    println!("   • Medical: heart rate + blood pressure + oxygen saturation");
    println!("   • Manufacturing: machine temp + vibration + power consumption");
    println!();

    Ok(())
}
