use anyhow::{Context, Result};
use candle_core::{IndexOp, Tensor};
use std::f32::consts::PI;
use tessera::backends::candle::get_device;
/// Time series similarity search using Chronos Bolt forecasts
///
/// This example demonstrates:
/// 1. Creating different time series patterns (trend, seasonal, random)
/// 2. Using forecasts as representations for similarity
/// 3. Computing similarity scores based on forecast behavior
/// 4. Visualizing full forecast vectors
///
/// Note: Chronos Bolt doesn't have dedicated embedding extraction,
/// but we can use forecast patterns as a form of similarity measure
///
/// Run with: cargo run --example timeseries_similarity_search
use tessera::timeseries::models::ChronosBolt;

/// Generate different time series patterns
fn generate_patterns(
    context_len: usize,
    device: &candle_core::Device,
) -> Result<(Tensor, Vec<String>)> {
    let mut all_series = Vec::new();
    let mut labels = Vec::new();

    // Pattern 1: Strong upward trend
    labels.push("Strong Upward Trend".to_string());
    let mut series = Vec::new();
    for t in 0..context_len {
        series.push(t as f32 * 0.05 + (t as f32 * 0.01).sin());
    }
    all_series.extend(series);

    // Pattern 2: Strong downward trend
    labels.push("Strong Downward Trend".to_string());
    let mut series = Vec::new();
    for t in 0..context_len {
        series.push(25.0 - t as f32 * 0.05 + (t as f32 * 0.01).sin());
    }
    all_series.extend(series);

    // Pattern 3: Seasonal (fast oscillation)
    labels.push("Fast Seasonal (period=20)".to_string());
    let mut series = Vec::new();
    for t in 0..context_len {
        series.push(5.0 * (2.0 * PI * t as f32 / 20.0).sin());
    }
    all_series.extend(series);

    // Pattern 4: Seasonal (slow oscillation)
    labels.push("Slow Seasonal (period=100)".to_string());
    let mut series = Vec::new();
    for t in 0..context_len {
        series.push(5.0 * (2.0 * PI * t as f32 / 100.0).sin());
    }
    all_series.extend(series);

    // Pattern 5: Trend + Seasonal (combined)
    labels.push("Trend + Seasonal".to_string());
    let mut series = Vec::new();
    for t in 0..context_len {
        let trend = t as f32 * 0.02;
        let seasonal = 3.0 * (2.0 * PI * t as f32 / 50.0).sin();
        series.push(trend + seasonal);
    }
    all_series.extend(series);

    // Pattern 6: Another trend + seasonal (should be similar to #5)
    labels.push("Trend + Seasonal (similar)".to_string());
    let mut series = Vec::new();
    for t in 0..context_len {
        let trend = t as f32 * 0.025 + 1.0; // Slightly different trend
        let seasonal = 2.5 * (2.0 * PI * t as f32 / 48.0).sin(); // Slightly different period
        series.push(trend + seasonal);
    }
    all_series.extend(series);

    // Pattern 7: Random walk
    labels.push("Random Walk".to_string());
    let mut series = Vec::new();
    let mut value = 10.0;
    for t in 0..context_len {
        value += (t as f32 * 0.1).sin() * 0.5 - 0.1;
        series.push(value);
    }
    all_series.extend(series);

    // Pattern 8: Flat with noise
    labels.push("Flat with Noise".to_string());
    let mut series = Vec::new();
    for t in 0..context_len {
        series.push(10.0 + (t as f32 * 0.3).sin() * 0.5);
    }
    all_series.extend(series);

    let batch_size = labels.len();
    let tensor = Tensor::from_vec(all_series, (batch_size, context_len), device)?;

    Ok((tensor, labels))
}

fn main() -> Result<()> {
    println!("\n{}\n", "=".repeat(80));
    println!("Time Series Similarity Search with Chronos Bolt Forecasts");
    println!("\n{}\n", "=".repeat(80));

    // Setup
    let device = get_device().context("Getting compute device")?;
    println!("[Device] Device: {:?}\n", device);

    let context_len = 2048; // Chronos Bolt context length

    // 1. Generate diverse time series patterns
    println!("[Data] Generating 8 different time series patterns...\n");
    let (input_tensor, labels) = generate_patterns(context_len, &device)?;

    let batch_size = labels.len();
    println!("[OK] Generated {} time series", batch_size);
    println!("   Context length: {}\n", context_len);

    // Show pattern descriptions
    println!("   Patterns:");
    for (i, label) in labels.iter().enumerate() {
        println!("   {} - {}", i + 1, label);
    }

    // 2. Load pre-trained model
    println!("\n[Loading] Downloading Chronos Bolt from HuggingFace...");
    println!("   Model: amazon/chronos-bolt-small (191 MB)");

    let mut model = ChronosBolt::from_pretrained("amazon/chronos-bolt-small", &device)?;

    println!("[OK] Model loaded!");
    println!("   Prediction length: 64\n");

    // 3. Generate forecasts as representations
    println!("[Forecast] Generating forecasts for similarity analysis...");
    let forecasts = model.forecast(&input_tensor)?;

    println!("[OK] Forecasts generated!");
    println!("   Shape: {:?}\n", forecasts.shape());

    // 4. Display FULL forecast vector for first pattern
    println!("[Vector] FULL FORECAST VECTOR for Pattern #1: 'Strong Upward Trend'");
    println!("{}", "-".repeat(80));

    let forecast_full: Vec<f32> = forecasts.i((0, ..))?.to_vec1()?;

    println!("   64-dimensional forecast vector:\n");
    for (i, chunk) in forecast_full.chunks(8).enumerate() {
        let values = chunk
            .iter()
            .map(|v| format!("{:>8.4}", v))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "   [t {:>2}-{:>2}]: {}",
            i * 8,
            (i * 8) + chunk.len() - 1,
            values
        );
    }

    // Compute stats
    let mean = forecast_full.iter().sum::<f32>() / forecast_full.len() as f32;
    let variance = forecast_full
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / forecast_full.len() as f32;
    let std_dev = variance.sqrt();

    println!("\n   Forecast statistics:");
    println!("   • Mean:    {:>8.4}", mean);
    println!("   • Std Dev: {:>8.4}", std_dev);
    println!(
        "   • Min:     {:>8.4}",
        forecast_full.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "   • Max:     {:>8.4}",
        forecast_full
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    // 5. Compute pairwise similarities based on forecasts
    println!("\n\n[Compute] Computing pairwise cosine similarities...\n");

    let mut sim_matrix = vec![vec![0.0; batch_size]; batch_size];

    for i in 0..batch_size {
        for j in 0..batch_size {
            let forecast_i = forecasts.i(i)?;
            let forecast_j = forecasts.i(j)?;

            let dot = (&forecast_i * &forecast_j)?.sum_all()?.to_scalar::<f32>()?;
            let norm_i = forecast_i.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            let norm_j = forecast_j.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            let sim = dot / (norm_i * norm_j);

            sim_matrix[i][j] = sim;
        }
    }

    // Display similarity matrix
    println!("[Data] FULL SIMILARITY MATRIX:");
    println!("{}", "-".repeat(80));
    println!("   (Cosine similarity: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)\n");

    // Header
    print!("        ");
    for i in 0..batch_size {
        print!("   P{:<2}", i + 1);
    }
    println!();

    print!("        ");
    for _ in 0..batch_size {
        print!("  -----");
    }
    println!();

    // Rows
    for i in 0..batch_size {
        print!("   P{:<2} |", i + 1);
        for j in 0..batch_size {
            let sim = sim_matrix[i][j];
            if i == j {
                print!("  {:.3}", sim); // Diagonal (self-similarity)
            } else if sim > 0.8 {
                print!("  \x1b[32m{:.3}\x1b[0m", sim); // High similarity (green)
            } else if sim < -0.5 {
                print!("  \x1b[31m{:.3}\x1b[0m", sim); // Negative similarity (red)
            } else {
                print!("  {:.3}", sim); // Normal
            }
        }
        println!("  | {}", labels[i]);
    }

    println!();

    // 6. Find most similar pairs
    println!("[Search] Most Similar Pattern Pairs:");
    println!("{}", "-".repeat(80));
    println!();

    let mut similarities = Vec::new();
    for i in 0..batch_size {
        for j in (i + 1)..batch_size {
            similarities.push((i, j, sim_matrix[i][j]));
        }
    }

    // Sort by similarity (descending)
    similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Show top 5 most similar pairs
    for (rank, (i, j, sim)) in similarities.iter().take(5).enumerate() {
        println!("   {}. Similarity: {:.4}", rank + 1, sim);
        println!("      Pattern {}: {}", i + 1, labels[*i]);
        println!("      Pattern {}: {}", j + 1, labels[*j]);
        println!();
    }

    // 7. Find most dissimilar pairs
    println!("[Search] Most Dissimilar Pattern Pairs:");
    println!("{}", "-".repeat(80));
    println!();

    // Show bottom 3 (most dissimilar)
    for (rank, (i, j, sim)) in similarities.iter().rev().take(3).enumerate() {
        println!("   {}. Similarity: {:.4}", rank + 1, sim);
        println!("      Pattern {}: {}", i + 1, labels[*i]);
        println!("      Pattern {}: {}", j + 1, labels[*j]);
        println!();
    }

    // 8. Similarity search example
    println!("[Extract] Similarity Search: Find patterns similar to 'Trend + Seasonal' (#5)");
    println!("{}", "-".repeat(80));
    println!();

    let query_idx = 4; // "Trend + Seasonal"
    let mut ranked = Vec::new();
    for i in 0..batch_size {
        if i != query_idx {
            ranked.push((i, sim_matrix[query_idx][i]));
        }
    }

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!(
        "   Query: Pattern {} - {}\n",
        query_idx + 1,
        labels[query_idx]
    );
    println!("   Top 3 most similar patterns:\n");

    for (rank, (i, sim)) in ranked.iter().take(3).enumerate() {
        println!("   {}. Pattern {} ({:.4} similarity)", rank + 1, i + 1, sim);
        println!("      {}", labels[*i]);
        println!();
    }

    println!("{}", "=".repeat(80));
    println!("[Complete] Example complete!");
    println!("{}", "=".repeat(80));
    println!();

    println!("[Insight] Key Insights:");
    println!("   • Time series with similar forecast behavior have high similarity");
    println!("   • Chronos Bolt forecasts capture temporal patterns and dynamics");
    println!("   • Can be used for clustering, classification, anomaly detection");
    println!("   • Forecast-based similarity works for pattern matching tasks");
    println!();

    println!("[Note] Alternative Approaches:");
    println!("   • For true embedding extraction, consider:");
    println!("     - Using encoder hidden states (requires model modification)");
    println!("     - Training a separate encoder on forecast reconstruction");
    println!("     - Using forecast statistics as features");
    println!();

    Ok(())
}
