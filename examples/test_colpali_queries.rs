use ndarray::Array2;
use std::path::Path;
use tessera::backends::candle::get_device;
use tessera::core::TokenEmbeddings;
use tessera::encoding::vision::ColPaliEncoder;
use tessera::models::ModelConfig;
use tessera::utils::similarity::max_sim;

fn main() -> anyhow::Result<()> {
    println!("=== ColPali Query Testing ===\n");

    let device = get_device()?;
    let config = ModelConfig::from_registry("colpali-v1.2")?;
    let encoder = ColPaliEncoder::new(config, device)?;

    let pdf_path = Path::new("examples/fixtures/attention_is_all_you_need.pdf");

    println!("Encoding PDF pages (this may take a moment)...");
    let page_embeddings = encoder.encode_pdf_document(pdf_path)?;

    let queries = vec![
        "transformer architecture",
        "encoder decoder",
        "self attention mechanism",
        "multi-head attention",
    ];

    for query in queries {
        println!("\n--- Query: \"{}\" ---", query);
        let query_emb = encoder.encode_text(query)?;

        let mut scores: Vec<(usize, f32)> = Vec::new();
        for (idx, page_emb) in page_embeddings.iter().enumerate() {
            let flat: Vec<f32> = page_emb.embeddings.iter().flatten().copied().collect();
            let page_array =
                Array2::from_shape_vec((page_emb.num_patches, page_emb.embedding_dim), flat)?;
            let page_token_emb = TokenEmbeddings::new(page_array, format!("Page {}", idx + 1))?;
            let score = max_sim(&query_emb, &page_token_emb)?;
            scores.push((idx + 1, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("Top 3:");
        for (rank, (page, score)) in scores.iter().take(3).enumerate() {
            println!("  {}. Page {} - {:.4}", rank + 1, page, score);
        }
    }

    Ok(())
}
