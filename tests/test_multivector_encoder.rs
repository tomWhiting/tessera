//! Serial, opt-in smoke tests for real multi-vector checkpoints.

use candle_core::Device;
use tessera::backends::candle::CandleBertEncoder;
use tessera::core::TokenEmbedder;
use tessera::models::ModelConfig;
use tessera::utils::similarity::max_sim;

#[test]
#[ignore = "requires remote ColBERT artifacts"]
fn test_colbert_small_single() {
    let config = ModelConfig::from_registry("colbert-small")
        .expect("ColBERT Small should remain a runnable catalog entry");
    let encoder =
        CandleBertEncoder::new(config, Device::Cpu).expect("ColBERT Small should load on CPU");

    let text = "A tessera is one tile in a mosaic.";
    let embedding = encoder
        .encode(text)
        .expect("ColBERT Small should encode one document");

    assert_eq!(embedding.embedding_dim, 96);
    assert_eq!(embedding.text, text);
    assert!(embedding.num_tokens > 0);
    for row in embedding.embeddings.rows() {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1.0e-4, "token norm was {norm}");
    }

    let self_score = max_sim(&embedding, &embedding).expect("self MaxSim should be defined");
    assert!(self_score.is_finite() && self_score > 0.0);
    let expected_self = embedding
        .embeddings
        .rows()
        .into_iter()
        .map(|row| row.dot(&row))
        .sum::<f32>();
    assert!((self_score - expected_self).abs() < 1.0e-3);
}
