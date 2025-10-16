//! Integration tests for CandleDenseEncoder

#![allow(unused_imports)]

use tessera::core::{DenseEncoder, Encoder};
use tessera::encoding::dense::CandleDenseEncoder;
use tessera::models::ModelConfig;
use candle_core::Device;

#[test]
#[ignore] // Requires downloading model - run with --ignored
fn test_dense_encoder_single() {
    // Use a small model for testing
    let config = ModelConfig::from_registry("bge-small-en-v1.5")
        .expect("Failed to load config");
    
    let device = Device::Cpu;
    let encoder = CandleDenseEncoder::new(config, device)
        .expect("Failed to create encoder");
    
    // Test single encoding
    let text = "Machine learning is amazing";
    let embedding = encoder.encode(text).expect("Failed to encode");
    
    // Verify embedding properties
    assert_eq!(embedding.dim(), 384); // BGE-small has 384 dimensions
    assert_eq!(embedding.text, text);
    
    // Verify embedding is not all zeros
    let sum: f32 = embedding.embedding.iter().sum();
    assert!(sum.abs() > 0.0, "Embedding should not be all zeros");
}

#[test]
#[ignore] // Requires downloading model - run with --ignored
fn test_dense_encoder_batch() {
    let config = ModelConfig::from_registry("bge-small-en-v1.5")
        .expect("Failed to load config");
    
    let device = Device::Cpu;
    let encoder = CandleDenseEncoder::new(config, device)
        .expect("Failed to create encoder");
    
    // Test batch encoding
    let texts = vec![
        "First document",
        "Second document",
        "Third document",
    ];
    let embeddings = encoder.encode_batch(&texts.iter().map(|s| s.as_ref()).collect::<Vec<_>>())
        .expect("Failed to batch encode");
    
    // Verify batch results
    assert_eq!(embeddings.len(), 3);
    
    for (i, embedding) in embeddings.iter().enumerate() {
        assert_eq!(embedding.dim(), 384);
        assert_eq!(embedding.text, texts[i]);
        
        let sum: f32 = embedding.embedding.iter().sum();
        assert!(sum.abs() > 0.0, "Embedding should not be all zeros");
    }
}

#[test]
#[ignore] // Requires downloading model - run with --ignored
fn test_dense_encoder_normalization() {
    // Load config with normalization enabled
    let config = ModelConfig::from_registry("bge-small-en-v1.5")
        .expect("Failed to load config");
    
    let device = Device::Cpu;
    let encoder = CandleDenseEncoder::new(config, device)
        .expect("Failed to create encoder");
    
    let text = "Test normalization";
    let embedding = encoder.encode(text).expect("Failed to encode");
    
    // BGE models use normalization by default
    // Compute L2 norm
    let norm: f32 = embedding.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    // Check if normalized (norm should be close to 1.0)
    assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized, got norm {}", norm);
}

#[test]
#[ignore] // Requires downloading model - run with --ignored
fn test_dense_encoder_pooling_strategy() {
    let config = ModelConfig::from_registry("bge-small-en-v1.5")
        .expect("Failed to load config");
    
    let device = Device::Cpu;
    let encoder = CandleDenseEncoder::new(config, device)
        .expect("Failed to create encoder");
    
    // Verify pooling strategy
    use tessera::core::PoolingStrategy;
    assert_eq!(encoder.pooling_strategy(), PoolingStrategy::Mean);
    
    // Verify embedding dimension
    assert_eq!(encoder.embedding_dim(), 384);
}

#[test]
fn test_dense_encoder_requires_pooling() {
    // Create a custom config without pooling strategy
    let config = ModelConfig::custom("bert-base-uncased", 768, 512);

    let device = Device::Cpu;
    let result = CandleDenseEncoder::new(config, device);

    // Should fail because pooling strategy is not set
    assert!(result.is_err());
    if let Err(err) = result {
        assert!(err.to_string().contains("pooling_strategy"));
    }
}
