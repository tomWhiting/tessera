# Phase 0 API Reference: Unified Encoder Traits and Utils

Quick reference guide for the new unified encoder trait hierarchy and utilities module.

## Encoder Trait Hierarchy

### Base Encoder Trait

```rust
pub trait Encoder {
    type Output;
    
    fn encode(&self, input: &str) -> Result<Self::Output>;
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Self::Output>>;
}
```

### Multi-Vector Encoder (ColBERT)

```rust
pub trait MultiVectorEncoder: Encoder<Output = TokenEmbeddings> {
    fn num_vectors(&self, text: &str) -> Result<usize>;
    fn embedding_dim(&self) -> usize;
}
```

**Example Usage:**
```rust
use tessera::core::{Encoder, MultiVectorEncoder};
use tessera::backends::CandleBertEncoder;

let encoder = CandleBertEncoder::new(config, device)?;

// Get embedding info without encoding
let num_tokens = encoder.num_vectors("Hello world")?;
let dim = encoder.embedding_dim();

// Encode
let embeddings = encoder.encode("Hello world")?;
```

### Dense Encoder (BERT-style pooled)

```rust
pub trait DenseEncoder: Encoder<Output = DenseEmbedding> {
    fn embedding_dim(&self) -> usize;
    fn pooling_strategy(&self) -> PoolingStrategy;
}
```

**Example Usage:**
```rust
// Future implementation
let encoder = DenseBertEncoder::new(config, device)?;
let embedding = encoder.encode("Hello world")?;
let strategy = encoder.pooling_strategy(); // Cls, Mean, or Max
```

### Sparse Encoder (SPLADE-style)

```rust
pub trait SparseEncoder: Encoder<Output = SparseEmbedding> {
    fn vocab_size(&self) -> usize;
    fn expected_sparsity(&self) -> f32;
}
```

---

## Utils Module

### Pooling Functions

```rust
use tessera::utils::{cls_pooling, mean_pooling, max_pooling};
use ndarray::Array2;

let token_embeddings: Array2<f32> = /* from model */;
let attention_mask: Vec<i64> = vec![1, 1, 1, 0, 0]; // 3 real, 2 padding

// CLS pooling (first token)
let cls_vec = cls_pooling(&token_embeddings, &attention_mask);

// Mean pooling (average of real tokens)
let mean_vec = mean_pooling(&token_embeddings, &attention_mask);

// Max pooling (element-wise max)
let max_vec = max_pooling(&token_embeddings, &attention_mask);
```

### Similarity Functions

```rust
use tessera::utils::{cosine_similarity, dot_product, euclidean_distance, max_sim};
use ndarray::array;

let a = array![1.0, 2.0, 3.0];
let b = array![4.0, 5.0, 6.0];

// Cosine similarity [-1, 1]
let cos_sim = cosine_similarity(&a, &b)?;

// Dot product (unbounded)
let dot = dot_product(&a, &b)?;

// Euclidean distance [0, ∞)
let dist = euclidean_distance(&a, &b)?;

// MaxSim for multi-vector embeddings
use tessera::core::TokenEmbeddings;
let query: TokenEmbeddings = /* ... */;
let doc: TokenEmbeddings = /* ... */;
let score = max_sim(&query, &doc)?;
```

### Normalization Functions

```rust
use tessera::utils::{l2_norm, l2_normalize};
use ndarray::array;

let v = array![3.0, 4.0];

// Compute L2 norm (magnitude)
let norm = l2_norm(&v); // 5.0

// Normalize to unit length
let normalized = l2_normalize(&v); // [0.6, 0.8]
```

### Batching Functions

```rust
use tessera::utils::{pad_sequences, create_attention_mask};

let sequences = vec![
    vec![1, 2, 3],
    vec![4, 5],
    vec![6, 7, 8, 9],
];

// Pad to max length (4)
let padded = pad_sequences(&sequences, 0);
// [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]]

// Create attention masks
let masks = create_attention_mask(&padded, 0);
// [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
```

---

## New Types

### DenseEmbedding

```rust
use tessera::core::DenseEmbedding;
use ndarray::array;

let embedding = DenseEmbedding::new(
    array![1.0, 2.0, 3.0],
    "Hello world".to_string()
);

println!("Dimension: {}", embedding.dim());
println!("Vector: {:?}", embedding.embedding);
println!("Text: {}", embedding.text);
```

### SparseEmbedding

```rust
use tessera::core::SparseEmbedding;

let embedding = SparseEmbedding::new(
    vec![(10, 0.5), (42, 0.8), (100, 0.3)], // (index, weight) pairs
    30522, // vocab size
    "machine learning".to_string()
);

println!("Non-zero count: {}", embedding.nnz()); // 3
println!("Sparsity: {:.4}", embedding.sparsity()); // 0.9999
```

### PoolingStrategy

```rust
use tessera::core::PoolingStrategy;

let strategy = PoolingStrategy::Mean;

match strategy {
    PoolingStrategy::Cls => println!("Using [CLS] token"),
    PoolingStrategy::Mean => println!("Using mean pooling"),
    PoolingStrategy::Max => println!("Using max pooling"),
}
```

---

## Migration from Legacy APIs

### Renamed Types

```rust
// OLD (deprecated)
use tessera::backends::CandleEncoder;

// NEW
use tessera::backends::CandleBertEncoder;
```

### Moved Functions

```rust
// OLD (deprecated)
use tessera::core::max_sim;

// NEW
use tessera::utils::max_sim;
```

### Generic Programming

```rust
// OLD: Concrete types only
fn process_embeddings(encoder: &CandleEncoder) { /* ... */ }

// NEW: Generic over encoder types
use tessera::core::{Encoder, TokenEmbeddings};

fn process_embeddings<E: Encoder<Output = TokenEmbeddings>>(
    encoder: &E
) -> Result<()> {
    let embeddings = encoder.encode("test")?;
    Ok(())
}
```

---

## Complete Example

```rust
use tessera::{
    core::{Encoder, MultiVectorEncoder, TokenEmbeddings},
    backends::{candle::get_device, CandleBertEncoder},
    models::ModelConfig,
    utils::{max_sim, mean_pooling, l2_normalize},
};

fn main() -> anyhow::Result<()> {
    // Load encoder
    let config = ModelConfig::distilbert_base_uncased();
    let device = get_device()?;
    let encoder = CandleBertEncoder::new(config, device)?;
    
    // Check model info
    println!("Embedding dimension: {}", encoder.embedding_dim());
    
    // Encode texts
    let query = encoder.encode("What is Rust?")?;
    let doc = encoder.encode("Rust is a systems programming language")?;
    
    // Compute similarity
    let score = max_sim(&query, &doc)?;
    println!("MaxSim score: {}", score);
    
    // Use pooling to get dense vectors
    let query_dense = mean_pooling(&query.embeddings, &vec![1; query.num_tokens]);
    let doc_dense = mean_pooling(&doc.embeddings, &vec![1; doc.num_tokens]);
    
    // Normalize and compare
    let query_norm = l2_normalize(&query_dense);
    let doc_norm = l2_normalize(&doc_dense);
    
    Ok(())
}
```

---

## Testing Your Code

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tessera::utils::{cosine_similarity, l2_normalize};
    use ndarray::array;
    
    #[test]
    fn test_normalized_similarity() {
        let a = array![3.0, 4.0];
        let b = array![6.0, 8.0];
        
        let a_norm = l2_normalize(&a);
        let b_norm = l2_normalize(&b);
        
        let sim = cosine_similarity(&a_norm, &b_norm).unwrap();
        assert!((sim - 1.0).abs() < 1e-6); // Same direction
    }
}
```

---

## Key Files

- `src/core/embeddings.rs`: Trait definitions and types
- `src/utils/pooling.rs`: Pooling implementations
- `src/utils/similarity.rs`: Similarity functions
- `src/utils/normalization.rs`: Normalization utilities
- `src/utils/batching.rs`: Batching utilities
- `src/backends/candle/encoder.rs`: CandleBertEncoder implementation

---

## Documentation

Full API documentation:
```bash
cargo doc --open
```

Run tests:
```bash
cargo test
```

Check examples:
```bash
cargo check --examples
```

---

**Version:** Phase 0 (Tasks 0.2 & 0.3)
**Last Updated:** 2025-10-16
