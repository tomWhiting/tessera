# Tessera API Quick Reference

## Type Overview

| Type | Purpose | Output | Quantization | Similarity |
|------|---------|--------|--------------|------------|
| `TesseraMultiVector` | ColBERT token embeddings | `TokenEmbeddings` | ✓ Binary | MaxSim |
| `TesseraDense` | Single pooled vector | `DenseEmbedding` | ✗ | Cosine |
| `Tessera` | Auto-detect factory | Enum variant | Varies | Varies |

## Basic Usage

### Multi-Vector (ColBERT)
```rust
use tessera::TesseraMultiVector;

let embedder = TesseraMultiVector::new("colbert-v2")?;
let embeddings = embedder.encode("query text")?;
println!("{} tokens × {} dims", embeddings.num_tokens, embeddings.embedding_dim);

let score = embedder.similarity("query", "document")?;
```

### Dense (Sentence Embeddings)
```rust
use tessera::TesseraDense;

let embedder = TesseraDense::new("bge-base-en-v1.5")?;
let embedding = embedder.encode("query text")?;
println!("{} dimensions", embedding.dim());

let score = embedder.similarity("query", "document")?;
```

### Auto-Detection
```rust
use tessera::Tessera;

let embedder = Tessera::new("colbert-v2")?;  // Auto-detects MultiVector

match embedder {
    Tessera::MultiVector(mv) => { /* ColBERT API */ }
    Tessera::Dense(d) => { /* Dense API */ }
}
```

## Builder API

### Multi-Vector Builder
```rust
use tessera::{TesseraMultiVectorBuilder, QuantizationConfig};
use candle_core::Device;

let embedder = TesseraMultiVectorBuilder::new()
    .model("jina-colbert-v2")
    .device(Device::Cpu)
    .dimension(128)  // Matryoshka
    .quantization(QuantizationConfig::Binary)
    .build()?;
```

### Dense Builder
```rust
use tessera::TesseraDenseBuilder;
use candle_core::Device;

let embedder = TesseraDenseBuilder::new()
    .model("bge-base-en-v1.5")
    .device(Device::Cpu)
    .dimension(384)  // Matryoshka if supported
    .build()?;
```

## Common Methods

### Encoding
```rust
// Single text
let embedding = embedder.encode("text")?;

// Batch processing
let embeddings = embedder.encode_batch(&["text1", "text2"])?;
```

### Similarity
```rust
let score = embedder.similarity("query", "document")?;
```

### Metadata
```rust
let model_id = embedder.model();
let dim = embedder.dimension();
```

## Quantization (Multi-Vector Only)

```rust
use tessera::{TesseraMultiVector, QuantizationConfig};

let embedder = TesseraMultiVector::builder()
    .model("colbert-v2")
    .quantization(QuantizationConfig::Binary)
    .build()?;

// Encode and quantize
let embeddings = embedder.encode("text")?;
let quantized = embedder.quantize(&embeddings)?;

// Or in one step
let quantized = embedder.encode_quantized("text")?;

// Similarity with quantized
let score = embedder.similarity_quantized(&query_quant, &doc_quant)?;
```

## Error Handling

```rust
use tessera::{Result, TesseraError};

fn example() -> Result<()> {
    let embedder = TesseraDense::new("model-id")?;
    // ...
    Ok(())
}

// Common errors
match embedder {
    Err(TesseraError::ModelNotFound { model_id }) => { /* ... */ }
    Err(TesseraError::UnsupportedDimension { model_id, requested, supported }) => { /* ... */ }
    Err(TesseraError::ConfigError(msg)) => { /* ... */ }
    _ => {}
}
```

## Model Registry

```rust
use tessera::model_registry::{get_model, ModelType};

if let Some(model) = get_model("colbert-v2") {
    println!("Type: {:?}", model.model_type);
    println!("Dim: {}", model.embedding_dim);
    println!("Context: {}", model.context_length);
}
```

## Imports

### Minimal
```rust
use tessera::{Result, TesseraMultiVector, TesseraDense};
```

### With Builder
```rust
use tessera::{
    Result,
    TesseraMultiVector, TesseraMultiVectorBuilder,
    TesseraDense, TesseraDenseBuilder,
    QuantizationConfig,
};
```

### With Factory
```rust
use tessera::{Result, Tessera};
```

### Full
```rust
use tessera::{
    Result, TesseraError,
    Tessera, TesseraDense, TesseraMultiVector,
    TesseraDenseBuilder, TesseraMultiVectorBuilder,
    QuantizationConfig, QuantizedEmbeddings,
    TokenEmbeddings, DenseEmbedding,
};
```
