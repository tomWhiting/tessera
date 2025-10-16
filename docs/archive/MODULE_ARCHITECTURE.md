# Tessera Module Architecture

## Module Hierarchy

```
tessera/
│
├── API Layer (User-Facing)
│   ├── api::Tessera          - Simple interface: Tessera::new("model")
│   └── api::TesseraBuilder   - Advanced interface: builder pattern
│
├── Encoding Layer (Paradigm-Specific)
│   ├── encoding::ColBERTEncoding      - Multi-vector token embeddings
│   ├── encoding::DenseEncoding        - Single-vector pooled embeddings
│   ├── encoding::SparseEncoding       - Sparse vocabulary embeddings
│   ├── encoding::TimeSeriesEncoding   - Temporal data encoding
│   └── encoding::VisionEncoding       - Image/document encoding
│
├── Quantization Layer (Compression)
│   ├── quantization::BinaryQuantization  - 1-bit (32x compression)
│   ├── quantization::Int8Quantization    - 8-bit (4x compression)
│   └── quantization::Int4Quantization    - 4-bit (8x compression)
│
├── Backend Layer (Inference Engines)
│   ├── backends::candle::CandleEncoder   - Production backend
│   └── backends::burn::BurnEncoder       - Experimental backend
│
├── Core Layer (Foundations)
│   ├── core::TokenEmbeddings    - Multi-vector representation
│   ├── core::TokenEmbedder      - Encoding trait
│   ├── core::Tokenizer          - Text tokenization
│   └── core::max_sim            - Similarity algorithms
│
├── Model Layer (Configuration)
│   ├── models::ModelConfig      - Model configuration
│   ├── models::ModelRegistry    - Model metadata
│   └── models::ModelLoader      - Model loading
│
└── Bindings Layer (Language Interop)
    ├── bindings::python::PyTessera    - Python API
    └── bindings::wasm::WasmTessera    - WebAssembly API
```

## Data Flow

### Encoding Pipeline

```
User Text
    ↓
[API Layer: Tessera::encode()]
    ↓
[Encoding Layer: ColBERTEncoding::encode()]
    ↓
[Backend Layer: CandleEncoder::encode()]
    ↓
[Core Layer: TokenEmbeddings]
    ↓
[Optional: Quantization]
    ↓
Embeddings (Vec<Vec<f32>> or quantized)
```

### Similarity Pipeline

```
Query Embeddings + Document Embeddings
    ↓
[Core Layer: max_sim()]
    ↓
Similarity Score (f32)
```

## Module Dependencies

```
┌─────────────────────────────────────────────────────────┐
│ bindings (Python, WASM)                                 │
│   depends on: api                                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ api (Tessera, TesseraBuilder)                           │
│   depends on: encoding, quantization, backends, core    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ encoding (ColBERT, Dense, Sparse, Vision, TimeSeries)   │
│   depends on: backends, core                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────┬────────────────────────────────────┐
│ quantization       │ backends (Candle, Burn)             │
│   depends on: core │   depends on: core                  │
└────────────────────┴────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ core (TokenEmbeddings, TokenEmbedder, Tokenizer)        │
│   No dependencies (foundation layer)                    │
└─────────────────────────────────────────────────────────┘
```

## Feature Matrix

| Module | Status | Priority | Dependencies |
|--------|--------|----------|--------------|
| core::* | ✓ Implemented | P0 | None |
| backends::candle | ✓ Implemented | P0 | core |
| backends::burn | ✓ Implemented | P3 | core |
| models::* | ✓ Implemented | P0 | None |
| encoding::colbert | TODO | P1 | backends, core |
| encoding::dense | TODO | P1 | backends, core |
| encoding::sparse | TODO | P2 | backends, core |
| encoding::timeseries | TODO | P3 | backends, core |
| encoding::vision | TODO | P3 | backends, core |
| quantization::binary | TODO | P2 | core |
| quantization::int8 | TODO | P2 | core |
| quantization::int4 | TODO | P3 | core |
| api::embedder | TODO | P1 | encoding, backends |
| api::builder | TODO | P2 | api::embedder |
| bindings::python | TODO | P4 | api |
| bindings::wasm | TODO | P4 | api |

## Interface Examples

### Simple API (Beginner-Friendly)

```rust
use tessera::Tessera;

// One line to get started
let embedder = Tessera::new("jina-colbert-v2")?;
let embeddings = embedder.encode("What is machine learning?")?;
```

### Builder API (Advanced Users)

```rust
use tessera::TesseraBuilder;
use tessera::quantization::BinaryQuantization;

let embedder = TesseraBuilder::new()
    .model("jina-colbert-v2")
    .device("metal")
    .dimension(96)  // Matryoshka
    .quantization(BinaryQuantization::new())
    .normalize(true)
    .build()?;
```

### Low-Level API (Library Authors)

```rust
use tessera::backends::CandleEncoder;
use tessera::core::{TokenEmbedder, max_sim};
use tessera::encoding::ColBERTEncoding;

// Direct backend usage
let encoder = CandleEncoder::new(config, device)?;
let embeddings = encoder.encode("text")?;

// Direct encoding usage
let encoding = ColBERTEncoding::new()?;
let embeddings = encoding.encode("text")?;

// Direct similarity computation
let score = max_sim(&query_embeddings, &doc_embeddings)?;
```

## Extension Points

### Adding a New Backend

1. Implement `core::TokenEmbedder` trait
2. Add module in `backends/`
3. Update `backends/mod.rs`
4. Add feature flag (optional)

### Adding a New Encoding

1. Create module in `encoding/`
2. Implement encoding logic
3. Update `encoding/mod.rs`
4. Wire into `api::Tessera`

### Adding a New Quantization

1. Implement `quantization::Quantization` trait
2. Add module in `quantization/`
3. Update `quantization/mod.rs`
4. Optimize distance computation

### Adding a New Language Binding

1. Create module in `bindings/`
2. Wrap `api::Tessera`
3. Add feature flag
4. Add FFI dependencies
5. Create language-specific package

## Performance Considerations

### Memory Layout

```
Multi-Vector Embeddings (ColBERT):
  Vec<Vec<f32>>
  - Outer vec: tokens (dynamic length)
  - Inner vec: dimensions (fixed, e.g., 128)
  - Memory: tokens * dims * 4 bytes

Quantized (Binary):
  Vec<u64>
  - Packed bits: 64 dims per word
  - Memory: tokens * (dims / 64) * 8 bytes
  - Compression: 32x reduction

Dense Embeddings:
  Vec<f32>
  - Single vector: dims (fixed, e.g., 768)
  - Memory: dims * 4 bytes
```

### Computation Paths

```
Fast Path (Binary Quantized):
  XOR + popcount
  ~1-5 ns per comparison

Medium Path (Int8):
  int8 dot product
  ~10-50 ns per comparison

Slow Path (Float32):
  float32 dot product
  ~50-200 ns per comparison

MaxSim Path (Multi-Vector):
  O(query_tokens * doc_tokens * dims)
  Optimized with SIMD
```

## Testing Strategy

### Unit Tests
- Each module tests its own functionality
- Mock dependencies when needed
- Test error paths

### Integration Tests
- End-to-end encoding pipelines
- Cross-backend compatibility
- Quantization accuracy

### Benchmark Tests
- Throughput (texts/second)
- Latency (milliseconds/text)
- Memory usage (MB/text)
- Compression ratio

## Documentation Structure

```
docs/
├── api/              - API reference (generated)
├── guides/
│   ├── quickstart.md
│   ├── encoding.md
│   ├── quantization.md
│   └── backends.md
├── examples/
│   ├── basic.rs
│   ├── advanced.rs
│   ├── quantization.rs
│   └── custom_backend.rs
└── benchmarks/
    ├── throughput.rs
    ├── memory.rs
    └── accuracy.rs
```

---

This architecture provides:
- **Clarity**: Each module has a clear purpose
- **Flexibility**: Easy to extend and customize
- **Performance**: Optimized data paths
- **Usability**: Simple for beginners, powerful for experts
