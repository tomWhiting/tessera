# Tessera Embedding Library - Architecture & PyO3 Bindings Preparation

## Executive Summary

Tessera is a production-ready multi-paradigm embedding library for Rust with support for 5 embedding paradigms and 23 models. The codebase is well-structured with a clear separation between core types, encoders, and high-level APIs. Python bindings via PyO3 are planned but not yet implemented (currently a stub).

## Project Structure

```
hypiler/
├── Cargo.toml                    # Main manifest with python feature flag
├── build.rs                      # Build script for model registry generation
├── models.json                   # Model metadata and registry (40KB+)
├── src/
│   ├── lib.rs                    # Library root, exports main API
│   ├── main.rs                   # CLI binary
│   ├── error.rs                  # Structured error types (TesseraError enum)
│   ├── api/                      # High-level user-facing API ⭐
│   │   ├── mod.rs                # Module exports
│   │   ├── embedder.rs           # Main embedder structs (1400+ lines)
│   │   └── builder.rs            # Builder pattern implementations (830+ lines)
│   ├── core/                     # Core abstractions & types
│   │   ├── embeddings.rs         # TokenEmbeddings, embedding type traits
│   │   ├── similarity.rs         # MaxSim, cosine similarity algorithms
│   │   └── tokenizer.rs          # Tokenizer abstraction
│   ├── backends/                 # Model inference backends
│   │   ├── candle/               # Candle backend (primary)
│   │   └── burn/                 # Burn backend (experimental)
│   ├── encoding/                 # Paradigm-specific encoders
│   │   ├── dense.rs              # Dense single-vector encoding
│   │   ├── sparse.rs             # Sparse vocabulary encoding
│   │   ├── vision.rs             # Vision-language encoding (ColPali)
│   │   └── mod.rs
│   ├── models/                   # Model loading and config
│   ├── timeseries/               # Time series forecasting (Chronos Bolt)
│   ├── quantization/             # Embedding compression (binary, int8, int4)
│   ├── bindings/                 # Language bindings
│   │   ├── mod.rs                # Bindings module root
│   │   └── python.rs             # PyO3 stubs (not implemented yet) ⭐
│   ├── vision/                   # Vision processing utilities
│   └── utils/                    # Utility functions
├── examples/                     # 40+ usage examples
│   ├── dense_semantic_search.rs
│   ├── batch_processing.rs
│   ├── colpali_demo.rs
│   ├── timeseries_basic_forecasting.rs
│   └── (30+ more...)
└── tests/                        # Comprehensive test suite
```

## Public API Surface

### 1. Main Entry Points

#### Unified Factory Enum: `Tessera`
```rust
pub enum Tessera {
    Dense(TesseraDense),
    MultiVector(TesseraMultiVector),
    Sparse(TesseraSparse),
    Vision(TesseraVision),
    TimeSeries(TesseraTimeSeries),
}

impl Tessera {
    pub fn new(model_id: &str) -> Result<Self>  // Auto-detects model type
}
```

**Usage Pattern:**
- Auto-detects model type from registry
- Returns appropriate variant for each model
- Pattern matching on result to use specific API

### 2. Specific Embedder Classes

#### A. Multi-Vector Embeddings: `TesseraMultiVector`
**Models:** ColBERT, Jina-ColBERT, NomicBERT-MultiVector

**Key Methods:**
```rust
impl TesseraMultiVector {
    pub fn new(model_id: &str) -> Result<Self>
    pub fn builder() -> TesseraMultiVectorBuilder
    pub fn encode(&self, text: &str) -> Result<TokenEmbeddings>
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<TokenEmbeddings>>
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32>
    pub fn dimension(&self) -> usize
    pub fn model(&self) -> &str
    pub fn quantize(&self, embeddings: &TokenEmbeddings) -> Result<QuantizedEmbeddings>
    pub fn encode_quantized(&self, text: &str) -> Result<QuantizedEmbeddings>
    pub fn similarity_quantized(&self, query: &QuantizedEmbeddings, doc: &QuantizedEmbeddings) -> Result<f32>
}
```

**Output Types:**
- `TokenEmbeddings`: Contains `Array2<f32>` embeddings matrix (num_tokens × embedding_dim)

#### B. Dense Embeddings: `TesseraDense`
**Models:** BGE, Nomic, GTE, Qwen, Jina-v3

**Key Methods:**
```rust
impl TesseraDense {
    pub fn new(model_id: &str) -> Result<Self>
    pub fn builder() -> TesseraDenseBuilder
    pub fn encode(&self, text: &str) -> Result<DenseEmbedding>
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>>
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32>
    pub fn dimension(&self) -> usize
    pub fn model(&self) -> &str
}
```

**Output Types:**
- `DenseEmbedding`: Contains `Vec<f32>` single pooled vector (embedding_dim,)

#### C. Sparse Embeddings: `TesseraSparse`
**Models:** SPLADE CoCondenser, SPLADE-pp

**Key Methods:**
```rust
impl TesseraSparse {
    pub fn new(model_id: &str) -> Result<Self>
    pub fn builder() -> TesseraSparseBuilder
    pub fn encode(&self, text: &str) -> Result<SparseEmbedding>
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<SparseEmbedding>>
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32>
    pub fn vocab_size(&self) -> usize
    pub fn model(&self) -> &str
}
```

**Output Types:**
- `SparseEmbedding`: Contains sparse vector with `Vec<(index, weight)>` pairs

#### D. Vision-Language Embeddings: `TesseraVision`
**Models:** ColPali v1.2, v1.3

**Key Methods:**
```rust
impl TesseraVision {
    pub fn new(model_id: &str) -> Result<Self>
    pub fn builder() -> TesseraVisionBuilder
    pub fn encode_document(&self, image_path: &str) -> Result<VisionEmbedding>
    pub fn encode_query(&self, text: &str) -> Result<TokenEmbeddings>
    pub fn search(&self, query: &TokenEmbeddings, document: &VisionEmbedding) -> Result<f32>
    pub fn search_document(&self, query_text: &str, image_path: &str) -> Result<f32>
    pub fn embedding_dim(&self) -> usize
    pub fn num_patches(&self) -> usize
    pub fn model(&self) -> &str
}
```

**Output Types:**
- `VisionEmbedding`: Contains patch-level embeddings
- `TokenEmbeddings`: Query token embeddings

#### E. Time Series Forecasting: `TesseraTimeSeries`
**Models:** Chronos Bolt (small, base, large)

**Key Methods:**
```rust
impl TesseraTimeSeries {
    pub fn new(model_id: &str) -> Result<Self>
    pub fn builder() -> TesseraTimeSeriesBuilder
    pub fn forecast(&mut self, context: &Tensor) -> Result<Tensor>
    pub fn forecast_quantiles(&mut self, context: &Tensor) -> Result<Tensor>
    pub fn prediction_length(&self) -> usize
    pub fn context_length(&self) -> usize
    pub fn quantiles(&self) -> &[f32]
    pub fn model(&self) -> &str
}
```

**Input/Output:**
- Input: `Tensor` of shape `[batch, context_length]`
- Output: `Tensor` of shape `[batch, prediction_length]` or `[batch, prediction_length, num_quantiles]`

### 3. Builder Pattern Classes

All builders follow similar pattern:

```rust
pub struct TesseraMultiVectorBuilder {
    model_id: Option<String>,
    device: Option<Device>,
    dimension: Option<usize>,
    quantization: Option<QuantizationConfig>,
}

impl TesseraMultiVectorBuilder {
    pub fn new() -> Self
    pub fn model(self, id: &str) -> Self
    pub fn device(self, device: Device) -> Self
    pub fn dimension(self, dim: usize) -> Self
    pub fn quantization(self, quant: QuantizationConfig) -> Self
    pub fn build(self) -> Result<TesseraMultiVector>
}
```

**Quantization Config:**
```rust
pub enum QuantizationConfig {
    None,
    Binary,           // 32x compression, 95%+ accuracy
    #[allow(dead_code)]
    Int8,             // 4x compression (Phase 2)
    #[allow(dead_code)]
    Int4,             // 8x compression (Phase 2)
}
```

### 4. Core Output Types

#### TokenEmbeddings (Multi-Vector)
```rust
pub struct TokenEmbeddings {
    pub embeddings: Array2<f32>,  // (num_tokens, embedding_dim)
    pub text: String,
    pub num_tokens: usize,
    pub embedding_dim: usize,
}
```

#### DenseEmbedding
```rust
pub struct DenseEmbedding {
    pub embedding: Vec<f32>,
    pub dim: usize,
}
```

#### SparseEmbedding
```rust
pub struct SparseEmbedding {
    pub weights: Vec<(usize, f32)>,  // (vocab_index, weight) pairs
    pub vocab_size: usize,
    pub sparsity: f32,               // Percentage of zero dimensions
}
```

#### VisionEmbedding
```rust
pub struct VisionEmbedding {
    pub embeddings: Vec<Vec<f32>>,  // [num_patches][embedding_dim]
    pub num_patches: usize,
    pub embedding_dim: usize,
    pub source: Option<String>,
}
```

#### QuantizedEmbeddings
```rust
pub struct QuantizedEmbeddings {
    pub quantized: Vec<BinaryVector>,  // 1-bit per dimension
    pub original_dim: usize,
    pub num_tokens: usize,
}

impl QuantizedEmbeddings {
    pub fn memory_bytes(&self) -> usize
    pub fn compression_ratio(&self) -> f32
}
```

### 5. Error Handling

```rust
pub enum TesseraError {
    ModelNotFound { model_id: String },
    ModelLoadError { model_id: String, source: anyhow::Error },
    EncodingError { context: String, source: anyhow::Error },
    UnsupportedDimension { model_id: String, requested: usize, supported: Vec<usize> },
    DeviceError(String),
    QuantizationError(String),
    TokenizationError(tokenizers::Error),
    ConfigError(String),
    DimensionMismatch { expected: usize, actual: usize },
    MatryoshkaError(String),
    IoError(std::io::Error),
    TensorError(candle_core::Error),
    Other(anyhow::Error),
}

pub type Result<T> = std::result::Result<T, TesseraError>;
```

## Model Registry

### models.json Structure
- **Version:** 1.0
- **Model Categories:** multi_vector, dense, sparse, vision_language, timeseries
- **Metadata per Model:**
  - id, type, name, huggingface_id
  - organization, release_date
  - architecture (type, variant, has_projection)
  - specs (parameters, embedding_dim, hidden_dim, context_length, vocab_size)
  - files (tokenizer, config, weights)
  - capabilities (languages, modalities, multi_vector, quantization)
  - performance (BEIR, MS-MARCO scores)
  - license, description

### Example Models in Registry
- **ColBERT:** colbert-v2, colbert-small
- **Dense:** bge-base-en-v1.5, nomic-embed-text-v1, gte-large-en-v1.5
- **Sparse:** splade-cocondenser, splade-pp-en-v1
- **Vision:** colpali-v1.3-hf, colpali-v1.2-hf
- **TimeSeries:** chronos-bolt-small, chronos-bolt-base

## Typical Usage Patterns from Examples

### Pattern 1: Simple Single-Text Encoding
```rust
let embedder = TesseraDense::new("bge-base-en-v1.5")?;
let embedding = embedder.encode("Hello, world!")?;
println!("Dimension: {}", embedding.dim());
```

### Pattern 2: Batch Processing
```rust
let texts = vec!["text1", "text2", "text3"];
let embeddings = embedder.encode_batch(&texts)?;
for emb in embeddings {
    println!("Got {} dimensions", emb.dim());
}
```

### Pattern 3: Similarity Search
```rust
let embedder = TesseraDense::new("bge-base-en-v1.5")?;
let score = embedder.similarity("query", "document")?;
println!("Similarity: {:.4}", score);
```

### Pattern 4: Multi-Vector with MaxSim
```rust
let embedder = TesseraMultiVector::new("colbert-v2")?;
let query_emb = embedder.encode("What is ML?")?;
let doc_emb = embedder.encode("Machine learning is...")?;
let score = embedder.similarity("What is ML?", "Machine learning is...")?;
```

### Pattern 5: Builder with Configuration
```rust
let embedder = TesseraMultiVector::builder()
    .model("jina-colbert-v2")
    .device(Device::Cpu)
    .dimension(96)
    .quantization(QuantizationConfig::Binary)
    .build()?;
```

### Pattern 6: Vision-Language (ColPali)
```rust
let vision = TesseraVision::new("colpali-v1.3-hf")?;
let doc_emb = vision.encode_document("invoice.jpg")?;
let query_emb = vision.encode_query("total amount")?;
let score = vision.search(&query_emb, &doc_emb)?;
```

### Pattern 7: Time Series Forecasting
```rust
let forecaster = TesseraTimeSeries::new("chronos-bolt-small")?;
let context = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
let forecast = forecaster.forecast(&context)?;
let quantiles = forecaster.forecast_quantiles(&context)?;
```

### Pattern 8: Factory Pattern (Auto-Detection)
```rust
let embedder = Tessera::new("colbert-v2")?;
match embedder {
    Tessera::MultiVector(mv) => { /* ... */ },
    Tessera::Dense(d) => { /* ... */ },
    Tessera::Vision(v) => { /* ... */ },
    _ => {},
}
```

## Existing Python-Related Files

### Current Status
- **src/bindings/python.rs** - Currently only a stub (87 lines)
  - Contains placeholder `PyTessera` struct
  - Includes commented-out PyO3 implementation outline
  - NOT compiled by default (requires `python` feature)

### Cargo.toml Feature Flags
```toml
[features]
default = ["pdf"]
metal = ["candle-core/metal", ...]
cuda = ["candle-core/cuda", ...]
pdf = ["pdf2image"]
python = []           # PyO3 bindings - dependencies NOT added yet
wasm = []             # WebAssembly bindings - dependencies NOT added yet
```

### Notes
- PyO3 dependencies NOT yet in Cargo.toml
- No numpy integration yet
- No type hints (.pyi files) generated
- No maturin setup for building Python wheels

## Key Architectural Patterns

### 1. Builder Pattern
- All embedders support builder() method
- Enables progressive disclosure (simple → advanced)
- Type-safe configuration

### 2. Trait Hierarchy for Encoders
```rust
pub trait Encoder {
    fn encode(&self, text: &str) -> Result<T>;
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<T>>;
}

pub trait MultiVectorEncoder: Encoder { /* ... */ }
pub trait DenseEncoder: Encoder { /* ... */ }
pub trait SparseEncoder: Encoder { /* ... */ }
pub trait VisionEncoder: Encoder { /* ... */ }
pub trait TimeSeriesEncoder { /* ... */ }
```

### 3. Factory with Enum Dispatch
- `Tessera::new()` auto-detects and creates appropriate variant
- Enables polymorphic usage without trait objects

### 4. Device Auto-Selection
```rust
// Priority: Metal (Apple Silicon) > CUDA (NVIDIA) > CPU
let device = crate::backends::candle::get_device()?;
```

### 5. Quantization as Optional Post-Processing
- Binary quantization: 32x compression
- Works on TokenEmbeddings after encoding
- Separate similarity method for quantized embeddings

## Critical Types for Python Binding

### Essential to Expose
1. **Tessera** (enum) - Factory
2. **TesseraDense** - Dense embeddings
3. **TesseraMultiVector** - Multi-vector embeddings
4. **TesseraSparse** - Sparse embeddings
5. **TesseraVision** - Vision-language embeddings
6. **TesseraTimeSeries** - Time series forecasting
7. **TokenEmbeddings** - Multi-vector output
8. **DenseEmbedding** - Dense output
9. **SparseEmbedding** - Sparse output
10. **VisionEmbedding** - Vision output
11. **QuantizedEmbeddings** - Quantized output
12. **TesseraError** (enum) - Error handling
13. **Device** - from candle_core
14. **Tensor** - from candle_core (for time series)

### NumPy Interop Points
- `TokenEmbeddings.embeddings: Array2<f32>` → numpy.ndarray (2D)
- `DenseEmbedding.embedding: Vec<f32>` → numpy.ndarray (1D)
- `SparseEmbedding.weights: Vec<(usize, f32)>` → COO sparse matrix or tuple of arrays
- `TesseraTimeSeries.forecast() -> Tensor` → numpy.ndarray

## Example Model Usage

### Multi-Vector (ColBERT):
```
Input: "What is machine learning?"
Output: TokenEmbeddings with shape (7, 128)
        [CLS] → [1, 128]
        What  → [1, 128]
        is    → [1, 128]
        ml    → [1, 128]
        ...
```

### Dense (BGE):
```
Input: "What is machine learning?"
Output: DenseEmbedding with shape (768,)
        Single pooled vector
```

### Sparse (SPLADE):
```
Input: "What is machine learning?"
Output: SparseEmbedding
        [(101, 0.8), (2034, 0.5), (5021, 0.3), ...]
        Vocabulary indices with weights
        99%+ sparsity
```

### Vision (ColPali):
```
Input: "invoice.jpg" (448×448)
Output: VisionEmbedding with shape (1024, 128)
        1024 patches (14×14 layout)
        128 dimensions each
```

### Time Series (Chronos):
```
Input: Tensor [batch=4, context=2048]
Output (forecast): Tensor [batch=4, prediction=64]
Output (quantiles): Tensor [batch=4, prediction=64, quantiles=9]
```

## What Needs Implementation for PyO3

1. **Add PyO3 dependencies to Cargo.toml**
   ```toml
   [dependencies]
   pyo3 = { version = "0.20", features = ["extension-module"] }
   numpy = "0.20"
   ```

2. **Implement PyO3 wrapper classes** in src/bindings/python.rs
   - PyTessera (factory)
   - PyTesseraDense
   - PyTesseraMultiVector
   - PyTesseraSparse
   - PyTesseraVision
   - PyTesseraTimeSeries

3. **Implement NumPy interop**
   - Convert Array2<f32> → PyArray2
   - Convert Vec<f32> → PyArray1
   - Handle sparse vectors (tuple of arrays or COO format)
   - Convert Tensor → PyArray

4. **Error mapping**
   - Convert TesseraError → Python exceptions

5. **Build configuration**
   - Create pyproject.toml for maturin
   - Generate .pyi type stub files

## Files Key for Understanding

**Essential Reading Order:**
1. `/src/lib.rs` - Exports
2. `/src/api/embedder.rs` - Main API classes (1400 lines)
3. `/src/api/builder.rs` - Builder pattern (830 lines)
4. `/src/core/embeddings.rs` - Core types
5. `/src/error.rs` - Error types
6. `/examples/dense_semantic_search.rs` - Usage example
7. `/models.json` - Model registry structure

**For PyO3 Implementation:**
1. `/src/bindings/python.rs` - Stub to implement
2. `/src/bindings/mod.rs` - Module structure

## Summary of Public API for Python

**Total Classes to Expose:** 11
**Total Methods:** ~60
**Supported Models:** 23
**Embedding Paradigms:** 5
**Error Types:** 11

**Key Statistics:**
- Batch processing: 5-10x speedup
- Binary quantization: 32x compression
- Model sizes: 33M-3B parameters
- Supported languages: 100+
- Vision: Handles tables, charts, handwriting
- Time series: 9 quantile levels for uncertainty
