# Tessera Architecture Review

**Version:** 1.0  
**Date:** January 2025  
**Reviewer:** Technical Architecture Analysis  
**Scope:** Integration analysis, trait design, missing components, architectural improvements

---

## Executive Summary

Tessera's architecture shows strong foundational design with clear separation of concerns across encoding, backends, quantization, and API layers. However, several critical integration gaps exist, particularly around:

1. **Quantization compatibility with multi-vector encodings** - Binary quantization design assumes fixed-size vectors, incompatible with variable-length ColBERT outputs
2. **Missing trait unification** - No unified `Encoder` trait causes type system fragmentation across paradigms
3. **Matryoshka implementation ambiguity** - Unclear where dimension truncation occurs relative to projection layers
4. **Missing utility infrastructure** - No normalization, pooling, or similarity utilities beyond MaxSim

The architecture is **technically sound for Phase 1 (ColBERT-only)** but requires significant refactoring before Phase 2 (multi-paradigm support).

**Priority Fixes Required:**
- Unified trait hierarchy for all encoding paradigms
- Quantization abstraction over variable-length embeddings
- Common utility layer for pooling, normalization, similarity
- Clear Matryoshka implementation strategy

---

## Section 1: Integration Analysis

### 1.1 Quantization + Encoding Paradigms

#### CRITICAL ISSUE: Binary Quantization Incompatible with Multi-Vector

**Problem:**  
`src/quantization/mod.rs` defines quantization as:
```rust
fn quantize(&self, embeddings: &[f32]) -> Self::Output;
```

This assumes a **single fixed-size vector**. ColBERT produces **N vectors of dimension D** (where N varies per text). The current design cannot quantize multi-vector outputs.

**Specific Conflicts:**

1. **ColBERT** (`src/encoding/colbert.rs`): Returns `Vec<Vec<f32>>` (variable N, fixed D)
   - Binary quantization expects `&[f32]` (single vector)
   - **Cannot quantize**: Type mismatch

2. **Dense** (`src/encoding/dense.rs`): Returns `Vec<f32>` (single vector)
   - Binary quantization works: Compatible

3. **Sparse** (`src/encoding/sparse.rs`): Returns `HashMap<u32, f32>`
   - Binary quantization incompatible: Sparse structure lost
   - **Does not make sense**: Sparsity is the point, binarization destroys vocabulary mapping

4. **Time Series** (`src/encoding/timeseries.rs`): Returns `Vec<Vec<f32>>` (patches)
   - Same issue as ColBERT: Variable-length

5. **Vision** (`src/encoding/vision.rs`): Returns `Vec<Vec<f32>>` (patches)
   - Same issue as ColBERT: Variable-length patches

**Impact Severity:** HIGH - Blocks Phase 2 implementation

**Recommendations:**

**Option A: Separate Quantization Traits (Preferred)**
```rust
// Dense/single-vector quantization
pub trait VectorQuantization {
    type Output;
    fn quantize(&self, embedding: &[f32]) -> Self::Output;
    fn distance(&self, a: &Self::Output, b: &Self::Output) -> f32;
}

// Multi-vector quantization (ColBERT, time series, vision)
pub trait MultiVectorQuantization {
    type Output;
    fn quantize_multi(&self, embeddings: &[Vec<f32>]) -> Self::Output;
    fn distance_multi(&self, a: &Self::Output, b: &Self::Output) -> f32;
}
```

**Option B: Unified Trait with Enum Output**
```rust
pub enum QuantizedEmbedding {
    Single(BinaryEmbedding),
    Multi(Vec<BinaryEmbedding>),
    Sparse(SparseQuantized),
}

pub trait Quantization {
    fn quantize_dense(&self, embedding: &[f32]) -> BinaryEmbedding;
    fn quantize_multi(&self, embeddings: &[Vec<f32>]) -> Vec<BinaryEmbedding>;
    // Sparse quantization doesn't make sense - skip
}
```

**Option C: Quantize at Token Level (Most Elegant)**
```rust
// Quantize individual vectors, composition handles multi-vector
impl BinaryQuantization {
    fn quantize_vector(&self, v: &[f32]) -> BinaryEmbedding;
}

// Users compose for multi-vector:
let quantized_tokens: Vec<BinaryEmbedding> = 
    token_embeddings.iter()
        .map(|v| quantizer.quantize_vector(v))
        .collect();
```

**Verdict:** Implement Option C for simplicity. Quantization operates on individual vectors. Multi-vector structures (ColBERT, vision) are collections of quantized vectors.

---

#### Matryoshka + Encoding Paradigms

**Problem:**  
Matryoshka dimension truncation location is ambiguous. For ColBERT with projection:
- **Option 1:** Apply projection (768 → 128), THEN truncate (128 → 96)
- **Option 2:** Truncate hidden states (768 → 96), THEN project (96 → 96)

**Analysis:**

From `models.json`, Jina-ColBERT-v2:
```json
"specs": {
    "embedding_dim": {
        "default": 768,
        "matryoshka": {
            "supported": [64, 96, 128, 256, 384, 512, 768]
        }
    }
}
```

This model has **no projection layer** (`has_projection: false`). The Matryoshka dimensions are **hidden state truncations**, not post-projection.

For models **with projection** (ColBERT v2: 768 → 128):
- Matryoshka must happen **before projection** (truncate 768 → 96, then project 96 → output_dim)
- OR have **multiple projection matrices** (768 → 128, 768 → 96, etc.)

**Current Implementation Gap:**  
`src/backends/candle/encoder.rs` applies projection but has no Matryoshka logic:
```rust
if let Some(ref projection) = self.projection {
    output = output.broadcast_matmul(&projection_t)?;
}
```

**Recommendation:**
```rust
// 1. Extract hidden states [batch, seq, hidden_size]
let hidden = self.model.forward(&token_ids, &attention_mask)?;

// 2. Apply Matryoshka FIRST (truncate hidden dimension)
let hidden_truncated = if let Some(matryoshka_dim) = self.config.matryoshka_dim {
    hidden.narrow(2, 0, matryoshka_dim)?  // Truncate last dimension
} else {
    hidden
};

// 3. THEN apply projection (if present)
let output = if let Some(ref projection) = self.projection {
    // Projection matrix must match truncated dimension
    hidden_truncated.broadcast_matmul(&projection.narrow(1, 0, matryoshka_dim)?.t()?)?
} else {
    hidden_truncated
};
```

**Impact:** MEDIUM - Affects API design and model loading

---

#### Sparse Embeddings + Quantization/Matryoshka

**Sparse + Binary Quantization:** **Does NOT make sense**
- Sparse embeddings are vocabulary-sized (30K dims, 99% zero)
- Binary quantization produces dense bit vectors (loses sparsity)
- Destroys interpretability and inverted index compatibility
- **Verdict:** Sparse should not support binary quantization

**Sparse + Matryoshka:** **Does NOT make sense**
- Matryoshka truncates dimensions (keep first N dims)
- Sparse embeddings are in vocabulary space (dimensions = word IDs)
- Truncating vocabulary IDs is semantically meaningless
- **Verdict:** Sparse should not support Matryoshka

**Recommendation:** Add compile-time or runtime guards preventing incompatible combinations.

---

#### Time Series + Quantization

**Time Series Embeddings:** Multi-vector (patches)
- Same structure as ColBERT: `Vec<Vec<f32>>` where each patch gets a vector
- Binary quantization should work with Option C approach (quantize per patch)
- Matryoshka makes sense: Truncate patch embedding dimension

**Vision + Quantization**

**Vision Embeddings (ColPali):** Multi-vector (patches)
- Same as time series: `Vec<Vec<f32>>` where each image patch gets a vector
- Binary quantization compatible with per-vector approach
- Matryoshka makes sense: Truncate patch dimension

**Verdict:** With Option C quantization design, both work cleanly.

---

### 1.2 Backend + Encoding Compatibility

#### Candle Backend Analysis

**Current Support:**
- ✅ BERT variants (standard, DistilBERT, JinaBERT) - IMPLEMENTED
- ✅ ColBERT projection layers - IMPLEMENTED  
- ❌ Vision transformers (SigLIP, ViT) - NOT IMPLEMENTED
- ❌ Time series models (TinyTimeMixer, TimesFM) - NOT IMPLEMENTED
- ❌ Sparse MLM heads - NOT IMPLEMENTED

**From `src/backends/candle/encoder.rs`:**
```rust
enum BertVariant {
    Bert(candle_transformers::models::bert::BertModel),
    DistilBert(candle_transformers::models::distilbert::DistilBertModel),
    JinaBert(candle_transformers::models::jina_bert::BertModel),
}
```

This is **BERT-specific**, not a general encoder. Adding vision/timeseries requires:
1. Expanding `BertVariant` enum (gets messy)
2. OR creating separate encoder types (`VisionEncoder`, `TimeSeriesEncoder`)

**Recommendation:** Rename `CandleEncoder` → `CandleBertEncoder` and create:
- `CandleVisionEncoder` for ColPali/ViT
- `CandleTimeSeriesEncoder` for TTM/TimesFM
- Common trait: `CandleBackend` or similar

**Candle Vision Support:**
Check if `candle-transformers` has vision models:
- `candle_transformers::models::vit` - Vision Transformer (likely exists)
- `candle_transformers::models::siglip` - SigLIP (may not exist)

If missing, must implement from scratch or use ONNX fallback.

#### Burn Backend Analysis

**Current Status:** Stub implementation (`src/backends/burn/encoder.rs`)

**Burn Capabilities:**
- Supports custom architectures (BERT not built-in)
- Requires manual implementation of all model variants
- More flexible but more work

**Recommendation:** Deprioritize Burn backend until Candle limitations found. Candle is more mature.

---

### 1.3 Trait Boundary Issues

#### CRITICAL: No Unified Encoder Trait

**Current Situation:**
- `TokenEmbedder` trait in `src/core/embeddings.rs` - Returns `TokenEmbeddings` (multi-vector)
- No trait for dense embeddings (single vector)
- No trait for sparse embeddings (vocabulary distribution)
- No trait for time series embeddings
- No trait for vision embeddings

**Problem:**  
Generic code cannot abstract over encoding types. Users must know exact encoder type at compile time.

**Example API Impossibility:**
```rust
// CANNOT DO THIS - no common trait
fn encode_with_any<E: Encoder>(encoder: E, text: &str) -> Embedding {
    encoder.encode(text)
}
```

**Current Workaround:** Enums
```rust
pub enum Encoder {
    ColBERT(ColBERTEncoding),
    Dense(DenseEncoding),
    Sparse(SparseEncoding),
}
```

This works but loses type safety and forces runtime dispatch.

**Recommendation: Unified Trait Hierarchy**

```rust
/// Base trait for all encoders
pub trait Encoder {
    type Output;
    fn encode(&self, input: &str) -> Result<Self::Output>;
}

/// Multi-vector outputs (ColBERT, vision, time series)
pub trait MultiVectorEncoder: Encoder<Output = MultiVectorEmbedding> {
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<MultiVectorEmbedding>>;
}

/// Single-vector outputs (dense embeddings)
pub trait DenseEncoder: Encoder<Output = DenseEmbedding> {
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<DenseEmbedding>>;
}

/// Sparse outputs (SPLADE)
pub trait SparseEncoder: Encoder<Output = SparseEmbedding> {
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<SparseEmbedding>>;
}

/// Unified embedding type with variants
pub enum Embedding {
    MultiVector(MultiVectorEmbedding),
    Dense(DenseEmbedding),
    Sparse(SparseEmbedding),
}
```

**Benefits:**
- Generic code possible over `Encoder` trait
- Subtraits provide paradigm-specific guarantees
- Type-safe but flexible

---

#### TokenEmbedder vs Embedder Distinction

**Current:**
- `TokenEmbedder` in `src/core/embeddings.rs` - Token-level only
- No `Embedder` trait exists

**Issue:**  
Dense encodings don't implement `TokenEmbedder` (they don't produce token-level embeddings). Need separate trait.

**Recommendation:**  
Use the unified hierarchy above. `TokenEmbedder` becomes `MultiVectorEncoder`.

---

### 1.4 Module Dependency Analysis

**Dependency Graph:**
```
api/          → encoding/, quantization/, backends/, models/
encoding/     → backends/, core/
quantization/ → (standalone)
backends/     → core/, models/
core/         → (standalone - only ndarray)
models/       → (standalone - only serde)
bindings/     → api/
```

**Dependency Flow:**
```
core/ (foundation)
  ↓
models/ ← backends/ ← encoding/ ← api/ ← bindings/
  ↓           ↓
quantization/ ←┘
```

**Analysis:**
- ✅ No circular dependencies
- ✅ Core is standalone (good)
- ✅ Models is standalone (good)
- ✅ Quantization is standalone (good)
- ⚠️ Encoding layer is thin - mostly TODOs
- ⚠️ API layer depends on everything - could become God object

**Concern: API Layer Coupling**

`src/api/mod.rs` will depend on all encoding types:
```rust
use crate::encoding::{ColBERTEncoding, DenseEncoding, SparseEncoding, TimeSeriesEncoding, VisionEncoding};
use crate::quantization::{BinaryQuantization, Int8Quantization, Int4Quantization};
use crate::backends::{CandleEncoder, BurnEncoder};
```

This creates tight coupling. Any new encoding type requires API changes.

**Recommendation: Plugin Architecture**

```rust
pub trait EncodingPlugin {
    fn supports_model(&self, model_type: &str) -> bool;
    fn create_encoder(&self, config: &ModelConfig) -> Result<Box<dyn Encoder>>;
}

pub struct TesseraBuilder {
    plugins: Vec<Box<dyn EncodingPlugin>>,
}

impl TesseraBuilder {
    pub fn register_plugin(&mut self, plugin: Box<dyn EncodingPlugin>) {
        self.plugins.push(plugin);
    }
    
    pub fn build(&self) -> Result<Tessera> {
        // Find plugin that supports the model type
        let plugin = self.plugins.iter()
            .find(|p| p.supports_model(&self.model_type))
            .ok_or_else(|| anyhow!("No plugin for model type"))?;
        
        let encoder = plugin.create_encoder(&self.config)?;
        Ok(Tessera { encoder })
    }
}
```

**Benefits:**
- API layer doesn't know about specific encodings
- Adding new encodings doesn't change API code
- Clean separation of concerns

---

## Section 2: Architectural Assessment

### 2.1 Strengths

**1. Clear Layering**
- Core abstractions separate from implementations
- Backends isolated from encoding logic
- Clean separation enables testing and swapping components

**2. Type Safety**
- Strong use of Result types for error handling
- ndarray for mathematical correctness
- Leverages Rust's type system effectively

**3. Documentation Quality**
- Module-level documentation is excellent
- Clear explanations of why each component exists
- Good use of examples in doc comments

**4. Backend Abstraction**
- Support for multiple backends (Candle, Burn) shows good foresight
- Device handling abstracted properly
- HuggingFace Hub integration clean

**5. Model Registry Design**
- Compile-time model metadata generation
- Type-safe model configuration
- JSON-based registry easy to extend

**6. Current ColBERT Implementation**
- Working implementation demonstrates feasibility
- Handles multiple BERT variants (BERT, DistilBERT, JinaBERT)
- Projection layer support present
- MaxSim implementation correct and tested

---

### 2.2 Weaknesses

**1. Incomplete Trait Hierarchy** (Critical)
- No unified `Encoder` trait across paradigms
- Cannot write generic code over different embedding types
- Forces runtime dispatch via enums or lots of duplication

**2. Quantization Design Flaw** (Critical)
- Assumes single fixed-size vectors
- Incompatible with multi-vector outputs (ColBERT, vision, time series)
- Requires redesign before Phase 2

**3. Missing Matryoshka Strategy** (High)
- No implementation plan for dimension truncation
- Unclear interaction with projection layers
- No model registry support for Matryoshka metadata

**4. Sparse Embedding Orphan** (Medium)
- Sparse paradigm incompatible with quantization/Matryoshka
- No inverted index utilities planned
- Unclear how to integrate with retrieval systems

**5. API Layer Coupling Risk** (Medium)
- Builder pattern will tightly couple to all encoding types
- No plugin or registry pattern for extensibility
- Adding new encodings requires API changes

**6. Missing Common Utilities** (Medium)
- No pooling implementation (needed for dense encodings)
- No normalization utilities (L2 norm mentioned but not implemented)
- No similarity functions beyond MaxSim
- No batch padding utilities

**7. Error Handling Lacks Domain Specificity** (Low)
- Using `anyhow::Result` everywhere
- No custom error types for domain-specific failures
- Makes error handling in bindings (Python, WASM) harder

---

### 2.3 Improvement Opportunities

**1. Unified Trait Hierarchy**
Implement the `Encoder` trait family outlined in Section 1.3.

**2. Quantization Redesign**
Adopt per-vector quantization (Option C) to support all paradigms.

**3. Common Utilities Module**
Create `src/utils/` with:
- `pooling.rs` - CLS, mean, max pooling
- `normalization.rs` - L2 normalization, standardization
- `similarity.rs` - Cosine, dot product, Euclidean distance
- `batching.rs` - Padding, masking utilities

**4. Matryoshka Implementation**
Add Matryoshka support to:
- Model registry (`matryoshka_dims` field)
- Model config loading
- Backend encoder (truncation logic)
- API builder (`.dimension()` method)

**5. Custom Error Types**
Define domain-specific errors:
```rust
#[derive(Debug, thiserror::Error)]
pub enum TesseraError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Device unavailable: {0}")]
    DeviceUnavailable(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Quantization not supported for encoding type: {0}")]
    QuantizationNotSupported(String),
}
```

**6. Async Support Planning**
Phase 4 includes async APIs. Plan now:
- Model loading is I/O bound (async)
- Inference is CPU/GPU bound (spawn_blocking)
- Batch processing benefits from async streams

**7. Performance Instrumentation**
Add tracing/metrics hooks:
- Encode timing
- Model loading time
- Memory usage tracking
- Batch size optimization

---

## Section 3: Missing Components

### 3.1 Critical Missing Utilities

#### Pooling Implementations

**Needed for:** Dense embeddings (Phase 2)

**Location:** Create `src/core/pooling.rs`

**Required Functions:**
```rust
pub fn cls_pooling(token_embeddings: &Array2<f32>) -> Result<Array1<f32>>;
pub fn mean_pooling(token_embeddings: &Array2<f32>, attention_mask: &[u32]) -> Result<Array1<f32>>;
pub fn max_pooling(token_embeddings: &Array2<f32>) -> Result<Array1<f32>>;
```

**Status:** NOT PLANNED in completion plan, but REQUIRED for dense encodings.

---

#### Normalization Utilities

**Needed for:** All embedding types (ColBERT, dense, vision)

**Location:** Create `src/core/normalization.rs`

**Required Functions:**
```rust
pub fn l2_normalize(embedding: &mut [f32]);
pub fn l2_normalize_batch(embeddings: &mut Array2<f32>);
pub fn standardize(embedding: &mut [f32], mean: f32, std: f32);
```

**Status:** MENTIONED in docs but NOT IMPLEMENTED.

---

#### Extended Similarity Functions

**Current:** Only MaxSim implemented (`src/core/similarity.rs`)

**Needed for:** Dense embeddings, evaluation, hybrid search

**Missing:**
```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32>;
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32>;
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32>;
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> Result<f32>;
```

**Status:** NOT PLANNED, but essential for dense embeddings and evaluation.

---

#### Batch Padding/Masking

**Needed for:** Batch processing (Phase 1)

**Location:** Extend `src/core/tokenizer.rs` or create `src/core/batching.rs`

**Required:**
```rust
pub fn pad_batch(token_ids: &[Vec<u32>], pad_token: u32) -> (Array2<u32>, Array2<u32>);
pub fn create_attention_mask(token_ids: &[Vec<u32>], pad_token: u32) -> Array2<u32>;
```

**Status:** MENTIONED in Phase 1 but no implementation module specified.

---

### 3.2 Infrastructure Gaps

#### Device Auto-Detection

**Needed for:** API simplicity (`.device("auto")`)

**Current:** `src/backends/candle/device.rs` has `get_device()` but not smart detection

**Missing:**
```rust
pub fn auto_detect_device() -> Device {
    if metal_available() {
        Device::new_metal(0).unwrap()
    } else if cuda_available() {
        Device::new_cuda(0).unwrap()
    } else {
        Device::Cpu
    }
}

pub fn metal_available() -> bool;
pub fn cuda_available() -> bool;
```

**Status:** NOT IMPLEMENTED, mentioned in Phase 1 API section.

---

#### Model Caching Strategy

**Needed for:** Fast model loading, disk space management

**Missing:**
- HuggingFace Hub cache management
- Model version pinning
- Cache invalidation strategy
- Shared cache across instances

**Current:** Uses default hf-hub cache, no management

**Recommendation:** Document cache location, add cache clearing utility.

---

#### Memory Management

**Needed for:** Large model handling, OOM prevention

**Missing:**
- Model memory footprint estimation
- OOM prediction/prevention
- Model unloading/reloading
- Memory-mapped model loading (Candle supports this)

**Status:** NOT PLANNED explicitly.

---

### 3.3 Missing Feature Infrastructure

#### Hybrid Search Support

**Mentioned in:** COMPLETION_PLAN.md ("Hybrid search (dense + sparse)")

**Missing Components:**
- Score fusion algorithms (RRF, linear combination)
- Multi-encoder orchestration
- Result merging utilities

**Recommendation:** Add `src/search/` module with:
- `fusion.rs` - Score combination strategies
- `hybrid.rs` - Multi-encoder query execution

---

#### Inverted Index Integration (Sparse)

**Needed for:** Sparse embeddings (Phase 2)

**Missing:**
- Sparse vector → posting list conversion
- Term weight serialization
- Index building utilities
- Integration with search engines (Tantivy, Elasticsearch)

**Location:** Create `src/sparse/` module with:
- `index.rs` - Inverted index building
- `retrieval.rs` - Sparse search algorithms

---

#### Matryoshka Dimension Metadata

**Needed for:** Matryoshka support (Phase 1-2)

**Missing from `models.json`:**
```json
"matryoshka": {
    "enabled": true,
    "method": "truncate",  // or "project"
    "supported_dims": [64, 96, 128, 256, 512, 768],
    "training_dims": [64, 96, 128, 256, 512, 768]
}
```

**Current:** Some models have `matryoshka` field, but no `method` specification.

**Recommendation:** Add `method` field indicating truncation vs projection approach.

---

#### Configuration Management

**Needed for:** Production deployments

**Missing:**
- Environment variable configuration
- Config file loading (YAML, TOML)
- Runtime configuration updates
- Validation of configuration combinations

**Example:**
```toml
# tessera.toml
[model]
id = "jina-colbert-v2"
dimension = 96

[device]
type = "auto"
fallback = "cpu"

[performance]
batch_size = 32
cache_size = 1000
```

---

## Section 4: Recommendations

### 4.1 Priority Improvements

#### PRIORITY 1: Quantization Redesign (Before Phase 2)

**Action Items:**
1. Refactor `src/quantization/mod.rs` to per-vector quantization
2. Update `BinaryQuantization`, `Int8Quantization`, `Int4Quantization` to new design
3. Add tests for multi-vector quantization
4. Document incompatible combinations (sparse + binary)

**Estimated Effort:** 1-2 days  
**Blocks:** Phase 2 implementation

---

#### PRIORITY 2: Unified Trait Hierarchy (Before Phase 2)

**Action Items:**
1. Define `Encoder` trait family in `src/core/encoder.rs`
2. Refactor existing `TokenEmbedder` to `MultiVectorEncoder`
3. Add `DenseEncoder`, `SparseEncoder` traits
4. Update API to use trait objects or enum wrappers
5. Add trait bounds to builder pattern

**Estimated Effort:** 2-3 days  
**Blocks:** API generalization, multi-paradigm support

---

#### PRIORITY 3: Common Utilities Module (During Phase 2)

**Action Items:**
1. Create `src/core/pooling.rs` with CLS, mean, max pooling
2. Create `src/core/normalization.rs` with L2 norm
3. Extend `src/core/similarity.rs` with cosine, dot, Euclidean
4. Create `src/core/batching.rs` with padding utilities
5. Add comprehensive tests for all utilities

**Estimated Effort:** 2-3 days  
**Blocks:** Dense encoding implementation

---

#### PRIORITY 4: Matryoshka Implementation Strategy (Phase 1-2)

**Action Items:**
1. Add `matryoshka_method` field to model registry
2. Implement truncation logic in Candle encoder
3. Update API builder with `.dimension()` support
4. Add validation for supported dimensions
5. Test with Jina-ColBERT-v2 at multiple dimensions

**Estimated Effort:** 1-2 days  
**Blocks:** Dimension flexibility feature

---

### 4.2 Architectural Changes

#### Change 1: Rename CandleEncoder → CandleBertEncoder

**Rationale:** Current name implies general-purpose encoder, but it's BERT-specific.

**Action:**
- Rename `src/backends/candle/encoder.rs` → `bert_encoder.rs`
- Update struct name: `CandleEncoder` → `CandleBertEncoder`
- Prepare for `CandleVisionEncoder`, `CandleTimeSeriesEncoder`

---

#### Change 2: Introduce Utils Module

**Create:** `src/utils/` with:
- `pooling.rs`
- `normalization.rs`
- `batching.rs`
- `device.rs` (move device detection here)

**Update:** `src/lib.rs` to export common utils

---

#### Change 3: Add Custom Error Types

**Create:** `src/error.rs` with `TesseraError` enum

**Update:** All `Result<T>` to `Result<T, TesseraError>`

**Benefits:**
- Better error messages
- Easier Python/WASM bindings
- More structured error handling

---

#### Change 4: Plugin Architecture for Encodings

**Rationale:** Decouple API from specific encoding implementations

**Implementation:**
- Define `EncodingPlugin` trait
- Register plugins in builder
- Dynamic dispatch to appropriate encoder

**Benefits:**
- API doesn't change when adding encodings
- Third-party encodings possible
- Cleaner separation of concerns

---

### 4.3 Simplifications

#### Simplification 1: Remove Burn Backend (Short-term)

**Rationale:**
- Burn support is incomplete (stub only)
- Candle is working and mature
- Maintaining two backends doubles effort

**Recommendation:**
- Comment out Burn in Cargo.toml
- Keep code but don't test/maintain
- Revisit if Candle limitations found

**Effort Saved:** 20-30% on backend work

---

#### Simplification 2: Deprioritize Sparse Embeddings

**Rationale:**
- Sparse has unique constraints (no quantization, no Matryoshka)
- Requires inverted index infrastructure (not planned)
- Less demand than dense/multi-vector

**Recommendation:**
- Move sparse to Phase 3 or later
- Focus Phase 2 on dense + multi-vector
- Implement sparse when inverted index integration clear

**Effort Saved:** 1-2 weeks in Phase 2

---

#### Simplification 3: API Builder - Start Simple

**Current Plan:** Full builder with all options

**Simpler Approach:**
```rust
impl Tessera {
    pub fn new(model_id: &str) -> Result<Self>;  // Auto everything
    pub fn with_device(model_id: &str, device: Device) -> Result<Self>;
    pub fn with_config(config: TesseraConfig) -> Result<Self>;  // Advanced
}
```

**Benefits:**
- 90% of users just call `new()`
- Advanced users use `with_config()`
- Fewer builder methods to maintain

---

### 4.4 Specific Technical Concerns

#### Concern 1: Binary Quantization + Variable-Length Multi-Vector

**Question:** Does binary quantization work with N vectors per document?

**Answer:** YES, with per-vector quantization design.

**Implementation:**
```rust
// Quantize each token vector independently
let quantized: Vec<BinaryEmbedding> = token_embeddings
    .iter()
    .map(|token_vec| quantizer.quantize_vector(token_vec))
    .collect();

// Distance: Apply MaxSim over Hamming distances
fn maxsim_binary(query: &[BinaryEmbedding], doc: &[BinaryEmbedding]) -> f32 {
    query.iter().map(|q| {
        doc.iter().map(|d| hamming_similarity(q, d)).max().unwrap()
    }).sum()
}
```

**Status:** Feasible, requires quantization redesign.

---

#### Concern 2: Matryoshka + Projection Layers

**Question:** Project THEN truncate, or truncate THEN project?

**Answer:** Depends on model training.

**Analysis:**
- **Jina-ColBERT-v2:** No projection, so truncate hidden states directly
- **ColBERT v2:** Has projection (768 → 128), trained on 128-dim output
  - Matryoshka would require **retraining** or **truncating 128 dims**
  - Original ColBERT NOT trained with Matryoshka

**Recommendation:**
- For models with Matryoshka support: Truncate as trained (check model card)
- For models without: Only support full dimension
- Add validation: Reject dimension if not in `supported_dims`

---

#### Concern 3: Sparse + Dense API Coexistence

**Question:** Can sparse (30K dims) and dense (768 dims) coexist cleanly?

**Answer:** YES, with enum wrapper or trait objects.

**Option 1: Enum**
```rust
pub enum Embedding {
    Dense(Vec<f32>),
    Sparse(HashMap<u32, f32>),
    MultiVector(Vec<Vec<f32>>),
}
```

**Option 2: Trait Objects**
```rust
pub trait Embedding {
    fn similarity(&self, other: &dyn Embedding) -> Result<f32>;
}

impl Embedding for DenseEmbedding { ... }
impl Embedding for SparseEmbedding { ... }
```

**Recommendation:** Use enum for simplicity. Trait objects add complexity without clear benefit.

---

## Section 5: Testing Strategy Gaps

### Missing Test Coverage

**Current:** Only MaxSim has tests

**Needed:**
1. **Quantization tests:**
   - Accuracy retention measurement
   - Round-trip encoding/decoding
   - Distance computation correctness

2. **Pooling tests:**
   - CLS pooling correctness
   - Mean pooling with masks
   - Max pooling edge cases

3. **Normalization tests:**
   - L2 norm verification
   - Batch normalization

4. **Integration tests:**
   - End-to-end encoding pipeline
   - Model loading and inference
   - Batch processing correctness

5. **Performance tests:**
   - Batch vs single throughput
   - Quantization speedup
   - Memory usage

**Recommendation:** Add `tests/` directory with integration tests separate from unit tests.

---

## Section 6: Documentation Gaps

### Missing Documentation

**Current:** Good module docs, no user guides

**Needed:**
1. **Architecture guide** (this document is a start)
2. **Contributing guide** (how to add new encodings)
3. **Performance tuning guide** (batch sizes, quantization choices)
4. **Model selection guide** (when to use which model)
5. **Deployment guide** (Docker, cloud, edge)

**Recommendation:** Create `docs/guides/` directory with Markdown guides.

---

## Appendix: Action Plan Summary

### Phase 0: Pre-Phase 1 Refactoring (1 week)

**Before starting Phase 1 batch processing:**
1. ✅ Rename `CandleEncoder` → `CandleBertEncoder`
2. ✅ Create `src/utils/` module with pooling, normalization, similarity
3. ✅ Redesign quantization to per-vector approach
4. ✅ Add custom error types (`TesseraError`)
5. ✅ Define unified `Encoder` trait hierarchy
6. ✅ Add Matryoshka metadata to model registry

### Phase 1: Modified (4 weeks)

**Batch Processing:**
- Implement in `CandleBertEncoder` with new batching utils

**Binary Quantization:**
- Use new per-vector design
- Test with ColBERT multi-vector

**Matryoshka:**
- Implement truncation in encoder
- Validate against model registry

**API Simplification:**
- Use simplified builder approach
- Auto-device detection

### Phase 2: Modified (5 weeks)

**Dense Embeddings:**
- Use pooling utils from `utils/`
- Implement `DenseEncoder` trait

**Python Bindings:**
- Use `TesseraError` for clean error handling

**Skip Sparse:**
- Move to Phase 3 or later

### Phase 3+: As Planned

Vision, time series, hyperbolic proceed as outlined.

---

## Conclusion

Tessera's architecture is **fundamentally sound** with excellent documentation and clear separation of concerns. However, several **critical integration issues** must be resolved before Phase 2:

1. **Quantization redesign** - Current design incompatible with multi-vector
2. **Unified trait hierarchy** - No common abstraction across paradigms
3. **Common utilities** - Missing pooling, normalization, similarity functions
4. **Matryoshka strategy** - Implementation location ambiguous

**Recommendation:** Allocate 1 week for architectural refactoring before starting Phase 1 implementation. This upfront investment will prevent costly rework and enable clean multi-paradigm support.

The scaffolded code is well-structured and ready for implementation once these architectural issues are addressed.

**Overall Assessment: STRONG FOUNDATION, NEEDS REFINEMENT BEFORE SCALING**
