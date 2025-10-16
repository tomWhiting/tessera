# Phase 0: Architectural Refactoring

**Duration:** 1 week
**Goal:** Fix critical architectural issues before scaling to multiple paradigms
**Status:** In Progress

---

## Overview

Before implementing Phase 1 (batch processing, quantization, expanded models), we must address fundamental architectural issues discovered during the technical review. These issues would cause costly rework if not resolved upfront.

### Why Phase 0?

The current architecture works well for ColBERT but has design flaws that prevent clean multi-paradigm support:

1. **Quantization trait is single-vector only** - breaks with ColBERT's variable-length multi-vector
2. **No unified encoder abstraction** - can't write generic code across paradigms
3. **Missing common utilities** - pooling, normalization, similarity functions needed repeatedly
4. **Matryoshka unclear** - truncation logic not specified
5. **Error types generic** - using `anyhow::Result` everywhere lacks structure

Fixing these now enables clean Phase 1-4 implementation.

---

## Task 0.1: Redesign Quantization for Multi-Vector

**File:** `src/quantization/mod.rs`, `src/quantization/binary.rs`

### Current Problem

```rust
// Current trait - assumes single fixed-size vector
pub trait Quantization {
    type Output;
    fn quantize(&self, embeddings: &[f32]) -> Self::Output;
    fn distance(&self, a: &Self::Output, b: &Self::Output) -> f32;
}
```

This breaks with ColBERT which outputs `Vec<Vec<f32>>` (variable number of vectors).

### Solution

**Per-Vector Quantization Approach:**

```rust
// Redesigned trait - works per vector
pub trait Quantization {
    type Output;

    // Quantize a single vector
    fn quantize_vector(&self, vector: &[f32]) -> Self::Output;

    // Dequantize back to float (for exact search)
    fn dequantize_vector(&self, quantized: &Self::Output) -> Vec<f32>;

    // Distance between two quantized vectors
    fn distance(&self, a: &Self::Output, b: &Self::Output) -> f32;
}

// For multi-vector embeddings
impl BinaryQuantization {
    fn quantize_multi(&self, vectors: &[Vec<f32>]) -> Vec<BinaryVector> {
        vectors.iter().map(|v| self.quantize_vector(v)).collect()
    }
}
```

### Changes Required

**1. Update `src/quantization/mod.rs`:**
- Redesign `Quantization` trait with per-vector methods
- Add helper methods for multi-vector quantization
- Document the design decision

**2. Update `src/quantization/binary.rs`:**
- Implement new trait design
- Add `quantize_multi` for multi-vector
- Add `hamming_distance_multi` for MaxSim over Hamming

**3. Update stubs for int8/int4:**
- `src/quantization/int8.rs`
- `src/quantization/int4.rs`
- Use same per-vector approach

### Success Criteria

- [ ] Quantization trait works with single vectors (dense)
- [ ] Works with multi-vector (ColBERT)
- [ ] Works with time series (patch-level)
- [ ] Works with vision (patch embeddings)
- [ ] Binary quantization maintains 32x compression, 95%+ accuracy
- [ ] All tests pass

---

## Task 0.2: Create Unified Encoder Trait Hierarchy

**File:** `src/core/embeddings.rs`

### Current Problem

Only `TokenEmbedder` exists (multi-vector specific). No abstraction for dense, sparse, or other paradigms. Cannot write generic code over different encoder types.

### Solution

**Trait Hierarchy:**

```rust
/// Base trait for all encoders
pub trait Encoder {
    type Output;

    fn encode(&self, input: &str) -> Result<Self::Output>;
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Self::Output>>;
}

/// Multi-vector encoder (ColBERT)
pub trait MultiVectorEncoder: Encoder<Output = TokenEmbeddings> {
    fn num_vectors(&self, text: &str) -> Result<usize>;
}

/// Single-vector encoder (dense embeddings)
pub trait DenseEncoder: Encoder<Output = DenseEmbedding> {
    fn embedding_dim(&self) -> usize;
}

/// Sparse encoder (SPLADE)
pub trait SparseEncoder: Encoder<Output = SparseEmbedding> {
    fn vocab_size(&self) -> usize;
    fn sparsity(&self, embedding: &SparseEmbedding) -> f32;
}
```

### Changes Required

**1. Update `src/core/embeddings.rs`:**
- Define base `Encoder` trait
- Define paradigm-specific subtraits
- Add `DenseEmbedding` type (single vector)
- Add `SparseEmbedding` type (sparse vector)
- Keep `TokenEmbeddings` for multi-vector

**2. Update `src/backends/candle/encoder.rs`:**
- Implement `MultiVectorEncoder` for `CandleEncoder`
- Rename to `CandleBertEncoder` (prepare for vision/timeseries variants)

**3. Update existing code:**
- `src/encoding/colbert.rs` - use `MultiVectorEncoder`
- Examples remain unchanged (concrete types work)

### Success Criteria

- [ ] Base `Encoder` trait defined
- [ ] Subtraits for multi-vector, dense, sparse
- [ ] Backward compatible (existing code works)
- [ ] Can write generic functions over `Encoder`
- [ ] All tests pass

---

## Task 0.3: Add Utils Module (Pooling, Similarity, Normalization)

**Files:** `src/utils/mod.rs`, `src/utils/pooling.rs`, `src/utils/similarity.rs`, `src/utils/normalization.rs`

### Current Problem

Missing common utilities needed across paradigms:
- Pooling (CLS, mean, max) - required for dense encodings
- Similarity (cosine, dot, Euclidean) - only MaxSim exists
- Normalization (L2 norm) - mentioned but not implemented
- Batching utilities (padding, masking) - needed for batch processing

### Solution

Create comprehensive utilities module with reusable components.

### Changes Required

**1. Create `src/utils/mod.rs`:**
```rust
//! Common utilities for embedding operations.
//!
//! Provides reusable components used across different encoding paradigms:
//! - Pooling strategies (CLS, mean, max)
//! - Similarity functions (cosine, dot product, Euclidean, MaxSim)
//! - Normalization (L2 norm, standardization)
//! - Batching utilities (padding, masking)

pub mod pooling;
pub mod similarity;
pub mod normalization;
pub mod batching;

pub use pooling::{cls_pooling, mean_pooling, max_pooling};
pub use similarity::{cosine_similarity, dot_product, euclidean_distance};
pub use normalization::l2_normalize;
```

**2. Implement `src/utils/pooling.rs`:**
- `cls_pooling(embeddings, mask)` - Extract first token
- `mean_pooling(embeddings, mask)` - Average with mask weighting
- `max_pooling(embeddings, mask)` - Element-wise max

**3. Implement `src/utils/similarity.rs`:**
- Move `max_sim` from `src/core/similarity.rs` here
- Add `cosine_similarity(a, b)` - dot(a,b) / (||a|| ||b||)
- Add `dot_product(a, b)`
- Add `euclidean_distance(a, b)`

**4. Implement `src/utils/normalization.rs`:**
- `l2_normalize(vec)` - Divide by L2 norm
- `l2_norm(vec)` - Compute L2 norm
- `standardize(vec)` - Zero mean, unit variance

**5. Implement `src/utils/batching.rs`:**
- `pad_sequences(sequences, pad_token)` - Pad to max length
- `create_attention_mask(sequences)` - Generate masks
- `unpad_sequences(padded, masks)` - Remove padding

### Success Criteria

- [ ] All pooling strategies implemented and tested
- [ ] All similarity functions match reference implementations
- [ ] Normalization is numerically stable
- [ ] Batching utilities handle edge cases (empty sequences, single token)
- [ ] Comprehensive documentation with examples
- [ ] All tests pass

---

## Task 0.4: Implement Matryoshka Truncation Logic

**Files:** `src/models/registry.rs`, `src/core/embeddings.rs`, `src/utils/matryoshka.rs`

### Current Problem

Unclear when/how to truncate embeddings for Matryoshka:
- Models with projection (ColBERT v2): Truncate hidden states before projection?
- Models without projection (Jina-ColBERT): Truncate output directly?
- What about dense models?

### Solution

**Strategy per model type:**

**Type 1: With Projection (ColBERT v2)**
```
BERT (768-dim) → Truncate to D → Project (D → 128) → Output (128-dim)
```

**Type 2: Without Projection (Jina-ColBERT)**
```
BERT (768-dim) → Truncate directly → Output (D-dim)
```

**Type 3: Dense Models (BGE, Nomic)**
```
BERT (768-dim) → Pool → Truncate → Output (D-dim)
```

### Changes Required

**1. Add to `models.json`:**
```json
"matryoshka": {
  "strategy": "truncate_hidden" | "truncate_output" | "truncate_pooled",
  "dimensions": [64, 128, 256, 512, 768]
}
```

**2. Create `src/utils/matryoshka.rs`:**
```rust
pub enum MatryoshkaStrategy {
    TruncateHidden,   // Before projection
    TruncateOutput,   // After projection
    TruncatePooled,   // After pooling (dense)
}

pub fn apply_matryoshka(
    embeddings: &Tensor,
    target_dim: usize,
    strategy: MatryoshkaStrategy
) -> Result<Tensor>;
```

**3. Update `src/backends/candle/encoder.rs`:**
- Check model's Matryoshka strategy
- Apply truncation at correct point in pipeline

**4. Add validation in `build.rs`:**
- Ensure Matryoshka strategy matches model architecture

### Success Criteria

- [ ] Truncation works for all three strategies
- [ ] Jina-ColBERT 96/64-dim variants produce correct outputs
- [ ] Nomic Embed works at 64, 128, 256, 512, 768
- [ ] Validation prevents incorrect configurations
- [ ] All tests pass

---

## Task 0.5: Add Custom Error Types

**File:** `src/error.rs`

### Current Problem

Using `anyhow::Result` everywhere. Good for prototyping, but production libraries need structured errors for:
- Better error messages
- Programmatic error handling
- Error categorization
- Debugging

### Solution

Create domain-specific error types.

### Changes Required

**1. Create `src/error.rs`:**
```rust
//! Error types for Tessera.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TesseraError {
    #[error("Model not found in registry: {model_id}")]
    ModelNotFound { model_id: String },

    #[error("Failed to load model {model_id}: {source}")]
    ModelLoadError {
        model_id: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Encoding failed for text: {context}")]
    EncodingError {
        context: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Unsupported dimension {requested} for model {model_id}. Supported: {supported:?}")]
    UnsupportedDimension {
        model_id: String,
        requested: usize,
        supported: Vec<usize>,
    },

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Quantization error: {0}")]
    QuantizationError(String),

    // ... more variants
}

pub type Result<T> = std::result::Result<T, TesseraError>;
```

**2. Add dependency:**
```toml
[dependencies]
thiserror = "1.0"
```

**3. Update all `Result` types:**
- Change `anyhow::Result` → `crate::Result` (our `TesseraError`)
- Keep `anyhow` internally, convert to `TesseraError` at boundaries
- Update `src/lib.rs` to export error types

### Success Criteria

- [ ] All public APIs use `TesseraError`
- [ ] Error messages are helpful and actionable
- [ ] Internal code can still use `anyhow` with `.map_err()`
- [ ] All tests pass
- [ ] Documentation includes error handling examples

---

## Task 0.6: Deprioritize Burn Backend

**Files:** `src/backends/burn/*`, documentation

### Action

**DO NOT remove** - keep for future potential (training, fine-tuning capabilities).

**Changes:**
1. Add note in `src/backends/mod.rs` that Burn is experimental/deprioritized
2. Focus all Phase 0-1 work on Candle backend only
3. Skip Burn implementations until Candle path is proven

**Documentation:**
```rust
//! - [`burn`]: Experimental backend (DEPRIORITIZED).
//!   Currently incomplete. Future work may leverage Burn's training
//!   capabilities for model fine-tuning. Production use should rely
//!   on Candle backend.
```

---

## Phase 0 Task Breakdown

### Week 1: Refactoring

**Day 1-2: Quantization Redesign**
- [x] Redesign trait (per-vector approach)
- [x] Update binary quantization implementation
- [x] Update int8/int4 stubs
- [x] Tests for multi-vector compatibility

**Day 2-3: Encoder Trait Hierarchy**
- [x] Define base `Encoder` trait
- [x] Define subtraits (MultiVector, Dense, Sparse)
- [x] Update existing implementations
- [x] Backward compatibility verified

**Day 3-4: Utils Module**
- [x] Create module structure
- [x] Implement pooling (CLS, mean, max)
- [x] Implement similarity functions
- [x] Implement normalization
- [x] Comprehensive tests

**Day 4-5: Matryoshka Logic**
- [x] Add strategy to models.json schema
- [x] Implement truncation utilities
- [x] Integrate with encoders
- [x] Test with Jina variants

**Day 5: Error Types**
- [x] Create TesseraError enum
- [x] Add thiserror dependency
- [x] Update public APIs
- [x] Update examples

**Day 5: Review & Integration**
- [x] Separate specialist reviews all work
- [x] Integration testing
- [x] Documentation updated
- [x] Commit Phase 0 complete

---

## Deliverables

### Code
- [x] Redesigned quantization in `src/quantization/mod.rs`
- [x] Unified encoder traits in `src/core/embeddings.rs`
- [x] Utils module: `src/utils/{pooling, similarity, normalization, batching}.rs`
- [x] Matryoshka logic in `src/utils/matryoshka.rs`
- [x] Custom errors in `src/error.rs`

### Documentation
- [x] Architecture decisions documented
- [x] Trait design rationale explained
- [x] Migration notes (if APIs changed)

### Tests
- [x] Quantization with multi-vector
- [x] Pooling correctness
- [x] Similarity function accuracy
- [x] Matryoshka truncation
- [x] Error type coverage

### Success Criteria
- [ ] All architectural issues resolved
- [ ] Clean foundation for Phase 1
- [ ] Backward compatible where possible
- [ ] Zero warnings from clippy
- [ ] All tests passing
- [ ] Documentation comprehensive

---

**Status:** Ready for implementation
**Next:** Begin Task 0.1 (Quantization Redesign)
