# Tessera Performance Optimization Plan

## Overview

This document outlines a comprehensive plan for implementing the remaining performance optimizations in Tessera, based on exhaustive research of current best practices (February 2026).

---

## Phase 1: SIMD Optimizations (CPU Operations)

**Goal:** 2-5x speedup for CPU-side operations (pooling, similarity, normalization)

### Research Findings

- **`std::simd` is NOT stable in 2026** and won't be for the foreseeable future—avoid it
- **Recommended approach:** Use `pulp` or `macerator` crates for portable SIMD with runtime dispatch
- **For embedding-specific operations:** `SimSIMD` offers up to 200x faster dot products

### Implementation Plan

#### 1.1 Add Dependencies

```toml
# Cargo.toml additions
[dependencies]
pulp = "0.18"           # Runtime SIMD dispatch (stable Rust)
# OR
simsimd = "5.0"         # Ultra-optimized similarity metrics (if Rust bindings mature)

# For ndarray BLAS acceleration on Apple Silicon
[target.'cfg(target_os = "macos")'.dependencies]
blas-src = { version = "0.10", features = ["accelerate"] }
ndarray = { version = "0.16", features = ["blas"] }
```

#### 1.2 Optimize Pooling Operations (`src/utils/pooling.rs`)

Replace current scalar loops with SIMD-accelerated versions:

- `mean_pooling`: Use `pulp` for vectorized accumulation
- `max_pooling`: Use `pulp` for SIMD max reduction
- `cls_pooling`: Already optimal (single row copy)
- `last_token_pooling`: Already optimal (single row copy)

**Priority:** HIGH - These are hot-path operations called for every embedding

#### 1.3 Optimize Similarity Functions (`src/utils/similarity.rs`)

- `cosine_similarity`: Fused SIMD dot product + norm
- `dot_product`: Direct SIMD implementation
- `euclidean_distance`: SIMD squared difference accumulation
- `max_sim`: Most critical—use streaming SIMD without full matrix allocation

**Priority:** HIGH - `max_sim` is critical for ColBERT retrieval performance

#### 1.4 Optimize Normalization (`src/utils/normalization.rs`)

- `l2_norm`: SIMD squared sum + single sqrt
- `l2_normalize`: Fused SIMD norm + division

**Priority:** MEDIUM - Called once per embedding output

### Testing Strategy

- Benchmark before/after with criterion
- Test across input sizes (64, 128, 384, 768, 1024 dimensions)
- Verify numerical equivalence with existing implementations

---

## Phase 2: Sparse Encoder Batch Processing

**Goal:** Enable true batch encoding for SPLADE models (currently 100% serial)

### Research Findings

- Use **token-count budgeting** rather than fixed batch counts
- Store sparse vectors as parallel arrays: `indices: Vec<u32>`, `values: Vec<f32>`
- Use **COO format during construction**, convert to **CSR for batch operations**
- `max_active_dims` limiting can reduce memory by ~43% with minimal accuracy loss

### Implementation Plan

#### 2.1 Add Sparse Matrix Dependencies

```toml
[dependencies]
sprs = "0.11"                    # Pure Rust sparse linear algebra
# OR
scirs2-sparse = { version = "0.3", features = ["parallel", "simd"] }
```

#### 2.2 Implement Batch Tokenization (`src/encoding/sparse.rs`)

```
Pipeline:
1. Batch tokenize all inputs (already exists)
2. Pad to uniform length within batch
3. Single forward pass through model
4. Extract sparse outputs in parallel
5. Apply max_active_dims limiting if configured
```

#### 2.3 Memory-Efficient Output Handling

- Pre-sort inputs by length to minimize padding
- Use bounded batch sizes based on token count (not item count)
- Implement streaming output for very large batches

#### 2.4 Configuration Interface

```rust
pub struct SparseBatchConfig {
    pub max_batch_tokens: usize,      // Total tokens per batch (e.g., 8192)
    pub max_active_dims: Option<usize>, // Limit non-zero dims (e.g., 128)
    pub presort_by_length: bool,      // Enable length-based sorting
}
```

### Testing Strategy

- Compare output equivalence with serial encoding
- Benchmark throughput: items/second at various batch sizes
- Memory profiling: peak VRAM usage vs batch size
- Test with variable-length inputs

---

## Phase 3: Tensor Memory Pooling

**Goal:** Reduce GPU memory pressure through buffer reuse

### Research Findings

- **Metal Resource Heaps** are the primary pooling mechanism for Apple Silicon
- Use **aliasable resources** for intermediate tensors with non-overlapping lifetimes
- **Triple buffering** pattern prevents CPU-GPU pipeline stalls
- **Static memory planning** at model load time enables optimal pre-allocation

### Implementation Plan

#### 3.1 Memory Pool Architecture

```
Three-Tier System:
├── Tier 1: Static Pool (60-70% VRAM)
│   └── Model weights, permanent allocations
├── Tier 2: Dynamic Arena (20-25% VRAM)
│   └── Intermediate activations, batch buffers
└── Tier 3: Scratch Buffer (5-10% VRAM)
    └── Operator-specific temporary allocations
```

#### 3.2 Storage Mode Strategy (Metal-Specific)

| Data Type | Storage Mode | Rationale |
|-----------|--------------|-----------|
| Weights | Private | GPU-only, never modified |
| KV Cache | Managed | Occasional CPU readback |
| Batch Inputs | Shared | Frequent CPU upload |
| Activations | Private | GPU-only, temporary |

#### 3.3 Implementation Components

**a) Tensor Pool Manager**
```rust
pub struct TensorPool {
    heap: MTLHeap,
    free_list: Vec<TensorSlot>,
    allocation_stats: AllocationStats,
}

impl TensorPool {
    fn allocate(&mut self, shape: &[usize], dtype: DType) -> PooledTensor;
    fn release(&mut self, tensor: PooledTensor);
    fn defragment(&mut self);  // Periodic compaction
}
```

**b) Lifetime Analysis (Optional, Phase 3b)**
- Analyze computation graph at model load
- Build tensor lifetime map using ASAP/ALAP scheduling
- Pre-compute buffer assignments

#### 3.4 Configuration Interface

```rust
pub struct MemoryConfig {
    pub total_pool_size_mb: usize,    // Total VRAM budget
    pub static_pool_ratio: f32,       // 0.6-0.8 for weights
    pub enable_defragmentation: bool,
    pub defrag_interval_secs: u64,
}
```

### Testing Strategy

- Track allocation counts before/after
- Monitor peak VRAM usage under load
- Benchmark sustained throughput (multi-batch sequences)
- Test memory recovery after large batch processing

---

## Phase 4: Parallel Processing with Rayon

**Goal:** Parallelize CPU-intensive preprocessing

### Research Findings

- **PDF rendering:** Parallelize across documents, NOT within documents (Pdfium mutex limitation)
- **Image preprocessing:** Use `par_iter()` with `with_min_len()` tuning
- **CPU-GPU coordination:** Separate Rayon pool from async GPU layer, use bounded queues
- **Thread count:** Start at logical cores, reduce if GPU becomes bottleneck

### Implementation Plan

#### 4.1 Add Rayon Dependency

```toml
[dependencies]
rayon = "1.10"
```

#### 4.2 Parallel PDF Processing (`src/utils/pdf.rs`)

**Current (serial):**
```rust
pub fn render_all_pages(&self, pdf_path: &Path, dpi: u32) -> Result<Vec<DynamicImage>> {
    (0..count).map(|i| self.render_page(pdf_path, i, dpi)).collect()
}
```

**Optimized (parallel across pages with batching):**
```rust
use rayon::prelude::*;

pub fn render_all_pages(&self, pdf_path: &Path, dpi: u32) -> Result<Vec<DynamicImage>> {
    let pdf = PDF::from_file(pdf_path)?;
    let count = pdf.page_count();

    // Parallelize page rendering with minimum batch size
    (0..count)
        .into_par_iter()
        .with_min_len(4)  // Batch at least 4 pages per task
        .map(|i| self.render_single_page(&pdf, i, dpi))
        .collect()
}
```

**Note:** Due to Pdfium's mutex, true parallelism requires multiple PDF instances or document-level parallelism.

#### 4.3 Parallel Image Preprocessing (`src/vision/preprocessing.rs`)

```rust
pub fn preprocess_batch(&self, images: &[DynamicImage], device: &Device) -> Result<Tensor> {
    let preprocessed: Vec<_> = images
        .par_iter()
        .with_min_len(8)  // Batch at least 8 images per task
        .map(|img| self.preprocess_single(img))
        .collect::<Result<Vec<_>>>()?;

    // Stack into batch tensor
    Tensor::stack(&preprocessed, 0)
}
```

#### 4.4 Thread Pool Configuration

```rust
// In lib.rs or config module
pub fn configure_thread_pool(config: &ThreadConfig) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        }))
        .build_global()?;
    Ok(())
}
```

#### 4.5 CPU-GPU Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Rayon Pool     │     │  Bounded Queue   │     │  Async GPU      │
│  (CPU Preproc)  │ ──► │  (Backpressure)  │ ──► │  (Inference)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

- Bounded queue prevents memory explosion if CPU outpaces GPU
- Backpressure naturally throttles preprocessing
- GPU executor runs in separate async runtime (tokio)

### Testing Strategy

- Benchmark preprocessing throughput: images/second
- Measure GPU utilization with different thread counts
- Test queue depth under various load patterns
- Profile for contention bottlenecks

---

## Phase 5: Vision Encoder Improvements

**Goal:** Batch image encoding and caching

### Implementation Plan

#### 5.1 Batch Image Encoding

```rust
pub fn encode_images(&self, image_paths: &[&Path]) -> Result<Vec<VisionEmbedding>> {
    // 1. Parallel preprocessing (Rayon)
    let preprocessed = self.preprocess_batch(image_paths)?;

    // 2. Single batched forward pass
    let batch_output = self.model.forward(&preprocessed)?;

    // 3. Extract individual embeddings
    self.extract_batch_embeddings(batch_output, image_paths)
}
```

#### 5.2 Image Preprocessing Cache

```rust
pub struct ImageCache {
    cache: LruCache<PathBuf, Tensor>,
    max_size_mb: usize,
}

impl ImageCache {
    pub fn get_or_preprocess(&mut self, path: &Path, processor: &ImageProcessor) -> Result<Tensor> {
        if let Some(tensor) = self.cache.get(path) {
            return Ok(tensor.clone());
        }
        let tensor = processor.preprocess_from_path(path)?;
        self.cache.put(path.to_path_buf(), tensor.clone());
        Ok(tensor)
    }
}
```

---

## Phase 6: Adaptive Batch Sizing

**Goal:** Automatically determine optimal batch size based on available GPU memory

### Implementation Plan

#### 6.1 Memory Estimation

```rust
pub struct BatchSizer {
    per_sample_memory: usize,  // Estimated bytes per sample
    available_memory: usize,   // Total available VRAM
    safety_margin: f32,        // 0.85-0.90
}

impl BatchSizer {
    pub fn optimal_batch_size(&self) -> usize {
        let usable = (self.available_memory as f32 * self.safety_margin) as usize;
        usable / self.per_sample_memory
    }

    pub fn calibrate(&mut self, device: &Device) {
        // Run test batches to measure actual memory usage
        // Binary search for maximum stable batch size
    }
}
```

#### 6.2 Dynamic Adjustment

```rust
pub fn encode_adaptive(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>> {
    let batch_size = self.batch_sizer.optimal_batch_size();

    texts.chunks(batch_size)
        .map(|chunk| self.encode_batch(chunk))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect()
}
```

---

## Implementation Timeline

| Phase | Estimated Effort | Priority | Impact |
|-------|------------------|----------|--------|
| Phase 1: SIMD | 2-3 days | HIGH | 2-5x CPU speedup |
| Phase 2: Sparse Batch | 3-4 days | HIGH | 10-50x sparse encoding speedup |
| Phase 3: Memory Pool | 4-5 days | MEDIUM | Reduced thermal, higher throughput |
| Phase 4: Rayon | 2-3 days | MEDIUM | 2-4x preprocessing speedup |
| Phase 5: Vision Batch | 2-3 days | MEDIUM | 5-10x vision encoding speedup |
| Phase 6: Adaptive Batch | 1-2 days | LOW | Stability improvement |

**Total:** ~15-20 days of implementation work

---

## Dependencies to Add

```toml
[dependencies]
# SIMD (choose one)
pulp = "0.18"
# simsimd = "5.0"  # When Rust bindings stabilize

# Parallelization
rayon = "1.10"

# Caching
lru = "0.12"

# Optional: Sparse matrix operations
sprs = "0.11"

# Optional: BLAS for Apple Silicon
[target.'cfg(target_os = "macos")'.dependencies]
blas-src = { version = "0.10", features = ["accelerate"] }
```

---

## Benchmarking Strategy

Create `benches/` directory with criterion benchmarks:

```
benches/
├── pooling_bench.rs      # SIMD pooling operations
├── similarity_bench.rs   # SIMD similarity functions
├── dense_batch_bench.rs  # Dense encoder throughput
├── sparse_batch_bench.rs # Sparse encoder throughput
├── vision_batch_bench.rs # Vision encoder throughput
└── e2e_bench.rs          # End-to-end pipeline
```

Run with: `cargo bench --features metal`

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Dense batch encode (1000 texts) | ~10s | ~3-5s |
| Sparse batch encode (1000 texts) | ~60s (serial) | ~5-10s |
| Vision encode (100 images) | ~30s | ~5-10s |
| GPU memory pressure | HIGH | LOW |
| Thermal throttling | Frequent | Rare |
| Peak VRAM usage | Unbounded | Configurable |

---

## Next Steps

1. **Review this plan** - discuss priorities and approach
2. **Start with Phase 1** (SIMD) - highest impact, lowest risk
3. **Add benchmarking infrastructure** - measure before optimizing
4. **Iterate** - implement, measure, refine

---

*Generated: February 2026*
*Based on research of current Rust ecosystem best practices*
