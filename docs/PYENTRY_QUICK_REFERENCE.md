# PyO3 Python Bindings - Quick Reference Guide

## API Classes to Implement

### 1. Factory Class: `Tessera`

| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `__new__(model_id: str)` | model_id: str | Tessera enum variant | Auto-detects type |
| `__repr__()` | - | str | Debug representation |

**Factory Returns:**
- Dense("bge-base-en-v1.5") → TesseraDense
- MultiVector("colbert-v2") → TesseraMultiVector
- Sparse("splade-cocondenser") → TesseraSparse
- Vision("colpali-v1.3-hf") → TesseraVision
- TimeSeries("chronos-bolt-small") → TesseraTimeSeries

---

### 2. Dense Embedder: `TesseraDense`

| Method | Input | Output | Shape |
|--------|-------|--------|-------|
| `__new__(model_id: str)` | model_id: str | TesseraDense | - |
| `encode(text: str)` | text: str | ndarray | (embedding_dim,) |
| `encode_batch(texts: List[str])` | texts: List[str] | List[ndarray] | [(dim,), ...] |
| `similarity(text_a: str, text_b: str)` | 2x strings | float | Cosine sim |
| `dimension()` | - | int | 384, 768, etc. |
| `model()` | - | str | Model ID |
| `builder()` | - | TesseraDenseBuilder | - |

**Supported Models:** bge-base-en-v1.5, bge-large-en-v1.5, nomic-embed-text-v1, gte-large-en-v1.5, qwen-2.5-0.5b

---

### 3. Multi-Vector Embedder: `TesseraMultiVector`

| Method | Input | Output | Shape |
|--------|-------|--------|-------|
| `__new__(model_id: str)` | model_id: str | TesseraMultiVector | - |
| `encode(text: str)` | text: str | ndarray | (num_tokens, embedding_dim) |
| `encode_batch(texts: List[str])` | texts: List[str] | List[ndarray] | [(N, D), ...] |
| `similarity(text_a: str, text_b: str)` | 2x strings | float | MaxSim score |
| `dimension()` | - | int | 96, 128, 384 |
| `model()` | - | str | Model ID |
| `quantize(embeddings: ndarray)` | (N, D) ndarray | QuantizedEmbeddings | Compressed |
| `encode_quantized(text: str)` | text: str | QuantizedEmbeddings | Compressed |
| `similarity_quantized(q, d)` | 2x QuantizedEmbeddings | float | Fast sim |
| `builder()` | - | TesseraMultiVectorBuilder | - |

**Supported Models:** colbert-v2, colbert-small, jina-colbert-v2, jina-colbert-v3

---

### 4. Sparse Embedder: `TesseraSparse`

| Method | Input | Output | Shape |
|--------|-------|--------|-------|
| `__new__(model_id: str)` | model_id: str | TesseraSparse | - |
| `encode(text: str)` | text: str | SparseEmbedding | Sparse |
| `encode_batch(texts: List[str])` | texts: List[str] | List[SparseEmbedding] | Sparse list |
| `similarity(text_a: str, text_b: str)` | 2x strings | float | Dot product |
| `vocab_size()` | - | int | 30522 |
| `model()` | - | str | Model ID |
| `builder()` | - | TesseraSparseBuilder | - |

**Supported Models:** splade-cocondenser, splade-pp-en-v1

---

### 5. Vision Embedder: `TesseraVision`

| Method | Input | Output | Shape |
|--------|-------|--------|-------|
| `__new__(model_id: str)` | model_id: str | TesseraVision | - |
| `encode_document(path: str)` | image_path: str | VisionEmbedding | (num_patches, 128) |
| `encode_query(text: str)` | text: str | ndarray | (num_tokens, 128) |
| `search(query, document)` | 2x embeddings | float | MaxSim score |
| `search_document(text: str, path: str)` | text & path | float | Convenience |
| `embedding_dim()` | - | int | 128 |
| `num_patches()` | - | int | 1024 |
| `model()` | - | str | Model ID |
| `builder()` | - | TesseraVisionBuilder | - |

**Supported Models:** colpali-v1.3-hf, colpali-v1.2-hf

---

### 6. Time Series Forecaster: `TesseraTimeSeries`

| Method | Input | Output | Shape |
|--------|-------|--------|-------|
| `__new__(model_id: str)` | model_id: str | TesseraTimeSeries | - |
| `forecast(context: ndarray)` | (batch, 2048) | ndarray | (batch, 64) |
| `forecast_quantiles(context: ndarray)` | (batch, 2048) | ndarray | (batch, 64, 9) |
| `prediction_length()` | - | int | 64 |
| `context_length()` | - | int | 2048 |
| `quantiles()` | - | List[float] | [0.1, 0.2, ..., 0.9] |
| `model()` | - | str | Model ID |
| `builder()` | - | TesseraTimeSeriesBuilder | - |

**Supported Models:** chronos-bolt-small, chronos-bolt-base, chronos-bolt-large

---

## Output Types

### DenseEmbedding (Python)
```python
embedding: np.ndarray  # Shape: (embedding_dim,) float32
dim: int               # Dimensionality
```

### TokenEmbeddings (Python)
```python
embeddings: np.ndarray  # Shape: (num_tokens, embedding_dim) float32
text: str              # Original input text
num_tokens: int        # Number of tokens
embedding_dim: int     # Dimension per token
shape: Tuple[int, int] # (num_tokens, embedding_dim)
```

### SparseEmbedding (Python)
```python
# Option 1: Tuple format
indices: np.ndarray    # Shape: (nnz,) int32
values: np.ndarray     # Shape: (nnz,) float32
vocab_size: int        # 30522
sparsity: float        # 0.0-1.0

# Option 2: Dict format (alternative)
weights: Dict[int, float]  # {vocab_idx: weight, ...}
vocab_size: int
sparsity: float
```

### VisionEmbedding (Python)
```python
embeddings: np.ndarray  # Shape: (num_patches, embedding_dim) float32
num_patches: int        # 1024 for 448x448 image
embedding_dim: int      # 128
source: Optional[str]   # Image file path
```

### QuantizedEmbeddings (Python)
```python
quantized: List[np.ndarray]  # List of binary vectors
original_dim: int            # Original dim before quantization
num_tokens: int              # Number of token vectors
memory_bytes() -> int        # Memory usage
compression_ratio() -> float # Compression ratio
```

---

## Builder Classes

### Pattern (All builders similar)

```python
builder = TesseraDenseBuilder()
builder = builder.model("bge-base-en-v1.5")  # Required
builder = builder.device("metal")             # Optional
builder = builder.dimension(384)              # Optional (Matryoshka)
embedder = builder.build()                    # Returns embedder
```

### Builders Available
- `TesseraDenseBuilder`
- `TesseraMultiVectorBuilder` 
- `TesseraSparseBuilder`
- `TesseraVisionBuilder`
- `TesseraTimeSeriesBuilder`

---

## Error Handling (Python Exception Mapping)

| Rust Error | Python Exception |
|------------|------------------|
| `ModelNotFound` | `RuntimeError` |
| `ModelLoadError` | `RuntimeError` |
| `EncodingError` | `RuntimeError` |
| `UnsupportedDimension` | `ValueError` |
| `DeviceError` | `RuntimeError` |
| `QuantizationError` | `ValueError` |
| `TokenizationError` | `ValueError` |
| `ConfigError` | `ValueError` |
| `DimensionMismatch` | `ValueError` |
| `MatryoshkaError` | `ValueError` |
| `IoError` | `IOError` |
| `TensorError` | `RuntimeError` |

---

## Device Options (Python)

```python
# Auto-select (default)
embedder = TesseraDense.new("model-id")

# Explicit device (if binding Device to Python)
from tessera import Device
embedder = TesseraDense.builder() \
    .model("model-id") \
    .device(Device.Metal)  # or Device.Cuda, Device.Cpu
    .build()
```

---

## Quantization Options (Python)

```python
from tessera import QuantizationConfig

# For multi-vector only
embedder = TesseraMultiVector.builder() \
    .model("colbert-v2") \
    .quantization(QuantizationConfig.Binary)  # 32x compression
    .build()

# Usage
embeddings = embedder.encode("text")
quantized = embedder.quantize(embeddings)
score = embedder.similarity_quantized(q_quantized, d_quantized)
```

---

## Batch Processing (Python)

```python
embedder = TesseraDense.new("bge-base-en-v1.5")

texts = [
    "First document",
    "Second document", 
    "Third document"
]

# Batch encoding (5-10x faster than sequential)
embeddings = embedder.encode_batch(texts)

# Returns list of ndarrays
for emb in embeddings:
    print(emb.shape)  # (embedding_dim,)
```

---

## Matryoshka Dimensions (Python)

```python
# Jina ColBERT with reduced dimension
embedder = TesseraMultiVector.builder() \
    .model("jina-colbert-v2") \
    .dimension(96)  # Instead of 768
    .build()

embeddings = embedder.encode("text")
print(embeddings.shape)  # (num_tokens, 96) instead of (num_tokens, 768)
```

---

## Model Registry Quick List

### Multi-Vector Models
- colbert-v2 (110M params)
- colbert-small (33M params)
- jina-colbert-v2 (137M params)
- jina-colbert-v3 (250M params)
- nomic-bert-multivector (137M params)

### Dense Models
- bge-base-en-v1.5 (110M, 768-dim)
- bge-large-en-v1.5 (335M, 1024-dim)
- nomic-embed-text-v1 (137M, 768-dim)
- gte-large-en-v1.5 (335M, 1024-dim)
- qwen-2.5-0.5b (100M, 1024-dim)
- jina-embeddings-v3-base (570M, 1024-dim)
- snowflake-arctic-embed-large (735M, 1024-dim)

### Sparse Models
- splade-cocondenser (110M)
- splade-pp-en-v1 (110M)

### Vision Models
- colpali-v1.3-hf (3B params, 128-dim)
- colpali-v1.2-hf (3B params, 128-dim)

### Time Series Models
- chronos-bolt-small (70M)
- chronos-bolt-base (200M)
- chronos-bolt-large (500M)

---

## Example: Complete Python Usage

```python
from tessera import Tessera, TesseraDense, TesseraMultiVector
from tessera import QuantizationConfig
import numpy as np

# Factory pattern
embedder = Tessera("colbert-v2")
if isinstance(embedder, TesseraMultiVector):
    # Single encoding
    embeddings = embedder.encode("Query text")
    print(embeddings.shape)  # (num_tokens, 128)
    
    # Batch encoding
    texts = ["Doc 1", "Doc 2"]
    batch_emb = embedder.encode_batch(texts)
    
    # Similarity
    score = embedder.similarity("Query", "Document")
    print(f"Score: {score:.4f}")
    
    # Quantization
    quantized = embedder.quantize(embeddings)
    print(f"Compression: {quantized.compression_ratio():.1f}x")

# Dense embedder
dense = TesseraDense("bge-base-en-v1.5")
dense_emb = dense.encode("text")
print(dense_emb.shape)  # (768,)

# With builder
sparse = TesseraSparse.builder() \
    .model("splade-cocondenser") \
    .build()
sparse_emb = sparse.encode("text")
print(f"Sparsity: {sparse_emb.sparsity * 100:.1f}%")
```

---

## NumPy Array Shapes Summary

| Embedder | Output Type | Shape |
|----------|------------|-------|
| Dense | ndarray | (embedding_dim,) |
| MultiVector | ndarray | (num_tokens, embedding_dim) |
| Sparse | (indices, values) | (nnz,), (nnz,) |
| Vision (doc) | ndarray | (1024, embedding_dim) |
| Vision (query) | ndarray | (num_tokens, embedding_dim) |
| TimeSeries | ndarray | (batch, prediction_length) |
| TimeSeries (quant) | ndarray | (batch, prediction_length, 9) |
| QuantizedMulti | list of arrays | [(num_tokens,), ...] |

---

## Type Hints (Python .pyi file)

```python
from typing import List, Optional, Tuple, Dict
import numpy as np
from enum import Enum

class Device(Enum):
    Cpu: Device
    Metal: Device
    Cuda: Device

class QuantizationConfig(Enum):
    None: QuantizationConfig
    Binary: QuantizationConfig
    Int8: QuantizationConfig
    Int4: QuantizationConfig

class DenseEmbedding:
    embedding: np.ndarray
    dim: int

class TokenEmbeddings:
    embeddings: np.ndarray
    text: str
    num_tokens: int
    embedding_dim: int
    def shape(self) -> Tuple[int, int]: ...

class SparseEmbedding:
    indices: np.ndarray
    values: np.ndarray
    vocab_size: int
    sparsity: float

class VisionEmbedding:
    embeddings: np.ndarray
    num_patches: int
    embedding_dim: int
    source: Optional[str]

class QuantizedEmbeddings:
    original_dim: int
    num_tokens: int
    def memory_bytes(self) -> int: ...
    def compression_ratio(self) -> float: ...

class TesseraDense:
    @classmethod
    def new(cls, model_id: str) -> TesseraDense: ...
    def encode(self, text: str) -> np.ndarray: ...
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]: ...
    def similarity(self, text_a: str, text_b: str) -> float: ...
    def dimension(self) -> int: ...
    def model(self) -> str: ...

class TesseraMultiVector:
    @classmethod
    def new(cls, model_id: str) -> TesseraMultiVector: ...
    def encode(self, text: str) -> np.ndarray: ...
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]: ...
    def similarity(self, text_a: str, text_b: str) -> float: ...
    def dimension(self) -> int: ...
    def model(self) -> str: ...
    def quantize(self, embeddings: np.ndarray) -> QuantizedEmbeddings: ...
    def encode_quantized(self, text: str) -> QuantizedEmbeddings: ...
    def similarity_quantized(
        self, query: QuantizedEmbeddings, doc: QuantizedEmbeddings
    ) -> float: ...

# ... similar for TesseraSparse, TesseraVision, TesseraTimeSeries
```
