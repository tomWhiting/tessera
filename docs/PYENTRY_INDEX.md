# Tessera PyO3 Python Bindings - Entry Point Index

## Overview

This directory contains comprehensive documentation for implementing PyO3 Python bindings for the Tessera multi-paradigm embedding library. The codebase has been thoroughly explored and cataloged for maximum efficiency in binding development.

## Documentation Files

### 1. TESSERA_ARCHITECTURE_PYENTRY.md (18KB)
**Primary architecture and design document**

Comprehensive guide covering:
- Complete project structure and module organization
- Full public API surface (11 classes, ~60 methods)
- Detailed class documentation with method signatures
- Core output types and data structures
- Error handling and exception types
- Model registry structure and available models
- 8 typical usage patterns from examples
- Key architectural patterns (Factory, Builder, Device auto-selection)
- Critical types required for Python binding
- Implementation roadmap for PyO3

**Best for:** Understanding overall architecture, planning implementation phases

### 2. PYENTRY_QUICK_REFERENCE.md (12KB)
**Implementation quick reference guide**

Detailed reference for:
- API class methods in table format
- Output types with Python representations
- Builder pattern documentation
- Error mapping (Rust → Python exceptions)
- Device and quantization options
- Complete model registry
- NumPy array shape reference
- Type hints template (.pyi file)
- Complete end-to-end usage examples

**Best for:** During implementation, looking up specific signatures, testing

## Key Statistics

| Metric | Value |
|--------|-------|
| Total classes to expose | 11 |
| Total methods to bind | ~60 |
| Supported models | 23 |
| Embedding paradigms | 5 |
| Output types | 5 |
| Error types | 11 |

## Implementation Checklist

- [ ] Add PyO3 dependencies to Cargo.toml
  ```toml
  pyo3 = { version = "0.20", features = ["extension-module"] }
  numpy = "0.20"
  ```

- [ ] Implement PyTessera (factory enum) in src/bindings/python.rs
  - Auto-detect model type from registry
  - Return appropriate variant wrapper

- [ ] Implement wrapper classes
  - [ ] PyTesseraDense
  - [ ] PyTesseraMultiVector
  - [ ] PyTesseraSparse
  - [ ] PyTesseraVision
  - [ ] PyTesseraTimeSeries

- [ ] Implement builder classes
  - [ ] PyTesseraDenseBuilder
  - [ ] PyTesseraMultiVectorBuilder
  - [ ] PyTesseraSparseBuilder
  - [ ] PyTesseraVisionBuilder
  - [ ] PyTesseraTimeSeriesBuilder

- [ ] Implement output types
  - [ ] PyTokenEmbeddings (wrap TokenEmbeddings → ndarray)
  - [ ] PyDenseEmbedding (wrap Vec<f32> → ndarray)
  - [ ] PySparseEmbedding (wrap sparse format)
  - [ ] PyVisionEmbedding (wrap patch embeddings)
  - [ ] PyQuantizedEmbeddings (wrap binary quantized)

- [ ] Implement error mapping
  - [ ] TesseraError → Python exceptions
  - [ ] Create custom exception classes

- [ ] NumPy interop
  - [ ] Array2<f32> → PyArray2
  - [ ] Vec<f32> → PyArray1
  - [ ] Tensor → PyArray3
  - [ ] Sparse vector representation

- [ ] Build configuration
  - [ ] Create pyproject.toml for maturin
  - [ ] Generate .pyi type stub files
  - [ ] Test wheel building

- [ ] Testing
  - [ ] Unit tests for each class
  - [ ] Integration tests
  - [ ] Performance benchmarks
  - [ ] Example scripts

## File Structure

### Rust Source Files to Implement
- `/src/bindings/python.rs` - Main PyO3 implementation (currently stub)
- `/src/bindings/mod.rs` - Bindings module structure (already set up)

### Configuration Files to Create
- `pyproject.toml` - Maturin build configuration
- `tessera.pyi` - Python type hints

### Test Files (Optional)
- `tests/test_python_bindings.py` - Python integration tests
- `examples/python_example.py` - Usage examples

## Quick Start for Developers

### Phase 1: Setup (1-2 hours)
1. Read TESSERA_ARCHITECTURE_PYENTRY.md
2. Review Cargo.toml current structure
3. Review src/bindings/python.rs stub code

### Phase 2: Core Implementation (2-3 days)
1. Add dependencies to Cargo.toml
2. Implement wrapper classes (start with TesseraDense)
3. Implement NumPy interop for basic arrays
4. Error mapping

### Phase 3: Complete Coverage (1-2 days)
1. Remaining wrapper classes
2. All output types
3. Builder classes
4. Advanced features (quantization, vision)

### Phase 4: Polish & Testing (1 day)
1. Type hints generation
2. pyproject.toml setup
3. Documentation and examples
4. Performance testing

## Key Architectural Patterns to Preserve

### 1. Factory Pattern
- `Tessera.new(model_id)` returns appropriate variant
- Auto-detection based on model registry

### 2. Builder Pattern
- All embedders support `.builder()`
- Fluent interface: `.model().device().build()`

### 3. Lazy Loading
- Models downloaded on first use from HuggingFace
- Device selection happens at runtime

### 4. Type Safety
- Different embedding types have distinct output types
- Paradigm-specific methods (e.g., quantize only for MultiVector)

## NumPy Integration Points

### Dense Output
```python
# Input: "text"
# Output: np.ndarray shape (embedding_dim,) dtype float32
embedder.encode("text") → ndarray with shape (768,)
```

### Multi-Vector Output
```python
# Input: "text"
# Output: np.ndarray shape (num_tokens, embedding_dim) dtype float32
embedder.encode("text") → ndarray with shape (7, 128)
```

### Sparse Output
```python
# Input: "text"
# Option 1: Separate indices/values arrays
embeddings: Tuple[np.ndarray, np.ndarray]  # (indices, values)

# Option 2: COO sparse matrix
from scipy.sparse import coo_matrix
sparse_matrix = coo_matrix(...)
```

### Vision Output
```python
# Input: "path/to/image.jpg"
# Output: np.ndarray shape (1024, 128) dtype float32
embedder.encode_document("image.jpg") → ndarray with shape (1024, 128)
```

### Time Series Output
```python
# Input: ndarray shape (batch, 2048)
# Output: ndarray shape (batch, 64)
embedder.forecast(data) → ndarray with shape (1, 64)
```

## Error Handling Strategy

Map Rust errors to Python exceptions:

```python
ModelNotFound          → RuntimeError("Model 'X' not found in registry")
ModelLoadError         → RuntimeError("Failed to load model 'X'...")
EncodingError          → RuntimeError("Encoding failed: ...")
UnsupportedDimension   → ValueError("Unsupported dimension 100 for 'X'...")
DeviceError            → RuntimeError("Device error: ...")
QuantizationError      → ValueError("Quantization error: ...")
ConfigError            → ValueError("Invalid configuration: ...")
DimensionMismatch      → ValueError("Dimension mismatch...")
TensorError            → RuntimeError("Tensor operation failed...")
```

## Performance Targets for Python Bindings

- Encoding: <100ms per document on GPU (same as Rust)
- Batch processing: 5-10x speedup vs sequential (maintained from Rust)
- Memory overhead: <5% over Rust (due to Python object wrapper)
- Startup time: <2s for first model load (download + init)

## Dependencies

### Required
- pyo3 0.20.x - PyO3 framework
- numpy - NumPy interop
- Rust 1.70+

### Optional
- maturin - Python package builder
- black, mypy - Python code quality

## References

### Important Source Files
- `/src/api/embedder.rs` - Main API classes
- `/src/api/builder.rs` - Builder implementations
- `/src/core/embeddings.rs` - Output types
- `/src/error.rs` - Error types
- `/models.json` - Model registry

### Examples
- `/examples/dense_semantic_search.rs` - Dense usage
- `/examples/batch_processing.rs` - Batch patterns
- `/examples/colpali_demo.rs` - Vision usage
- `/examples/timeseries_basic_forecasting.rs` - Time series

## Next Steps

1. Start with TESSERA_ARCHITECTURE_PYENTRY.md for full context
2. Use PYENTRY_QUICK_REFERENCE.md during implementation
3. Begin with TesseraDense wrapper (simplest paradigm)
4. Expand to remaining embedder types
5. Implement advanced features (quantization, builders)

## Questions & Debugging

### Common Issues During Implementation

**Issue: Array conversion failing**
- Check ndarray shape and stride
- Use .as_slice_memory_order() for safe conversion
- Verify dtype matches (must be float32)

**Issue: Error not caught properly**
- Ensure all Result-returning functions use map_err()
- Convert anyhow::Error to PyErr
- Test error paths in Python

**Issue: Performance degradation**
- Minimize copying between Rust/Python
- Use zero-copy conversions where possible
- Profile with py-spy

**Issue: Type hints not recognized**
- Generate .pyi with pyo3-stub-gen
- Place in package root
- Verify mypy can find them

## Success Criteria

- All 11 classes fully implemented and exposed
- All ~60 methods working correctly
- NumPy interop seamless (zero-copy where possible)
- Error messages informative
- Performance within 5% of Rust
- Type hints complete (.pyi file)
- Test coverage >80%
- Documentation complete

---

**Document Generated:** 2025-10-18
**Status:** Exploration Complete, Ready for Implementation
**Estimated Implementation Time:** 4-6 days (depending on parallelization)
