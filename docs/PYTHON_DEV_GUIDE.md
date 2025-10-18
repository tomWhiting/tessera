# Python Bindings Development Guide

Quick reference for developing and testing Python bindings for Tessera.

## Development Workflow

### 1. Build and Install (Development Mode)

Using UV (recommended):
```bash
# Create virtual environment if needed
uv venv

# Build and install in development mode
uv run maturin develop --features python --release

# Run tests
uv run python test_python_bindings.py
```

Using pip directly:
```bash
# Build wheel
maturin build --features python --release

# Install the wheel
pip install --force-reinstall target/wheels/tessera-0.1.0-*.whl

# Run tests
python test_python_bindings.py
```

### 2. Quick Rebuild Cycle

```bash
# After making changes to src/bindings/python.rs:

# 1. Check compilation
cargo check --features python

# 2. Build release wheel
maturin build --features python --release

# 3. Reinstall
pip install --force-reinstall target/wheels/tessera-0.1.0-*.whl

# 4. Test
python test_python_bindings.py
```

### 3. Interactive Testing

```bash
# Start Python REPL
python3

# Try it out
>>> from tessera import TesseraMultiVector
>>> embedder = TesseraMultiVector("colbert-v2")
>>> embeddings = embedder.encode("test")
>>> print(embeddings.shape)
```

## Project Structure

```
tessera/
├── src/
│   └── bindings/
│       └── python.rs          # Python bindings implementation
├── Cargo.toml                  # Rust dependencies, lib config
├── pyproject.toml             # Python package config, maturin setup
├── test_python_bindings.py    # Test suite
└── target/
    └── wheels/                # Built wheels (after maturin build)
```

## Key Files

### Cargo.toml
```toml
[lib]
crate-type = ["cdylib", "rlib"]

[features]
python = ["dep:pyo3", "dep:numpy"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"], optional = true }
numpy = { version = "0.22", optional = true }
```

### pyproject.toml
```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "tessera"
features = ["python"]
```

### src/bindings/python.rs
```rust
#[pymodule]
fn tessera(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTesseraMultiVector>()?;
    Ok(())
}
```

## Adding New Embedder Bindings

To add bindings for another embedder (e.g., TesseraDense):

### 1. Add Python wrapper struct

```rust
#[pyclass(name = "TesseraDense")]
pub struct PyTesseraDense {
    inner: crate::api::TesseraDense,
}
```

### 2. Implement methods

```rust
#[pymethods]
impl PyTesseraDense {
    #[new]
    fn new(model_id: &str) -> PyResult<Self> {
        let inner = crate::api::TesseraDense::new(model_id)
            .map_err(tessera_error_to_pyerr)?;
        Ok(Self { inner })
    }

    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Py<PyArray1<f32>>> {
        let embedding = self.inner.encode(text)
            .map_err(tessera_error_to_pyerr)?;
        // Convert DenseEmbedding to PyArray1
        todo!("Implement conversion")
    }

    // ... other methods
}
```

### 3. Register in module

```rust
#[pymodule]
fn tessera(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTesseraMultiVector>()?;
    m.add_class::<PyTesseraDense>()?;  // Add new class
    Ok(())
}
```

### 4. Add tests

```python
def test_dense_embeddings():
    from tessera import TesseraDense

    embedder = TesseraDense("bge-base-en-v1.5")
    embedding = embedder.encode("test")

    assert embedding.ndim == 1
    assert embedding.dtype == np.float32
```

## Common Issues and Solutions

### Issue: "Module not found" after build

**Solution**: Reinstall the wheel
```bash
pip install --force-reinstall target/wheels/tessera-*.whl
```

### Issue: "Couldn't find virtualenv"

**Solution**: Use maturin build + pip install
```bash
maturin build --features python --release
pip install target/wheels/tessera-*.whl
```

### Issue: Lifetime errors with PyArray

**Solution**: Return `Py<PyArray2<f32>>` (owned) instead of `&PyArray2<f32>` (borrowed)
```rust
fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Py<PyArray2<f32>>> {
    // ... create array_bound
    Ok(array_bound.unbind())  // Return owned Py<>
}
```

### Issue: "method not found: from_vec2"

**Solution**: Use `from_vec2_bound` for PyO3 0.22
```rust
let array_bound = PyArray2::from_vec2_bound(py, &rows)?;
Ok(array_bound.unbind())
```

## Testing Checklist

Before committing changes:

- [ ] `cargo check --features python` passes
- [ ] `cargo test --features python` passes (if unit tests added)
- [ ] `maturin build --features python --release` succeeds
- [ ] `test_python_bindings.py` passes all tests
- [ ] No warnings about unused imports/variables
- [ ] Documentation strings are complete
- [ ] Error handling covers all failure cases

## Performance Tips

### 1. Use Release Builds
```bash
maturin build --features python --release  # Much faster than debug
```

### 2. Batch Processing
```python
# Slow - sequential
for text in texts:
    embeddings.append(embedder.encode(text))

# Fast - batched
embeddings = embedder.encode_batch(texts)
```

### 3. Pre-allocate when possible
```python
# Pre-allocate NumPy arrays if you know sizes
results = np.empty((len(texts), 128), dtype=np.float32)
```

## Debugging

### Enable Rust Backtraces
```bash
RUST_BACKTRACE=1 python test_python_bindings.py
```

### Python Debugger
```python
import pdb; pdb.set_trace()
# Step through Python code
```

### Print Debugging in Rust
```rust
eprintln!("Debug: embeddings shape = {:?}", embeddings.shape());
```

## Resources

- **PyO3 Documentation**: https://pyo3.rs/
- **Maturin Guide**: https://www.maturin.rs/
- **NumPy Integration**: https://docs.rs/numpy/latest/numpy/
- **Rust Error Handling**: https://doc.rust-lang.org/book/ch09-00-error-handling.html

## Quick Commands Reference

```bash
# Check Rust compilation
cargo check --features python

# Build wheel
maturin build --features python --release

# Install locally
pip install --force-reinstall target/wheels/tessera-*.whl

# Run tests
python test_python_bindings.py

# Interactive testing
python3
>>> from tessera import TesseraMultiVector

# Clean build artifacts
cargo clean
rm -rf target/wheels/

# Format Rust code
cargo fmt

# Lint Rust code
cargo clippy --features python
```
