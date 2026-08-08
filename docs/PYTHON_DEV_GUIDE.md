# Python bindings development guide

Tessera's optional PyO3 module exposes the active dense, sparse, multi-vector,
and vision façades plus the immutable `ResourcePolicy`. Time-series models are
not exported.

All model paths are currently experimental. The normal Python test lane is
model-free; model-marked smoke tests download and run one checkpoint at a time.

## Development environment

Create a virtual environment, install the development dependencies and Maturin,
then build the extension in place:

```bash
uv venv
uv sync --extra dev
uv pip install "maturin>=1.7,<2.0"
uv run maturin develop
```

The package metadata in `pyproject.toml` enables Cargo's `python` feature for
Maturin. A release wheel can be built with:

```bash
uv run maturin build --release
```

## Test lanes

Run the hermetic, model-free Python contract after each binding change:

```bash
uv run pytest tests/python/test_python_bindings.py -m "not model"
```

Model-marked tests access Hugging Face and can consume substantial memory. Run
one explicitly selected smoke test at a time, outside the normal pull-request
lane:

```bash
uv run pytest tests/python/test_python_bindings.py \
  -m model -k dense_bge_base_smoke
```

Do not run the whole model marker as a convenient “all models” check. Dense,
ColBERT, SPLADE, and especially ColPali checkpoints can overlap in memory if a
runner or notebook keeps embedders alive.

## Module layout

```text
src/bindings/python.rs                 module registration
src/bindings/python/conversion.rs      Rust-to-NumPy conversion
src/bindings/python/resource_policy.rs Python ResourcePolicy wrapper
src/bindings/python/dense.rs           TesseraDense
src/bindings/python/multivector.rs     TesseraMultiVector
src/bindings/python/sparse.rs          TesseraSparse
src/bindings/python/vision.rs          TesseraVision
tests/python/test_python_bindings.py   model-free contract and opt-in smokes
```

The extension releases the GIL around inference. Batch inputs are validated in
Python before Rust constructs model tensors, and all wrappers forward an
optional keyword-only `resource_policy` to the Rust builder.

## API smoke example

```python
from tessera import TesseraDense

embedder = TesseraDense("bge-base-en-v1.5")
embedding = embedder.encode("semantic retrieval")

print(embedding.shape)
print(embedder.model())
```

Creating the embedder can download a remote checkpoint. The model is
`Experimental`, so successful construction on one machine is not yet a stable
compatibility guarantee.

## Resource policy

Python defaults match Rust: 1 MiB of pre-tokenizer input, 512 tokens per
sequence, 16 batch items, 2,048 padded token cells, 1,048,576 attention cells,
and a 2 GiB F32 model-parameter estimate.

Policies are immutable:

```python
from tessera import ResourcePolicy, TesseraVision

vision_policy = ResourcePolicy().with_max_model_bytes(12_000_000_000)
vision = TesseraVision("colpali-v1.2", resource_policy=vision_policy)
```

The larger model budget only passes the parameter preflight. It is not a peak
memory estimate and does not make the ColPali path safe on a given machine.
Likewise, raising the sequence and attention limits for an 8K model does not
account for heads, layers, temporary tensors, or allocator overhead.

## Adding or changing a wrapper

1. Put the class in its own file under `src/bindings/python/`.
2. Convert results through the shared conversion helpers.
3. Release the GIL around model inference.
4. Accept `resource_policy` as a keyword-only constructor argument and pass it
   to the matching Rust builder.
5. Register the class in `src/bindings/python.rs`.
6. Add model-free constructor, validation, and shape-contract tests. Keep any
   networked inference proof behind `@pytest.mark.model`.

## Repository checks

```bash
cargo run --locked --offline -p tessera-xtask -- all
cargo fmt --all -- --check
cargo clippy --locked --all-targets --features python -- -D warnings
cargo test --locked --offline --all-targets --features python
uv run pytest tests/python/test_python_bindings.py -m "not model"
```

Handwritten Rust and Python files must remain at or below 500 lines. The model
smoke lane is separate because it is networked, slow, and hardware-sensitive.
