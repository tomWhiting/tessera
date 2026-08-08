# Python bindings development guide

Tessera's optional PyO3 module exposes the active dense, sparse, multi-vector,
and vision façades plus the immutable `ResourcePolicy`. Time-series models are
not exported.

All model paths are currently experimental. Python test discovery is entirely
model-free; remote checkpoint execution is handled by the separate local
certification tool.

## Development environment

Create a virtual environment, install the locked development dependencies, and
then build the extension in place. The `dev` extra includes Maturin:

```bash
uv venv
uv sync --extra dev
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
uv run pytest tests/python/test_python_bindings.py
```

Do not add remote model downloads to ordinary Python tests. To exercise a real
checkpoint, use the repository's certification workflow instead:

```bash
cargo run --locked -p tessera-xtask --features certification -- \
  cert fetch --model jina-embeddings-v2-small-en
cargo run --locked --offline --release -p tessera-xtask \
  --features certification -- cert run \
  --model jina-embeddings-v2-small-en --device cpu --profile smoke --repeat 2
```

The tool verifies pinned artifacts, then runs fresh offline CPU child processes
serially with the checked model-specific resource policy. See the
[certification guide](../certification/README.md) before attempting larger
checkpoints such as Snowflake or ColPali.

## Module layout

```text
src/bindings/python.rs                 module registration
src/bindings/python/conversion.rs      Rust-to-NumPy conversion
src/bindings/python/resource_policy.rs Python ResourcePolicy wrapper
src/bindings/python/dense.rs           TesseraDense
src/bindings/python/multivector.rs     TesseraMultiVector
src/bindings/python/sparse.rs          TesseraSparse
src/bindings/python/vision.rs          TesseraVision
tests/python/test_python_bindings.py   model-free binding contract
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

vision_policy = (
    ResourcePolicy()
    .with_max_sequence_tokens(2_048)
    .with_max_batch_items(1)
    .with_max_batch_tokens(2_048)
    .with_max_attention_cells(4_194_304)
    .with_max_activation_bytes(1024 * 1024 * 1024)
    .with_max_model_bytes(12 * 1024 * 1024 * 1024)
)
vision = TesseraVision("colpali-v1.2", resource_policy=vision_policy)
```

ColPali retains 1,024 visual positions plus prompt tokens, so this is an
explicit high-memory opt-in for its sequence, batch, attention, activation,
and model-parameter preflights. Passing those estimated ceilings is not a peak
memory measurement and does not make the path safe on a given machine. The
estimator cannot account for every temporary tensor, allocator, driver, or
other process allocation.

## Adding or changing a wrapper

1. Put the class in its own file under `src/bindings/python/`.
2. Convert results through the shared conversion helpers.
3. Release the GIL around model inference.
4. Accept `resource_policy` as a keyword-only constructor argument and pass it
   to the matching Rust builder.
5. Register the class in `src/bindings/python.rs`.
6. Add model-free constructor, validation, and shape-contract tests. Put real
   checkpoint inference assertions in the certification specification and
   harness rather than `pytest`.

## Repository checks

```bash
./scripts/check
cargo clippy --locked --all-targets --no-default-features --features python -- -D warnings
cargo test --locked --offline --all-targets --no-default-features --features python
uv run pytest tests/python/test_python_bindings.py
```

Handwritten Rust and Python files must remain at or below 500 lines. The model
certification lane is separate because it is networked during fetch, slow, and
hardware-sensitive.
