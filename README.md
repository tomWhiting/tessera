# Tessera

Tessera is a Rust-first embedding library being revived around four retrieval
representations: dense, sparse, multi-vector, and vision-language embeddings.
It uses Candle for inference and offers optional Python bindings through PyO3.

> [!IMPORTANT]
> Tessera is currently an alpha-quality revival, not a production-ready model
> suite. The registry contains 22 model entries, but support metadata is
> deliberately conservative: 10 entries have an `Experimental` runtime path,
> 12 are `CatalogOnly`, and none are `Supported` yet.

`Experimental` means that a Tessera adapter and immutable checkpoint pin exist,
but the path still needs repeatable offline inference and quality validation.
`CatalogOnly` means the metadata is retained for discovery while builders
reject execution. An entry will move to `Supported` only after it satisfies the
checked certification evidence contract.

## Current priorities

- Dense embeddings for general semantic retrieval
- Sparse SPLADE-style vocabulary vectors for hybrid and lexical retrieval
- Multi-vector ColBERT-style embeddings with bounded-memory MaxSim scoring
- Vision-language patch embeddings from image inputs
- Bounded input, job, output, activation, model-load, and inference admission
  plus a two-worker CPU default
- Rust and Python APIs backed by the same model registry

Time-series forecasting is not an active runtime. The retained Chronos code
depends on APIs from an old Candle fork and is quarantined while Tessera uses
stock Candle 0.11. See [the legacy time-series note](docs/legacy/TIMESERIES.md)
for the exact incompatibilities and reactivation criteria.

## Build locally

The checked-in Rust 1.97.1 toolchain and lockfile are part of the reproducible
build. The package declares Rust 1.97 as its minimum because lower toolchains
are not part of the tested release contract. From a full repository checkout,
run the complete model-free gate locally before committing:

```bash
./scripts/check
```

The script checks formatting, generated-registry and 500-line policies, strict
Clippy, model-free Rust tests, documentation tests, and the Python lockfile
without downloading model artifacts. The script and unpublished `tessera-xtask`
runner are repository tooling and are not included in the crates.io archive.

Optional acceleration and bindings are feature-gated:

```bash
cargo build --locked                        # Core library; no default features
cargo build --locked --features metal       # Apple Metal
cargo build --locked --features cuda        # NVIDIA CUDA
cargo build --locked --features python      # PyO3 extension module
cargo build --locked --features pdf         # PDF plumbing; requires Poppler
```

PDF rendering is deliberately opt-in because it adds a native Poppler runtime
dependency. It is not needed for image inputs or the Python wheel.

For local Python development, use Maturin so the extension and Python package
are built together:

```bash
uv sync --extra dev
uv run maturin develop
uv run pytest tests/python/test_python_bindings.py
```

Python tests are model-free. Remote checkpoint execution belongs to the
explicit local certification workflow, not ordinary test discovery.

## Rust quick start

Creating an embedder may download model artifacts from Hugging Face. The BGE
path below is `Experimental`, so treat it as a smoke test rather than a stable
compatibility promise.

```rust,no_run
use tessera::TesseraDense;

fn main() -> tessera::Result<()> {
    let embedder = TesseraDense::new("bge-base-en-v1.5")?;
    let embedding = embedder.encode("A tessera is one tile in a mosaic.")?;
    println!("{} dimensions", embedding.dim());
    Ok(())
}
```

The high-level APIs follow the representation rather than pretending every
model is interchangeable:

```rust,no_run
use tessera::{ResourcePolicy, TesseraMultiVector, TesseraSparse, TesseraVision};

fn main() -> tessera::Result<()> {
    let multi = TesseraMultiVector::new("colbert-small")?;
    let sparse = TesseraSparse::new("splade-pp-en-v1")?;

    let token_vectors = multi.encode("late interaction retrieval")?;
    let term_weights = sparse.encode("learned lexical expansion")?;

    // ColPali combines 1,024 visual positions with prompt tokens. These
    // illustrative F32 ceilings are an explicit high-memory opt-in.
    let vision_policy = ResourcePolicy::default()
        .with_max_sequence_tokens(2_048)
        .with_max_batch_items(1)
        .with_max_batch_tokens(2_048)
        .with_max_attention_cells(4_194_304)
        .with_max_activation_bytes(1024 * 1024 * 1024)
        .with_max_model_bytes(12 * 1024 * 1024 * 1024);
    let vision = TesseraVision::builder()
        .model("colpali-v1.2")
        .resource_policy(vision_policy)
        .build()?;
    let page_vectors = vision.encode_document("page.png")?;

    println!(
        "{} token vectors, {} sparse terms, {} image patches",
        token_vectors.num_tokens(),
        term_weights.nnz(),
        page_vectors.num_patches()
    );
    Ok(())
}
```

All three checkpoints in this example are currently `Experimental`. ColPali is
a large F32 load whose visual sequence, attention, activation estimate, and
model parameters exceed conservative defaults. The example only demonstrates
the required opt-in shape; it is not evidence that the model fits a particular
machine.

## Resource policy

Tessera's high-level builders preflight registered-model and request-shape
limits before their main model input tensors are allocated. The default
`ResourcePolicy` is intentionally conservative:

| Limit | Default |
|---|---:|
| UTF-8 input bytes in one sequence, before tokenization | 1 MiB |
| Tokens in one sequence, including special tokens | 512 |
| Items in one batch | 16 |
| Padded token cells (`batch items × longest sequence`) | 2,048 |
| Attention cells (`batch items × longest sequence²`) | 1,048,576 |
| Inputs in one logical encode job, across all chunks | 1,024 |
| Aggregate UTF-8 input bytes in one logical job | 64 MiB |
| Retained embedding values from one collecting API | 64 MiB |
| Estimated live inference scratch bytes per forward pass | 512 MiB |
| Estimated resident model parameter bytes (default F32) | 2 GiB |

The batch limits apply to one tensor forward pass; the job and output limits
also bound work accumulated across internal chunks. The activation limit is a
conservative estimate derived from the pinned transformer configuration and
load dtype, not a measurement of total process or accelerator memory.

These limits are not model capabilities. A registered 8K model still uses the
512-token default until the caller opts in. Sequence limits can be raised only
up to the selected model's registered context window:

```rust,no_run
use tessera::{ResourcePolicy, TesseraDense};

fn main() -> tessera::Result<()> {
    let single_document_8k = ResourcePolicy::default()
        .with_max_sequence_tokens(8_192)
        .with_max_batch_items(1)
        .with_max_batch_tokens(8_192)
        .with_max_attention_cells(67_108_864)
        .with_max_activation_bytes(8 * 1024 * 1024 * 1024);

    let embedder = TesseraDense::builder()
        .model("jina-embeddings-v2-small-en")
        .resource_policy(single_document_8k)
        .build()?;

    let _embedding = embedder.encode("long document")?;
    Ok(())
}
```

> [!WARNING]
> This policy only allows the request to pass Tessera's shape preflight; it is
> not evidence that 8K inference is safe on the target machine. Full attention
> is quadratic in sequence length. The 67,108,864-cell allowance above is for
> one 8,192 × 8,192 matrix before attention heads, model layers, temporary
> tensors, allocator overhead, and other process memory are considered. The
> 8 GiB scratch allowance is illustrative, not a certification result. Even a
> one-item 8K request can exhaust CPU, GPU, or Metal shared memory. Measure the
> selected model on the target hardware and raise the limit only when that risk
> is acceptable.

`max_model_bytes` bounds both one model and the prospective aggregate estimated
parameter bytes retained by Tessera encoders. Raise it deliberately when
loading a model whose selected dtype exceeds 2 GiB or when admitting multiple
distinct models. For example, a 3B-parameter checkpoint has an approximately
12 GB F32 parameter estimate before allocator overhead. This admission budget
prevents accidental loads, but it is not a full peak-memory estimator.

### CPU worker ceiling

Before Tessera constructs the first CPU encoder, it attempts to configure
Candle's process-global Rayon and barrier pools with a ceiling of two workers
(or fewer when the machine exposes less parallelism). This conservative default
avoids a Tessera model load silently requesting every CPU core.

Applications that deliberately want a higher ceiling must call
`configure_cpu_threads` during single-threaded startup, before constructing any
Tessera builder or allowing other code to initialize those pools. The first
Tessera configuration call wins for the lifetime of the process:

```rust,no_run
use candle_core::Device;
use tessera::{configure_cpu_threads, TesseraDense};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    configure_cpu_threads(8)?;
    let embedder = TesseraDense::builder()
        .model("bge-base-en-v1.5")
        .device(Device::Cpu)
        .build()?;

    let _embedding = embedder.encode("explicit CPU worker opt-in")?;
    Ok(())
}
```

Thread setup is best-effort because an already initialized external Rayon pool
cannot be resized through environment variables. It also does not cap every
thread, allocation, or accelerator operation in the process.

### Inference concurrency

Tessera admits only one Candle forward pass at a time across dense, sparse,
multi-vector, and vision encoders in the process, including calls arriving
through separate Python instances. By default, at most 16 callers may wait and
each may wait for at most 30 seconds; excess callers and expired waits return
structured errors. Call `tessera::runtime::configure_inference_gate` before the
first forward pass to choose different process-wide bounds. The exclusive gate
prevents overlapping Tessera forward passes, but it does not govern allocations
made by other libraries in the process.

### Model residency

Each dense, sparse, multi-vector, or vision constructor atomically reserves its
estimated parameter bytes before tokenizer, Hub, or artifact I/O. Reservations
are keyed by the immutable model revision, physical Candle device, and model
dtype. A second retained instance with the same key is rejected with guidance
to reuse or drop the existing embedder; Tessera does not currently share model
tensors between instances.

A different key is admitted only when its estimate plus existing Tessera
reservations fits the requesting policy's `max_model_bytes`. Giving that request
a higher policy deliberately raises its prospective aggregate ceiling. The
reservation is released automatically if construction fails and after the
encoder's tensors are dropped. Registry parameter counts remain estimates and
do not include every allocator, activation, driver, or non-Tessera allocation.

## Model catalog

`models.json` is the source of truth, including immutable checkpoint revisions
and exact runtime artifact filenames. Build-time generation exposes the same
metadata through `tessera::model_registry`. `get_model` is catalog-complete;
use `ModelInfo::is_runnable()` or `runnable_models()` before selecting a model
for execution.

### Experimental runtime paths (10)

| Representation | Model IDs |
|---|---|
| Dense | `bge-base-en-v1.5`, `jina-embeddings-v2-small-en`, `jina-embeddings-v2-base-en`, `nomic-embed-v1.5`, `snowflake-arctic-l` |
| Multi-vector | `colbert-small`, `colbert-v2` |
| Sparse | `splade-pp-en-v1`, `splade-pp-en-v2` |
| Vision-language | `colpali-v1.2` |

### Catalog-only entries (12)

| Representation | Model IDs |
|---|---|
| Dense | `jina-embeddings-v2-base-code`, `jina-embeddings-v3` |
| Multi-vector | `bge-m3-multi`, `gte-modern-colbert`, `jina-colbert-v2-64`, `jina-colbert-v2-96`, `jina-colbert-v2` |
| Vision-language | `colpali-v1.3-hf` |
| Sparse | `minicoil-v1`, `splade-v3` |
| Time series | `chronos-bolt-small`, `timesfm-1.0-200m` |

There are currently no `Supported` entries. This distinction is intentional:
catalog breadth is useful, but metadata alone is not runtime support.

The maintained [documentation index](docs/README.md) links the registry,
Python-development, notebook, and legacy compatibility guides. Dated phase
reports and research roadmaps are retained separately as historical snapshots.

## Representation notes

### Dense

One vector per text. Pooling is model-specific and is selected from registry
metadata. Dense vectors are suited to approximate nearest-neighbour indexes,
clustering, and broad semantic retrieval.

### Multi-vector

One vector per token, scored with late interaction. Tessera's MaxSim path uses
bounded working memory instead of materializing a full query-by-document score
matrix. Binary quantization exists as an explicit option; no quality-retention
or throughput claim is made without a checked benchmark.

### Sparse

Vocabulary-sized SPLADE-style weights intended for inverted indexes and hybrid
retrieval. Pooling stays on the model device and transfers only the pooled
vocabulary vector to the host.

### Vision-language

Patch embeddings from document page images and text-query embeddings for
late-interaction retrieval. The public `TesseraVision` façade currently accepts
an image path. The opt-in `pdf` feature provides PDF rendering plumbing used by
lower layers, but it is not yet a high-level PDF-document API and requires a
system Poppler installation.

## Python

The Python module mirrors the active dense, multi-vector, sparse, and vision
façades. Its API tests are model-free and run without downloads; checkpoint
execution is kept out of Python test discovery.

```python
from tessera import TesseraDense

embedder = TesseraDense("bge-base-en-v1.5")
embedding = embedder.encode("semantic retrieval")
print(embedding.shape)
```

Chronos and TimesFM are not exported by the active Python module.

## Project checks

Handwritten Rust and Python files are kept to 500 lines or fewer. The repository
task runner checks that policy as part of the complete local gate:

```bash
./scripts/check
```

Model certification is a separate, opt-in local operation. It downloads exactly
one pinned checkpoint into a dedicated cache, verifies artifact sizes and
SHA-256 hashes, and runs each model in a fresh resource-bounded CPU process.
Start with the [certification guide](certification/README.md); certification
evidence is required before any entry can move from `Experimental` to
`Supported`.

## License

Tessera's source code and repository documentation are licensed under the
Apache License, Version 2.0. See [LICENSE](LICENSE).

That license does **not** grant rights to third-party model checkpoints. The
crate and Python package do not bundle model weights; checkpoints obtained from
Hugging Face or another provider remain subject to their upstream licenses and
terms. The catalog includes permissive, non-commercial, and model-specific
licenses, so check the pinned model repository before downloading or deploying
a checkpoint. The `license` value in `models.json` is discovery metadata; the
upstream model card and license files are authoritative.
