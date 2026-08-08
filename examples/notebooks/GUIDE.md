# Notebook guide

This guide covers the runnable embedding comparison and the retained legacy
forecasting notice. The examples exercise experimental model adapters and
should be treated as development tools.

## Prerequisites

- Python 3.9 or newer;
- `uv`;
- network access for the first Hugging Face download; and
- enough memory for the selected models and their intermediate tensors.

Install the project and notebook dependencies from the repository root:

```bash
uv sync --extra dev
```

## Embedding comparison

Launch it with either command:

```bash
uv run marimo edit examples/notebooks/embedding_comparison.py
./examples/notebooks/run.sh embedding
```

The notebook offers exactly three model choices:

| Paradigm | Registry ID | Representation | Scoring in the notebook |
| --- | --- | --- | --- |
| Dense | `bge-base-en-v1.5` | One 768-dimensional vector | Dot product |
| Multi-vector | `colbert-v2` | One vector per token | Late-interaction MaxSim |
| Sparse | `splade-pp-en-v1` | Weighted vocabulary entries | Sparse dot product |

These are experimental registry entries. Catalog-only entries such as the
Jina ColBERT variants or `colpali-v1.3-hf` are intentionally not substitutions:
their checkpoints do not match the current adapters.

### Memory model

The default dropdown selection is BGE, and the notebook constructs exactly that
one model. Choosing ColBERT or SPLADE replaces the active single-model path.
The only route to simultaneous construction is the unchecked-by-default
checkbox labelled `HIGH MEMORY`.

When explicitly enabled, high-memory mode retains all three model objects,
embedding collections, and UMAP reducers to make the side-by-side comparison
interactive. Unchecking it returns the reactive graph to one model, but Python
or device allocators may not immediately return reserved memory to the system;
restart the notebook after memory pressure.

The runtime's conservative policy limits raw input bytes, sequence length,
batch shape, attention cells, and estimated parameter storage. It also admits
the prospective aggregate parameter estimate across retained Tessera encoders
against the requesting model-byte policy. Those are preflight guards, not a
peak process-memory estimate. They cannot account for attention heads and
temporaries, Python arrays, plotting data, allocator overhead, Metal shared
memory, or other applications. Start with the defaults and monitor the
operating system's memory-pressure indicator.

For constrained machines, leave high-memory mode disabled, inspect one
paradigm, restart before switching after memory pressure, and avoid increasing
sequence and batch limits at the same time.

### Reading the results

- Dense retrieval compares one pooled semantic vector per item.
- Multi-vector retrieval preserves token-level vectors and uses a maximum
  document match for each query token.
- Sparse retrieval represents learned vocabulary activations and can be used
  with an inverted-index-style system.

In high-memory mode, the three raw scores have different meanings and scales.
Compare rankings, not numeric scores across columns.

The UMAP panels are each fitted independently. Distances and coordinates from
one panel therefore cannot be compared directly with another. The dataset is a
small teaching corpus, so use an external labelled corpus and retrieval metrics
before drawing quality conclusions.

### Customizing the corpus

The data lives in `embedding_comparison_data.py`. `DATASET` maps category names
to text lists, while `COLORS_MAP` supplies plot colors. Preserve at least enough
samples for the configured UMAP neighbor count, or lower `n_neighbors` in the
notebook.

The model IDs are set in the model-loading cell. Choose only entries marked
`experimental` or `supported` by the generated registry. A catalog entry is
discoverable metadata, not a promise that a builder can run it.

## Legacy forecasting notice

`timeseries_forecasting.py` is intentionally a small, non-executable notice.
The previous Chronos implementation depended on private and fork-specific
Candle T5 APIs and is not linked into the active API. TimesFM has no runtime
adapter.

You can open the notice with:

```bash
uv run marimo edit examples/notebooks/timeseries_forecasting.py
```

It does not download a time-series model or produce forecasts. The retained
design history is documented in
[`docs/legacy/TIMESERIES.md`](../../docs/legacy/TIMESERIES.md).

## Marimo workflow

Marimo cells are reactive: a cell reruns when an input value changes. Avoid
redefining the same top-level variable in several cells. Expensive downstream
cells, including model inference and UMAP fitting, may rerun after an upstream
edit, so wait for a run to finish before changing another control.

Useful commands:

```bash
# Edit interactively
uv run marimo edit examples/notebooks/embedding_comparison.py

# Check that the notebook is valid Python
uv run python -m py_compile examples/notebooks/embedding_comparison.py
```

## Troubleshooting

### Import or build failure

Run `uv sync --extra dev` from the repository root. The Python extension is
built from the current Rust tree, so a compiler error must be fixed rather than
worked around by installing an unrelated `tessera` package.

### Model download failure

Confirm network access and Hugging Face availability. A partially populated
local cache may need to be retried, but the notebook does not guarantee offline
operation unless all required artifacts are already present.

### Resource-limit failure

The error reports the measured and allowed values. Reduce input length or batch
size first. Raising a `ResourcePolicy` is an explicit opt-in and should reflect
the actual machine budget.

### Slow execution

The notebook performs three model passes plus three UMAP fits. First runs also
download model artifacts. This is expected work, not a benchmark result or a
documented cache-speed guarantee.

### Process or desktop becomes unresponsive

Stop the notebook process. Relaunch with other heavy applications closed, use
CPU, and restructure the comparison to hold one model at a time. Do not solve
aggregate memory pressure by removing all per-request limits.
