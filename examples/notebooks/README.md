# Tessera Marimo notebooks

These [Marimo](https://marimo.io) notebooks are exploratory examples for
Tessera's Python bindings. They are useful for manual evaluation, but they are
not benchmarks or production validation.

## Current status

| Notebook | Status | Purpose |
| --- | --- | --- |
| `embedding_comparison.py` | Experimental | Compare dense, multi-vector, and sparse retrieval |
| `timeseries_forecasting.py` | Quarantined notice | Explain why the legacy Chronos demo is not runnable |

The embedding notebook offers three registry entries with experimental runtime
adapters:

- `bge-base-en-v1.5` for one dense vector per text;
- `colbert-v2` for token-level multi-vector embeddings; and
- `splade-pp-en-v1` for sparse vocabulary-space embeddings.

Only the selected model is constructed by default; BGE is the initial
selection. Its artifacts are downloaded from Hugging Face on first use.
Experimental means a matching Tessera adapter exists; it does not mean the
remote checkpoint has a golden-output or quality guarantee.

> **High-memory opt-in:** the notebook constructs all three models only after
> the user checks the conspicuous `HIGH MEMORY` checkbox. That mode retains
> three model objects, three embedding sets, and three UMAP reducers. Tessera's
> resource policy preflights individual raw inputs, request shapes, attention
> cells, and estimated parameter bytes, but it does not impose an aggregate or
> peak-memory budget across them. Leave the checkbox clear unless the target
> machine has been monitored with this workload.

## Install and launch

From the repository root:

```bash
uv sync --extra dev
uv run marimo edit examples/notebooks/embedding_comparison.py
```

The launcher script is equivalent:

```bash
./examples/notebooks/run.sh embedding
```

Launching the legacy entry opens its quarantine explanation, not a
forecaster:

```bash
./examples/notebooks/run.sh timeseries
```

Chronos and TimesFM remain catalog metadata only. See
[`docs/legacy/TIMESERIES.md`](../../docs/legacy/TIMESERIES.md) for the missing
runtime work and the validation required before reactivation.

## What the embedding notebook does

The example uses 90 short documents across six categories. It:

1. constructs and embeds with the selected paradigm only by default;
2. optionally activates all three through an explicit high-memory checkbox;
3. projects each active representation separately with UMAP;
4. embeds an interactive text query;
5. scores dense and sparse vectors with dot products and multi-vector results
   with late interaction; and
6. displays one result panel, or three panels in high-memory mode.

UMAP is a visualization aid, not evidence of retrieval quality. The corpus is
synthetic and small, and scores from different paradigms are not calibrated to
one another.

## Troubleshooting

- If `marimo` is missing, run `uv sync --extra dev` again.
- Model construction requires network access on the first uncached run.
- A resource-limit error is deliberate. Shorten the input/batch or configure a
  larger `ResourcePolicy` only when the machine has enough memory.
- If the process approaches memory pressure, stop it, relaunch, and leave the
  high-memory checkbox clear rather than raising every limit.

For notebook mechanics, see [GUIDE.md](GUIDE.md). For the shortest command and
API reference, see [QUICKREF.md](QUICKREF.md).
