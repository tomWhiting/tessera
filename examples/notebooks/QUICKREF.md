# Notebook quick reference

## Setup and launch

```bash
uv sync --extra dev
uv run marimo edit examples/notebooks/embedding_comparison.py
```

Or:

```bash
./examples/notebooks/run.sh embedding
```

The `timeseries` launcher opens a quarantine notice; it does not run Chronos.

## Models used by the comparison

```text
dense        bge-base-en-v1.5
multi-vector colbert-v2
sparse       splade-pp-en-v1
```

All three are currently experimental adapters. BGE is selected initially and
exactly one model is constructed by default. The unchecked `HIGH MEMORY`
checkbox is the only path that constructs all three simultaneously.

## Python API shape

```python
from tessera import TesseraDense

# Construct one selected facade by default.
embedder = TesseraDense("bge-base-en-v1.5")
vector = embedder.encode("example text")
```

The other notebook choices construct `TesseraMultiVector("colbert-v2")` or
`TesseraSparse("splade-pp-en-v1")` instead. High-memory mode constructs all
three and displays them side by side. Do not compare raw scores between
paradigms as if they shared one scale.

## Safety checklist

- Start with the default resource limits.
- Expect first use to download remote model artifacts.
- Leave the `HIGH MEMORY` checkbox clear on a constrained machine.
- If opting in, monitor memory: Tessera checks aggregate estimated parameter
  bytes, not the full peak memory of three models plus Python and plotting.
- Treat UMAP as visualization, not retrieval evaluation.
- Do not substitute a catalog-only ID just because it appears in the registry.

## Validation commands

```bash
uv run python -m py_compile \
  examples/notebooks/embedding_comparison.py \
  examples/notebooks/embedding_comparison_data.py \
  examples/notebooks/timeseries_forecasting.py
```

For more context, see [GUIDE.md](GUIDE.md).
