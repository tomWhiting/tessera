# Notebook status summary

The notebook area now has one experimental embedding comparison and one honest
legacy quarantine notice.

## Embedding comparison

`embedding_comparison.py` offers three representations over the same small
synthetic corpus:

| Paradigm | Model ID | Output |
| --- | --- | --- |
| Dense | `bge-base-en-v1.5` | One pooled vector per text |
| Multi-vector | `colbert-v2` | Token-level vectors |
| Sparse | `splade-pp-en-v1` | Weighted vocabulary entries |

The adapters are experimental, not production-certified. BGE is selected at
startup and default execution constructs exactly one model. The dropdown can
select either other paradigm without constructing the remaining two.

Side-by-side comparison requires checking a conspicuous, unchecked-by-default
`HIGH MEMORY` control. Only that path keeps three model instances, three
embedding collections, and three UMAP reducers live concurrently. Per-request
resource policies do not add up peak memory across those objects.

## Time-series example

`timeseries_forecasting.py` is no longer a forecasting demo. It explains that
the Chronos path depended on APIs from an old Candle fork and that TimesFM has
no implementation. Both remain catalog-only metadata. The legacy source is
documented in [`docs/legacy/TIMESERIES.md`](../../docs/legacy/TIMESERIES.md).

## What is and is not demonstrated

The embedding notebook demonstrates:

- construction of dense, multi-vector, and sparse Python facades;
- corpus and query encoding;
- paradigm-appropriate scoring; and
- interactive plotting with Marimo, Plotly, and UMAP.

It does not demonstrate:

- every model listed in the registry;
- support for catalog-only checkpoints;
- production throughput or memory bounds;
- calibrated score comparisons across paradigms;
- retrieval quality on a representative labelled corpus; or
- a working time-series runtime.

## Run

```bash
uv sync --extra dev
uv run marimo edit examples/notebooks/embedding_comparison.py
```

Use [README.md](README.md) for setup and safety notes, [GUIDE.md](GUIDE.md) for
interpretation and troubleshooting, and [QUICKREF.md](QUICKREF.md) for the
shortest reference.
