# Notebook index

## Files

| File | Role |
| --- | --- |
| `embedding_comparison.py` | Runnable experimental comparison of three embedding paradigms |
| `embedding_comparison_data.py` | Synthetic corpus and plot colors used by that notebook |
| `timeseries_forecasting.py` | Non-executable legacy quarantine notice |
| `run.sh` | Convenience launcher |
| `README.md` | Status, setup, and safety overview |
| `GUIDE.md` | Detailed workflow and interpretation notes |
| `QUICKREF.md` | Commands and API snippets |
| `SUMMARY.md` | Scope and evidence boundaries |

## Runnable comparison

The embedding notebook offers these three experimental selections:

- dense: `bge-base-en-v1.5`;
- multi-vector: `colbert-v2`; and
- sparse: `splade-pp-en-v1`.

The default is single-model mode with BGE selected. Changing the dropdown
constructs one of the other two instead. The notebook constructs all three only
after the user checks the explicit `HIGH MEMORY` opt-in. This set is not a claim
that every catalog entry is runnable; catalog-only models are rejected by the
active builders when no compatible runtime adapter exists.

The opted-in simultaneous comparison has a meaningful memory cost. Resource
limits protect individual loads and requests, not the total of three resident
models plus Python and visualization allocations. Keep the checkbox clear on
constrained machines.

## Quarantined notebook

The former time-series example no longer imports a forecasting facade or runs
Chronos. Its file remains as a visible explanation so old links fail honestly
rather than implying that the fork-dependent runtime still works. TimesFM is
also catalog metadata only.

See [`docs/legacy/TIMESERIES.md`](../../docs/legacy/TIMESERIES.md) for the
reactivation requirements.

## Start here

```bash
uv sync --extra dev
uv run marimo edit examples/notebooks/embedding_comparison.py
```

Then read:

1. [README.md](README.md) for the current status and safety warning;
2. [GUIDE.md](GUIDE.md) before interpreting plots or changing limits; and
3. [QUICKREF.md](QUICKREF.md) for a compact command/API reference.

## Evidence boundary

The notebooks demonstrate API shape and permit manual exploration. They do not
establish model quality, checkpoint parity, throughput, memory ceilings, or
production readiness. Any such claim needs reproducible model smokes,
golden-output checks, labelled retrieval evaluation, and measurements on the
target hardware.
