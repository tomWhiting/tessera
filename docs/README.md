# Tessera documentation

Tessera is an alpha-quality revival. The root [README](../README.md) is the
primary status page and the checked-in [`models.json`](../models.json) is the
source of truth for model metadata and support tiers.

## Maintained documentation

- [Model catalog](models/supported_models.md) explains `Supported`,
  `Experimental`, and `CatalogOnly` without treating catalog breadth as runtime
  compatibility.
- [Registry architecture](architecture/model_registry.md) describes build-time
  generation and the public lookup API.
- [Registry quick start](guides/quick_start_registry.md) shows how to inspect
  the catalog and select only runnable entries.
- [Python development guide](PYTHON_DEV_GUIDE.md) covers the current split PyO3
  bindings and model-free test lane.
- [Time-series legacy note](legacy/TIMESERIES.md) records why Chronos and
  TimesFM are not active runtimes.
- [Notebook guide](../examples/notebooks/README.md) covers the exploratory
  Marimo examples and their explicit memory warnings.

The generated registry currently has 22 entries: 10 `Experimental`, 12
`CatalogOnly`, and no `Supported` entries. `Experimental` means an adapter path
exists but still needs repeatable remote-checkpoint and output validation.
`CatalogOnly` means metadata is discoverable but Tessera rejects execution.

## Historical material

The repository's `docs/archive/` directory, the dated `vision_board` research
survey, old phase and completion plans, PyEntry design briefs, implementation
postmortems, and test-summary reports are retained as project history.
Prominent banners identify those snapshots. They must not be used as current
support or benchmark evidence.

Cargo packaging excludes that historical corpus. Maintained documentation,
including this index and the legacy compatibility note, remains in the crate so
links from the packaged README continue to resolve.
