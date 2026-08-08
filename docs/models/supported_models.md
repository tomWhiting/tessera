# Model catalog and support tiers

The filename is retained for existing links, but this is a catalog rather than
a list of fully supported models. Tessera currently has no `Supported` entries.
The authoritative per-model metadata and notes live in
[`models.json`](../../models.json).

## Current totals

| Tier | Count | Meaning |
|---|---:|---|
| `Supported` | 0 | Passed the project's repeatable compatibility and output-validation bar |
| `Experimental` | 10 | Runtime adapter and immutable checkpoint pin exist; inference and output quality remain provisional |
| `CatalogOnly` | 12 | Metadata only; builders reject execution |

Use `tessera::model_registry::runnable_models()` to get the `Supported` and
`Experimental` entries. Use `get_model()` for catalog-complete discovery, then
inspect `support_tier`, `support_note`, and `is_runnable()`.

## Experimental adapter paths

| Representation | Model IDs |
|---|---|
| Dense | `bge-base-en-v1.5`, `jina-embeddings-v2-small-en`, `jina-embeddings-v2-base-en`, `nomic-embed-v1.5`, `snowflake-arctic-l` |
| Multi-vector | `colbert-small`, `colbert-v2` |
| Sparse | `splade-pp-en-v1`, `splade-pp-en-v2` |
| Vision-language | `colpali-v1.2` |

These entries are runnable in the narrow sense that Tessera exposes an adapter.
They are not a claim that every referenced checkpoint currently loads, produces
correct embeddings, or meets a quality or performance target.

## Catalog-only metadata

| Representation | Model IDs |
|---|---|
| Dense | `jina-embeddings-v2-base-code`, `jina-embeddings-v3` |
| Multi-vector / unified | `bge-m3-multi`, `colpali-v1.3-hf`, `gte-modern-colbert`, `jina-colbert-v2`, `jina-colbert-v2-64`, `jina-colbert-v2-96` |
| Sparse | `minicoil-v1`, `splade-v3` |
| Time series | `chronos-bolt-small`, `timesfm-1.0-200m` |

Catalog-only entries cover architectures or artifact layouts for which the
current adapters are incompatible or absent. The generated support note states
the concrete reason for each entry. Builders reject these IDs before model
artifacts are downloaded.

## Context windows

Several entries advertise 8,192-token contexts. That value is descriptive
metadata, not a safe default. Tessera defaults to 512 tokens and 1,048,576
attention cells; raising both limits only passes preflight. Full attention is
quadratic and multiplies across heads, layers, and temporary tensors, so an 8K
request can still exhaust CPU, GPU, or Metal shared memory.

## Promotion criteria

An entry should move to `Supported` only after the checked certification
contract passes: byte-verified pinned artifacts, two matching clean offline CPU
runs from one source commit, enforced peak-RSS evidence, checked shape and
numerical behavior, and a pinned official-reference fingerprint.

No current entry meets that bar. See the
[local certification guide](../../certification/README.md) for the explicit
fetch, run, readiness, and purge workflow. Metadata completeness alone is not
runtime support.

See the [registry quick start](../guides/quick_start_registry.md) for selection
examples and the [registry architecture](../architecture/model_registry.md) for
the generated API and validation flow.
