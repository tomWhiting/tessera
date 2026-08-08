# Model registry architecture

Tessera generates a typed Rust catalog from the checked-in `models.json` during
the build. The catalog is intentionally broader than runtime support: metadata
can remain useful even when the current Candle adapters cannot load a model.

## Data flow

```text
models.json
    -> build.rs + build_support/* (parse, validate, generate)
    -> OUT_DIR/model_registry.rs
    -> src/models/registry.rs
    -> tessera::model_registry
```

No JSON parsing or registry file access occurs at runtime. Build failures catch
invalid metadata before generated code is compiled.

## Support is explicit

Every entry has a non-empty support note and one of three tiers:

- `SupportTier::Supported`: the runtime path has passed the project's support
  and verification bar.
- `SupportTier::Experimental`: an adapter and immutable checkpoint pin exist,
  but repeatable inference and output quality remain provisional.
- `SupportTier::CatalogOnly`: metadata is retained, but Tessera cannot execute
  the model with its current adapters.

There are currently no `Supported` entries. Catalog lookup therefore must not
be treated as proof that a model can run.

```rust
use tessera::model_registry::{get_model, runnable_models, SupportTier};

let entry = get_model("jina-colbert-v2").expect("catalog entry");
assert_eq!(entry.support_tier, SupportTier::CatalogOnly);
assert!(!entry.is_runnable());

for model in runnable_models() {
    println!("{}: {:?} - {}", model.id, model.support_tier, model.support_note);
}
```

`get_model` and `MODEL_REGISTRY` are catalog-complete. `runnable_models()` and
`ModelInfo::is_runnable()` exclude catalog-only entries. The high-level builders
also reject catalog-only models before downloading artifacts.

## Generated API

The generated module exposes:

- `ModelInfo`, `ModelType`, `SupportTier`, `EmbeddingDimension`, and pooling
  metadata types;
- one constant per model plus `MODEL_REGISTRY`;
- `get_model` and `get_model_by_hf_id`;
- `runnable_models`;
- filters by type, organization, language, maximum default dimension, and
  Matryoshka capability.

Model IDs are strings, so an unknown ID is a normal runtime lookup miss:

```rust
use tessera::model_registry::get_model;

assert!(get_model("not-a-model").is_none());
```

## Adding or changing an entry

1. Edit `models.json`.
2. Supply accurate architecture, dimensions, context, artifacts, license, and
   capability metadata.
3. Set `support.tier` and a concrete `support.note`. New metadata does not earn
   runtime support by itself.
4. Run the registry and source-size policies:

   ```bash
   cargo run --locked --offline -p tessera-xtask -- all
   ```

5. Run the complete local, model-free repository gate:

   ```bash
   ./scripts/check
   ```

6. For a runnable model, update its checked certification specification and run
   the explicit local certification workflow. A tier promotion requires the
   complete evidence contract documented in
   [`certification/README.md`](../../certification/README.md).

The build validates unique IDs, positive and internally consistent dimensions,
Hugging Face repository IDs, projection metadata, and support metadata. See
`build_support/validation.rs` for the executable contract rather than copying
the schema into another document.

## Context windows and resource limits

`ModelInfo::context_length` records model metadata; it is not a memory-safety
promise. Tessera's default resource policy remains capped at 512 tokens and
1,048,576 attention cells. Raising those limits only passes request preflight;
full attention still scales quadratically and can exhaust host or accelerator
memory. See the root README before opting into a larger window.
