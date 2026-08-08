# Model registry quick start

The registry is both a catalog and a support contract. Use catalog lookup for
discovery and the runnable filter when choosing a model for execution.

## Inspect one entry

```rust
use tessera::model_registry::get_model;

let model = get_model("colbert-small").expect("registered model");
println!("tier: {:?}", model.support_tier);
println!("note: {}", model.support_note);
println!("context: {}", model.context_length);
println!("dimensions: {:?}", model.embedding_dim.supported_dimensions());
```

`get_model` may return `CatalogOnly` metadata. Check `model.is_runnable()`
before sending the ID to a builder.

## List runtime candidates

```rust
use tessera::model_registry::runnable_models;

for model in runnable_models() {
    println!("{} ({:?})", model.id, model.support_tier);
}
```

The current runnable list contains experimental adapters as well as any future
supported adapters. It does not mean every remote checkpoint has passed a smoke
test.

## Build an embedder

```rust,no_run
use tessera::TesseraDense;

fn main() -> tessera::Result<()> {
    let embedder = TesseraDense::builder()
        .model("bge-base-en-v1.5")
        .build()?;
    let vector = embedder.encode("registry-backed dense embedding")?;
    println!("{} dimensions", vector.dim());
    Ok(())
}
```

Construction can download checkpoint files. `bge-base-en-v1.5` is currently
`Experimental`, so this is a smoke-test path rather than a compatibility
guarantee.

## Query metadata

```rust
use tessera::model_registry::{models_by_type, ModelType};

let dense = models_by_type(ModelType::Dense);
for model in dense {
    println!("{}: {:?}", model.id, model.support_tier);
}
```

Other filters cover organization, language, maximum default embedding
dimension, Matryoshka capability, and Hugging Face repository ID.

## Large contexts

A registered 8K context window is model metadata, not the default request size
and not proof that an 8K request fits in memory. Tessera defaults to 512 tokens
and 1,048,576 attention cells. Review the resource-policy warning in the
[root README](../../README.md) before raising those limits.

See the [model catalog](../models/supported_models.md) for support-tier meaning
and the [registry architecture](../architecture/model_registry.md) for the
generation and validation contract.
