# Model Registry - Quick Start Guide

## Using the Registry

### Access a Model Constant

```rust
use hypiler::model_registry::COLBERT_V2;

println!("Model: {}", COLBERT_V2.name);
println!("Dimensions: {}", COLBERT_V2.embedding_dim);
println!("HuggingFace: {}", COLBERT_V2.huggingface_id);
```

### Look Up a Model by ID

```rust
use hypiler::model_registry::get_model;

let model = get_model("colbert-small")
    .expect("Model not found");

println!("Found: {}", model.name);
```

### Create a Config from Registry

```rust
use hypiler::ModelConfig;

// From registry by ID
let config = ModelConfig::from_registry("jina-colbert-v2")?;

// Or use existing convenience methods (still work)
let config = ModelConfig::colbert_v2();
```

### List All Models

```rust
use hypiler::model_registry::MODEL_REGISTRY;

for model in MODEL_REGISTRY {
    println!("{}: {} dims, {} context",
        model.name,
        model.embedding_dim,
        model.context_length
    );
}
```

### Query by Type

```rust
use hypiler::model_registry::{models_by_type, ModelType};

let colbert_models = models_by_type(ModelType::Colbert);
for model in colbert_models {
    println!("{}: {} dims", model.name, model.embedding_dim);
}
```

### Query by Language

```rust
use hypiler::model_registry::models_by_language;

let english_models = models_by_language("en");
println!("Found {} English models", english_models.len());
```

### Query by Dimension

```rust
use hypiler::model_registry::models_by_max_embedding_dim;

let compact = models_by_max_embedding_dim(128);
println!("Found {} models with <= 128 dims", compact.len());
```

### Get Matryoshka Models

```rust
use hypiler::model_registry::models_with_matryoshka;

let matryoshka = models_with_matryoshka();
for model in matryoshka {
    println!("{}: {:?}", model.name, model.matryoshka_dims);
}
```

## Available Models

| ID | Name | Dims | Context | Params | Languages |
|----|------|------|---------|--------|-----------|
| `colbert-v2` | ColBERT v2 | 128 | 512 | 110M | 1 |
| `colbert-small` | ColBERT Small | 96 | 512 | 33M | 1 |
| `jina-colbert-v2` | Jina ColBERT v2 | 768 | 8192 | 560M | 89 |
| `jina-colbert-v2-96` | Jina ColBERT v2 (96-dim) | 96 | 8192 | 560M | 89 |
| `jina-colbert-v2-64` | Jina ColBERT v2 (64-dim) | 64 | 8192 | 560M | 89 |

## Adding a New Model

1. Edit `models.json`:

```json
{
  "id": "my-new-model",
  "type": "colbert",
  "name": "My New Model",
  "huggingface_id": "org/my-new-model",
  "organization": "My Org",
  "release_date": "2024",
  "architecture": {
    "type": "bert",
    "variant": "bert-base",
    "has_projection": true,
    "projection_dims": 128
  },
  "specs": {
    "parameters": "110M",
    "embedding_dim": 128,
    "hidden_dim": 768,
    "context_length": 512,
    "max_position_embeddings": 512,
    "vocab_size": 30522
  },
  "files": {
    "tokenizer": "tokenizer.json",
    "config": "config.json",
    "weights": {
      "safetensors": "model.safetensors",
      "pytorch": "pytorch_model.bin"
    }
  },
  "capabilities": {
    "languages": ["en"],
    "modalities": ["text"],
    "multi_vector": true,
    "quantization": ["fp32", "fp16"]
  },
  "performance": {
    "beir_avg": 0.50,
    "ms_marco_mrr10": 0.35
  },
  "license": "MIT",
  "description": "Description of my model"
}
```

2. Rebuild:

```bash
cargo build
```

3. Use it:

```rust
use hypiler::model_registry::MY_NEW_MODEL;
println!("New model: {}", MY_NEW_MODEL.name);

// Or by ID
let config = ModelConfig::from_registry("my-new-model")?;
```

## Examples

Run the examples to see the registry in action:

```bash
# Comprehensive API demo
cargo run --example model_registry_demo

# Practical similarity scoring
cargo run --example registry_similarity

# Complete showcase
cargo run --example registry_showcase
```

## Documentation

- **SUPPORTED_MODELS.md** - Complete model catalog
- **MODEL_REGISTRY.md** - Full system documentation
- **REGISTRY_IMPLEMENTATION.md** - Implementation details

## API Reference

### Types

- `ModelType` - Enum of model categories
- `ModelInfo` - Struct with all model metadata

### Constants

- `COLBERT_V2` - Stanford ColBERT v2
- `COLBERT_SMALL` - Answer.AI ColBERT Small
- `JINA_COLBERT_V2` - Jina ColBERT v2 (full)
- `JINA_COLBERT_V2_96` - Jina ColBERT v2 (96-dim Matryoshka)
- `JINA_COLBERT_V2_64` - Jina ColBERT v2 (64-dim Matryoshka)
- `MODEL_REGISTRY` - Array of all models

### Functions

- `get_model(id)` - Get model by ID
- `models_by_type(type)` - Filter by type
- `models_by_organization(org)` - Filter by organization
- `models_by_language(lang)` - Filter by language
- `models_by_max_embedding_dim(max)` - Filter by dimension
- `models_with_matryoshka()` - Get Matryoshka models

## Benefits

- **Type Safe** - Compiler catches invalid model references
- **Zero Overhead** - All compile-time constants
- **Easy Maintenance** - Add models by editing JSON
- **Comprehensive** - 20+ metadata fields per model
- **Well Tested** - 14 tests, all passing
- **Well Documented** - 4 comprehensive guides

## Getting Help

1. Check the examples: `cargo run --example model_registry_demo`
2. Read the docs: **MODEL_REGISTRY.md**
3. View model catalog: **SUPPORTED_MODELS.md**
4. Run tests: `cargo test models::registry`

---

For detailed information, see **MODEL_REGISTRY.md**.
