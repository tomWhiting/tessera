# Supported Models

This document lists all models available in the Tessera/Hypiler model registry. The registry is generated at compile time from `models.json` and provides type-safe access to model metadata.

## Overview

The model registry contains **5 ColBERT models** optimized for late-interaction retrieval:

- 1 original Stanford ColBERT model
- 1 compact ColBERT variant
- 3 Jina AI multilingual variants (including Matryoshka representations)

## Usage

### Accessing Models

```rust
use hypiler::model_registry::{get_model, COLBERT_V2};
use hypiler::ModelConfig;

// Access constant directly
println!("Model: {}", COLBERT_V2.name);
println!("Dimensions: {}", COLBERT_V2.embedding_dim);

// Load by ID
let model = get_model("colbert-small").expect("Model not found");
println!("Found: {}", model.name);

// Create config from registry
let config = ModelConfig::from_registry("jina-colbert-v2")?;
```

### Querying Models

```rust
use hypiler::model_registry::{models_by_type, models_by_language, ModelType};

// Get all ColBERT models
let colbert_models = models_by_type(ModelType::Colbert);

// Get models supporting a language
let english_models = models_by_language("en");

// Get compact models
let compact = models_by_max_embedding_dim(128);

// Get Matryoshka-enabled models
let matryoshka = models_with_matryoshka();
```

## Model Catalog

### ColBERT v2

**ID:** `colbert-v2`  
**HuggingFace:** `colbert-ir/colbertv2.0`  
**Organization:** Stanford NLP  
**License:** MIT

Original ColBERT v2 from Stanford, baseline for late interaction retrieval. Uses BERT-base with projection layer to 128 dimensions.

- **Parameters:** 110M
- **Embedding Dimensions:** 128
- **Context Length:** 512 tokens
- **Architecture:** bert-base with projection
- **Languages:** English
- **Benchmarks:**
  - BEIR Average: 0.52
  - MS MARCO MRR@10: 0.39

### ColBERT Small

**ID:** `colbert-small`  
**HuggingFace:** `answerdotai/answerai-colbert-small-v1`  
**Organization:** Answer.AI  
**License:** Apache-2.0

Compact ColBERT variant based on DistilBERT. Recommended for development and testing due to smaller size and faster inference.

- **Parameters:** 33M
- **Embedding Dimensions:** 96
- **Context Length:** 512 tokens
- **Architecture:** distilbert-base with projection
- **Languages:** English
- **Benchmarks:**
  - BEIR Average: 0.45
  - MS MARCO MRR@10: 0.32

### Jina ColBERT v2

**ID:** `jina-colbert-v2`  
**HuggingFace:** `jinaai/jina-colbert-v2`  
**Organization:** Jina AI  
**License:** Apache-2.0

Multilingual ColBERT supporting 89 languages with extended 8K context length. Based on Jina BERT v2 architecture without projection layer.

- **Parameters:** 560M
- **Embedding Dimensions:** 768
- **Context Length:** 8192 tokens
- **Architecture:** jina-bert-v2-base-en (no projection)
- **Languages:** 89 languages including English, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Chinese, Japanese, Korean, Arabic, Hindi, and many more
- **Benchmarks:**
  - BEIR Average: 0.54
  - MS MARCO MRR@10: 0.42

### Jina ColBERT v2 (96-dim Matryoshka)

**ID:** `jina-colbert-v2-96`  
**HuggingFace:** `jinaai/jina-colbert-v2`  
**Organization:** Jina AI  
**License:** Apache-2.0

Jina ColBERT v2 using Matryoshka representation at 96 dimensions for compact storage with minimal quality loss.

- **Parameters:** 560M
- **Embedding Dimensions:** 96 (Matryoshka)
- **Available Dimensions:** 768, 512, 384, 256, 128, 96, 64
- **Context Length:** 8192 tokens
- **Architecture:** jina-bert-v2-base-en (Matryoshka)
- **Languages:** 89 languages
- **Benchmarks:**
  - BEIR Average: 0.53
  - MS MARCO MRR@10: 0.41

### Jina ColBERT v2 (64-dim Matryoshka)

**ID:** `jina-colbert-v2-64`  
**HuggingFace:** `jinaai/jina-colbert-v2`  
**Organization:** Jina AI  
**License:** Apache-2.0

Jina ColBERT v2 using Matryoshka representation at 64 dimensions for maximum compactness.

- **Parameters:** 560M
- **Embedding Dimensions:** 64 (Matryoshka)
- **Available Dimensions:** 768, 512, 384, 256, 128, 96, 64
- **Context Length:** 8192 tokens
- **Architecture:** jina-bert-v2-base-en (Matryoshka)
- **Languages:** 89 languages
- **Benchmarks:**
  - BEIR Average: 0.51
  - MS MARCO MRR@10: 0.39

## Model Comparison

| Model | Dims | Context | Params | BEIR | MRR@10 | Languages | License |
|-------|------|---------|--------|------|--------|-----------|---------|
| ColBERT v2 | 128 | 512 | 110M | 0.52 | 0.39 | 1 | MIT |
| ColBERT Small | 96 | 512 | 33M | 0.45 | 0.32 | 1 | Apache-2.0 |
| Jina ColBERT v2 | 768 | 8192 | 560M | 0.54 | 0.42 | 89 | Apache-2.0 |
| Jina ColBERT v2-96 | 96 | 8192 | 560M | 0.53 | 0.41 | 89 | Apache-2.0 |
| Jina ColBERT v2-64 | 64 | 8192 | 560M | 0.51 | 0.39 | 89 | Apache-2.0 |

## Adding New Models

To add a new model to the registry:

1. Edit `models.json` in the project root
2. Add a new model entry with all required fields
3. Run `cargo build` - the model will be automatically included

### Required Fields

```json
{
  "id": "unique-model-id",
  "type": "colbert",
  "name": "Display Name",
  "huggingface_id": "org/model-name",
  "organization": "Organization Name",
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
    "beir_avg": 0.52,
    "ms_marco_mrr10": 0.39
  },
  "license": "MIT",
  "description": "Model description here"
}
```

## Registry API

### Constants

Each model has a corresponding constant:

- `COLBERT_V2`
- `COLBERT_SMALL`
- `JINA_COLBERT_V2`
- `JINA_COLBERT_V2_96`
- `JINA_COLBERT_V2_64`

### Functions

- `get_model(id: &str) -> Option<&ModelInfo>` - Get model by ID
- `models_by_type(model_type: ModelType) -> Vec<&ModelInfo>` - Filter by type
- `models_by_organization(org: &str) -> Vec<&ModelInfo>` - Filter by organization
- `models_by_language(lang: &str) -> Vec<&ModelInfo>` - Filter by language support
- `models_by_max_embedding_dim(max: usize) -> Vec<&ModelInfo>` - Filter by dimension
- `models_with_matryoshka() -> Vec<&ModelInfo>` - Get Matryoshka models

## Architecture

The model registry is implemented using build-time code generation:

1. **models.json** - Source of truth for all model metadata
2. **build.rs** - Parses JSON and generates Rust code at compile time
3. **model_registry.rs** - Generated code with type-safe constants and functions
4. **ModelConfig** - Integration point for existing code

This approach provides:

- Zero runtime overhead - all metadata is compile-time constants
- Type safety - impossible to reference non-existent models
- Easy maintenance - just edit JSON to add models
- Comprehensive metadata - all model information in one place

## Future Extensions

The registry system is designed to scale to hundreds of models. Future additions may include:

- Dense embedding models
- Sparse embedding models
- Timeseries models
- Geometric embedding models
- Cross-modal models

Adding models is as simple as editing `models.json` and rebuilding.
