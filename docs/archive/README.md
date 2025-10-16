# Tessera

**Multi-vector embeddings and geometric representations for Rust**

Production-ready embedding library featuring ColBERT, time series models, and exotic geometries.

## Why Tessera?

- **GPU acceleration that works** - Metal on Apple Silicon, CUDA support
- **Multi-vector models** - ColBERT, ColPali for fine-grained retrieval
- **Pure Rust** - Single binary, no Python runtime required
- **Latest 2024-2025 models** - Jina ColBERT v2, modern architectures
- **Build-time model registry** - Type-safe model metadata at compile time
- **Matryoshka dimension support** - Flexible embedding sizes (64-768 dims)
- **Exotic embeddings** - Hyperbolic, time series, geometric models (roadmap)

## Quick Start

```rust
use tessera::{
    backends::CandleEncoder,
    core::{TokenEmbedder, max_sim},
    model_registry,
};

fn main() -> anyhow::Result<()> {
    // Load model from registry
    let model = model_registry::get_model("colbert-small")
        .expect("Model not found");
    
    let device = tessera::backends::candle::get_device()?;
    let config = tessera::models::ModelConfig::from_registry(model);
    let encoder = CandleEncoder::new(config, device)?;
    
    // Encode and compute similarity
    let query = encoder.encode("What is machine learning?")?;
    let doc = encoder.encode("Machine learning is a subset of AI")?;
    
    let score = max_sim(&query, &doc)?;
    println!("Similarity: {:.4}", score);
    
    Ok(())
}
```

## Supported Models

### Multi-Vector Models (ColBERT)

| Model | Dims | Context | Parameters | Languages | Notes |
|-------|------|---------|------------|-----------|-------|
| `colbert-v2` | 128 | 512 | 110M | EN | Original Stanford ColBERT |
| `colbert-small` | 96 | 512 | 33M | EN | Fast, lightweight |
| `jina-colbert-v2` | 64-768* | 8192 | 560M | 89 languages | Matryoshka support |
| `jina-colbert-v2-96` | 96 | 8192 | 560M | 89 languages | Optimized for compactness |
| `jina-colbert-v2-64` | 64 | 8192 | 560M | 89 languages | Maximum efficiency |

*Supports Matryoshka dimensions: 64, 96, 128, 256, 384, 512, 768

For complete model details, see [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md)

## Features

### Token-level ColBERT Embeddings

ColBERT produces one vector per token, enabling fine-grained semantic matching:

```rust
// Each token gets its own embedding vector
let embeddings = encoder.encode("machine learning algorithms")?;
// Shape: [num_tokens, embedding_dim]
```

MaxSim scoring finds the best match for each query token:

```
MaxSim(Q, D) = Σ(i=1 to n) max(j=1 to m) qi · dj
```

### Metal GPU Acceleration

Optimized for Apple Silicon:

```toml
[dependencies]
tessera = { version = "0.1.0", features = ["metal"] }
```

```bash
cargo run --example basic_similarity --features metal
```

### Build-time Model Registry

Type-safe model metadata generated at compile time:

```rust
use tessera::model_registry::{self, ModelType};

// Get specific model
let model = model_registry::get_model("colbert-v2")?;
println!("Dimensions: {}", model.embedding_dim.default_dim());

// Query by criteria
let colbert_models = model_registry::models_by_type(ModelType::Colbert);
let multilingual = model_registry::models_by_language("zh");
let matryoshka = model_registry::models_with_matryoshka();

// Filter by embedding size
let compact = model_registry::models_by_max_embedding_dim(96);
```

### Matryoshka Dimension Support

Flexible embedding sizes with minimal quality loss:

```rust
let model = model_registry::get_model("jina-colbert-v2")?;

// Check supported dimensions
if model.embedding_dim.supports_dimension(96) {
    println!("96 dimensions supported!");
}

// Get all available dimensions
let dims = model.embedding_dim.supported_dimensions();
// [64, 96, 128, 256, 384, 512, 768]
```

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
tessera = "0.1.0"

# For GPU acceleration
tessera = { version = "0.1.0", features = ["metal"] }  # Apple Silicon
tessera = { version = "0.1.0", features = ["cuda"] }   # NVIDIA GPUs
```

## CLI Usage

```bash
# Basic similarity
cargo run -- --query "What is machine learning?" \
             --document "Machine learning is a subset of AI"

# Use specific model
cargo run -- --model jina-colbert-v2-96 \
             --query "machine learning" \
             --document "artificial intelligence"

# With Metal acceleration
cargo run --features metal -- --query "..." --document "..."
```

## Architecture

```
tessera/
├── src/
│   ├── core/           # Core abstractions
│   │   ├── embeddings.rs   # TokenEmbeddings, TokenEmbedder trait
│   │   ├── similarity.rs   # MaxSim algorithm
│   │   └── tokenizer.rs    # Tokenization
│   │
│   ├── backends/       # Backend implementations
│   │   ├── candle/         # Candle backend (production-ready)
│   │   └── burn/           # Burn backend (experimental)
│   │
│   └── models/         # Model configuration
│       ├── config.rs       # Model configs
│       ├── loader.rs       # HuggingFace Hub
│       └── registry.rs     # Generated registry (build.rs)
│
├── models.json         # Model metadata (source of truth)
├── build.rs            # Generates model registry at compile time
│
├── docs/               # Documentation
│   ├── architecture/   # System design docs
│   ├── models/         # Model documentation
│   ├── guides/         # How-to guides
│   └── vision_board/   # Future model categories
│
└── examples/           # Usage examples
```

## Examples

```bash
# Basic similarity with ColBERT
cargo run --example basic_similarity --features metal

# Comprehensive demo (longer text, embeddings)
cargo run --example comprehensive_demo --features metal

# Model registry usage
cargo run --example model_registry_demo

# Registry-based similarity
cargo run --example registry_similarity --features metal
```

## Model Categories (Registry)

Tessera organizes models by category in `models.json`:

- **multi_vector**: ColBERT, ColPali (token-level embeddings)
- **dense**: BERT, GTE, E5 (single vector per input) - Coming soon
- **sparse**: SPLADE, uniCOIL (vocabulary-sized vectors) - Coming soon
- **timeseries**: TimesFM, TTM, Chronos - Coming soon
- **geometric**: Hyperbolic, spherical, quaternion - Coming soon

The registry is generated at compile time by `build.rs` from `models.json`, providing type-safe access to model metadata.

## Roadmap

### Complete (v0.1)
- [x] ColBERT inference with projection layers
- [x] MaxSim similarity scoring
- [x] Metal GPU acceleration
- [x] Multiple ColBERT models (v2, Small, Jina)
- [x] Build-time model registry
- [x] Matryoshka dimension support
- [x] Category-based model organization

### In Progress
- [ ] Batch processing
- [ ] Binary quantization
- [ ] Dense embeddings (BERT, GTE, E5)

### Planned
- [ ] ColPali (vision-language multi-vector)
- [ ] Sparse models (SPLADE, uniCOIL)
- [ ] Time series models (TimesFM, TTM, Chronos)
- [ ] Hyperbolic embeddings
- [ ] Python bindings
- [ ] Vector database integration

## Documentation

- [Supported Models](SUPPORTED_MODELS.md) - Complete model specifications
- [Architecture](docs/architecture/) - System design and build system
- [Guides](docs/guides/) - Quick start, GPU acceleration, adding models
- [Vision Board](docs/vision_board/) - Future model categories

## Performance

- **Model caching**: HuggingFace Hub caches models locally (`~/.cache/huggingface/`)
- **First run**: Downloads models (one-time cost)
- **Subsequent runs**: Uses cached models (fast)
- **Metal acceleration**: 5-10x speedup on Apple Silicon M1/M2/M3
- **Memory efficient**: Matryoshka models allow quality/size tradeoffs

## Contributing

Contributions welcome! Please ensure:
- Code compiles without warnings
- Tests pass
- Documentation is updated
- Follow Rust idioms and project structure

## License

MIT

## Credits

- **ColBERT**: Stanford NLP (Omar Khattab et al.)
- **Jina ColBERT**: Jina AI
- **ColBERT Small**: Answer.AI
- **Candle**: Hugging Face
- **Burn**: Burn ML community
