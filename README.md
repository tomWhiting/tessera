# Tessera

**Multi-paradigm embedding library for Rust**

Tessera is a high-performance embedding library supporting multiple embedding paradigms with native GPU acceleration (Metal, CUDA) and a type-safe API.

## Status

🚧 **In Development** - Phase 2 Complete (Core Embedding Types)

## Features

### Embedding Paradigms

- **Multi-Vector (ColBERT)** - Token-level embeddings with MaxSim scoring
- **Dense (BERT)** - Single-vector embeddings with pooling strategies
- **Sparse (SPLADE)** - Vocabulary-sized sparse vectors for interpretable search

### Capabilities

- **26 Production Models** - ColBERT, BGE, Nomic, GTE, Snowflake, SPLADE variants
- **GPU Acceleration** - Metal (macOS), CUDA (Linux), CPU fallback
- **Batch Processing** - 5-10x throughput for large-scale encoding
- **Binary Quantization** - 32x compression for multi-vector embeddings
- **Matryoshka Dimensions** - Variable-precision embeddings
- **Type-Safe API** - Factory pattern with compile-time guarantees

## Quick Start

```rust
use tessera::TesseraDense;

// Dense embeddings
let embedder = TesseraDense::new("bge-base-en-v1.5")?;
let embedding = embedder.encode("Hello, world!")?;

// Similarity search
let score = embedder.similarity("query", "document")?;
```

## Architecture

- **Zero-copy operations** - Minimal data movement
- **Production-ready** - Comprehensive error handling
- **Well-tested** - 103 tests, zero failures
- **Documented** - Extensive API documentation

## Documentation

See `docs/COMPLETION_PLAN.md` for detailed implementation notes and roadmap.

## License

[To be determined]

## Status

Phase 2 Complete:
- ✅ Multi-vector embeddings (ColBERT)
- ✅ Dense embeddings (BERT pooling)
- ✅ Sparse embeddings (SPLADE)
- ⏳ Python bindings (Phase 2.3)
- ⏳ Vision-language embeddings (Phase 3)
- ⏳ Time series embeddings (Phase 3)
