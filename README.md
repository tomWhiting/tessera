# Tessera

**Multi-paradigm embedding library for Rust**

Tessera is a high-performance embedding library supporting five embedding paradigms with native GPU acceleration (Metal, CUDA) and a type-safe API.

## Status

✅ **Phase 3.2 Complete** - Vision-Language + Time Series Forecasting

## Features

### Embedding Paradigms

- **Multi-Vector (ColBERT)** - Token-level embeddings with MaxSim scoring
- **Dense (BERT)** - Single-vector embeddings with pooling strategies
- **Sparse (SPLADE)** - Vocabulary-sized sparse vectors for interpretable search
- **Vision-Language (ColPali)** - OCR-free document search with late interaction
- **Time Series (Chronos Bolt)** - Zero-shot probabilistic forecasting

### Capabilities

- **23 Production Models** - ColBERT, BGE, Nomic, GTE, Snowflake, Qwen3, Jina, SPLADE, ColPali, Chronos Bolt
- **GPU Acceleration** - Metal (macOS), CUDA (Linux), CPU fallback
- **Batch Processing** - 5-10x throughput for large-scale encoding
- **Binary Quantization** - 32x compression for multi-vector embeddings
- **Matryoshka Dimensions** - Variable-precision embeddings
- **PDF Document Search** - Native PDF rendering with Poppler
- **Probabilistic Forecasting** - 9 quantile levels for uncertainty quantification
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

Phase 3.2 Complete:
- ✅ Multi-vector embeddings (ColBERT)
- ✅ Dense embeddings (BERT pooling)
- ✅ Sparse embeddings (SPLADE)
- ✅ Vision-language embeddings (ColPali)
- ✅ Time series forecasting (Chronos Bolt)
- ⏳ Python bindings (Phase 2.3)
- ⏳ Hyperbolic embeddings (Phase 3.3)
