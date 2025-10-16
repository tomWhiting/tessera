# Tessera Vision Board: Comprehensive Embedding Models

This vision board outlines the complete landscape of embedding models and representations that Tessera aims to support.

## Parts

1. [Multi-Vector Models](part1_multi_vector.md) - ColBERT, ColPali (token-level)
2. [Sparse Models](part2_sparse.md) - SPLADE, uniCOIL (vocabulary-sized vectors)
3. [Time Series Models](part3_timeseries.md) - TimesFM, TTM, Chronos
4. [Exotic Geometries](part4_exotic.md) - Hyperbolic, spherical, quaternion
5. [Dense Models](part5_dense.md) - BERT, GTE, E5 (conclusion)

## Current Status

- **Part 1 (Multi-Vector)**: Implemented - ColBERT v2, ColBERT Small, Jina ColBERT v2
- **Parts 2-5**: Planned for future releases

## Model Categories

### 1. Multi-Vector (Current)
Dense token-level embeddings where each token gets its own vector. Enables fine-grained semantic matching with late interaction.

**Examples**: ColBERT, ColPali
**Status**: Production-ready

### 2. Sparse (Planned)
Vocabulary-sized sparse vectors with efficient storage and retrieval.

**Examples**: SPLADE, uniCOIL, SparseEmbed
**Status**: Roadmap

### 3. Time Series (Planned)
Foundation models for time series embeddings and forecasting.

**Examples**: TimesFM, TTM, Chronos, Lag-Llama
**Status**: Roadmap

### 4. Geometric (Planned)
Non-Euclidean embeddings for hierarchical and geometric relationships.

**Examples**: Hyperbolic, Spherical, Quaternion, Lorentz
**Status**: Roadmap

### 5. Dense (Planned)
Traditional single-vector embeddings for semantic similarity.

**Examples**: BERT, GTE, E5, BGE, Sentence Transformers
**Status**: Roadmap

## Design Philosophy

Tessera aims to be the **most comprehensive embedding library** across all representation types:

1. **Production-ready**: Real implementations, no placeholders
2. **GPU-accelerated**: Metal, CUDA, and optimized CPU backends
3. **Type-safe**: Build-time model registry with compile-time guarantees
4. **Flexible**: Support for multiple backends (Candle, Burn)
5. **Documented**: Extensive documentation and examples

## Contributing

We welcome contributions for any model category! See the individual parts for specific model details and implementation guidelines.
