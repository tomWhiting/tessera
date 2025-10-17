//! Core abstractions and algorithms for Tessera.
//!
//! This module contains the fundamental types, traits, and algorithms
//! that underpin all embedding operations:
//!
//! - [`embeddings`]: Core embedding types and traits
//! - [`similarity`]: Similarity scoring algorithms (MaxSim, cosine, etc.)
//! - [`tokenizer`]: Text tokenization abstractions
//!
//! These abstractions are paradigm-agnostic and used across all
//! embedding types (multi-vector, dense, sparse, time series).
//!
//! # Token-Level Embeddings
//!
//! The core design centers around token-level embeddings that enable
//! late interaction similarity scoring (MaxSim):
//!
//! - `TokenEmbeddings`: Represents token-level embeddings for a text sequence
//! - `TokenEmbedder`: Trait for models that produce token-level embeddings
//!
//! # Similarity Algorithms
//!
//! - `max_sim`: MaxSim similarity for multi-vector embeddings
//! - Cosine similarity for dense embeddings
//! - Hamming distance for binary quantized embeddings
//!
//! # Tokenization
//!
//! - `Tokenizer`: Abstraction over HuggingFace tokenizers
//! - Consistent interface across different models

pub mod embeddings;
pub mod similarity;
pub mod tokenizer;

// Export legacy types for backward compatibility
pub use embeddings::{TokenEmbedder, TokenEmbeddings};
pub use tokenizer::Tokenizer;

// Export new unified trait hierarchy
pub use embeddings::{
    DenseEmbedding, DenseEncoder, Encoder, MultiVectorEncoder, PoolingStrategy, SparseEmbedding,
    SparseEncoder, VisionEmbedding, VisionEncoder,
};
