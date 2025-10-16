//! Core abstractions for token-level embeddings and similarity scoring.
//!
//! This module provides the fundamental types and traits for implementing
//! ColBERT-style late interaction similarity scoring:
//!
//! - `TokenEmbeddings`: Represents token-level embeddings for a text sequence
//! - `TokenEmbedder`: Trait for models that produce token-level embeddings
//! - `max_sim`: MaxSim similarity scoring function
//! - `Tokenizer`: Abstraction over HuggingFace tokenizers

pub mod embeddings;
pub mod similarity;
pub mod tokenizer;

pub use embeddings::{TokenEmbedder, TokenEmbeddings};
pub use similarity::max_sim;
pub use tokenizer::Tokenizer;
