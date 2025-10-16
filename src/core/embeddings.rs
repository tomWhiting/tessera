//! Token-level embeddings for ColBERT-style late interaction.
//!
//! This module provides types and traits for working with token-level embeddings,
//! which are essential for ColBERT's MaxSim scoring mechanism.

use anyhow::Result;
use ndarray::Array2;

/// Token-level embeddings representing a sequence of tokens.
///
/// Each row represents a single token's embedding vector.
/// Shape: (num_tokens, embedding_dim)
#[derive(Debug, Clone)]
pub struct TokenEmbeddings {
    /// The embedding matrix (num_tokens x embedding_dim)
    pub embeddings: Array2<f32>,
    /// The original input text
    pub text: String,
    /// Number of tokens in the sequence
    pub num_tokens: usize,
    /// Dimensionality of each embedding vector
    pub embedding_dim: usize,
}

impl TokenEmbeddings {
    /// Creates a new TokenEmbeddings instance.
    ///
    /// # Arguments
    /// * `embeddings` - The embedding matrix (num_tokens x embedding_dim)
    /// * `text` - The original input text
    ///
    /// # Returns
    /// A new TokenEmbeddings instance with validated dimensions
    pub fn new(embeddings: Array2<f32>, text: String) -> Result<Self> {
        let shape = embeddings.shape();
        let num_tokens = shape[0];
        let embedding_dim = shape[1];

        anyhow::ensure!(
            num_tokens > 0,
            "Token embeddings must contain at least one token"
        );
        anyhow::ensure!(
            embedding_dim > 0,
            "Embedding dimension must be greater than zero"
        );

        Ok(Self {
            embeddings,
            text,
            num_tokens,
            embedding_dim,
        })
    }

    /// Returns the shape of the embedding matrix as (num_tokens, embedding_dim)
    pub fn shape(&self) -> (usize, usize) {
        (self.num_tokens, self.embedding_dim)
    }
}

/// Trait for models that can encode text into token-level embeddings.
///
/// Implementors should produce embeddings suitable for ColBERT-style
/// late interaction scoring.
pub trait TokenEmbedder {
    /// Encodes the input text into token-level embeddings.
    ///
    /// # Arguments
    /// * `text` - The input text to encode
    ///
    /// # Returns
    /// Token-level embeddings for the input text
    fn encode(&self, text: &str) -> Result<TokenEmbeddings>;
}
