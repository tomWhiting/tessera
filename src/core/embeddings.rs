//! Embedding types and encoder trait abstractions.
//!
//! This module provides types and traits for working with embeddings across
//! different encoding paradigms:
//!
//! - Multi-vector embeddings (ColBERT): Token-level embeddings for late interaction
//! - Dense embeddings (BERT, BGE, Nomic): Single pooled vector per input
//! - Sparse embeddings (SPLADE): Vocabulary-space sparse vectors
//! - Vision embeddings (ColPali): Patch-level image embeddings for vision-language retrieval
//!
//! # Trait Hierarchy
//!
//! The `Encoder` trait provides the base abstraction for all encoders,
//! with specialized subtraits for each paradigm:
//!
//! - `MultiVectorEncoder`: Produces token-level embeddings (ColBERT-style)
//! - `DenseEncoder`: Produces single pooled vectors (BERT-style)
//! - `SparseEncoder`: Produces sparse vocabulary vectors (SPLADE-style)
//! - `VisionEncoder`: Produces patch-level embeddings (ColPali-style)
//!
//! This hierarchy enables writing generic code over different encoder types
//! while maintaining paradigm-specific functionality.

use anyhow::Result;
use ndarray::{Array1, Array2};

#[cfg(feature = "timeseries")]
mod timeseries;
mod vision;

#[cfg(feature = "timeseries")]
pub use timeseries::{TimeSeriesEmbedding, TimeSeriesEncoder};
pub use vision::{VisionEmbedding, VisionEncoder};

/// Token-level embeddings representing a sequence of tokens.
///
/// Each row represents a single token's embedding vector.
/// Shape: (`num_tokens`, `embedding_dim`)
#[derive(Debug, Clone)]
pub struct TokenEmbeddings {
    /// The embedding matrix (`num_tokens` x `embedding_dim`)
    pub embeddings: Array2<f32>,
    /// The original input text
    pub text: String,
    /// Number of tokens in the sequence
    pub num_tokens: usize,
    /// Dimensionality of each embedding vector
    pub embedding_dim: usize,
}

impl TokenEmbeddings {
    /// Creates a new `TokenEmbeddings` instance.
    ///
    /// # Arguments
    /// * `embeddings` - The embedding matrix (`num_tokens` x `embedding_dim`)
    /// * `text` - The original input text
    ///
    /// # Returns
    /// A new `TokenEmbeddings` instance with validated dimensions
    ///
    /// # Errors
    /// Returns an error if embeddings contain NaN or Inf values
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

        // Validate no NaN or Inf values
        anyhow::ensure!(
            embeddings.iter().all(|v| v.is_finite()),
            "Token embeddings contain NaN or Inf values"
        );

        Ok(Self {
            embeddings,
            text,
            num_tokens,
            embedding_dim,
        })
    }

    /// Returns the shape of the embedding matrix as (`num_tokens`, `embedding_dim`)
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.num_tokens, self.embedding_dim)
    }
}

/// Trait for models that can encode text into token-level embeddings.
///
/// Implementors should produce embeddings suitable for ColBERT-style
/// late interaction scoring.
///
/// # Note
/// This trait is maintained for backward compatibility. New code should use
/// the `MultiVectorEncoder` trait instead.
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

// ============================================================================
// Unified Encoder Trait Hierarchy
// ============================================================================

/// Base trait for all encoders.
///
/// Provides a common interface for encoding text/data into embeddings,
/// regardless of the embedding paradigm (dense, sparse, multi-vector, etc.).
///
/// # Type Parameters
/// * `Output` - The type of embeddings produced by this encoder
///
/// # Example
/// ```no_run
/// use tessera::core::{Encoder, TokenEmbeddings};
/// use anyhow::Result;
///
/// fn process_text<E: Encoder<Output = TokenEmbeddings>>(
///     encoder: &E,
///     text: &str
/// ) -> Result<E::Output> {
///     encoder.encode(text)
/// }
/// ```
pub trait Encoder {
    /// Output embedding type produced by this encoder
    type Output;

    /// Encode a single input text into embeddings.
    ///
    /// # Arguments
    /// * `input` - The text to encode
    ///
    /// # Returns
    /// Embeddings for the input text
    ///
    /// # Errors
    /// Returns an error if encoding fails (tokenization, model inference, etc.)
    fn encode(&self, input: &str) -> Result<Self::Output>;

    /// Encode multiple inputs in batch.
    ///
    /// # Arguments
    /// * `inputs` - Slice of text inputs to encode
    ///
    /// # Returns
    /// Vector of embeddings, one per input
    ///
    /// # Errors
    /// Returns an error if batch encoding fails
    ///
    /// # Note
    /// Default implementation calls `encode` for each input sequentially.
    /// Backend implementations should override this with optimized batching.
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Self::Output>> {
        inputs.iter().map(|&text| self.encode(text)).collect()
    }
}

/// Multi-vector encoder producing token-level embeddings (ColBERT-style).
///
/// Each input produces multiple vectors (one per token), enabling
/// fine-grained late-interaction matching via `MaxSim`.
///
/// # Characteristics
/// - Variable-length output (depends on tokenization)
/// - Token-level granularity
/// - Designed for late interaction scoring
/// - Typically 64-128 dimensions per token
///
/// # Example Models
/// - `colbert-v2`
/// - `colbert-small`
pub trait MultiVectorEncoder: Encoder<Output = TokenEmbeddings> {
    /// Get the number of vectors that would be produced for a given text.
    ///
    /// This corresponds to the number of tokens after tokenization.
    ///
    /// # Arguments
    /// * `text` - The input text to analyze
    ///
    /// # Returns
    /// Number of token vectors that will be produced
    ///
    /// # Errors
    /// Returns an error if tokenization fails
    fn num_vectors(&self, text: &str) -> Result<usize>;

    /// Get the embedding dimension per token vector.
    ///
    /// # Returns
    /// Dimensionality of each token embedding (e.g., 128 for `ColBERT` v2)
    fn embedding_dim(&self) -> usize;
}

/// Dense single-vector embedding.
///
/// Represents a single pooled vector for an entire input text,
/// produced by pooling token embeddings (CLS, mean, max, etc.).
#[derive(Debug, Clone)]
pub struct DenseEmbedding {
    /// The embedding vector
    pub embedding: Array1<f32>,
    /// Original input text
    pub text: String,
}

impl DenseEmbedding {
    /// Creates a new dense embedding with validation.
    ///
    /// # Arguments
    /// * `embedding` - The embedding vector
    /// * `text` - The original input text
    ///
    /// # Returns
    /// A new `DenseEmbedding` instance, or error if embedding contains NaN/Inf
    ///
    /// # Errors
    /// Returns an error if embedding contains non-finite values
    pub fn new(embedding: Array1<f32>, text: String) -> anyhow::Result<Self> {
        anyhow::ensure!(
            embedding.iter().all(|v| v.is_finite()),
            "Dense embedding contains NaN or Inf values"
        );
        Ok(Self { embedding, text })
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.embedding.len()
    }
}

/// Single-vector encoder producing pooled embeddings (BERT-style).
///
/// Each input is encoded to a single vector via a pooling strategy
/// (CLS token, mean pooling, max pooling).
///
/// # Characteristics
/// - Fixed-length output (one vector per input)
/// - Sentence/document-level granularity
/// - Efficient for large-scale retrieval (single vector comparison)
/// - Typically 384-1024 dimensions
///
/// # Example Models
/// - `bge-base-en-v1.5`
/// - `jina-embeddings-v2-small-en`
/// - `nomic-embed-v1.5`
pub trait DenseEncoder: Encoder<Output = DenseEmbedding> {
    /// Get the embedding dimension.
    ///
    /// # Returns
    /// Dimensionality of the output embedding vector
    fn embedding_dim(&self) -> usize;

    /// Get the pooling strategy used by this encoder.
    ///
    /// # Returns
    /// The pooling strategy (CLS, mean, max)
    fn pooling_strategy(&self) -> PoolingStrategy;
}

/// Pooling strategy for dense encodings.
///
/// Determines how token-level embeddings are aggregated into a single vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use the `[CLS]` token embedding (first token).
    ///
    /// Common in BERT-style models where `[CLS]` is trained to represent
    /// the entire sequence.
    Cls,

    /// Average all token embeddings (weighted by attention mask).
    ///
    /// Produces a centroid representation of all tokens, ignoring padding.
    /// Most common strategy in sentence transformers.
    Mean,

    /// Element-wise maximum across all token embeddings.
    ///
    /// Captures the most salient features from any token position.
    Max,

    /// Use the last valid token's embedding.
    ///
    /// Used by decoder-based embedding models (e.g., Qwen3-Embedding)
    /// where bidirectional attention allows the final token to accumulate
    /// information from the entire sequence.
    LastToken,
}

/// Sparse vocabulary-space embedding.
///
/// Represents an input as a sparse vector over the vocabulary space. The
/// observed sparsity depends on the model, input, and filtering threshold.
#[derive(Debug, Clone)]
pub struct SparseEmbedding {
    /// Sparse vector as (index, weight) pairs.
    ///
    /// Only non-zero dimensions are stored. Indices correspond to
    /// vocabulary token IDs.
    pub weights: Vec<(usize, f32)>,

    /// Total vocabulary size (dimension of full dense vector).
    pub vocab_size: usize,

    /// Original input text
    pub text: String,
}

impl SparseEmbedding {
    /// Creates a new sparse embedding with validation.
    ///
    /// # Arguments
    /// * `weights` - Non-zero (index, weight) pairs
    /// * `vocab_size` - Total vocabulary size
    /// * `text` - Original input text
    ///
    /// # Returns
    /// A new `SparseEmbedding` instance, or an error if validation fails
    ///
    /// # Errors
    /// Returns an error if any weight index >= vocab_size or if any weight is NaN/Inf
    pub fn new(weights: Vec<(usize, f32)>, vocab_size: usize, text: String) -> Result<Self> {
        // Validate all indices are within bounds
        for &(idx, _) in &weights {
            anyhow::ensure!(
                idx < vocab_size,
                "Sparse embedding index {} exceeds vocabulary size {}",
                idx,
                vocab_size
            );
        }

        // Validate no NaN or Inf weights
        for &(idx, weight) in &weights {
            anyhow::ensure!(
                weight.is_finite(),
                "Sparse embedding weight at index {} is not finite: {}",
                idx,
                weight
            );
        }

        Ok(Self {
            weights,
            vocab_size,
            text,
        })
    }

    /// Get the number of non-zero dimensions.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.weights.len()
    }

    /// Calculate the sparsity level (0.0 = dense, 1.0 = all zeros).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.nnz() as f32 / self.vocab_size as f32)
    }
}

/// Sparse encoder producing vocabulary-space embeddings (SPLADE-style).
///
/// Each input produces a sparse vector over the model vocabulary, compatible
/// with inverted-index retrieval structures.
///
/// # Characteristics
/// - High-dimensional output (the model's vocabulary size)
/// - Input-dependent sparsity
/// - Inverted-index compatible
/// - Learned term weighting (vs. fixed like BM25)
///
/// # Example Models
/// - `splade-pp-en-v1`
/// - `splade-pp-en-v2`
pub trait SparseEncoder: Encoder<Output = SparseEmbedding> {
    /// Get the vocabulary size (dimension of full dense vector).
    ///
    /// # Returns
    /// Total vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get the expected sparsity level for this encoder.
    ///
    /// # Returns
    /// Expected sparsity (0.0 = dense, 1.0 = all zeros)
    ///
    /// This is a guideline value; actual sparsity varies by input.
    fn expected_sparsity(&self) -> f32;
}
