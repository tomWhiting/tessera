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
/// fine-grained late-interaction matching via MaxSim.
///
/// # Characteristics
/// - Variable-length output (depends on tokenization)
/// - Token-level granularity
/// - Designed for late interaction scoring
/// - Typically 64-128 dimensions per token
///
/// # Example Models
/// - ColBERT v2 (colbert-ir/colbertv2.0)
/// - Jina ColBERT (jinaai/jina-colbert-v2)
/// - AnswerAI ColBERT Small (answerdotai/answerai-colbert-small-v1)
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
    /// Dimensionality of each token embedding (e.g., 128 for ColBERT v2)
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
    /// Creates a new dense embedding.
    ///
    /// # Arguments
    /// * `embedding` - The embedding vector
    /// * `text` - The original input text
    ///
    /// # Returns
    /// A new DenseEmbedding instance
    pub fn new(embedding: Array1<f32>, text: String) -> Self {
        Self { embedding, text }
    }

    /// Get the embedding dimension.
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
/// - sentence-transformers/all-MiniLM-L6-v2
/// - BAAI/bge-base-en-v1.5
/// - nomic-ai/nomic-embed-text-v1
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
    /// Use the [CLS] token embedding (first token).
    ///
    /// Common in BERT-style models where [CLS] is trained to represent
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
}

/// Sparse vocabulary-space embedding.
///
/// Represents an input as a sparse vector over the vocabulary space,
/// with most dimensions zero (99%+ sparsity typical).
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
    /// Creates a new sparse embedding.
    ///
    /// # Arguments
    /// * `weights` - Non-zero (index, weight) pairs
    /// * `vocab_size` - Total vocabulary size
    /// * `text` - Original input text
    ///
    /// # Returns
    /// A new SparseEmbedding instance
    pub fn new(weights: Vec<(usize, f32)>, vocab_size: usize, text: String) -> Self {
        Self {
            weights,
            vocab_size,
            text,
        }
    }

    /// Get the number of non-zero dimensions.
    pub fn nnz(&self) -> usize {
        self.weights.len()
    }

    /// Calculate the sparsity level (0.0 = dense, 1.0 = all zeros).
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.nnz() as f32 / self.vocab_size as f32)
    }
}

/// Sparse encoder producing vocabulary-space embeddings (SPLADE-style).
///
/// Each input produces a sparse vector over the vocabulary (30K+ dimensions,
/// 99%+ sparsity), compatible with inverted indexes for efficient retrieval.
///
/// # Characteristics
/// - High-dimensional output (vocabulary size, typically 30K+)
/// - Extremely sparse (99%+ zero values)
/// - Inverted-index compatible
/// - Learned term weighting (vs. fixed like BM25)
///
/// # Example Models
/// - naver/splade-cocondenser-ensembledistil
/// - naver/splade_v2_max
/// - naver/splade_v2_distil
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

/// Vision embedding representation for ColPali-style vision-language models.
///
/// Represents an image as a collection of patch embeddings, similar to how
/// TokenEmbeddings represents text as token embeddings. Each patch corresponds
/// to a spatial region of the input image (e.g., 448×448 → 32×32 patches = 1024 total).
///
/// Used for vision-language retrieval where queries are text and documents
/// are page images (PDFs, scans, screenshots). Compatible with late interaction
/// scoring (MaxSim) similar to ColBERT.
///
/// # Example Models
/// - wanghaofan/colpali-v1.2
/// - vidore/colpali-v1
#[derive(Debug, Clone)]
pub struct VisionEmbedding {
    /// Patch embeddings: shape [num_patches, embedding_dim]
    ///
    /// Typically 1024 patches for 448×448 images (32×32 grid).
    /// Each patch embedding is a vector of dimension `embedding_dim`.
    pub embeddings: Vec<Vec<f32>>,

    /// Number of patches in the image grid.
    ///
    /// For ColPali with 448×448 input, this is 32×32 = 1024 patches.
    pub num_patches: usize,

    /// Embedding dimension per patch.
    ///
    /// Typically 128 for ColPali (matches ColBERT dimension for compatibility).
    pub embedding_dim: usize,

    /// Optional: Source image path or identifier.
    ///
    /// Used for tracking the origin of the image embedding.
    pub source: Option<String>,
}

impl VisionEmbedding {
    /// Create a new vision embedding.
    ///
    /// # Arguments
    /// * `embeddings` - The patch embeddings (shape [num_patches, embedding_dim])
    /// * `num_patches` - Number of patches (typically 1024 for ColPali)
    /// * `embedding_dim` - Dimension per patch (typically 128 for ColPali)
    /// * `source` - Optional source image path/identifier
    ///
    /// # Returns
    /// A new VisionEmbedding instance
    pub fn new(
        embeddings: Vec<Vec<f32>>,
        num_patches: usize,
        embedding_dim: usize,
        source: Option<String>,
    ) -> Self {
        Self {
            embeddings,
            num_patches,
            embedding_dim,
            source,
        }
    }

    /// Get the number of patches.
    ///
    /// # Returns
    /// Number of patches in this image embedding
    pub fn num_patches(&self) -> usize {
        self.num_patches
    }

    /// Get the embedding dimension per patch.
    ///
    /// # Returns
    /// Dimensionality of each patch embedding vector
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the source image path/identifier if available.
    ///
    /// # Returns
    /// Optional reference to the source identifier
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Get the shape of the embedding matrix as (num_patches, embedding_dim).
    ///
    /// # Returns
    /// Tuple of (number of patches, embedding dimension)
    pub fn shape(&self) -> (usize, usize) {
        (self.num_patches, self.embedding_dim)
    }
}

/// Vision encoder producing patch-level embeddings (ColPali-style).
///
/// Encodes images into multi-vector representations where each vector
/// corresponds to a spatial patch. This enables late interaction scoring
/// with text queries for vision-language retrieval, similar to ColBERT's
/// token-level interactions.
///
/// # Characteristics
/// - Variable-length output (depends on image size)
/// - Patch-level granularity (typically 32×32 = 1024 patches per image)
/// - Designed for late interaction scoring with text queries
/// - Typically 128-384 dimensions per patch
/// - Compatible with MaxSim scoring used in ColBERT
///
/// # Example Models
/// - wanghaofan/colpali-v1.2
/// - vidore/colpali-v1
pub trait VisionEncoder: Encoder<Output = VisionEmbedding> {
    /// Get the number of patches per image.
    ///
    /// # Returns
    /// Number of patches the encoder produces per image.
    /// Typically 1024 for ColPali (32×32 grid of 14×14 pixel patches from 448×448 images).
    fn num_patches(&self) -> usize;

    /// Get the embedding dimension per patch.
    ///
    /// # Returns
    /// Dimensionality of each patch embedding vector.
    /// Typically 128 for ColPali (matches ColBERT dimension for compatibility).
    fn embedding_dim(&self) -> usize;

    /// Get the input image resolution.
    ///
    /// # Returns
    /// Tuple of (width, height) in pixels that the encoder expects.
    /// Typically (448, 448) for ColPali.
    fn image_resolution(&self) -> (u32, u32);
}

/// Time series embedding representation for time series foundation models.
///
/// Represents time series data as fixed-size embedding vectors suitable for
/// similarity search, clustering, and retrieval. Unlike forecasting outputs,
/// embeddings are designed to capture the temporal patterns in a compressed
/// representation for downstream tasks.
///
/// # Example Models
/// - amazon/chronos-bolt-small
/// - google/timesfm-1.0-200m
#[derive(Debug, Clone)]
pub struct TimeSeriesEmbedding {
    /// Embedding vectors: [num_series, embedding_dim]
    ///
    /// For batch processing, this contains embeddings for multiple time series.
    /// Each row represents the embedding for one time series.
    pub embeddings: Vec<Vec<f32>>,

    /// Number of time series in the batch.
    pub num_series: usize,

    /// Embedding dimension (e.g., 512 for Chronos Bolt).
    pub embedding_dim: usize,

    /// Optional: Original time series lengths before padding.
    ///
    /// Useful for tracking which series were padded/truncated during preprocessing.
    pub original_lengths: Option<Vec<usize>>,

    /// Optional: Source identifier for tracking data origin.
    pub source: Option<String>,
}

impl TimeSeriesEmbedding {
    /// Create a new time series embedding.
    ///
    /// # Arguments
    /// * `embeddings` - The embedding vectors [num_series, embedding_dim]
    /// * `num_series` - Number of time series in the batch
    /// * `embedding_dim` - Dimension of each embedding vector
    /// * `original_lengths` - Optional original lengths before preprocessing
    /// * `source` - Optional source identifier
    ///
    /// # Returns
    /// A new TimeSeriesEmbedding instance
    pub fn new(
        embeddings: Vec<Vec<f32>>,
        num_series: usize,
        embedding_dim: usize,
        original_lengths: Option<Vec<usize>>,
        source: Option<String>,
    ) -> Self {
        Self {
            embeddings,
            num_series,
            embedding_dim,
            original_lengths,
            source,
        }
    }

    /// Get the number of time series in this embedding.
    pub fn num_series(&self) -> usize {
        self.num_series
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the shape of the embedding matrix as (num_series, embedding_dim).
    pub fn shape(&self) -> (usize, usize) {
        (self.num_series, self.embedding_dim)
    }

    /// Get the source identifier if available.
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }
}

/// Time series encoder producing fixed-size embeddings from temporal data.
///
/// Encodes time series into fixed-size vector representations suitable for
/// similarity search, clustering, and retrieval. These models can typically
/// also perform forecasting, but the primary use case is embedding extraction.
///
/// # Characteristics
/// - Fixed-length output (one vector per time series)
/// - Temporal pattern compression
/// - Designed for similarity-based retrieval
/// - Typically 192-1280 dimensions
/// - Context lengths from 512 to 2048+ timesteps
///
/// # Example Models
/// - amazon/chronos-bolt-small (512-dim)
/// - google/timesfm-1.0-200m (1280-dim)
pub trait TimeSeriesEncoder: Encoder<Output = TimeSeriesEmbedding> {
    /// Get the embedding dimension.
    ///
    /// # Returns
    /// Dimensionality of the output embedding vector
    fn embedding_dim(&self) -> usize;

    /// Get the context length (maximum input timesteps).
    ///
    /// # Returns
    /// Maximum number of timesteps the encoder can process
    fn context_length(&self) -> usize;

    /// Get the prediction length (forecast horizon).
    ///
    /// # Returns
    /// Number of future timesteps the model can predict (if forecasting is supported)
    fn prediction_length(&self) -> usize;

    /// Forecast future values from historical data.
    ///
    /// # Arguments
    /// * `input` - Historical time series data [batch, channels, timesteps]
    ///
    /// # Returns
    /// Predicted future values [batch, channels, prediction_length]
    ///
    /// # Errors
    /// Returns error if forecasting fails or is not supported
    fn forecast(&self, input: &candle_core::Tensor) -> Result<candle_core::Tensor>;

    /// Extract embeddings for similarity search.
    ///
    /// # Arguments
    /// * `input` - Time series data [batch, channels, timesteps]
    ///
    /// # Returns
    /// Fixed-size embeddings [batch, embedding_dim]
    ///
    /// # Errors
    /// Returns error if embedding extraction fails
    fn extract_embeddings(&self, input: &candle_core::Tensor) -> Result<candle_core::Tensor>;
}
