//! Main Tessera embedder interface.
//!
//! Provides the primary user-facing API for encoding text into
//! embeddings. Supports both dense single-vector and multi-vector
//! encoders with automatic model type detection.
//!
//! # Example
//!
//! ```ignore
//! use tessera::Tessera;
//!
//! // Auto-detect model type and create appropriate encoder
//! let embedder = Tessera::new("colbert-v2")?;  // Creates MultiVector variant
//! let embedder = Tessera::new("bge-base-en-v1.5")?;  // Creates Dense variant
//!
//! // Or use specific types directly
//! use tessera::{TesseraMultiVector, TesseraDense};
//!
//! let mv_embedder = TesseraMultiVector::new("colbert-v2")?;
//! let dense_embedder = TesseraDense::new("bge-base-en-v1.5")?;
//! ```

use crate::api::{
    TesseraDenseBuilder, TesseraMultiVectorBuilder, TesseraSparseBuilder, TesseraTimeSeriesBuilder,
    TesseraVisionBuilder,
};
use crate::backends::CandleBertEncoder;
use crate::core::{
    DenseEmbedding, DenseEncoder, Encoder, SparseEmbedding, TokenEmbedder, TokenEmbeddings,
    VisionEmbedding,
};
use crate::encoding::dense::CandleDenseEncoder;
use crate::encoding::sparse::CandleSparseEncoder;
use crate::encoding::vision::ColPaliEncoder;
use crate::error::{Result, TesseraError};
use crate::models::registry::{get_model, ModelType};
use crate::quantization::{
    binary::BinaryVector, multi_vector_distance, quantize_multi, BinaryQuantization,
};
use crate::timeseries::models::ChronosBolt;
use crate::utils::similarity::max_sim;
use candle_core::Tensor;
use std::path::Path;

/// Binary quantized multi-vector embeddings.
///
/// Represents token embeddings compressed to 1-bit per dimension,
/// providing 32x compression with 95%+ accuracy retention.
#[derive(Debug, Clone)]
pub struct QuantizedEmbeddings {
    /// Quantized token vectors
    pub quantized: Vec<BinaryVector>,
    /// Original embedding dimension (before quantization)
    pub original_dim: usize,
    /// Number of token vectors
    pub num_tokens: usize,
}

impl QuantizedEmbeddings {
    /// Memory usage in bytes.
    ///
    /// Returns the total memory footprint including vector data and overhead.
    #[must_use] pub fn memory_bytes(&self) -> usize {
        self.quantized.iter().map(super::super::quantization::binary::BinaryVector::memory_bytes).sum()
    }

    /// Compression ratio compared to float32.
    ///
    /// Returns how much smaller the quantized representation is
    /// compared to the original float32 embeddings.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ratio = quantized.compression_ratio();
    /// println!("Compressed {:.1}x smaller", ratio);  // ~32.0x
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f32 {
        let float_bytes = self.num_tokens * self.original_dim * 4;
        let quantized_bytes = self.memory_bytes();
        if quantized_bytes == 0 {
            return 0.0;
        }
        float_bytes as f32 / quantized_bytes as f32
    }
}

/// Multi-vector embedder for ColBERT-style token-level embeddings.
///
/// Produces token-level embeddings suitable for late interaction scoring
/// via `MaxSim`. Each input text generates multiple vectors (one per token).
///
/// Thread-safe and can be shared across threads.
pub struct TesseraMultiVector {
    /// Backend encoder (currently Candle only)
    encoder: CandleBertEncoder,
    /// Model identifier from registry
    model_id: String,
    /// Optional quantizer for compression
    quantizer: Option<BinaryQuantization>,
}

impl TesseraMultiVector {
    /// Create a new embedder with default configuration.
    ///
    /// This is the simplest way to create an embedder - it automatically:
    /// - Looks up the model in the registry
    /// - Selects the best available device (Metal > CUDA > CPU)
    /// - Downloads the model from `HuggingFace` if needed
    /// - Initializes the encoder
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry (e.g., "colbert-v2", "jina-colbert-v2")
    ///
    /// # Returns
    ///
    /// Initialized embedder ready for use.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - Model cannot be downloaded or loaded
    /// - Device initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraMultiVector;
    ///
    /// let embedder = TesseraMultiVector::new("colbert-v2")?;
    /// let embeddings = embedder.encode("What is machine learning?")?;
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        // Use builder with just model ID
        TesseraMultiVectorBuilder::new().model(model_id).build()
    }

    /// Create a builder for advanced configuration.
    ///
    /// Use this for advanced use cases like:
    /// - Specifying a custom device
    /// - Setting Matryoshka dimensions
    /// - Enabling binary quantization
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraMultiVector;
    /// use candle_core::Device;
    ///
    /// let embedder = TesseraMultiVector::builder()
    ///     .model("jina-colbert-v2")
    ///     .device(Device::Cpu)
    ///     .build()?;
    /// ```
    #[must_use] pub const fn builder() -> TesseraMultiVectorBuilder {
        TesseraMultiVectorBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) const fn from_encoder(
        encoder: CandleBertEncoder,
        model_id: String,
        quantizer: Option<BinaryQuantization>,
    ) -> Self {
        Self {
            encoder,
            model_id,
            quantizer,
        }
    }

    /// Encode a single text into embeddings.
    ///
    /// Returns token-level embeddings suitable for ColBERT-style late interaction.
    /// Each token gets its own embedding vector.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode
    ///
    /// # Returns
    ///
    /// `TokenEmbeddings` containing the embedding matrix and metadata.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tokenization fails
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embeddings = embedder.encode("What is machine learning?")?;
    /// println!("Encoded to {} vectors of {} dimensions",
    ///     embeddings.num_tokens,
    ///     embeddings.embedding_dim);
    /// ```
    pub fn encode(&self, text: &str) -> Result<TokenEmbeddings> {
        TokenEmbedder::encode(&self.encoder, text).map_err(|e| TesseraError::EncodingError {
            context: format!("Failed to encode text: '{text}'"),
            source: e,
        })
    }

    /// Encode multiple texts in a batch.
    ///
    /// More efficient than calling `encode()` repeatedly due to
    /// batched inference on GPU. Achieves 5-10x speedup for batch sizes of 100+.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to encode
    ///
    /// # Returns
    ///
    /// Vector of `TokenEmbeddings`, one per input text.
    ///
    /// # Errors
    ///
    /// Returns error if encoding any text fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embeddings = embedder.encode_batch(&[
    ///     "First document",
    ///     "Second document",
    /// ])?;
    /// ```
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<TokenEmbeddings>> {
        Encoder::encode_batch(&self.encoder, texts).map_err(|e| TesseraError::EncodingError {
            context: format!("Failed to encode batch of {} texts", texts.len()),
            source: e,
        })
    }

    /// Compute similarity between two texts.
    ///
    /// Convenience method that encodes both texts and computes `MaxSim` similarity.
    /// `MaxSim` is the standard similarity metric for `ColBERT` multi-vector embeddings.
    ///
    /// # Arguments
    ///
    /// * `text_a` - First text
    /// * `text_b` - Second text
    ///
    /// # Returns
    ///
    /// Similarity score (higher = more similar). Typically in range [0, 1] for
    /// normalized embeddings.
    ///
    /// # Errors
    ///
    /// Returns error if encoding or similarity computation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let score = embedder.similarity(
    ///     "What is machine learning?",
    ///     "Machine learning is a subset of AI"
    /// )?;
    /// println!("Similarity: {:.4}", score);
    /// ```
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
        let emb_a = self.encode(text_a)?;
        let emb_b = self.encode(text_b)?;

        max_sim(&emb_a, &emb_b).map_err(|e| TesseraError::EncodingError {
            context: "Failed to compute similarity".to_string(),
            source: e,
        })
    }

    /// Get the embedding dimension.
    ///
    /// Returns the dimensionality of each token's embedding vector.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Embedding dimension: {}", embedder.dimension());
    /// ```
    pub fn dimension(&self) -> usize {
        use crate::core::MultiVectorEncoder;
        self.encoder.embedding_dim()
    }

    /// Get the model identifier.
    ///
    /// Returns the model ID from the registry (e.g., "colbert-v2").
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Using model: {}", embedder.model());
    /// ```
    pub fn model(&self) -> &str {
        &self.model_id
    }

    /// Quantize embeddings to binary representation (32x compression).
    ///
    /// Converts float32 embeddings to 1-bit binary representation,
    /// providing 32x memory reduction with 95%+ accuracy retention.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Full-precision embeddings to quantize
    ///
    /// # Returns
    ///
    /// Quantized embeddings with compression metadata.
    ///
    /// # Errors
    ///
    /// Returns error if no quantizer is configured. Use
    /// `.quantization(QuantizationConfig::Binary)` in the builder.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::{TesseraMultiVector, QuantizationConfig};
    ///
    /// let embedder = TesseraMultiVector::builder()
    ///     .model("colbert-v2")
    ///     .quantization(QuantizationConfig::Binary)
    ///     .build()?;
    ///
    /// let embeddings = embedder.encode("What is machine learning?")?;
    /// let quantized = embedder.quantize(&embeddings)?;
    ///
    /// println!("Compression: {:.1}x", quantized.compression_ratio());
    /// ```
    pub fn quantize(&self, embeddings: &TokenEmbeddings) -> Result<QuantizedEmbeddings> {
        #[allow(clippy::option_if_let_else)]
        match &self.quantizer {
            Some(q) => {
                // Convert Array2 to Vec<Vec<f32>> for quantization
                let vectors: Vec<Vec<f32>> = (0..embeddings.num_tokens)
                    .map(|i| embeddings.embeddings.row(i).to_vec())
                    .collect();

                let quantized = quantize_multi(q, &vectors);
                Ok(QuantizedEmbeddings {
                    quantized,
                    original_dim: embeddings.embedding_dim,
                    num_tokens: embeddings.num_tokens,
                })
            }
            None => Err(TesseraError::QuantizationError(
                "No quantizer configured. Use .quantization(QuantizationConfig::Binary) in builder"
                    .to_string(),
            )),
        }
    }

    /// Encode and quantize in one step.
    ///
    /// Convenience method that combines encoding and quantization.
    /// More efficient than calling `encode()` then `quantize()` separately
    /// when you only need the quantized representation.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode and quantize
    ///
    /// # Returns
    ///
    /// Quantized embeddings ready for similarity computation.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails or no quantizer is configured.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let quantized = embedder.encode_quantized("What is ML?")?;
    /// println!("Encoded {} tokens", quantized.num_tokens);
    /// ```
    pub fn encode_quantized(&self, text: &str) -> Result<QuantizedEmbeddings> {
        let embeddings = self.encode(text)?;
        self.quantize(&embeddings)
    }

    /// Compute similarity between quantized embeddings using Hamming distance.
    ///
    /// Uses the `MaxSim` algorithm adapted for binary embeddings:
    /// - Distance computed via XOR + popcount (Hamming distance)
    /// - For each query vector, find max similarity with document vectors
    /// - Sum across all query vectors
    ///
    /// This is significantly faster than float32 `MaxSim` while maintaining
    /// 95%+ accuracy.
    ///
    /// # Arguments
    ///
    /// * `query` - Quantized query embeddings
    /// * `document` - Quantized document embeddings
    ///
    /// # Returns
    ///
    /// Similarity score (higher = more similar). Scale is different from
    /// float32 `MaxSim` but ranking is preserved.
    ///
    /// # Errors
    ///
    /// Returns error if no quantizer is configured.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query = embedder.encode_quantized("What is ML?")?;
    /// let doc = embedder.encode_quantized("Machine learning is AI")?;
    /// let score = embedder.similarity_quantized(&query, &doc)?;
    /// println!("Similarity: {:.4}", score);
    /// ```
    pub fn similarity_quantized(
        &self,
        query: &QuantizedEmbeddings,
        document: &QuantizedEmbeddings,
    ) -> Result<f32> {
        #[allow(clippy::option_if_let_else)]
        match &self.quantizer {
            Some(q) => {
                let score = multi_vector_distance(q, &query.quantized, &document.quantized);
                Ok(score)
            }
            None => Err(TesseraError::QuantizationError(
                "No quantizer configured. Use .quantization(QuantizationConfig::Binary) in builder"
                    .to_string(),
            )),
        }
    }
}

// ============================================================================
// Dense Single-Vector Embedder
// ============================================================================

/// Dense single-vector embedder for traditional sentence embeddings.
///
/// Produces a single pooled vector per input text via strategies like
/// CLS token, mean pooling, or max pooling. Suitable for semantic search
/// and classification tasks.
///
/// Thread-safe and can be shared across threads.
pub struct TesseraDense {
    /// Backend encoder (Candle dense encoder)
    encoder: CandleDenseEncoder,
    /// Model identifier from registry
    model_id: String,
}

impl TesseraDense {
    /// Create a new dense embedder with default configuration.
    ///
    /// This is the simplest way to create a dense embedder - it automatically:
    /// - Looks up the model in the registry
    /// - Selects the best available device (Metal > CUDA > CPU)
    /// - Downloads the model from `HuggingFace` if needed
    /// - Initializes the encoder with appropriate pooling strategy
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry (e.g., "bge-base-en-v1.5", "nomic-embed-text-v1")
    ///
    /// # Returns
    ///
    /// Initialized embedder ready for use.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - Model is not a dense model type
    /// - Model cannot be downloaded or loaded
    /// - Device initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraDense;
    ///
    /// let embedder = TesseraDense::new("bge-base-en-v1.5")?;
    /// let embedding = embedder.encode("What is machine learning?")?;
    /// assert_eq!(embedding.dim(), 768);
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        // Use builder with just model ID
        TesseraDenseBuilder::new().model(model_id).build()
    }

    /// Create a builder for advanced configuration.
    ///
    /// Use this for advanced use cases like:
    /// - Specifying a custom device
    /// - Setting Matryoshka dimensions
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraDense;
    /// use candle_core::Device;
    ///
    /// let embedder = TesseraDense::builder()
    ///     .model("bge-base-en-v1.5")
    ///     .device(Device::Cpu)
    ///     .build()?;
    /// ```
    #[must_use] pub const fn builder() -> TesseraDenseBuilder {
        TesseraDenseBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) const fn from_encoder(encoder: CandleDenseEncoder, model_id: String) -> Self {
        Self { encoder, model_id }
    }

    /// Encode a single text into a dense embedding.
    ///
    /// Returns a single pooled vector representing the entire input text.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode
    ///
    /// # Returns
    ///
    /// `DenseEmbedding` containing the pooled embedding vector.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tokenization fails
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedding = embedder.encode("What is machine learning?")?;
    /// println!("Encoded to {} dimensions", embedding.dim());
    /// ```
    pub fn encode(&self, text: &str) -> Result<DenseEmbedding> {
        <CandleDenseEncoder as Encoder>::encode(&self.encoder, text).map_err(|e| {
            TesseraError::EncodingError {
                context: format!("Failed to encode text: '{text}'"),
                source: e,
            }
        })
    }

    /// Encode multiple texts in a batch.
    ///
    /// More efficient than calling `encode()` repeatedly due to
    /// batched inference on GPU. Achieves 5-10x speedup for batch sizes of 100+.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to encode
    ///
    /// # Returns
    ///
    /// Vector of `DenseEmbedding`, one per input text.
    ///
    /// # Errors
    ///
    /// Returns error if encoding any text fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embeddings = embedder.encode_batch(&[
    ///     "First document",
    ///     "Second document",
    /// ])?;
    /// ```
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>> {
        <CandleDenseEncoder as Encoder>::encode_batch(&self.encoder, texts).map_err(|e| {
            TesseraError::EncodingError {
                context: format!("Failed to encode batch of {} texts", texts.len()),
                source: e,
            }
        })
    }

    /// Compute cosine similarity between two texts.
    ///
    /// Convenience method that encodes both texts and computes cosine similarity.
    /// For normalized embeddings, this is equivalent to dot product.
    ///
    /// # Arguments
    ///
    /// * `text_a` - First text
    /// * `text_b` - Second text
    ///
    /// # Returns
    ///
    /// Similarity score (higher = more similar). Typically in range [-1, 1],
    /// or [0, 1] for normalized embeddings.
    ///
    /// # Errors
    ///
    /// Returns error if encoding or similarity computation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let score = embedder.similarity(
    ///     "What is machine learning?",
    ///     "Machine learning is a subset of AI"
    /// )?;
    /// println!("Similarity: {:.4}", score);
    /// ```
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
        let emb_a = self.encode(text_a)?;
        let emb_b = self.encode(text_b)?;

        // Compute cosine similarity (dot product for normalized embeddings)
        let dot_product: f32 = emb_a
            .embedding
            .iter()
            .zip(emb_b.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        Ok(dot_product)
    }

    /// Get the embedding dimension.
    ///
    /// Returns the dimensionality of the output embedding vector.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Embedding dimension: {}", embedder.dimension());
    /// ```
    pub fn dimension(&self) -> usize {
        self.encoder.embedding_dim()
    }

    /// Get the model identifier.
    ///
    /// Returns the model ID from the registry (e.g., "bge-base-en-v1.5").
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Using model: {}", embedder.model());
    /// ```
    pub fn model(&self) -> &str {
        &self.model_id
    }
}

// ============================================================================
// Sparse Embedding Encoder (SPLADE)
// ============================================================================

/// Sparse embedder for SPLADE-style vocabulary-sized embeddings.
///
/// Produces sparse vectors where most dimensions are zero (99%+ sparsity).
/// Suitable for interpretable search and inverted index integration.
///
/// Thread-safe and can be shared across threads.
pub struct TesseraSparse {
    /// Backend encoder (Candle sparse encoder)
    encoder: CandleSparseEncoder,
    /// Model identifier from registry
    model_id: String,
}

impl TesseraSparse {
    /// Create a new sparse embedder with default configuration.
    ///
    /// This is the simplest way to create a sparse embedder - it automatically:
    /// - Looks up the model in the registry
    /// - Selects the best available device (Metal > CUDA > CPU)
    /// - Downloads the model from `HuggingFace` if needed
    /// - Initializes the encoder with MLM head
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry (e.g., "splade-pp-en-v1", "splade-pp-en-v2")
    ///
    /// # Returns
    ///
    /// Initialized embedder ready for use.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - Model is not a sparse model type
    /// - Model cannot be downloaded or loaded
    /// - Device initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraSparse;
    ///
    /// let embedder = TesseraSparse::new("splade-cocondenser")?;
    /// let embedding = embedder.encode("What is machine learning?")?;
    /// println!("Sparsity: {:.2}%", embedding.sparsity() * 100.0);
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        TesseraSparseBuilder::new().model(model_id).build()
    }

    /// Create a builder for advanced configuration.
    ///
    /// Use this for advanced use cases like:
    /// - Specifying a custom device
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraSparse;
    /// use candle_core::Device;
    ///
    /// let embedder = TesseraSparse::builder()
    ///     .model("splade-cocondenser")
    ///     .device(Device::Cpu)
    ///     .build()?;
    /// ```
    #[must_use] pub const fn builder() -> TesseraSparseBuilder {
        TesseraSparseBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) const fn from_encoder(encoder: CandleSparseEncoder, model_id: String) -> Self {
        Self { encoder, model_id }
    }

    /// Encode a single text into a sparse embedding.
    ///
    /// Returns a sparse vector with vocabulary-sized dimensions (30522 for BERT).
    /// Typical sparsity: 99%+ (only ~100-200 non-zero dimensions).
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode
    ///
    /// # Returns
    ///
    /// `SparseEmbedding` containing sparse vector representation.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tokenization fails
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedding = embedder.encode("What is machine learning?")?;
    /// println!("Non-zero dimensions: {}", embedding.nnz());
    /// println!("Sparsity: {:.2}%", embedding.sparsity() * 100.0);
    /// ```
    pub fn encode(&self, text: &str) -> Result<SparseEmbedding> {
        <CandleSparseEncoder as Encoder>::encode(&self.encoder, text).map_err(|e| {
            TesseraError::EncodingError {
                context: format!("Failed to encode text: '{text}'"),
                source: e,
            }
        })
    }

    /// Encode multiple texts in a batch.
    ///
    /// More efficient than calling `encode()` repeatedly.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to encode
    ///
    /// # Returns
    ///
    /// Vector of `SparseEmbedding`, one per input text.
    ///
    /// # Errors
    ///
    /// Returns error if encoding any text fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embeddings = embedder.encode_batch(&[
    ///     "First document",
    ///     "Second document",
    /// ])?;
    /// ```
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<SparseEmbedding>> {
        <CandleSparseEncoder as Encoder>::encode_batch(&self.encoder, texts).map_err(|e| {
            TesseraError::EncodingError {
                context: format!("Failed to encode batch of {} texts", texts.len()),
                source: e,
            }
        })
    }

    /// Compute dot product similarity between two texts.
    ///
    /// Convenience method that encodes both texts and computes sparse dot product.
    /// For sparse vectors, this is the standard similarity metric.
    ///
    /// # Arguments
    ///
    /// * `text_a` - First text
    /// * `text_b` - Second text
    ///
    /// # Returns
    ///
    /// Similarity score (higher = more similar).
    ///
    /// # Errors
    ///
    /// Returns error if encoding or similarity computation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let score = embedder.similarity(
    ///     "What is machine learning?",
    ///     "Machine learning is a subset of AI"
    /// )?;
    /// println!("Similarity: {:.4}", score);
    /// ```
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
        let emb_a = self.encode(text_a)?;
        let emb_b = self.encode(text_b)?;

        // Sparse dot product
        let mut score = 0.0;
        for (idx_a, weight_a) in &emb_a.weights {
            if let Some(&(_, weight_b)) = emb_b.weights.iter().find(|(idx_b, _)| idx_b == idx_a) {
                score += weight_a * weight_b;
            }
        }

        Ok(score)
    }

    /// Get the vocabulary size (embedding dimension).
    ///
    /// Returns the full vocabulary dimension (typically 30522 for BERT).
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Vocab size: {}", embedder.vocab_size());
    /// ```
    pub fn vocab_size(&self) -> usize {
        use crate::core::SparseEncoder;
        self.encoder.vocab_size()
    }

    /// Get the model identifier.
    ///
    /// Returns the model ID from the registry (e.g., "splade-cocondenser").
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Using model: {}", embedder.model());
    /// ```
    pub fn model(&self) -> &str {
        &self.model_id
    }
}

// ============================================================================
// Vision-Language Encoder (ColPali)
// ============================================================================

/// Vision-language embedder for `ColPali` document retrieval.
///
/// Encodes document page images as multi-vector patch embeddings and enables
/// text queries to search visually through documents without OCR.
///
/// Thread-safe and can be shared across threads (except for encoding operations
/// which require exclusive access due to interior mutability).
pub struct TesseraVision {
    /// Backend encoder (`ColPali` encoder)
    encoder: ColPaliEncoder,
    /// Model identifier from registry
    model_id: String,
}

impl TesseraVision {
    /// Create a new vision-language embedder with default configuration.
    ///
    /// This is the simplest way to create a vision embedder - it automatically:
    /// - Looks up the model in the registry
    /// - Selects the best available device (Metal > CUDA > CPU)
    /// - Downloads the model from `HuggingFace` if needed (3B params, ~5.88 GB)
    /// - Initializes the `PaliGemma` vision-language model
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry (e.g., "colpali-v1.3-hf", "colpali-v1.2")
    ///
    /// # Returns
    ///
    /// Initialized embedder ready for document and query encoding.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - Model is not a vision-language model type
    /// - Model cannot be downloaded or loaded
    /// - Device initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraVision;
    ///
    /// let embedder = TesseraVision::new("colpali-v1.3-hf")?;
    /// let doc_emb = embedder.encode_document("invoice.jpg")?;
    /// let query_emb = embedder.encode_query("What is the total amount?")?;
    /// let score = embedder.search(&query_emb, &doc_emb)?;
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        TesseraVisionBuilder::new().model(model_id).build()
    }

    /// Create a builder for advanced configuration.
    #[must_use] pub const fn builder() -> TesseraVisionBuilder {
        TesseraVisionBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) const fn from_encoder(encoder: ColPaliEncoder, model_id: String) -> Self {
        Self { encoder, model_id }
    }

    /// Encode a document image into patch embeddings.
    ///
    /// Returns multi-vector representation where each vector corresponds to
    /// an image patch (14×14 pixels). Typically produces 1024 patch embeddings
    /// for 448×448 images.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to document image (PNG, JPEG, etc.)
    ///
    /// # Returns
    ///
    /// `VisionEmbedding` containing patch embeddings (shape: [1024, 128]).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Image cannot be loaded
    /// - Image preprocessing fails
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let doc_emb = embedder.encode_document("invoice.jpg")?;
    /// println!("Patches: {}, Dim: {}", doc_emb.num_patches(), doc_emb.embedding_dim());
    /// ```
    pub fn encode_document(&self, image_path: &str) -> Result<VisionEmbedding> {
        let path = Path::new(image_path);
        self.encoder
            .encode_image(path)
            .map_err(|e| TesseraError::EncodingError {
                context: format!("Failed to encode document image: '{image_path}'"),
                source: e,
            })
    }

    /// Encode a text query into token embeddings.
    ///
    /// Returns multi-vector representation where each vector corresponds to
    /// a query token. Compatible with late interaction (`MaxSim`) scoring
    /// against document patch embeddings.
    ///
    /// # Arguments
    ///
    /// * `text` - Query text
    ///
    /// # Returns
    ///
    /// `TokenEmbeddings` containing query token embeddings.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tokenization fails
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query_emb = embedder.encode_query("What is the total amount?")?;
    /// println!("Query tokens: {}", query_emb.num_tokens);
    /// ```
    pub fn encode_query(&self, text: &str) -> Result<TokenEmbeddings> {
        self.encoder
            .encode_text(text)
            .map_err(|e| TesseraError::EncodingError {
                context: format!("Failed to encode query text: '{text}'"),
                source: e,
            })
    }

    /// Compute late interaction score between query and document.
    ///
    /// Uses `MaxSim` scoring: for each query token, find maximum similarity
    /// across all document patches, then sum across query tokens.
    ///
    /// # Arguments
    ///
    /// * `query` - Query token embeddings
    /// * `document` - Document patch embeddings
    ///
    /// # Returns
    ///
    /// Similarity score (higher = more similar).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query_emb = embedder.encode_query("total amount")?;
    /// let doc_emb = embedder.encode_document("invoice.jpg")?;
    /// let score = embedder.search(&query_emb, &doc_emb)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the embeddings cannot be processed or if dimensions are mismatched.
    pub fn search(&self, query: &TokenEmbeddings, document: &VisionEmbedding) -> Result<f32> {
        // Convert VisionEmbedding to format compatible with max_sim
        // max_sim expects (&TokenEmbeddings, &TokenEmbeddings) but we can adapt it

        // Create a TokenEmbeddings-like structure from VisionEmbedding
        // We need to convert Vec<Vec<f32>> to Array2<f32>
        let doc_array = ndarray::Array2::from_shape_vec(
            (document.num_patches, document.embedding_dim),
            document.embeddings.iter().flatten().copied().collect(),
        )
        .map_err(|e| TesseraError::EncodingError {
            context: "Failed to convert document embeddings to array".to_string(),
            source: e.into(),
        })?;

        let doc_embeddings = TokenEmbeddings {
            embeddings: doc_array,
            num_tokens: document.num_patches,
            embedding_dim: document.embedding_dim,
            text: document.source.clone().unwrap_or_default(),
        };

        max_sim(query, &doc_embeddings).map_err(|e| TesseraError::EncodingError {
            context: "Failed to compute MaxSim score".to_string(),
            source: e,
        })
    }

    /// Convenience method: search with text query and image path.
    ///
    /// Encodes both query and document, then computes similarity.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails or the image cannot be read.
    pub fn search_document(&self, query_text: &str, image_path: &str) -> Result<f32> {
        let query_emb = self.encode_query(query_text)?;
        let doc_emb = self.encode_document(image_path)?;
        self.search(&query_emb, &doc_emb)
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        use crate::core::VisionEncoder;
        self.encoder.embedding_dim()
    }

    /// Get the number of patches per image.
    pub fn num_patches(&self) -> usize {
        use crate::core::VisionEncoder;
        self.encoder.num_patches()
    }

    /// Get the model identifier.
    pub fn model(&self) -> &str {
        &self.model_id
    }
}

// ============================================================================
// Time Series Forecasting (Chronos Bolt)
// ============================================================================

/// Time series embedder for Chronos Bolt forecasting.
///
/// Provides probabilistic time series forecasting using Amazon's Chronos Bolt
/// T5-based foundation model. Produces quantile predictions for uncertainty
/// quantification and point forecasts (median).
///
/// Thread-safe and can be shared across threads.
pub struct TesseraTimeSeries {
    /// Backend encoder (`ChronosBolt` model)
    encoder: ChronosBolt,
    /// Model identifier from registry
    model_id: String,
}

impl TesseraTimeSeries {
    /// Create a new time series forecaster with default configuration.
    ///
    /// This is the simplest way to create a forecaster - it automatically:
    /// - Looks up the model in the registry
    /// - Selects the best available device (Metal > CUDA > CPU)
    /// - Downloads the model from `HuggingFace` if needed
    /// - Initializes the T5-based forecasting model
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry (e.g., "chronos-bolt-small")
    ///
    /// # Returns
    ///
    /// Initialized forecaster ready for time series predictions.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - Model is not a time series model type
    /// - Model cannot be downloaded or loaded
    /// - Device initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraTimeSeries;
    /// use candle_core::Tensor;
    ///
    /// let forecaster = TesseraTimeSeries::new("chronos-bolt-small")?;
    /// let data = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
    /// let forecast = forecaster.forecast(&data)?;
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        TesseraTimeSeriesBuilder::new().model(model_id).build()
    }

    /// Create a builder for advanced configuration.
    ///
    /// Use this for advanced use cases like:
    /// - Specifying a custom device
    /// - Setting custom context/prediction lengths
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraTimeSeries;
    /// use candle_core::Device;
    ///
    /// let forecaster = TesseraTimeSeries::builder()
    ///     .model("chronos-bolt-small")
    ///     .device(Device::Cpu)
    ///     .build()?;
    /// ```
    #[must_use] pub const fn builder() -> TesseraTimeSeriesBuilder {
        TesseraTimeSeriesBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) const fn from_encoder(encoder: ChronosBolt, model_id: String) -> Self {
        Self { encoder, model_id }
    }

    /// Generate point forecast (median prediction).
    ///
    /// Returns the median quantile (50th percentile) as a point forecast.
    /// For uncertainty quantification, use `forecast_quantiles()` instead.
    ///
    /// # Arguments
    ///
    /// * `context` - Historical time series data [batch, `context_length`]
    ///
    /// # Returns
    ///
    /// Tensor of forecasted values [batch, `prediction_length`]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Context tensor has wrong shape
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let data = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
    /// let forecast = forecaster.forecast(&data)?;
    /// println!("Forecast shape: {:?}", forecast.shape());  // [1, 64]
    /// ```
    pub fn forecast(&mut self, context: &Tensor) -> Result<Tensor> {
        self.encoder
            .forecast(context)
            .map_err(|e| TesseraError::EncodingError {
                context: "Failed to generate forecast".to_string(),
                source: e,
            })
    }

    /// Generate probabilistic forecast with all quantiles.
    ///
    /// Returns predictions for all 9 quantiles (0.1, 0.2, ..., 0.9),
    /// enabling uncertainty quantification and prediction intervals.
    ///
    /// # Arguments
    ///
    /// * `context` - Historical time series data [batch, `context_length`]
    ///
    /// # Returns
    ///
    /// Tensor of quantile predictions [batch, `prediction_length`, `num_quantiles`]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Context tensor has wrong shape
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let data = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
    /// let quantiles = forecaster.forecast_quantiles(&data)?;
    /// println!("Quantiles shape: {:?}", quantiles.shape());  // [1, 64, 9]
    /// ```
    pub fn forecast_quantiles(&mut self, context: &Tensor) -> Result<Tensor> {
        self.encoder
            .predict_quantiles(context)
            .map_err(|e| TesseraError::EncodingError {
                context: "Failed to generate quantile predictions".to_string(),
                source: e,
            })
    }

    /// Get the prediction horizon length.
    ///
    /// Returns the number of timesteps forecasted.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Prediction length: {}", forecaster.prediction_length());
    /// ```
    #[must_use] pub const fn prediction_length(&self) -> usize {
        self.encoder.config.prediction_length
    }

    /// Get the context length.
    ///
    /// Returns the required input sequence length.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Context length: {}", forecaster.context_length());
    /// ```
    #[must_use] pub const fn context_length(&self) -> usize {
        self.encoder.config.context_length
    }

    /// Get the quantile levels.
    ///
    /// Returns the quantiles predicted by the model (typically [0.1, 0.2, ..., 0.9]).
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Quantiles: {:?}", forecaster.quantiles());
    /// ```
    #[must_use] pub fn quantiles(&self) -> &[f32] {
        &self.encoder.config.quantiles
    }

    /// Get the model identifier.
    ///
    /// Returns the model ID from the registry (e.g., "chronos-bolt-small").
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Using model: {}", forecaster.model());
    /// ```
    #[must_use] pub fn model(&self) -> &str {
        &self.model_id
    }
}

// ============================================================================
// Unified Factory Enum
// ============================================================================

/// Unified embedder that auto-detects model type.
///
/// This enum provides a smart factory pattern that automatically creates
/// the appropriate embedder variant (Dense, `MultiVector`, or Sparse) based on
/// the model type in the registry.
///
/// # Example
///
/// ```ignore
/// use tessera::Tessera;
///
/// // Auto-detects ColBERT model -> creates MultiVector variant
/// let colbert = Tessera::new("colbert-v2")?;
///
/// // Auto-detects dense model -> creates Dense variant
/// let bge = Tessera::new("bge-base-en-v1.5")?;
///
/// // Auto-detects sparse model -> creates Sparse variant
/// let splade = Tessera::new("splade-cocondenser")?;
///
/// // Auto-detects vision-language model -> creates Vision variant
/// let colpali = Tessera::new("colpali-v1.3-hf")?;
///
/// // Auto-detects time series model -> creates TimeSeries variant
/// let chronos = Tessera::new("chronos-bolt-small")?;
///
/// // Pattern match to use specific API
/// match colbert {
///     Tessera::MultiVector(mv) => {
///         let embeddings = mv.encode("query")?;
///         println!("Got {} tokens", embeddings.num_tokens);
///     }
///     Tessera::Dense(d) => {
///         let embedding = d.encode("query")?;
///         println!("Got {} dimensions", embedding.dim());
///     }
///     Tessera::Sparse(s) => {
///         let embedding = s.encode("query")?;
///         println!("Got {} non-zero dimensions", embedding.nnz());
///     }
///     Tessera::Vision(v) => {
///         let doc_emb = v.encode_document("invoice.jpg")?;
///         println!("Got {} patches", doc_emb.num_patches);
///     }
///     Tessera::TimeSeries(ts) => {
///         let forecast = ts.forecast(&data)?;
///         println!("Forecast {} timesteps", forecast.dims()[1]);
///     }
/// }
/// ```
pub enum Tessera {
    /// Dense single-vector embedder
    Dense(TesseraDense),
    /// Multi-vector ColBERT-style embedder
    MultiVector(TesseraMultiVector),
    /// Sparse SPLADE-style embedder
    Sparse(TesseraSparse),
    /// Vision-language ColPali-style embedder
    Vision(TesseraVision),
    /// Time series forecasting embedder
    TimeSeries(TesseraTimeSeries),
}

impl Tessera {
    /// Create a new embedder with automatic model type detection.
    ///
    /// Looks up the model in the registry and creates the appropriate
    /// embedder variant based on the model type:
    /// - Dense models -> `Tessera::Dense(TesseraDense)`
    /// - MultiVector/Colbert models -> `Tessera::MultiVector(TesseraMultiVector)`
    /// - Sparse models -> `Tessera::Sparse(TesseraSparse)`
    /// - `VisionLanguage` models -> `Tessera::Vision(TesseraVision)`
    /// - Timeseries models -> `Tessera::TimeSeries(TesseraTimeSeries)`
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry
    ///
    /// # Returns
    ///
    /// Tessera enum variant containing the appropriate embedder.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - Model type is not supported (e.g., Unified)
    /// - Model cannot be loaded
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::Tessera;
    ///
    /// let embedder = Tessera::new("colbert-v2")?;
    /// let embedder = Tessera::new("bge-base-en-v1.5")?;
    /// let embedder = Tessera::new("splade-cocondenser")?;
    /// let embedder = Tessera::new("colpali-v1.3-hf")?;
    /// let embedder = Tessera::new("chronos-bolt-small")?;
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        let model_info = get_model(model_id).ok_or_else(|| TesseraError::ModelNotFound {
            model_id: model_id.to_string(),
        })?;

        match model_info.model_type {
            ModelType::Dense => {
                let dense = TesseraDense::new(model_id)?;
                Ok(Self::Dense(dense))
            }
            ModelType::Colbert => {
                let mv = TesseraMultiVector::new(model_id)?;
                Ok(Self::MultiVector(mv))
            }
            ModelType::Sparse => {
                let sparse = TesseraSparse::new(model_id)?;
                Ok(Self::Sparse(sparse))
            }
            ModelType::VisionLanguage => {
                let vision = TesseraVision::new(model_id)?;
                Ok(Self::Vision(vision))
            }
            ModelType::Timeseries => {
                let timeseries = TesseraTimeSeries::new(model_id)?;
                Ok(Self::TimeSeries(timeseries))
            }
            ModelType::Unified => Err(TesseraError::ConfigError(
                "Model type 'Unified' is not yet supported. Currently supported: Dense, Colbert (MultiVector), Sparse, VisionLanguage, Timeseries".to_string()
            )),
        }
    }
}
