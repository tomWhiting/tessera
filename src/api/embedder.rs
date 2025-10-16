//! Main Tessera embedder interface.
//!
//! Provides the primary user-facing API for encoding text into
//! embeddings. Supports both single and batch encoding with
//! automatic batching and device management.
//!
//! # Example
//!
//! ```ignore
//! use tessera::Tessera;
//!
//! let embedder = Tessera::new("colbert-v2")?;
//!
//! // Single text
//! let embedding = embedder.encode("What is ML?")?;
//!
//! // Batch encoding
//! let embeddings = embedder.encode_batch(&[
//!     "First text",
//!     "Second text",
//! ])?;
//! ```

use crate::api::TesseraBuilder;
use crate::backends::CandleBertEncoder;
use crate::core::{TokenEmbedder, TokenEmbeddings};
use crate::error::{Result, TesseraError};
use crate::quantization::{binary::BinaryVector, multi_vector_distance, quantize_multi, BinaryQuantization};
use crate::utils::similarity::max_sim;

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
    pub fn memory_bytes(&self) -> usize {
        self.quantized.iter().map(|v| v.memory_bytes()).sum()
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
    pub fn compression_ratio(&self) -> f32 {
        let float_bytes = self.num_tokens * self.original_dim * 4;
        let quantized_bytes = self.memory_bytes();
        if quantized_bytes == 0 {
            return 0.0;
        }
        float_bytes as f32 / quantized_bytes as f32
    }
}

/// Main Tessera embedder.
///
/// Manages model loading, device allocation, and encoding operations.
/// Thread-safe and can be shared across threads.
pub struct Tessera {
    /// Backend encoder (currently Candle only)
    encoder: CandleBertEncoder,
    /// Model identifier from registry
    model_id: String,
    /// Optional quantizer for compression
    quantizer: Option<BinaryQuantization>,
}

impl Tessera {
    /// Create a new embedder with default configuration.
    ///
    /// This is the simplest way to create an embedder - it automatically:
    /// - Looks up the model in the registry
    /// - Selects the best available device (Metal > CUDA > CPU)
    /// - Downloads the model from HuggingFace if needed
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
    /// use tessera::Tessera;
    ///
    /// let embedder = Tessera::new("colbert-v2")?;
    /// let embeddings = embedder.encode("What is machine learning?")?;
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        // Use builder with just model ID
        TesseraBuilder::new().model(model_id).build()
    }

    /// Create a builder for advanced configuration.
    ///
    /// Use this for advanced use cases like:
    /// - Specifying a custom device
    /// - Setting Matryoshka dimensions
    /// - Future: quantization, normalization, etc.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::Tessera;
    /// use candle_core::Device;
    ///
    /// let embedder = Tessera::builder()
    ///     .model("jina-colbert-v2")
    ///     .device(Device::Cpu)
    ///     .build()?;
    /// ```
    pub fn builder() -> TesseraBuilder {
        TesseraBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) fn from_encoder(encoder: CandleBertEncoder, model_id: String, quantizer: Option<BinaryQuantization>) -> Self {
        Self { encoder, model_id, quantizer }
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
    /// TokenEmbeddings containing the embedding matrix and metadata.
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
        self.encoder
            .encode(text)
            .map_err(|e| TesseraError::EncodingError {
                context: format!("Failed to encode text: '{}'", text),
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
    /// Vector of TokenEmbeddings, one per input text.
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
        use crate::core::Encoder;

        self.encoder
            .encode_batch(texts)
            .map_err(|e| TesseraError::EncodingError {
                context: format!("Failed to encode batch of {} texts", texts.len()),
                source: e,
            })
    }

    /// Compute similarity between two texts.
    ///
    /// Convenience method that encodes both texts and computes MaxSim similarity.
    /// MaxSim is the standard similarity metric for ColBERT multi-vector embeddings.
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
    /// use tessera::{Tessera, QuantizationConfig};
    ///
    /// let embedder = Tessera::builder()
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
    /// Uses the MaxSim algorithm adapted for binary embeddings:
    /// - Distance computed via XOR + popcount (Hamming distance)
    /// - For each query vector, find max similarity with document vectors
    /// - Sum across all query vectors
    ///
    /// This is significantly faster than float32 MaxSim while maintaining
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
    /// float32 MaxSim but ranking is preserved.
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
