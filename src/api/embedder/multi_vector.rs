use super::quantized::QuantizedEmbeddings;
use crate::api::TesseraMultiVectorBuilder;
use crate::backends::CandleBertEncoder;
use crate::core::{Encoder, TokenEmbedder, TokenEmbeddings};
use crate::error::{Result, TesseraError};
use crate::quantization::{multi_vector_distance, quantize_multi, BinaryQuantization};
use crate::utils::similarity::max_sim;
use std::num::NonZeroUsize;

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
    /// Maximum items sent through one encoder forward pass
    batch_size: Option<NonZeroUsize>,
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
    /// * `model_id` - Model identifier from the registry (e.g., "colbert-v2", "colbert-small")
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
    /// - Enabling binary quantization
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraMultiVector;
    /// use candle_core::Device;
    ///
    /// let embedder = TesseraMultiVector::builder()
    ///     .model("colbert-v2")
    ///     .device(Device::Cpu)
    ///     .build()?;
    /// ```
    #[must_use]
    pub const fn builder() -> TesseraMultiVectorBuilder {
        TesseraMultiVectorBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) const fn from_encoder(
        encoder: CandleBertEncoder,
        model_id: String,
        quantizer: Option<BinaryQuantization>,
        batch_size: Option<NonZeroUsize>,
    ) -> Self {
        Self {
            encoder,
            model_id,
            quantizer,
            batch_size,
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
            context: format!("Failed to encode text ({} UTF-8 bytes)", text.len()),
            source: e,
        })
    }

    /// Encode multiple texts in a batch.
    ///
    /// Uses backend batching rather than one facade call per item. Throughput
    /// depends on input lengths, device, chunk size, and resource policy; no
    /// fixed speedup is guaranteed.
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
        let Some(batch_size) = self.batch_size else {
            return Encoder::encode_batch(&self.encoder, texts).map_err(|e| {
                TesseraError::EncodingError {
                    context: format!("Failed to encode batch of {} texts", texts.len()),
                    source: e,
                }
            });
        };

        let mut all_embeddings = Vec::with_capacity(texts.len());
        for (chunk_index, chunk) in texts.chunks(batch_size.get()).enumerate() {
            let chunk_embeddings = Encoder::encode_batch(&self.encoder, chunk).map_err(|e| {
                TesseraError::EncodingError {
                    context: format!(
                        "Failed to encode batch chunk {chunk_index} ({} texts)",
                        chunk.len()
                    ),
                    source: e,
                }
            })?;
            all_embeddings.extend(chunk_embeddings);
        }
        Ok(all_embeddings)
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

    /// Quantize embeddings to a one-bit representation.
    ///
    /// Converts float32 dimensions to sign bits. The packed payload is 32 times
    /// smaller than the corresponding float32 values before metadata and
    /// padding; retrieval quality must be measured for the target corpus.
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
    /// The implementation uses XOR and popcount instead of float32 dot
    /// products. Benchmark speed and ranking quality for the target workload.
    ///
    /// # Arguments
    ///
    /// * `query` - Quantized query embeddings
    /// * `document` - Quantized document embeddings
    ///
    /// # Returns
    ///
    /// Similarity score (higher = more similar). Scale is different from
    /// float32 `MaxSim`; rankings are not guaranteed to be identical.
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
