use crate::api::TesseraDenseBuilder;
use crate::core::{DenseEmbedding, DenseEncoder, Encoder};
use crate::encoding::dense::CandleDenseEncoder;
use crate::error::{Result, TesseraError};
use std::num::NonZeroUsize;

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
    /// Maximum batch size for encode_batch (None = no limit)
    batch_size: Option<NonZeroUsize>,
    /// Milliseconds to sleep between batches
    yield_ms: Option<u64>,
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
    /// * `model_id` - Model identifier from the registry (e.g., "bge-base-en-v1.5", "nomic-embed-v1.5")
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
    #[must_use]
    pub const fn builder() -> TesseraDenseBuilder {
        TesseraDenseBuilder::new()
    }

    /// Internal constructor used by builder (legacy, no batch options).
    #[allow(dead_code)]
    pub(crate) const fn from_encoder(encoder: CandleDenseEncoder, model_id: String) -> Self {
        Self {
            encoder,
            model_id,
            batch_size: None,
            yield_ms: None,
        }
    }

    /// Internal constructor with batch options.
    pub(crate) const fn from_encoder_with_options(
        encoder: CandleDenseEncoder,
        model_id: String,
        batch_size: Option<NonZeroUsize>,
        yield_ms: Option<u64>,
    ) -> Self {
        Self {
            encoder,
            model_id,
            batch_size,
            yield_ms,
        }
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
                context: format!("Failed to encode text ({} UTF-8 bytes)", text.len()),
                source: e,
            }
        })
    }

    /// Encode multiple texts in a batch.
    ///
    /// Uses backend batching rather than one facade call per item. Throughput
    /// depends on input lengths, device, chunk size, and resource policy; no
    /// fixed speedup is guaranteed.
    ///
    /// If `batch_size` was configured via the builder, texts are processed in
    /// chunks with optional yielding between batches to prevent GPU saturation.
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
        // If no batch_size configured, process all at once (original behavior)
        let Some(batch_size) = self.batch_size else {
            return <CandleDenseEncoder as Encoder>::encode_batch(&self.encoder, texts).map_err(
                |e| TesseraError::EncodingError {
                    context: format!("Failed to encode batch of {} texts", texts.len()),
                    source: e,
                },
            );
        };

        // Process in chunks with optional yielding
        let mut all_embeddings = Vec::with_capacity(texts.len());
        let yield_duration = self.yield_ms.map(std::time::Duration::from_millis);

        for (chunk_idx, chunk) in texts.chunks(batch_size.get()).enumerate() {
            // Yield between batches (not before the first one)
            if chunk_idx > 0 {
                if let Some(duration) = yield_duration {
                    std::thread::sleep(duration);
                }
            }

            // Process this chunk
            let chunk_embeddings =
                <CandleDenseEncoder as Encoder>::encode_batch(&self.encoder, chunk).map_err(
                    |e| TesseraError::EncodingError {
                        context: format!(
                            "Failed to encode batch chunk {} ({} texts)",
                            chunk_idx,
                            chunk.len()
                        ),
                        source: e,
                    },
                )?;

            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
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
