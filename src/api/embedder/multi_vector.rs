use crate::api::TesseraMultiVectorBuilder;
use crate::backends::CandleBertEncoder;
use crate::core::{Encoder, TokenEmbedder, TokenEmbeddings};
use crate::error::{Result, TesseraError};
use crate::quantization::BinaryQuantization;
use crate::runtime::{
    f32_output_bytes, ContextWindowConfig, JobTracker, ModelDType, ResourcePolicy,
};
use crate::utils::similarity::max_sim;
use std::num::NonZeroUsize;

mod quantization;

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
    /// Whole-job and collected-output limits.
    resource_policy: ResourcePolicy,
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
    /// let embeddings = embedder.encode_query("What is machine learning?")?;
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
        resource_policy: ResourcePolicy,
    ) -> Self {
        Self {
            encoder,
            model_id,
            quantizer,
            batch_size,
            resource_policy,
        }
    }

    /// Encode a single text with generic BERT tokenization.
    ///
    /// This compatibility path does not assign a ColBERT query/document role.
    /// Use [`Self::encode_query`] and [`Self::encode_document`] for retrieval.
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
    ///     embeddings.num_tokens(),
    ///     embeddings.embedding_dim());
    /// ```
    pub fn encode(&self, text: &str) -> Result<TokenEmbeddings> {
        self.encode_one(text, "generic", TokenEmbedder::encode)
    }

    fn encode_one<F>(&self, text: &str, role: &str, encode: F) -> Result<TokenEmbeddings>
    where
        F: Fn(&CandleBertEncoder, &str) -> anyhow::Result<TokenEmbeddings>,
    {
        let mut tracker = JobTracker::new(self.resource_policy);
        tracker.admit_input(text.len()).map_err(|error| {
            resource_error(&format!("ColBERT {role} input exceeds job limits"), error)
        })?;
        let embedding = encode(&self.encoder, text).map_err(|e| TesseraError::EncodingError {
            context: format!("Failed to encode text ({} UTF-8 bytes)", text.len()),
            source: e,
        })?;
        tracker
            .retain_output(token_output_bytes(&embedding))
            .map_err(|error| resource_error("ColBERT output exceeds collection limit", error))?;
        Ok(embedding)
    }

    /// Encode a query with `[Q]` framing and fixed-length `[MASK]` augmentation.
    pub fn encode_query(&self, text: &str) -> Result<TokenEmbeddings> {
        self.encode_one(text, "query", CandleBertEncoder::encode_query)
    }

    /// Encode a document with `[D]` framing and punctuation filtering.
    pub fn encode_document(&self, text: &str) -> Result<TokenEmbeddings> {
        self.encode_one(text, "document", CandleBertEncoder::encode_document)
    }

    /// Encodes a long document through bounded overlapping ColBERT windows.
    ///
    /// This is an aggregation mode rather than native full-context attention.
    /// Overlap is context-only: each non-punctuation content token contributes
    /// to the returned late-interaction matrix once.
    pub fn encode_document_windowed(
        &self,
        text: &str,
        config: ContextWindowConfig,
    ) -> Result<TokenEmbeddings> {
        self.encode_one(text, "windowed document", |encoder, input| {
            encoder.encode_document_windowed(input, config)
        })
    }

    /// Encode multiple texts with generic, un-typed BERT tokenization.
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
        self.encode_role_batch(texts, "generic", Encoder::encode_batch)
    }

    /// Encode multiple queries with reference ColBERT query semantics.
    pub fn encode_query_batch(&self, texts: &[&str]) -> Result<Vec<TokenEmbeddings>> {
        self.encode_role_batch(texts, "query", CandleBertEncoder::encode_query_batch)
    }

    /// Encode multiple documents with reference ColBERT document semantics.
    pub fn encode_document_batch(&self, texts: &[&str]) -> Result<Vec<TokenEmbeddings>> {
        self.encode_role_batch(texts, "document", CandleBertEncoder::encode_document_batch)
    }

    /// Encodes generic inputs in bounded chunks and yields each result.
    ///
    /// Tessera does not retain streamed embeddings. Whole-job item and input
    /// limits still apply, while the output ceiling is checked per yielded
    /// item.
    pub fn encode_stream<'a, I, C>(&self, texts: I, consume: C) -> Result<()>
    where
        I: IntoIterator<Item = &'a str>,
        C: FnMut(TokenEmbeddings) -> Result<()>,
    {
        self.encode_role_stream(texts, "generic", Encoder::encode_batch, consume)
    }

    /// Encodes ColBERT queries in bounded chunks and yields each result.
    pub fn encode_query_stream<'a, I, C>(&self, texts: I, consume: C) -> Result<()>
    where
        I: IntoIterator<Item = &'a str>,
        C: FnMut(TokenEmbeddings) -> Result<()>,
    {
        self.encode_role_stream(
            texts,
            "query",
            CandleBertEncoder::encode_query_batch,
            consume,
        )
    }

    /// Encodes ColBERT documents in bounded chunks and yields each result.
    pub fn encode_document_stream<'a, I, C>(&self, texts: I, consume: C) -> Result<()>
    where
        I: IntoIterator<Item = &'a str>,
        C: FnMut(TokenEmbeddings) -> Result<()>,
    {
        self.encode_role_stream(
            texts,
            "document",
            CandleBertEncoder::encode_document_batch,
            consume,
        )
    }

    fn encode_role_stream<'a, I, F, C>(
        &self,
        texts: I,
        role: &str,
        encode: F,
        mut consume: C,
    ) -> Result<()>
    where
        I: IntoIterator<Item = &'a str>,
        F: Fn(&CandleBertEncoder, &[&str]) -> anyhow::Result<Vec<TokenEmbeddings>>,
        C: FnMut(TokenEmbeddings) -> Result<()>,
    {
        let mut tracker = JobTracker::new(self.resource_policy);
        let chunk_size = self.batch_size.unwrap_or(NonZeroUsize::MIN).get();
        let mut chunk = Vec::with_capacity(chunk_size);
        let mut chunk_index = 0_usize;

        for text in texts {
            tracker.admit_input(text.len()).map_err(|error| {
                resource_error(&format!("ColBERT {role} stream exceeds job limits"), error)
            })?;
            chunk.push(text);
            if chunk.len() == chunk_size {
                consume_stream_chunk(
                    &self.encoder,
                    &chunk,
                    chunk_index,
                    role,
                    &encode,
                    &tracker,
                    &mut consume,
                )?;
                chunk.clear();
                chunk_index = chunk_index.saturating_add(1);
            }
        }
        if !chunk.is_empty() {
            consume_stream_chunk(
                &self.encoder,
                &chunk,
                chunk_index,
                role,
                &encode,
                &tracker,
                &mut consume,
            )?;
        }
        Ok(())
    }

    fn encode_role_batch<F>(
        &self,
        texts: &[&str],
        role: &str,
        encode: F,
    ) -> Result<Vec<TokenEmbeddings>>
    where
        F: Fn(&CandleBertEncoder, &[&str]) -> anyhow::Result<Vec<TokenEmbeddings>>,
    {
        let mut tracker = JobTracker::new(self.resource_policy);
        for text in texts {
            tracker.admit_input(text.len()).map_err(|error| {
                resource_error(&format!("ColBERT {role} batch exceeds job limits"), error)
            })?;
        }
        let chunk_size = self
            .batch_size
            .map_or_else(|| texts.len().max(1), NonZeroUsize::get);
        let mut all_embeddings = Vec::with_capacity(texts.len());
        for (chunk_index, chunk) in texts.chunks(chunk_size).enumerate() {
            let embeddings =
                encode(&self.encoder, chunk).map_err(|source| TesseraError::EncodingError {
                    context: format!(
                        "Failed to encode {role} batch chunk {chunk_index} ({} texts)",
                        chunk.len()
                    ),
                    source,
                })?;
            for embedding in &embeddings {
                tracker
                    .retain_output(token_output_bytes(embedding))
                    .map_err(|error| {
                        resource_error("ColBERT batch output exceeds collection limit", error)
                    })?;
            }
            all_embeddings.extend(embeddings);
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
    /// * `query` - Retrieval query
    /// * `document` - Candidate document
    ///
    /// # Returns
    ///
    /// Similarity score (higher = more similar). `MaxSim` is a sum over query
    /// vectors, so it is not bounded to the cosine-similarity range.
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
    pub fn similarity(&self, query: &str, document: &str) -> Result<f32> {
        let query = self.encode_query(query)?;
        let document = self.encode_document(document)?;
        self.search(&query, &document)
    }

    /// Score role-specific query and document embeddings with ColBERT `MaxSim`.
    pub fn search(&self, query: &TokenEmbeddings, document: &TokenEmbeddings) -> Result<f32> {
        max_sim(query, document).map_err(|e| TesseraError::EncodingError {
            context: "Failed to compute similarity".to_string(),
            source: e,
        })
    }

    /// Encode one query and a document batch with their respective roles, then
    /// return one `MaxSim` score per document in input order.
    pub fn search_documents(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let query = self.encode_query(query)?;
        self.encode_document_batch(documents)?
            .iter()
            .map(|document| self.search(&query, document))
            .collect()
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

    /// Parameter dtype selected for this model instance.
    #[must_use]
    pub const fn model_dtype(&self) -> ModelDType {
        self.encoder.model_dtype()
    }
}

fn token_output_bytes(embedding: &TokenEmbeddings) -> usize {
    f32_output_bytes(
        embedding
            .num_tokens()
            .saturating_mul(embedding.embedding_dim()),
    )
}

fn resource_error(context: &str, error: crate::runtime::ResourcePolicyError) -> TesseraError {
    TesseraError::EncodingError {
        context: context.to_string(),
        source: anyhow::Error::new(error),
    }
}

fn consume_stream_chunk<F, C>(
    encoder: &CandleBertEncoder,
    chunk: &[&str],
    chunk_index: usize,
    role: &str,
    encode: &F,
    tracker: &JobTracker,
    consume: &mut C,
) -> Result<()>
where
    F: Fn(&CandleBertEncoder, &[&str]) -> anyhow::Result<Vec<TokenEmbeddings>>,
    C: FnMut(TokenEmbeddings) -> Result<()>,
{
    let embeddings = encode(encoder, chunk).map_err(|source| TesseraError::EncodingError {
        context: format!("Failed to encode {role} stream chunk {chunk_index}"),
        source,
    })?;
    for embedding in embeddings {
        tracker
            .validate_streamed_output(token_output_bytes(&embedding))
            .map_err(|error| {
                resource_error("ColBERT streamed output exceeds per-item limit", error)
            })?;
        consume(embedding)?;
    }
    Ok(())
}
