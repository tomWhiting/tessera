use crate::api::TesseraSparseBuilder;
use crate::core::{Encoder, SparseEmbedding};
use crate::encoding::sparse::CandleSparseEncoder;
use crate::error::{Result, TesseraError};
use crate::runtime::{ContextWindowConfig, JobTracker, ModelDType, ResourcePolicy};

/// Sparse embedder for SPLADE-style vocabulary-sized embeddings.
///
/// Produces weighted vocabulary entries suitable for inverted-index
/// integration. Actual sparsity depends on the model, input, and filtering
/// threshold.
///
/// Thread-safe and can be shared across threads.
pub struct TesseraSparse {
    /// Backend encoder (Candle sparse encoder)
    encoder: CandleSparseEncoder,
    /// Model identifier from registry
    model_id: String,
    /// Whole-job and collected-output limits.
    resource_policy: ResourcePolicy,
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
    /// let embedder = TesseraSparse::new("splade-pp-en-v1")?;
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
    ///     .model("splade-pp-en-v1")
    ///     .device(Device::Cpu)
    ///     .build()?;
    /// ```
    #[must_use]
    pub const fn builder() -> TesseraSparseBuilder {
        TesseraSparseBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) const fn from_encoder(
        encoder: CandleSparseEncoder,
        model_id: String,
        resource_policy: ResourcePolicy,
    ) -> Self {
        Self {
            encoder,
            model_id,
            resource_policy,
        }
    }

    /// Encode a single text into a sparse embedding.
    ///
    /// Returns a sparse vector over the model's registered vocabulary.
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
        let mut tracker = JobTracker::new(self.resource_policy);
        tracker
            .admit_input(text.len())
            .map_err(|error| resource_error("Sparse input exceeds job limits", error))?;
        let embedding =
            <CandleSparseEncoder as Encoder>::encode(&self.encoder, text).map_err(|e| {
                TesseraError::EncodingError {
                    context: format!("Failed to encode text ({} UTF-8 bytes)", text.len()),
                    source: e,
                }
            })?;
        tracker
            .retain_output(sparse_output_bytes(embedding.nnz()))
            .map_err(|error| resource_error("Sparse output exceeds collection limit", error))?;
        Ok(embedding)
    }

    /// Encodes a long input in bounded windows and max-merges sparse weights.
    pub fn encode_windowed(
        &self,
        text: &str,
        config: ContextWindowConfig,
    ) -> Result<SparseEmbedding> {
        let mut tracker = JobTracker::new(self.resource_policy);
        tracker
            .admit_input(text.len())
            .map_err(|error| resource_error("Sparse windowed input exceeds job limits", error))?;
        let embedding = self
            .encoder
            .encode_windowed(text, config)
            .map_err(|source| TesseraError::EncodingError {
                context: format!(
                    "Failed to encode windowed sparse text ({} UTF-8 bytes)",
                    text.len()
                ),
                source,
            })?;
        tracker
            .retain_output(sparse_output_bytes(embedding.nnz()))
            .map_err(|error| resource_error("Sparse output exceeds collection limit", error))?;
        Ok(embedding)
    }

    /// Encode multiple texts in a batch.
    ///
    /// Processes inputs sequentially so each result is admitted against the
    /// cumulative output budget before Tessera retains it. Use
    /// [`Self::encode_stream`] when the complete result set need not be held in
    /// memory.
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
        collect_sparse_batch(texts, self.resource_policy, |item_index, text| {
            <CandleSparseEncoder as Encoder>::encode(&self.encoder, text).map_err(|source| {
                TesseraError::EncodingError {
                    context: format!(
                        "Failed to encode sparse batch item {item_index} ({} UTF-8 bytes)",
                        text.len()
                    ),
                    source,
                }
            })
        })
    }

    /// Encodes inputs sequentially and yields each sparse result without collection.
    pub fn encode_stream<'a, I, F>(&self, texts: I, mut consume: F) -> Result<()>
    where
        I: IntoIterator<Item = &'a str>,
        F: FnMut(SparseEmbedding) -> Result<()>,
    {
        let mut tracker = JobTracker::new(self.resource_policy);
        for text in texts {
            tracker
                .admit_input(text.len())
                .map_err(|error| resource_error("Sparse stream input exceeds job limits", error))?;
            let embedding = <CandleSparseEncoder as Encoder>::encode(&self.encoder, text).map_err(
                |source| TesseraError::EncodingError {
                    context: format!(
                        "Failed to encode sparse stream item ({} UTF-8 bytes)",
                        text.len()
                    ),
                    source,
                },
            )?;
            tracker
                .validate_streamed_output(sparse_output_bytes(embedding.nnz()))
                .map_err(|error| {
                    resource_error("Sparse streamed output exceeds per-item limit", error)
                })?;
            consume(embedding)?;
        }
        Ok(())
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

        Ok(sorted_sparse_dot(emb_a.entries(), emb_b.entries()))
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
    /// Returns the model ID from the registry (e.g., "splade-pp-en-v1").
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

const fn sparse_output_bytes(entries: usize) -> usize {
    entries.saturating_mul(std::mem::size_of::<(usize, f32)>())
}

fn resource_error(context: &str, error: crate::runtime::ResourcePolicyError) -> TesseraError {
    TesseraError::EncodingError {
        context: context.to_string(),
        source: anyhow::Error::new(error),
    }
}

fn collect_sparse_batch<F>(
    texts: &[&str],
    resource_policy: ResourcePolicy,
    mut encode: F,
) -> Result<Vec<SparseEmbedding>>
where
    F: FnMut(usize, &str) -> Result<SparseEmbedding>,
{
    let mut tracker = JobTracker::new(resource_policy);
    for text in texts {
        tracker
            .admit_input(text.len())
            .map_err(|error| resource_error("Sparse batch input exceeds job limits", error))?;
    }

    let mut embeddings = Vec::with_capacity(texts.len());
    for (item_index, text) in texts.iter().copied().enumerate() {
        let embedding = encode(item_index, text)?;
        tracker
            .retain_output(sparse_output_bytes(embedding.nnz()))
            .map_err(|error| {
                resource_error("Sparse batch output exceeds collection limit", error)
            })?;
        embeddings.push(embedding);
    }
    Ok(embeddings)
}

fn sorted_sparse_dot(left: &[(usize, f32)], right: &[(usize, f32)]) -> f32 {
    let (mut left_index, mut right_index, mut score) = (0, 0, 0.0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].0.cmp(&right[right_index].0) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                score = left[left_index].1.mul_add(right[right_index].1, score);
                left_index += 1;
                right_index += 1;
            }
        }
    }
    score
}

#[cfg(test)]
#[path = "sparse/tests.rs"]
mod tests;
