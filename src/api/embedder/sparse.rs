use crate::api::TesseraSparseBuilder;
use crate::core::{Encoder, SparseEmbedding};
use crate::encoding::sparse::CandleSparseEncoder;
use crate::error::{Result, TesseraError};

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
    pub(crate) const fn from_encoder(encoder: CandleSparseEncoder, model_id: String) -> Self {
        Self { encoder, model_id }
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
        <CandleSparseEncoder as Encoder>::encode(&self.encoder, text).map_err(|e| {
            TesseraError::EncodingError {
                context: format!("Failed to encode text ({} UTF-8 bytes)", text.len()),
                source: e,
            }
        })
    }

    /// Encode multiple texts in a batch.
    ///
    /// Uses the encoder's batch path. Measure throughput on the target device;
    /// the default trait implementation may process inputs sequentially.
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

        Ok(sorted_sparse_dot(&emb_a.weights, &emb_b.weights))
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
mod tests {
    use super::sorted_sparse_dot;

    #[test]
    fn sparse_dot_merges_sorted_indices() {
        let left = [(1, 2.0), (4, 3.0), (9, -1.0)];
        let right = [(0, 8.0), (4, 5.0), (7, 2.0), (9, 4.0)];

        assert!((sorted_sparse_dot(&left, &right) - 11.0).abs() < f32::EPSILON);
        assert!(sorted_sparse_dot(&[], &right).abs() < f32::EPSILON);
    }
}
