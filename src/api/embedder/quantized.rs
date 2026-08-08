use crate::error::{Result, TesseraError};
use crate::quantization::binary::BinaryVector;

/// Binary quantized multi-vector embeddings.
///
/// Represents token embeddings packed to one sign bit per dimension. Retrieval
/// quality must be measured for the selected model and dataset.
#[derive(Debug, Clone)]
pub struct QuantizedEmbeddings {
    /// Quantized token vectors
    quantized: Vec<BinaryVector>,
    /// Original embedding dimension (before quantization)
    original_dim: usize,
    /// Number of token vectors
    num_tokens: usize,
}

impl QuantizedEmbeddings {
    /// Construct a validated collection of binary token vectors.
    ///
    /// The token count and original dimension are derived from the vectors so
    /// callers cannot provide inconsistent metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is empty or its vector dimensions
    /// are inconsistent.
    pub fn new(quantized: Vec<BinaryVector>) -> Result<Self> {
        let Some(first) = quantized.first() else {
            return Err(TesseraError::QuantizationError(
                "Quantized embeddings must contain at least one token vector".to_string(),
            ));
        };
        let original_dim = first.dim();

        if let Some((index, vector)) = quantized
            .iter()
            .enumerate()
            .find(|(_, vector)| vector.dim() != original_dim)
        {
            return Err(TesseraError::QuantizationError(format!(
                "Quantized token vector {index} has dimension {}, expected {original_dim}",
                vector.dim()
            )));
        }

        let num_tokens = quantized.len();
        Ok(Self {
            quantized,
            original_dim,
            num_tokens,
        })
    }

    /// Return the packed token vectors.
    #[must_use]
    pub fn vectors(&self) -> &[BinaryVector] {
        &self.quantized
    }

    /// Return the original floating-point embedding dimension.
    #[must_use]
    pub const fn original_dim(&self) -> usize {
        self.original_dim
    }

    /// Return the number of token vectors.
    #[must_use]
    pub const fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Memory usage in bytes.
    ///
    /// Returns the packed payload size, excluding collection and metadata
    /// overhead.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.quantized.iter().fold(0, |total, vector| {
            total.saturating_add(vector.memory_bytes())
        })
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
        let float_bytes = self
            .num_tokens
            .saturating_mul(self.original_dim)
            .saturating_mul(std::mem::size_of::<f32>());
        let quantized_bytes = self.memory_bytes();
        if quantized_bytes == 0 {
            return 0.0;
        }
        float_bytes as f32 / quantized_bytes as f32
    }
}

#[cfg(test)]
#[path = "quantized/tests.rs"]
mod tests;
