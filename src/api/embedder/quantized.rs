use crate::quantization::binary::BinaryVector;

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
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.quantized.iter().map(BinaryVector::memory_bytes).sum()
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
