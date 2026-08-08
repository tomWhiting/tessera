/// Quantization configuration for embeddings.
///
/// Enables compression of embeddings for reduced memory footprint and
/// faster distance computation with minimal accuracy loss.
#[derive(Debug, Clone, Copy)]
pub enum QuantizationConfig {
    /// No quantization (full precision float32)
    None,
    /// Binary quantization (1-bit, 32x compression, 95%+ accuracy)
    ///
    /// Converts each dimension to a single bit (sign of the value).
    /// Provides maximum compression with acceptable accuracy for most
    /// retrieval tasks. Ideal for initial filtering + reranking workflows.
    Binary,
    /// Int8 quantization (8-bit, 4x compression) - Phase 2
    #[allow(dead_code)]
    Int8,
    /// Int4 quantization (4-bit, 8x compression) - Phase 2
    #[allow(dead_code)]
    Int4,
}
