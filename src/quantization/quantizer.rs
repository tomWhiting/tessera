use crate::error::Result;

/// Per-vector quantization interface.
///
/// Implementations must return higher distance values for more similar
/// vectors and reject inputs that cannot be represented safely.
pub trait Quantization {
    /// Quantized representation type.
    type Output;

    /// Quantize a single vector.
    ///
    /// # Errors
    ///
    /// Returns an error when the input cannot be represented safely.
    fn quantize_vector(&self, vector: &[f32]) -> Result<Self::Output>;

    /// Dequantize a vector to float32 values.
    ///
    /// # Errors
    ///
    /// Returns an error when the quantized representation is invalid.
    fn dequantize_vector(&self, quantized: &Self::Output) -> Result<Vec<f32>>;

    /// Compute similarity between two quantized vectors.
    ///
    /// # Errors
    ///
    /// Returns an error when the representations are incompatible.
    fn distance(&self, a: &Self::Output, b: &Self::Output) -> Result<f32>;
}
