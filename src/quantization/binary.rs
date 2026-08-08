//! Binary (1-bit) sign quantization.
//!
//! Implements binary quantization where each dimension is represented
//! as a single bit (sign of the original value):
//!
//! - **Encoding**: `sign(x) → {0, 1}` where 1 = positive, 0 = zero/negative
//! - **Storage**: Pack bits into u8 bytes (8 dimensions per byte)
//! - **Distance**: Hamming-based similarity via XOR + popcount
//!
//! # Compression
//!
//! - Original: 4 bytes per dimension (float32)
//! - Binary payload: 1 bit per dimension, rounded up to a whole byte
//! - 768-dim embedding: 3072 bytes → 96 bytes
//!
//! Distance uses one XOR and popcount per packed byte. Retrieval quality is
//! model- and dataset-dependent and must be measured for the target workload.
//!
//! # Example
//!
//! ```ignore
//! use tessera::quantization::BinaryQuantization;
//!
//! let quantizer = BinaryQuantization::new();
//! let vector = vec![0.5, -0.3, 0.8, -0.1];
//! let binary = quantizer.quantize_vector(&vector)?;
//! let restored = quantizer.dequantize_vector(&binary)?;
//! ```

use super::Quantization;
use crate::error::{Result, TesseraError};

/// Binary quantized vector representation.
///
/// Stores a vector as packed bits with 8 dimensions per byte.
/// Bit ordering: within each byte, bit i represents dimension (`byte_idx` * 8 + i).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryVector {
    /// Packed bits (8 dimensions per byte)
    data: Vec<u8>,
    /// Original dimension (before packing)
    dim: usize,
}

impl BinaryVector {
    /// Construct a binary vector from an existing packed representation.
    ///
    /// # Errors
    ///
    /// Returns an error when the dimension is zero, the packed length does
    /// not exactly match the dimension, or unused padding bits are set.
    pub fn from_packed(data: Vec<u8>, dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(quantization_error(
                "Binary vector dimension must be greater than zero",
            ));
        }

        let expected_bytes = dim.div_ceil(8);
        if data.len() != expected_bytes {
            return Err(quantization_error(format!(
                "Binary vector with dimension {dim} requires {expected_bytes} bytes, got {}",
                data.len()
            )));
        }

        let used_bits_in_last_byte = dim % 8;
        if used_bits_in_last_byte != 0 {
            let valid_mask = (1_u8 << used_bits_in_last_byte) - 1;
            if data.last().is_some_and(|byte| byte & !valid_mask != 0) {
                return Err(quantization_error(
                    "Binary vector has non-zero bits outside its declared dimension",
                ));
            }
        }

        Ok(Self { data, dim })
    }

    /// Return the packed bit payload.
    #[must_use]
    pub fn packed(&self) -> &[u8] {
        &self.data
    }

    /// Return the original vector dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Calculate memory usage in bytes.
    ///
    /// Returns the size of the packed bit data only, excluding Rust struct
    /// overhead and metadata.
    ///
    /// # Returns
    ///
    /// Bytes consumed by the packed bit data.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        // Only count the actual packed bit data for fair compression comparison
        // Struct overhead (Vec header + dim field) is constant and amortized
        self.data.len()
    }
}

/// Binary quantization implementation.
///
/// Converts float32 vectors to binary representation by thresholding at 0.0.
/// Positive values become 1, negative/zero values become 0.
pub struct BinaryQuantization;

impl BinaryQuantization {
    /// Create a new binary quantization instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for BinaryQuantization {
    fn default() -> Self {
        Self::new()
    }
}

impl Quantization for BinaryQuantization {
    type Output = BinaryVector;

    fn quantize_vector(&self, vector: &[f32]) -> Result<BinaryVector> {
        if vector.is_empty() {
            return Err(quantization_error(
                "Cannot quantize a vector with zero dimensions",
            ));
        }
        if let Some((index, value)) = vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(quantization_error(format!(
                "Cannot quantize non-finite value {value} at dimension {index}"
            )));
        }

        let dim = vector.len();
        let num_bytes = dim.div_ceil(8); // Round up to nearest byte
        let mut data = vec![0u8; num_bytes];

        for (i, &val) in vector.iter().enumerate() {
            if val > 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                data[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(BinaryVector { data, dim })
    }

    fn dequantize_vector(&self, quantized: &BinaryVector) -> Result<Vec<f32>> {
        let mut result = vec![0.0; quantized.dim];

        for i in 0..quantized.dim {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (quantized.data[byte_idx] >> bit_idx) & 1;
            result[i] = if bit == 1 { 1.0 } else { -1.0 };
        }

        Ok(result)
    }

    #[allow(clippy::cast_precision_loss)]
    fn distance(&self, a: &BinaryVector, b: &BinaryVector) -> Result<f32> {
        if a.dim != b.dim {
            return Err(TesseraError::DimensionMismatch {
                expected: a.dim,
                actual: b.dim,
            });
        }

        let hamming = a
            .data
            .iter()
            .zip(&b.data)
            .try_fold(0_u64, |total, (left, right)| {
                total
                    .checked_add(u64::from((left ^ right).count_ones()))
                    .ok_or_else(|| quantization_error("Binary Hamming distance overflowed"))
            })?;

        // Convert Hamming distance to similarity (lower distance = higher similarity)
        // For MaxSim, we want higher values for similar vectors
        // Similarity = dimension - hamming_distance
        Ok(a.dim as f32 - hamming as f32)
    }
}

#[cfg(test)]
#[path = "binary/tests.rs"]
mod tests;

fn quantization_error(message: impl Into<String>) -> TesseraError {
    TesseraError::QuantizationError(message.into())
}
