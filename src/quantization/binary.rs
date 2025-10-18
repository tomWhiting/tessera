//! Binary (1-bit) quantization for maximum compression.
//!
//! Implements binary quantization where each dimension is represented
//! as a single bit (sign of the original value):
//!
//! - **Encoding**: `sign(x) → {0, 1}` where 1 = positive, 0 = negative
//! - **Storage**: Pack bits into u8 bytes (8 dimensions per byte)
//! - **Distance**: Hamming-based similarity via XOR + popcount
//!
//! # Compression
//!
//! - Original: 4 bytes per dimension (float32)
//! - Binary: 1 bit per dimension (32x compression)
//! - 768-dim embedding: 3072 bytes → 96 bytes
//!
//! # Performance
//!
//! - Distance computation: O(n) XOR operations + popcount
//! - Cache-friendly: 32x more embeddings fit in memory
//! - Fast bit manipulation operations
//!
//! # Accuracy
//!
//! Despite aggressive compression, binary embeddings maintain:
//! - 95-97% of original ranking accuracy
//! - Preserved relative ordering for most queries
//! - Suitable for initial filtering + reranking workflows
//!
//! # Example
//!
//! ```ignore
//! use tessera::quantization::BinaryQuantization;
//!
//! let quantizer = BinaryQuantization::new();
//! let vector = vec![0.5, -0.3, 0.8, -0.1];
//! let binary = quantizer.quantize_vector(&vector);
//! let restored = quantizer.dequantize_vector(&binary);
//! ```

use super::Quantization;

/// Binary quantized vector representation.
///
/// Stores a vector as packed bits with 8 dimensions per byte.
/// Bit ordering: within each byte, bit i represents dimension (`byte_idx` * 8 + i).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryVector {
    /// Packed bits (8 dimensions per byte)
    pub data: Vec<u8>,
    /// Original dimension (before packing)
    pub dim: usize,
}

impl BinaryVector {
    /// Calculate memory usage in bytes.
    ///
    /// Returns the size of the packed bit data only, excluding Rust struct overhead.
    /// This provides an accurate measure of the data compression ratio.
    ///
    /// # Returns
    ///
    /// Bytes consumed by the packed bit data.
    #[must_use] pub fn memory_bytes(&self) -> usize {
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
    #[must_use] pub const fn new() -> Self {
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

    fn quantize_vector(&self, vector: &[f32]) -> BinaryVector {
        let dim = vector.len();
        let num_bytes = dim.div_ceil(8); // Round up to nearest byte
        let mut data = vec![0u8; num_bytes];

        for (i, &val) in vector.iter().enumerate() {
            if val >= 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                data[byte_idx] |= 1 << bit_idx;
            }
        }

        BinaryVector { data, dim }
    }

    fn dequantize_vector(&self, quantized: &BinaryVector) -> Vec<f32> {
        let mut result = vec![0.0; quantized.dim];

        for i in 0..quantized.dim {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (quantized.data[byte_idx] >> bit_idx) & 1;
            result[i] = if bit == 1 { 1.0 } else { -1.0 };
        }

        result
    }

    fn distance(&self, a: &BinaryVector, b: &BinaryVector) -> f32 {
        assert_eq!(
            a.dim, b.dim,
            "Binary vectors must have same dimension for distance computation"
        );

        let mut hamming = 0u32;
        let num_bytes = a.data.len().min(b.data.len());

        for i in 0..num_bytes {
            let xor = a.data[i] ^ b.data[i];
            hamming += xor.count_ones();
        }

        // Convert Hamming distance to similarity (lower distance = higher similarity)
        // For MaxSim, we want higher values for similar vectors
        // Similarity = dimension - hamming_distance
        let max_hamming = a.dim as f32;
        max_hamming - hamming as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_quantization_single_vector() {
        let quantizer = BinaryQuantization::new();
        let vector = vec![0.5, -0.3, 0.8, -0.1];

        let binary = quantizer.quantize_vector(&vector);
        assert_eq!(binary.dim, 4);

        // Verify bits: [+, -, +, -] -> [1, 0, 1, 0]
        let deq = quantizer.dequantize_vector(&binary);
        assert_eq!(deq, vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_binary_hamming_distance() {
        let quantizer = BinaryQuantization::new();
        let v1 = vec![1.0, 1.0, -1.0, -1.0];
        let v2 = vec![1.0, -1.0, -1.0, 1.0];

        let b1 = quantizer.quantize_vector(&v1);
        let b2 = quantizer.quantize_vector(&v2);

        let dist = quantizer.distance(&b1, &b2);
        // Hamming distance = 2 (differ in positions 1 and 3)
        // Similarity = 4 - 2 = 2
        assert_eq!(dist, 2.0);
    }

    #[test]
    fn test_binary_identical_vectors() {
        let quantizer = BinaryQuantization::new();
        let vector = vec![0.5, -0.3, 0.8, -0.1, 0.2];

        let b1 = quantizer.quantize_vector(&vector);
        let b2 = quantizer.quantize_vector(&vector);

        let dist = quantizer.distance(&b1, &b2);
        // Identical vectors should have zero Hamming distance
        // Similarity = 5 - 0 = 5
        assert_eq!(dist, 5.0);
    }

    #[test]
    fn test_binary_opposite_vectors() {
        let quantizer = BinaryQuantization::new();
        let v1 = vec![1.0, 1.0, 1.0, 1.0];
        let v2 = vec![-1.0, -1.0, -1.0, -1.0];

        let b1 = quantizer.quantize_vector(&v1);
        let b2 = quantizer.quantize_vector(&v2);

        let dist = quantizer.distance(&b1, &b2);
        // Completely opposite: Hamming distance = 4
        // Similarity = 4 - 4 = 0
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_multi_vector_quantization() {
        use crate::quantization::quantize_multi;

        let quantizer = BinaryQuantization::new();
        let vectors = vec![vec![0.5, -0.3], vec![0.8, 0.2], vec![-0.1, -0.9]];

        let quantized = quantize_multi(&quantizer, &vectors);
        assert_eq!(quantized.len(), 3);
        assert_eq!(quantized[0].dim, 2);
        assert_eq!(quantized[1].dim, 2);
        assert_eq!(quantized[2].dim, 2);
    }

    #[test]
    fn test_multi_vector_distance() {
        use crate::quantization::{multi_vector_distance, quantize_multi};

        let quantizer = BinaryQuantization::new();

        // Query: 2 vectors
        let query = vec![vec![1.0, 1.0], vec![-1.0, 1.0]];
        // Document: 2 vectors
        let document = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];

        let q_quant = quantize_multi(&quantizer, &query);
        let d_quant = quantize_multi(&quantizer, &document);

        let score = multi_vector_distance(&quantizer, &q_quant, &d_quant);

        // Query vector 0 [1, 1] vs Doc vectors:
        //   vs [1, -1]: Hamming=1, Sim=2-1=1
        //   vs [-1, 1]: Hamming=1, Sim=2-1=1
        //   Max = 1
        // Query vector 1 [-1, 1] vs Doc vectors:
        //   vs [1, -1]: Hamming=2, Sim=2-2=0
        //   vs [-1, 1]: Hamming=0, Sim=2-0=2
        //   Max = 2
        // Total = 1 + 2 = 3
        assert_eq!(score, 3.0);
    }

    #[test]
    fn test_binary_large_dimension() {
        let quantizer = BinaryQuantization::new();
        let vector: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let binary = quantizer.quantize_vector(&vector);
        assert_eq!(binary.dim, 128);
        assert_eq!(binary.data.len(), 16); // 128 bits = 16 bytes

        let restored = quantizer.dequantize_vector(&binary);
        assert_eq!(restored.len(), 128);
    }

    #[test]
    fn test_binary_non_multiple_of_8() {
        let quantizer = BinaryQuantization::new();
        let vector = vec![0.5, -0.3, 0.8]; // 3 dimensions

        let binary = quantizer.quantize_vector(&vector);
        assert_eq!(binary.dim, 3);
        assert_eq!(binary.data.len(), 1); // Rounds up to 1 byte

        let restored = quantizer.dequantize_vector(&binary);
        assert_eq!(restored.len(), 3);
        assert_eq!(restored, vec![1.0, -1.0, 1.0]);
    }
}
