//! Int4 (4-bit) quantization for aggressive compression.
//!
//! Implements 4-bit integer quantization with grouped quantization:
//!
//! - **Encoding**: `round((x - offset) / scale)` → 4 bits
//! - **Storage**: 0.5 bytes per dimension (2 values per byte)
//! - **Distance**: Optimized int4 operations with dequantization
//!
//! # Compression
//!
//! - Original: 4 bytes per dimension (float32)
//! - Int4: 0.5 bytes per dimension (8x compression)
//! - 768-dim embedding: 3072 bytes → 384 bytes
//!
//! # Grouped Quantization
//!
//! To maintain accuracy with only 4 bits, we use grouped quantization:
//! - Divide embedding into groups (e.g., 32 or 64 dims)
//! - Compute separate scale/offset per group
//! - Reduces quantization error at edges of range
//!
//! # Storage Layout
//!
//! Two 4-bit values packed per byte:
//! - High nibble: first value
//! - Low nibble: second value
//! - Requires careful bit manipulation for access
//!
//! # Performance
//!
//! - 8x memory reduction
//! - Requires dequantization for distance computation
//! - ~2% accuracy loss compared to float32
//!
//! # Example
//!
//! ```ignore
//! use tessera::quantization::Int4Quantization;
//!
//! let quantizer = Int4Quantization::new(32);
//! let vector = vec![0.5, -0.3, 0.8, -0.1];
//! let quantized = quantizer.quantize_vector(&vector);
//! ```

use super::Quantization;

/// Int4 quantized vector representation with grouped quantization.
#[derive(Debug, Clone)]
pub struct Int4Vector {
    /// Packed 4-bit values (2 per byte)
    pub values: Vec<u8>,
    /// Per-group scale factors
    pub scales: Vec<f32>,
    /// Per-group offsets
    pub offsets: Vec<f32>,
    /// Group size for quantization
    pub group_size: usize,
}

/// Int4 quantization with grouped quantization.
pub struct Int4Quantization {
    // TODO: Add fields:
    // - group_size: Number of dimensions per quantization group
    // - symmetric: Whether to use symmetric quantization
}

impl Int4Quantization {
    /// Create a new int4 quantization instance.
    ///
    /// # Arguments
    ///
    /// * `group_size` - Number of dimensions per quantization group
    #[must_use] pub fn new(_group_size: usize) -> Self {
        todo!("Implement int4 quantization initialization")
    }
}

impl Quantization for Int4Quantization {
    type Output = Int4Vector;

    fn quantize_vector(&self, _vector: &[f32]) -> Self::Output {
        todo!("Implement int4 grouped quantization with nibble packing")
    }

    fn dequantize_vector(&self, _quantized: &Self::Output) -> Vec<f32> {
        todo!("Implement int4 dequantization with grouped scale/offset")
    }

    fn distance(&self, _a: &Self::Output, _b: &Self::Output) -> f32 {
        todo!("Implement int4 distance with dequantization")
    }
}
