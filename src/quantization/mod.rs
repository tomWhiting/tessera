//! Quantization methods for embedding compression.
//!
//! Provides per-vector quantization that works with:
//! - Single vectors (dense embeddings)
//! - Multi-vector (ColBERT token embeddings)
//! - Patch embeddings (vision, time series)
//!
//! The design quantizes individual vectors, enabling composition
//! for multi-vector scenarios.
//!
//! # Quantization Methods
//!
//! - [`binary`]: 1-bit quantization (32x compression, 95%+ accuracy)
//! - [`int8`]: 8-bit quantization (4x compression, <1% accuracy loss)
//! - [`int4`]: 4-bit quantization (8x compression, ~2% accuracy loss)
//!
//! Each quantization method provides encoding (float32 → quantized),
//! decoding (quantized → float32), and distance computation optimized
//! for the quantized representation.
//!
//! # Performance Benefits
//!
//! Quantization reduces:
//! - Memory footprint (4x to 32x compression)
//! - Cache misses (more embeddings fit in cache)
//! - Distance computation cost (SIMD-optimized int operations)
//!
//! # Accuracy Trade-offs
//!
//! | Method | Compression | Typical Accuracy Loss |
//! |--------|-------------|----------------------|
//! | Binary | 32x         | 3-5%                 |
//! | Int8   | 4x          | <1%                  |
//! | Int4   | 8x          | 1-3%                 |
//!
//! # Single Vector Example
//!
//! ```ignore
//! use tessera::quantization::{BinaryQuantization, Quantization};
//!
//! let quantizer = BinaryQuantization::new();
//! let vector = vec![0.5, -0.3, 0.8, -0.1];
//! let quantized = quantizer.quantize_vector(&vector);
//! let restored = quantizer.dequantize_vector(&quantized);
//! ```
//!
//! # Multi-Vector Example
//!
//! ```ignore
//! use tessera::quantization::{BinaryQuantization, quantize_multi, multi_vector_distance};
//!
//! let quantizer = BinaryQuantization::new();
//! let query_vectors = vec![vec![0.5, -0.3], vec![0.8, 0.2]];
//! let doc_vectors = vec![vec![0.6, -0.2], vec![0.7, 0.3]];
//!
//! let q_quantized = quantize_multi(&quantizer, &query_vectors);
//! let d_quantized = quantize_multi(&quantizer, &doc_vectors);
//!
//! let similarity = multi_vector_distance(&quantizer, &q_quantized, &d_quantized);
//! ```

pub mod binary;
pub mod int4;
pub mod int8;

pub use binary::BinaryQuantization;
pub use int4::Int4Quantization;
pub use int8::Int8Quantization;

/// Trait for quantization methods.
///
/// Quantizes individual vectors (not entire multi-vector embeddings).
/// For multi-vector scenarios, apply to each vector independently using
/// helper functions like [`quantize_multi`] and [`multi_vector_distance`].
///
/// # Design Rationale
///
/// The per-vector approach enables:
/// - Single-vector embeddings (dense models)
/// - Multi-vector embeddings (ColBERT)
/// - Variable-length sequences (time series patches)
/// - Consistent interface across paradigms
///
/// # Implementation Notes
///
/// Implementors should ensure:
/// - `distance` returns higher values for more similar vectors
/// - `dequantize_vector` provides reasonable reconstruction
/// - Thread-safety for concurrent quantization
pub trait Quantization {
    /// Quantized representation type
    type Output;

    /// Quantize a single vector to the target representation.
    ///
    /// # Arguments
    ///
    /// * `vector` - Float32 vector to quantize
    ///
    /// # Returns
    ///
    /// Quantized representation
    ///
    /// # Example
    ///
    /// ```ignore
    /// let quantizer = BinaryQuantization::new();
    /// let vector = vec![0.5, -0.3, 0.8];
    /// let quantized = quantizer.quantize_vector(&vector);
    /// ```
    fn quantize_vector(&self, vector: &[f32]) -> Self::Output;

    /// Dequantize back to float32 representation.
    ///
    /// Useful for exact search, rescoring, or debugging. The reconstruction
    /// may not be perfect depending on the quantization method's precision.
    ///
    /// # Arguments
    ///
    /// * `quantized` - Quantized vector to restore
    ///
    /// # Returns
    ///
    /// Reconstructed float32 vector
    ///
    /// # Example
    ///
    /// ```ignore
    /// let quantizer = BinaryQuantization::new();
    /// let original = vec![0.5, -0.3, 0.8];
    /// let quantized = quantizer.quantize_vector(&original);
    /// let restored = quantizer.dequantize_vector(&quantized);
    /// ```
    fn dequantize_vector(&self, quantized: &Self::Output) -> Vec<f32>;

    /// Compute distance between two quantized vectors.
    ///
    /// Returns higher values for more similar vectors to maintain
    /// consistency with MaxSim and other similarity metrics.
    ///
    /// # Arguments
    ///
    /// * `a` - First quantized vector
    /// * `b` - Second quantized vector
    ///
    /// # Returns
    ///
    /// Distance score (higher = more similar)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let quantizer = BinaryQuantization::new();
    /// let v1 = quantizer.quantize_vector(&vec![0.5, -0.3]);
    /// let v2 = quantizer.quantize_vector(&vec![0.6, -0.2]);
    /// let similarity = quantizer.distance(&v1, &v2);
    /// ```
    fn distance(&self, a: &Self::Output, b: &Self::Output) -> f32;
}

/// Helper function for multi-vector quantization.
///
/// Applies quantization to each vector in a multi-vector embedding
/// independently, enabling use with ColBERT and other multi-vector
/// paradigms.
///
/// # Arguments
///
/// * `quantizer` - Quantization method to apply
/// * `vectors` - Multiple vectors to quantize
///
/// # Returns
///
/// Vector of quantized representations
///
/// # Example
///
/// ```ignore
/// use tessera::quantization::{BinaryQuantization, quantize_multi};
///
/// let quantizer = BinaryQuantization::new();
/// let vectors = vec![
///     vec![0.5, -0.3],
///     vec![0.8, 0.2],
///     vec![-0.1, -0.9],
/// ];
/// let quantized = quantize_multi(&quantizer, &vectors);
/// assert_eq!(quantized.len(), 3);
/// ```
pub fn quantize_multi<Q: Quantization>(quantizer: &Q, vectors: &[Vec<f32>]) -> Vec<Q::Output> {
    vectors
        .iter()
        .map(|v| quantizer.quantize_vector(v))
        .collect()
}

/// Helper function for multi-vector distance computation (MaxSim).
///
/// Computes the MaxSim distance over quantized multi-vector embeddings.
/// For each query vector, finds the maximum similarity with document
/// vectors, then sums across all query vectors.
///
/// # Arguments
///
/// * `quantizer` - Quantization method (for distance computation)
/// * `query` - Quantized query vectors
/// * `document` - Quantized document vectors
///
/// # Returns
///
/// MaxSim score (sum of max similarities)
///
/// # MaxSim Algorithm
///
/// ```text
/// MaxSim(Q, D) = Σ max(sim(q_i, d_j))
///                i  j
/// ```
///
/// Where:
/// - Q = query vectors (variable length)
/// - D = document vectors (variable length)
/// - For each query vector q_i, find max similarity with any doc vector d_j
///
/// # Example
///
/// ```ignore
/// use tessera::quantization::{BinaryQuantization, quantize_multi, multi_vector_distance};
///
/// let quantizer = BinaryQuantization::new();
/// let query = vec![vec![0.5, -0.3], vec![0.8, 0.2]];
/// let document = vec![vec![0.6, -0.2], vec![0.7, 0.3]];
///
/// let q_quant = quantize_multi(&quantizer, &query);
/// let d_quant = quantize_multi(&quantizer, &document);
///
/// let score = multi_vector_distance(&quantizer, &q_quant, &d_quant);
/// ```
pub fn multi_vector_distance<Q: Quantization>(
    quantizer: &Q,
    query: &[Q::Output],
    document: &[Q::Output],
) -> f32 {
    query
        .iter()
        .map(|q_vec| {
            document
                .iter()
                .map(|d_vec| quantizer.distance(q_vec, d_vec))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}
