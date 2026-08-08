//! Quantization methods for embedding compression.
//!
//! Provides per-vector quantization that works with:
//! - Single vectors (dense embeddings)
//! - Multi-vector (`ColBERT` token embeddings)
//! - Patch embeddings (vision, time series)
//!
//! The design quantizes individual vectors, enabling composition
//! for multi-vector scenarios.
//!
//! # Quantization Method
//!
//! Tessera currently exposes binary (1-bit) quantization. Int4 and Int8 were
//! previously advertised as placeholders; they have been removed until they
//! have complete, validated implementations.
//!
//! The binary quantizer provides encoding (float32 → sign bits), decoding
//! (sign bits → ±1.0), and Hamming-based similarity.
//!
//! # Storage
//!
//! A float32 vector uses four bytes per dimension. The packed binary payload
//! uses one bit per dimension, rounded up to a whole byte, excluding metadata.
//!
//! Retrieval quality is model- and dataset-dependent and must be measured for
//! the target workload.
//!
//! # Single Vector Example
//!
//! ```ignore
//! use tessera::quantization::{BinaryQuantization, Quantization};
//!
//! let quantizer = BinaryQuantization::new();
//! let vector = vec![0.5, -0.3, 0.8, -0.1];
//! let quantized = quantizer.quantize_vector(&vector)?;
//! let restored = quantizer.dequantize_vector(&quantized)?;
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
//! let q_quantized = quantize_multi(&quantizer, &query_vectors)?;
//! let d_quantized = quantize_multi(&quantizer, &doc_vectors)?;
//!
//! let similarity = multi_vector_distance(&quantizer, &q_quantized, &d_quantized)?;
//! ```

pub mod binary;
mod multi;
mod quantizer;

pub use binary::BinaryQuantization;
pub use multi::{multi_vector_distance, quantize_multi};
pub use quantizer::Quantization;
