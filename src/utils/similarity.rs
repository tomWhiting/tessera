//! Similarity and distance functions for embeddings.
//!
//! Provides common similarity metrics used across different embedding paradigms:
//!
//! - Cosine similarity: Normalized dot product (angle-based)
//! - Dot product: Raw inner product similarity
//! - Euclidean distance: L2 norm of difference vector
//! - `MaxSim`: Late interaction similarity for multi-vector embeddings
//!
//! These functions are the building blocks for retrieval, ranking, and clustering.

use anyhow::Result;
use ndarray::{linalg::general_mat_vec_mul, s, Array1};

use crate::core::TokenEmbeddings;
use crate::utils::normalization::l2_norm;

/// Number of document-token scores retained while computing `MaxSim`.
///
/// Keeping this deliberately small bounds the temporary score workspace while
/// still amortizing traversal of the document matrix.
const MAX_SIM_DOCUMENT_BLOCK_SIZE: usize = 64;

/// Cosine similarity between two vectors.
///
/// Computes the normalized dot product, measuring the angle between vectors
/// (range: -1 to 1, where 1 = identical direction, 0 = orthogonal, -1 = opposite).
///
/// Formula: `cos(θ) = (a · b) / (||a|| ||b||)`
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity score in [-1, 1]
///
/// # Errors
/// Returns an error if vectors have different lengths or if norms are zero
///
/// # Example
/// ```
/// use ndarray::array;
/// use tessera::utils::cosine_similarity;
///
/// let a = array![1.0, 0.0, 0.0];
/// let b = array![1.0, 0.0, 0.0];
/// let sim = cosine_similarity(&a, &b).unwrap();
/// assert!((sim - 1.0).abs() < 1e-6);  // Identical vectors
///
/// let c = array![0.0, 1.0, 0.0];
/// let sim = cosine_similarity(&a, &c).unwrap();
/// assert!((sim - 0.0).abs() < 1e-6);  // Orthogonal vectors
/// ```
pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> Result<f32> {
    anyhow::ensure!(
        a.len() == b.len(),
        "Vectors must have same length (got {} and {})",
        a.len(),
        b.len()
    );

    let dot = dot_product(a, b)?;
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / (norm_a * norm_b))
    }
}

/// Dot product between two vectors.
///
/// Computes the inner product, measuring vector similarity in the original space
/// (unbounded, higher = more similar).
///
/// Formula: `a · b = Σ(aᵢ × bᵢ)`
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Dot product similarity score
///
/// # Errors
/// Returns an error if vectors have different lengths
///
/// # Example
/// ```
/// use ndarray::array;
/// use tessera::utils::dot_product;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![4.0, 5.0, 6.0];
/// let dot = dot_product(&a, &b).unwrap();
/// // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// assert!((dot - 32.0).abs() < 1e-6);
/// ```
pub fn dot_product(a: &Array1<f32>, b: &Array1<f32>) -> Result<f32> {
    anyhow::ensure!(
        a.len() == b.len(),
        "Vectors must have same length (got {} and {})",
        a.len(),
        b.len()
    );
    Ok(a.dot(b))
}

/// Euclidean distance between two vectors.
///
/// Computes the L2 norm of the difference vector, measuring straight-line
/// distance in embedding space (range: [0, ∞), where 0 = identical).
///
/// Formula: `d(a, b) = ||a - b||₂ = √(Σ(aᵢ - bᵢ)²)`
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Euclidean distance (0 = identical, higher = more different)
///
/// # Errors
/// Returns an error if vectors have different lengths
///
/// # Example
/// ```
/// use ndarray::array;
/// use tessera::utils::euclidean_distance;
///
/// let a = array![0.0, 0.0];
/// let b = array![3.0, 4.0];
/// let dist = euclidean_distance(&a, &b).unwrap();
/// // Distance = √(3² + 4²) = √25 = 5.0
/// assert!((dist - 5.0).abs() < 1e-6);
/// ```
pub fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> Result<f32> {
    anyhow::ensure!(
        a.len() == b.len(),
        "Vectors must have same length (got {} and {})",
        a.len(),
        b.len()
    );
    let diff = a - b;
    Ok(diff.dot(&diff).sqrt())
}

/// Computes `MaxSim` similarity between query and document embeddings.
///
/// `MaxSim` algorithm (late interaction for multi-vector embeddings):
/// For each query token vector qᵢ:
///   1. Compute dot product with all document token vectors dⱼ
///   2. Take the maximum score across all document tokens
///   3. Sum these maximum scores across all query tokens
///
/// Formula: `MaxSim(Q, D) = Σᵢ max_ⱼ (qᵢ · dⱼ)`
///
/// This enables fine-grained token-level matching while remaining efficient
/// through late interaction (no cross-attention required).
///
/// # Arguments
/// * `query` - Query token embeddings
/// * `document` - Document token embeddings
///
/// # Returns
/// `MaxSim` similarity score (higher = more similar)
///
/// # Errors
/// Returns an error if embedding dimensions don't match
///
/// # Example
/// ```ignore
/// use tessera::{backends::CandleBertEncoder, core::TokenEmbedder, utils::max_sim};
/// # use anyhow::Result;
///
/// # fn example() -> Result<()> {
/// # let device = tessera::backends::candle::get_device()?;
/// # let config = tessera::models::ModelConfig::colbert_small();
/// # let encoder = CandleBertEncoder::new(config, device)?;
/// let query = encoder.encode("machine learning")?;
/// let doc = encoder.encode("deep learning and neural networks")?;
/// let score = max_sim(&query, &doc)?;
/// println!("Similarity: {}", score);
/// # Ok(())
/// # }
/// ```
pub fn max_sim(query: &TokenEmbeddings, document: &TokenEmbeddings) -> Result<f32> {
    anyhow::ensure!(
        query.embedding_dim == document.embedding_dim,
        "Query and document embedding dimensions must match (query: {}, document: {})",
        query.embedding_dim,
        document.embedding_dim
    );

    let query_matrix = &query.embeddings;
    let doc_matrix = &document.embeddings;

    // Retain only one running maximum per query token and a fixed-size block of
    // document scores. This avoids materializing either the full Q x D matrix
    // or a Q x block matrix.
    let mut max_sims = Array1::from_elem(query_matrix.nrows(), f32::NEG_INFINITY);
    let mut block_scores = Array1::zeros(MAX_SIM_DOCUMENT_BLOCK_SIZE);

    for document_start in (0..doc_matrix.nrows()).step_by(MAX_SIM_DOCUMENT_BLOCK_SIZE) {
        let document_end = (document_start + MAX_SIM_DOCUMENT_BLOCK_SIZE).min(doc_matrix.nrows());
        let document_block = doc_matrix.slice(s![document_start..document_end, ..]);
        let mut active_scores = block_scores.slice_mut(s![..document_block.nrows()]);

        for (query_index, query_row) in query_matrix.outer_iter().enumerate() {
            general_mat_vec_mul(1.0, &document_block, &query_row, 0.0, &mut active_scores);

            let block_max = active_scores
                .iter()
                .fold(f32::NEG_INFINITY, |acc, &score| acc.max(score));
            max_sims[query_index] = max_sims[query_index].max(block_max);
        }
    }

    Ok(max_sims.sum())
}

#[cfg(test)]
#[path = "similarity/tests.rs"]
mod tests;
