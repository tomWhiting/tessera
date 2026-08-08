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
/// Returns an error if vectors are empty, have different lengths, contain
/// non-finite values, or the calculation overflows. Zero vectors return 0.0.
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
    validate_vector_pair(a, b)?;

    let dot = a.dot(b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        Ok(0.0)
    } else {
        let similarity = dot / (norm_a * norm_b);
        anyhow::ensure!(
            similarity.is_finite(),
            "Cosine similarity exceeded the finite f32 range"
        );
        Ok(similarity)
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
/// Returns an error if vectors are empty, have different lengths, contain
/// non-finite values, or the calculation overflows
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
    validate_vector_pair(a, b)?;
    let score = a.dot(b);
    anyhow::ensure!(
        score.is_finite(),
        "Dot product exceeded the finite f32 range"
    );
    Ok(score)
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
/// Returns an error if vectors are empty, have different lengths, contain
/// non-finite values, or the calculation overflows
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
    validate_vector_pair(a, b)?;
    let diff = a - b;
    let distance = diff.dot(&diff).sqrt();
    anyhow::ensure!(
        distance.is_finite(),
        "Euclidean distance exceeded the finite f32 range"
    );
    Ok(distance)
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
/// Returns an error if either embedding is empty, metadata or dimensions do
/// not match, values are non-finite, or the calculation overflows
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
    validate_token_embeddings("Query", query)?;
    validate_token_embeddings("Document", document)?;
    anyhow::ensure!(
        query.embedding_dim() == document.embedding_dim(),
        "Query and document embedding dimensions must match (query: {}, document: {})",
        query.embedding_dim(),
        document.embedding_dim()
    );

    let query_matrix = query.matrix();
    let doc_matrix = document.matrix();

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

            let mut block_max = f32::NEG_INFINITY;
            for &score in &active_scores {
                anyhow::ensure!(
                    score.is_finite(),
                    "MaxSim dot product exceeded the finite f32 range"
                );
                block_max = block_max.max(score);
            }
            max_sims[query_index] = max_sims[query_index].max(block_max);
        }
    }

    let score = max_sims.sum();
    anyhow::ensure!(
        score.is_finite(),
        "MaxSim score exceeded the finite f32 range"
    );
    Ok(score)
}

fn validate_vector_pair(a: &Array1<f32>, b: &Array1<f32>) -> Result<()> {
    anyhow::ensure!(!a.is_empty(), "Vectors must not be empty");
    anyhow::ensure!(
        a.len() == b.len(),
        "Vectors must have same length (got {} and {})",
        a.len(),
        b.len()
    );
    anyhow::ensure!(
        a.iter().all(|value| value.is_finite()),
        "First vector contains NaN or Inf values"
    );
    anyhow::ensure!(
        b.iter().all(|value| value.is_finite()),
        "Second vector contains NaN or Inf values"
    );
    Ok(())
}

fn validate_token_embeddings(label: &str, embeddings: &TokenEmbeddings) -> Result<()> {
    let actual_tokens = embeddings.matrix().nrows();
    let actual_dim = embeddings.matrix().ncols();
    anyhow::ensure!(
        actual_tokens > 0,
        "{label} embeddings must contain at least one token"
    );
    anyhow::ensure!(
        actual_dim > 0,
        "{label} embedding dimension must be greater than zero"
    );
    anyhow::ensure!(
        embeddings.num_tokens() == actual_tokens,
        "{label} token metadata mismatch: declared {}, actual {actual_tokens}",
        embeddings.num_tokens()
    );
    anyhow::ensure!(
        embeddings.embedding_dim() == actual_dim,
        "{label} dimension metadata mismatch: declared {}, actual {actual_dim}",
        embeddings.embedding_dim()
    );
    anyhow::ensure!(
        embeddings.matrix().iter().all(|value| value.is_finite()),
        "{label} embeddings contain NaN or Inf values"
    );
    Ok(())
}

#[cfg(test)]
#[path = "similarity/tests.rs"]
mod tests;
