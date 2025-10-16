//! Similarity and distance functions for embeddings.
//!
//! Provides common similarity metrics used across different embedding paradigms:
//!
//! - Cosine similarity: Normalized dot product (angle-based)
//! - Dot product: Raw inner product similarity
//! - Euclidean distance: L2 norm of difference vector
//! - MaxSim: Late interaction similarity for multi-vector embeddings
//!
//! These functions are the building blocks for retrieval, ranking, and clustering.

use anyhow::Result;
use ndarray::{Array1, Axis};

use crate::core::TokenEmbeddings;
use crate::utils::normalization::l2_norm;

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

/// Computes MaxSim similarity between query and document embeddings.
///
/// MaxSim algorithm (late interaction for multi-vector embeddings):
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
/// MaxSim similarity score (higher = more similar)
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
/// # let config = tessera::models::ModelConfig::distilbert_base_uncased();
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

    // Get references to the embedding matrices
    let query_matrix = &query.embeddings;
    let doc_matrix = &document.embeddings;

    // Compute the similarity matrix: query_matrix × doc_matrix^T
    // Result shape: (num_query_tokens, num_doc_tokens)
    let similarity_matrix = query_matrix.dot(&doc_matrix.t());

    // For each query token (row), find the maximum similarity across all doc tokens
    let max_sims: Array1<f32> = similarity_matrix.map_axis(Axis(1), |row| {
        row.fold(f32::NEG_INFINITY, |acc, &val| acc.max(val))
    });

    // Sum all maximum similarities
    let total_score: f32 = max_sims.sum();

    Ok(total_score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = array![1.0, 0.0];
        let b = array![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let a = array![1.0, 2.0];
        let b = array![1.0, 2.0, 3.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_dot_product() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert!((dot - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_dimension_mismatch() {
        let a = array![1.0, 2.0];
        let b = array![1.0, 2.0, 3.0];
        assert!(dot_product(&a, &b).is_err());
    }

    #[test]
    fn test_euclidean_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let dist = euclidean_distance(&a, &b).unwrap();
        // √(3² + 4²) = 5
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        let dist = euclidean_distance(&a, &b).unwrap();
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_dimension_mismatch() {
        let a = array![1.0, 2.0];
        let b = array![1.0, 2.0, 3.0];
        assert!(euclidean_distance(&a, &b).is_err());
    }

    #[test]
    fn test_max_sim_simple() {
        // Create simple query embeddings (2 tokens, 3 dimensions each)
        let query_emb = array![
            [1.0, 0.0, 0.0], // Token 1
            [0.0, 1.0, 0.0], // Token 2
        ];
        let query = TokenEmbeddings::new(query_emb, "query text".to_string()).unwrap();

        // Create simple document embeddings (3 tokens, 3 dimensions each)
        let doc_emb = array![
            [1.0, 0.0, 0.0], // Token 1 (matches query token 1)
            [0.0, 0.5, 0.0], // Token 2 (partial match to query token 2)
            [0.0, 1.0, 0.0], // Token 3 (matches query token 2)
        ];
        let document = TokenEmbeddings::new(doc_emb, "document text".to_string()).unwrap();

        let score = max_sim(&query, &document).unwrap();

        // Query token 1 max similarity: max(1.0, 0.0, 0.0) = 1.0
        // Query token 2 max similarity: max(0.0, 0.5, 1.0) = 1.0
        // Total: 1.0 + 1.0 = 2.0
        assert!((score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_sim_dimension_mismatch() {
        let query_emb = array![[1.0, 0.0]];
        let query = TokenEmbeddings::new(query_emb, "query".to_string()).unwrap();

        let doc_emb = array![[1.0, 0.0, 0.0]];
        let document = TokenEmbeddings::new(doc_emb, "document".to_string()).unwrap();

        assert!(max_sim(&query, &document).is_err());
    }
}
