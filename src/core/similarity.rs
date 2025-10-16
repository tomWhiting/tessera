//! MaxSim similarity scoring for ColBERT-style late interaction.
//!
//! This module is maintained for backward compatibility. New code should use
//! `crate::utils::similarity` instead, which provides a comprehensive set of
//! similarity functions including MaxSim.
//!
//! # Deprecation Notice
//!
//! This module will be removed in a future version. Please migrate to:
//! ```
//! use tessera::utils::similarity::max_sim;
//! ```

use anyhow::Result;

use super::embeddings::TokenEmbeddings;

/// Computes MaxSim similarity between query and document embeddings.
///
/// MaxSim algorithm:
/// For each query token vector qi:
///   1. Compute dot product with all document token vectors dj
///   2. Take the maximum score across all document tokens
///   3. Sum these maximum scores across all query tokens
///
/// Formula: MaxSim(Q, D) = Σ(i=1 to n) max(j=1 to m) qi · dj
///
/// # Arguments
/// * `query` - Query token embeddings
/// * `document` - Document token embeddings
///
/// # Returns
/// The MaxSim similarity score (higher is more similar)
///
/// # Errors
/// Returns an error if the embedding dimensions don't match
///
/// # Deprecated
/// Use `crate::utils::similarity::max_sim` instead
#[deprecated(since = "0.2.0", note = "Use crate::utils::similarity::max_sim instead")]
pub fn max_sim(query: &TokenEmbeddings, document: &TokenEmbeddings) -> Result<f32> {
    // Delegate to the new implementation
    crate::utils::similarity::max_sim(query, document)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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

        let score = crate::utils::similarity::max_sim(&query, &document).unwrap();

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

        assert!(crate::utils::similarity::max_sim(&query, &document).is_err());
    }
}
