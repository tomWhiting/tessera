//! Pooling strategies for aggregating token embeddings.
//!
//! Provides multiple pooling methods to convert token-level embeddings
//! into single dense vectors:
//!
//! - CLS pooling: Extract [CLS] token (first token)
//! - Mean pooling: Average all tokens (weighted by attention mask)
//! - Max pooling: Element-wise maximum across tokens
//!
//! These are commonly used in dense encoders to produce sentence/document
//! embeddings from BERT-style models.

use ndarray::{Array1, Array2};

/// Extract CLS token embedding (first token).
///
/// The [CLS] token is typically trained to represent the entire sequence
/// in BERT-style models. This pooling strategy simply extracts the first
/// token's embedding.
///
/// # Arguments
/// * `token_embeddings` - Token embedding matrix (`num_tokens` × `embedding_dim`)
/// * `_attention_mask` - Attention mask (unused for CLS pooling, but kept for interface consistency)
///
/// # Returns
/// The first token's embedding vector
///
/// # Panics
/// Panics if the `token_embeddings` matrix is empty (no tokens)
///
/// # Example
/// ```
/// use ndarray::array;
/// use tessera::utils::cls_pooling;
///
/// let embeddings = array![
///     [1.0, 2.0, 3.0],  // CLS token
///     [4.0, 5.0, 6.0],  // Token 1
///     [7.0, 8.0, 9.0],  // Token 2
/// ];
/// let mask = vec![1, 1, 1];
///
/// let pooled = cls_pooling(&embeddings, &mask);
/// assert_eq!(pooled[0], 1.0);
/// assert_eq!(pooled[1], 2.0);
/// assert_eq!(pooled[2], 3.0);
/// ```
#[must_use] pub fn cls_pooling(token_embeddings: &Array2<f32>, _attention_mask: &[i64]) -> Array1<f32> {
    token_embeddings.row(0).to_owned()
}

/// Mean pooling weighted by attention mask.
///
/// Averages token embeddings, considering only tokens where `attention_mask` is 1
/// (ignoring padding tokens where mask is 0). This produces a centroid
/// representation of all meaningful tokens.
///
/// # Arguments
/// * `token_embeddings` - Token embedding matrix (`num_tokens` × `embedding_dim`)
/// * `attention_mask` - Binary mask indicating valid tokens (1 = valid, 0 = padding)
///
/// # Returns
/// Mean-pooled embedding vector
///
/// # Example
/// ```
/// use ndarray::array;
/// use tessera::utils::mean_pooling;
///
/// let embeddings = array![
///     [1.0, 2.0],  // Token 0 (valid)
///     [3.0, 4.0],  // Token 1 (valid)
///     [5.0, 6.0],  // Token 2 (padding - ignored)
/// ];
/// let mask = vec![1, 1, 0];  // Last token is padding
///
/// let pooled = mean_pooling(&embeddings, &mask);
/// // Mean of first two tokens: [(1+3)/2, (2+4)/2] = [2.0, 3.0]
/// assert!((pooled[0] - 2.0).abs() < 1e-6);
/// assert!((pooled[1] - 3.0).abs() < 1e-6);
/// ```
#[must_use] pub fn mean_pooling(token_embeddings: &Array2<f32>, attention_mask: &[i64]) -> Array1<f32> {
    let mut sum = Array1::zeros(token_embeddings.ncols());
    let mut count = 0;

    for (i, &mask) in attention_mask.iter().enumerate() {
        if mask == 1 && i < token_embeddings.nrows() {
            sum = sum + token_embeddings.row(i);
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        sum // All zeros if no valid tokens
    }
}

/// Element-wise max pooling across tokens.
///
/// For each dimension, takes the maximum value across all valid tokens
/// (where `attention_mask` is 1). This captures the most salient features
/// from any token position.
///
/// # Arguments
/// * `token_embeddings` - Token embedding matrix (`num_tokens` × `embedding_dim`)
/// * `attention_mask` - Binary mask indicating valid tokens (1 = valid, 0 = padding)
///
/// # Returns
/// Max-pooled embedding vector
///
/// # Example
/// ```
/// use ndarray::array;
/// use tessera::utils::max_pooling;
///
/// let embeddings = array![
///     [1.0, 5.0],  // Token 0 (valid)
///     [3.0, 2.0],  // Token 1 (valid)
///     [9.0, 9.0],  // Token 2 (padding - ignored)
/// ];
/// let mask = vec![1, 1, 0];  // Last token is padding
///
/// let pooled = max_pooling(&embeddings, &mask);
/// // Max of first two tokens: [max(1,3), max(5,2)] = [3.0, 5.0]
/// assert!((pooled[0] - 3.0).abs() < 1e-6);
/// assert!((pooled[1] - 5.0).abs() < 1e-6);
/// ```
#[must_use] pub fn max_pooling(token_embeddings: &Array2<f32>, attention_mask: &[i64]) -> Array1<f32> {
    let embedding_dim = token_embeddings.ncols();
    let mut result = Array1::from_elem(embedding_dim, f32::NEG_INFINITY);

    for (i, &mask) in attention_mask.iter().enumerate() {
        if mask == 1 && i < token_embeddings.nrows() {
            let row = token_embeddings.row(i);
            for (j, &val) in row.iter().enumerate() {
                result[j] = result[j].max(val);
            }
        }
    }

    // Replace -inf with 0.0 if no valid tokens were found
    for val in &mut result {
        if val.is_infinite() && *val < 0.0 {
            *val = 0.0;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cls_pooling() {
        let embeddings = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
        let mask = vec![1, 1, 1];

        let pooled = cls_pooling(&embeddings, &mask);

        assert_eq!(pooled.len(), 3);
        assert_eq!(pooled[0], 1.0);
        assert_eq!(pooled[1], 2.0);
        assert_eq!(pooled[2], 3.0);
    }

    #[test]
    fn test_mean_pooling_all_valid() {
        let embeddings = array![[1.0, 2.0], [3.0, 4.0],];
        let mask = vec![1, 1];

        let pooled = mean_pooling(&embeddings, &mask);

        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 2.0).abs() < 1e-6); // (1+3)/2
        assert!((pooled[1] - 3.0).abs() < 1e-6); // (2+4)/2
    }

    #[test]
    fn test_mean_pooling_with_padding() {
        let embeddings = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0], // This is padding
        ];
        let mask = vec![1, 1, 0];

        let pooled = mean_pooling(&embeddings, &mask);

        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 2.0).abs() < 1e-6); // (1+3)/2, ignoring 5
        assert!((pooled[1] - 3.0).abs() < 1e-6); // (2+4)/2, ignoring 6
    }

    #[test]
    fn test_mean_pooling_all_padding() {
        let embeddings = array![[1.0, 2.0], [3.0, 4.0],];
        let mask = vec![0, 0];

        let pooled = mean_pooling(&embeddings, &mask);

        assert_eq!(pooled.len(), 2);
        assert_eq!(pooled[0], 0.0);
        assert_eq!(pooled[1], 0.0);
    }

    #[test]
    fn test_max_pooling_all_valid() {
        let embeddings = array![[1.0, 5.0], [3.0, 2.0],];
        let mask = vec![1, 1];

        let pooled = max_pooling(&embeddings, &mask);

        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 3.0).abs() < 1e-6); // max(1, 3)
        assert!((pooled[1] - 5.0).abs() < 1e-6); // max(5, 2)
    }

    #[test]
    fn test_max_pooling_with_padding() {
        let embeddings = array![
            [1.0, 5.0],
            [3.0, 2.0],
            [9.0, 9.0], // This is padding
        ];
        let mask = vec![1, 1, 0];

        let pooled = max_pooling(&embeddings, &mask);

        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 3.0).abs() < 1e-6); // max(1, 3), ignoring 9
        assert!((pooled[1] - 5.0).abs() < 1e-6); // max(5, 2), ignoring 9
    }

    #[test]
    fn test_max_pooling_all_padding() {
        let embeddings = array![[1.0, 2.0], [3.0, 4.0],];
        let mask = vec![0, 0];

        let pooled = max_pooling(&embeddings, &mask);

        assert_eq!(pooled.len(), 2);
        assert_eq!(pooled[0], 0.0);
        assert_eq!(pooled[1], 0.0);
    }
}
