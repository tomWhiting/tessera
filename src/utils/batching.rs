//! Batching utilities for efficient processing of multiple inputs.
//!
//! Provides utilities for preparing batched inputs for neural networks:
//!
//! - Sequence padding: Pad variable-length sequences to uniform length
//! - Attention mask generation: Create masks for padded sequences
//!
//! These utilities enable efficient batch processing while correctly handling
//! variable-length inputs and ensuring padding tokens don't affect results.

/// Pad sequences to a uniform length.
///
/// Takes a list of variable-length sequences and pads them all to the length
/// of the longest sequence using the specified pad token.
///
/// # Arguments
/// * `sequences` - Slice of variable-length token ID sequences
/// * `pad_token` - Token ID to use for padding
///
/// # Returns
/// Vector of padded sequences, all having the same length
///
/// # Example
/// ```
/// use tessera::utils::pad_sequences;
///
/// let sequences = vec![
///     vec![1, 2, 3],
///     vec![4, 5],
///     vec![6, 7, 8, 9],
/// ];
///
/// let padded = pad_sequences(&sequences, 0);
///
/// // All sequences now have length 4 (length of longest)
/// assert_eq!(padded[0], vec![1, 2, 3, 0]);
/// assert_eq!(padded[1], vec![4, 5, 0, 0]);
/// assert_eq!(padded[2], vec![6, 7, 8, 9]);
/// ```
pub fn pad_sequences(sequences: &[Vec<u32>], pad_token: u32) -> Vec<Vec<u32>> {
    if sequences.is_empty() {
        return Vec::new();
    }

    // Find maximum length
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);

    // Pad all sequences to max length
    sequences
        .iter()
        .map(|seq| {
            let mut padded = seq.clone();
            padded.resize(max_len, pad_token);
            padded
        })
        .collect()
}

/// Create attention masks for padded sequences.
///
/// Generates binary masks indicating which positions are real tokens (1)
/// and which are padding (0). This allows models to ignore padding tokens
/// during attention computation.
///
/// # Arguments
/// * `sequences` - Slice of token ID sequences (potentially padded)
/// * `pad_token` - Token ID used for padding
///
/// # Returns
/// Vector of attention masks (1 = real token, 0 = padding)
///
/// # Example
/// ```
/// use tessera::utils::create_attention_mask;
///
/// let sequences = vec![
///     vec![1, 2, 3, 0, 0],  // Last 2 are padding
///     vec![4, 5, 6, 7, 0],  // Last 1 is padding
///     vec![8, 9, 0, 0, 0],  // Last 3 are padding
/// ];
///
/// let masks = create_attention_mask(&sequences, 0);
///
/// assert_eq!(masks[0], vec![1, 1, 1, 0, 0]);
/// assert_eq!(masks[1], vec![1, 1, 1, 1, 0]);
/// assert_eq!(masks[2], vec![1, 1, 0, 0, 0]);
/// ```
pub fn create_attention_mask(sequences: &[Vec<u32>], pad_token: u32) -> Vec<Vec<i64>> {
    sequences
        .iter()
        .map(|seq| {
            seq.iter()
                .map(|&token| if token == pad_token { 0 } else { 1 })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_sequences_basic() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];

        let padded = pad_sequences(&sequences, 0);

        assert_eq!(padded.len(), 3);
        assert_eq!(padded[0], vec![1, 2, 3, 0]);
        assert_eq!(padded[1], vec![4, 5, 0, 0]);
        assert_eq!(padded[2], vec![6, 7, 8, 9]);
    }

    #[test]
    fn test_pad_sequences_empty() {
        let sequences: Vec<Vec<u32>> = vec![];
        let padded = pad_sequences(&sequences, 0);
        assert!(padded.is_empty());
    }

    #[test]
    fn test_pad_sequences_single() {
        let sequences = vec![vec![1, 2, 3]];
        let padded = pad_sequences(&sequences, 0);
        assert_eq!(padded.len(), 1);
        assert_eq!(padded[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_pad_sequences_all_same_length() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let padded = pad_sequences(&sequences, 0);

        // Should remain unchanged
        assert_eq!(padded[0], vec![1, 2, 3]);
        assert_eq!(padded[1], vec![4, 5, 6]);
        assert_eq!(padded[2], vec![7, 8, 9]);
    }

    #[test]
    fn test_pad_sequences_custom_pad_token() {
        let sequences = vec![vec![1, 2], vec![3, 4, 5]];
        let padded = pad_sequences(&sequences, 999);

        assert_eq!(padded[0], vec![1, 2, 999]);
        assert_eq!(padded[1], vec![3, 4, 5]);
    }

    #[test]
    fn test_create_attention_mask_basic() {
        let sequences = vec![
            vec![1, 2, 3, 0, 0],
            vec![4, 5, 6, 7, 0],
            vec![8, 9, 0, 0, 0],
        ];

        let masks = create_attention_mask(&sequences, 0);

        assert_eq!(masks.len(), 3);
        assert_eq!(masks[0], vec![1, 1, 1, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 1, 1, 0]);
        assert_eq!(masks[2], vec![1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_create_attention_mask_no_padding() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5, 6]];

        let masks = create_attention_mask(&sequences, 0);

        assert_eq!(masks[0], vec![1, 1, 1]);
        assert_eq!(masks[1], vec![1, 1, 1]);
    }

    #[test]
    fn test_create_attention_mask_all_padding() {
        let sequences = vec![vec![0, 0, 0]];

        let masks = create_attention_mask(&sequences, 0);

        assert_eq!(masks[0], vec![0, 0, 0]);
    }

    #[test]
    fn test_create_attention_mask_custom_pad_token() {
        let sequences = vec![vec![1, 2, 999, 999], vec![3, 4, 5, 999]];

        let masks = create_attention_mask(&sequences, 999);

        assert_eq!(masks[0], vec![1, 1, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 1, 0]);
    }

    #[test]
    fn test_pad_and_mask_together() {
        // Typical workflow: pad then create mask
        let sequences = vec![vec![1, 2], vec![3, 4, 5, 6], vec![7]];

        let padded = pad_sequences(&sequences, 0);
        let masks = create_attention_mask(&padded, 0);

        // All should be length 4
        assert_eq!(padded[0].len(), 4);
        assert_eq!(padded[1].len(), 4);
        assert_eq!(padded[2].len(), 4);

        // Masks should reflect padding
        assert_eq!(masks[0], vec![1, 1, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 1, 1]);
        assert_eq!(masks[2], vec![1, 0, 0, 0]);
    }
}
