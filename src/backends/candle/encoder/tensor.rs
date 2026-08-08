use anyhow::{Context, Result};
use candle_core::{Tensor, D};
use ndarray::{Array2, Axis};

/// L2-normalizes each token independently across the embedding dimension.
pub(super) fn l2_normalize_tokens(output: &Tensor) -> Result<Tensor> {
    anyhow::ensure!(
        output.rank() >= 2,
        "Token embeddings must have at least two dimensions"
    );

    // Match torch.nn.functional.normalize(..., p=2, dim=-1, eps=1e-12).
    let norms = output
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .maximum(1e-12)?;
    output
        .broadcast_div(&norms)
        .context("L2-normalizing token embeddings")
}

/// Keeps exactly the token rows selected by a standard 1/0 attention mask.
pub(super) fn filter_by_attention_mask(
    embeddings: &Array2<f32>,
    attention_mask: &[u32],
) -> Result<Array2<f32>> {
    anyhow::ensure!(
        embeddings.nrows() == attention_mask.len(),
        "Token embedding rows ({}) must match attention-mask length ({})",
        embeddings.nrows(),
        attention_mask.len()
    );

    let selected_rows: Vec<usize> = attention_mask
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value == 1).then_some(index))
        .collect();
    Ok(embeddings.select(Axis(0), &selected_rows))
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use candle_core::{Device, Tensor};
    use ndarray::array;

    use super::{filter_by_attention_mask, l2_normalize_tokens};

    #[test]
    fn normalizes_each_token_independently_and_preserves_zero_rows() -> Result<()> {
        let input = Tensor::new(
            &[[[3_f32, 4.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]]],
            &Device::Cpu,
        )?;

        let rows = l2_normalize_tokens(&input)?.squeeze(0)?.to_vec2::<f32>()?;

        assert!((rows[0][0] - 0.6).abs() < 1e-6);
        assert!((rows[0][1] - 0.8).abs() < 1e-6);
        assert!((rows[1][2] - 1.0).abs() < 1e-6);
        assert_eq!(rows[2], [0.0, 0.0, 0.0]);
        assert!(rows.iter().flatten().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn filters_rows_by_mask_values_instead_of_assuming_right_padding() -> Result<()> {
        let embeddings = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];

        let filtered = filter_by_attention_mask(&embeddings, &[1, 0, 1, 0])?;

        assert_eq!(filtered, array![[1.0, 10.0], [3.0, 30.0]]);
        Ok(())
    }

    #[test]
    fn rejects_attention_masks_with_the_wrong_length() {
        let embeddings = array![[1.0, 2.0], [3.0, 4.0]];

        let error = filter_by_attention_mask(&embeddings, &[1]).unwrap_err();

        assert!(error
            .to_string()
            .contains("must match attention-mask length"));
    }
}
