use super::{max_pool_token_logits, splade_transform};
use anyhow::Result;
use candle_core::{Device, Tensor};

fn deterministic_logits(sequence_length: usize, vocab_size: usize, seed: usize) -> Result<Tensor> {
    let values = (0..sequence_length * vocab_size)
        .map(|index| {
            let mixed = (index * 37 + seed * 19) % 127;
            (mixed as f32 - 63.0) / 11.0
        })
        .collect::<Vec<_>>();
    Ok(Tensor::from_vec(
        values,
        (sequence_length, vocab_size),
        &Device::Cpu,
    )?)
}

fn naive_transform_then_pool(logits: &Tensor, attention_mask: &[u32]) -> Result<Vec<f32>> {
    let transformed = logits.relu()?.affine(1.0, 1.0)?.log()?;
    let rows = transformed.to_vec2::<f32>()?;
    let vocab_size = logits.dims()[1];
    let mut pooled = vec![f32::NEG_INFINITY; vocab_size];
    let mut has_valid_row = false;

    for (token_index, row) in rows.iter().enumerate() {
        if attention_mask
            .get(token_index)
            .is_none_or(|&value| value == 0)
        {
            continue;
        }

        has_valid_row = true;
        for (maximum, &value) in pooled.iter_mut().zip(row) {
            *maximum = maximum.max(value);
        }
    }

    if !has_valid_row {
        pooled.fill(0.0);
    }

    Ok(pooled)
}

fn optimized_pool_then_transform(logits: &Tensor, attention_mask: &[u32]) -> Result<Vec<f32>> {
    let vocab_size = logits.dims()[1];
    let pooled_logits = max_pool_token_logits(logits, attention_mask, vocab_size)?;
    Ok(splade_transform(&pooled_logits)?.to_vec1::<f32>()?)
}

fn assert_vectors_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "value {index} differs: pooled-then-transform={actual}, transform-then-pool={expected}"
        );
    }
}

#[test]
fn pool_then_transform_matches_naive_reference_across_masks() -> Result<()> {
    let cases = [
        (1, 1, vec![1]),
        (5, 7, vec![1, 1, 1, 1, 1]),
        (5, 7, vec![1, 0, 1, 0, 1]),
        (5, 7, vec![0, 1, 1, 0, 0]),
        (5, 7, vec![2, 0, 7]),
        (3, 7, vec![0, 0, 0, 1]),
        (4, 5, vec![]),
        (0, 7, vec![1]),
    ];

    for (case_index, (sequence_length, vocab_size, attention_mask)) in cases.into_iter().enumerate()
    {
        let logits = deterministic_logits(sequence_length, vocab_size, case_index + 1)?;
        let expected = naive_transform_then_pool(&logits, &attention_mask)?;
        let actual = optimized_pool_then_transform(&logits, &attention_mask)?;
        assert_vectors_close(&actual, &expected);
    }

    Ok(())
}

#[test]
fn masked_high_logits_do_not_affect_pooling() -> Result<()> {
    let logits = Tensor::new(
        &[
            [1.0_f32, -2.0, 0.5, -4.0],
            [100.0, 100.0, 100.0, 100.0],
            [0.25, 3.0, -1.0, -2.0],
        ],
        &Device::Cpu,
    )?;
    let attention_mask = [1, 0, 1];

    let expected = naive_transform_then_pool(&logits, &attention_mask)?;
    let actual = optimized_pool_then_transform(&logits, &attention_mask)?;
    assert_vectors_close(&actual, &expected);

    Ok(())
}

#[test]
fn max_pool_token_logits_validates_shape_and_vocabulary() -> Result<()> {
    let one_dimensional = Tensor::new(&[1.0_f32, 2.0], &Device::Cpu)?;
    let shape_error = max_pool_token_logits(&one_dimensional, &[1], 2).unwrap_err();
    assert!(shape_error
        .to_string()
        .contains("Expected 2D tensor [seq_len, vocab_size]"));

    let logits = deterministic_logits(2, 3, 1)?;
    let vocab_error = max_pool_token_logits(&logits, &[1, 1], 4).unwrap_err();
    assert!(vocab_error
        .to_string()
        .contains("Vocabulary size mismatch: expected 4, got 3"));

    Ok(())
}
