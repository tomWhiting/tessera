use super::encode_outcome_batch_with;
use crate::core::embeddings::{CountedDenseEmbedding, EmbeddingOutcome, EmbeddingRefusal};
use crate::core::tokenizer::tests::cut_tokenizer_with_policy;
use crate::runtime::ResourcePolicy;
use ndarray::array;
use std::cell::Cell;
use std::num::NonZeroUsize;

#[test]
fn accepted_outcome_source_is_tokenized_once_and_keeps_joined_tokens() {
    let policy = ResourcePolicy::new(16, 2, 32, usize::MAX);
    let tokenizer = cut_tokenizer_with_policy(policy);
    for (prompt, text, expected_ids) in [
        ("", "one", vec![10, 2, 11]),
        ("three ", "one", vec![10, 4, 2, 11]),
        ("on", "e two", vec![10, 2, 3, 11]),
    ] {
        let source_tokenizations = Cell::new(0);
        let encode_source = |text| {
            source_tokenizations.set(source_tokenizations.get() + 1);
            tokenizer.encode_with_prompt(prompt, text)
        };
        let outcomes = encode_outcome_batch_with(
            &[text],
            (
                tokenizer.validate_cut_configuration_with(&[prompt]),
                |text| match encode_source(text) {
                    Ok(_) => Ok(None),
                    Err(error) => error
                        .downcast_ref::<EmbeddingRefusal>()
                        .copied()
                        .map_or_else(|| Err(error), |refusal| Ok(Some(refusal))),
                },
            ),
            policy,
            1,
            NonZeroUsize::MIN,
            None,
            |accepted| {
                accepted
                    .iter()
                    .map(|text| {
                        let input = encode_source(text)?;
                        assert_eq!(input.token_ids, expected_ids);
                        assert_eq!(input.attention_mask, vec![1; expected_ids.len()]);
                        CountedDenseEmbedding::new(array![1.0], input.tokens_total)
                    })
                    .collect()
            },
        )
        .unwrap();
        let EmbeddingOutcome::Embedded(output) = &outcomes[0] else {
            panic!("accepted source was refused");
        };
        assert_eq!(output.tokens_total(), expected_ids.len());
        assert_eq!(source_tokenizations.get(), 1);
    }
}

#[test]
fn later_outcome_classification_error_prevents_every_forward() {
    let policy = ResourcePolicy::new(16, 2, 32, usize::MAX);
    let error = encode_outcome_batch_with(
        &["one", "two", "three"],
        (
            Ok(()),
            |text| {
                if text == "three" {
                    Err(anyhow::anyhow!("source tokenization failed"))
                } else {
                    Ok(None)
                }
            },
        ),
        policy,
        1,
        NonZeroUsize::MIN,
        None,
        |_| panic!("every source must be classified before any forward"),
    )
    .unwrap_err();
    let crate::TesseraError::EncodingError { context, source } = error else {
        panic!("classification error was not propagated");
    };
    assert_eq!(context, "Failed to classify input 2");
    assert_eq!(source.to_string(), "source tokenization failed");
}
