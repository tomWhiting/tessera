use super::encode_cut_batch_with;
use crate::core::embeddings::{CutDenseEmbedding, CutEmbeddingOutcome, EmbeddingRefusal};
use crate::core::tokenizer::tests::cut_tokenizer_with_policy;
use crate::runtime::ResourcePolicy;
use ndarray::array;
use std::num::NonZeroUsize;

#[test]
fn mixed_cut_batch_keeps_order_and_refusals_out_of_resource_totals() {
    let policy = ResourcePolicy::new(5, 2, 10, usize::MAX)
        .with_max_input_bytes_per_sequence(20)
        .with_max_job_items(2)
        .with_max_job_input_bytes(20)
        .with_max_output_bytes(8);
    let tokenizer = cut_tokenizer_with_policy(policy);
    let whitespace = " ".repeat(21);
    let texts = [
        "one",
        "",
        " \n\t ",
        "one two three one two",
        "one two three one",
        &whitespace,
        "\u{2003}\u{2003}",
    ];
    let mut embedded = Vec::new();
    let outcomes = encode_cut_batch_with(
        &texts,
        tokenizer.validate_cut_configuration(),
        policy,
        NonZeroUsize::new(2).unwrap(),
        None,
        |accepted| {
            accepted
                .iter()
                .map(|text| {
                    embedded.push(text.to_string());
                    let input = tokenizer.encode_cut(text)?;
                    CutDenseEmbedding::new(
                        array![1.0],
                        input.tokens_read(),
                        input.tokens_total,
                        input.cut,
                    )
                })
                .collect()
        },
    )
    .unwrap();
    assert_eq!(embedded, ["one", "one two three one"]);
    assert_eq!(outcomes.len(), texts.len());
    for (index, read, total, cut) in [(0, 3, 3, false), (4, 5, 6, true)] {
        let CutEmbeddingOutcome::Embedded(value) = &outcomes[index] else {
            panic!("accepted text was refused");
        };
        assert_eq!(value.tokens_read(), read);
        assert_eq!(value.tokens_total(), total);
        assert_eq!(value.cut(), cut);
    }
    for index in [1, 2, 5, 6] {
        let CutEmbeddingOutcome::Refused(refusal) = outcomes[index] else {
            panic!("empty text was embedded");
        };
        assert_eq!(refusal, EmbeddingRefusal::Empty);
        assert_eq!(refusal.code(), "embed_input_empty");
    }
    let CutEmbeddingOutcome::Refused(refusal) = outcomes[3] else {
        panic!("oversized text was embedded");
    };
    assert_eq!(
        refusal,
        EmbeddingRefusal::TooLarge {
            input_bytes: 21,
            limit: 20
        }
    );
    assert_eq!(refusal.code(), "embed_input_too_large");
    assert_whole_call_errors(policy);
}

fn assert_whole_call_errors(policy: ResourcePolicy) {
    for policy in [
        policy.with_max_job_items(1),
        policy.with_max_job_input_bytes(5),
    ] {
        let error = encode_cut_batch_with(
            &["one", "two"],
            Ok(()),
            policy,
            NonZeroUsize::MIN,
            None,
            |_| panic!("job limits must be checked before embedding"),
        )
        .unwrap_err();
        let crate::TesseraError::EncodingError { source, .. } = error else {
            panic!("job limits must fail the whole call");
        };
        assert!(source
            .downcast_ref::<crate::runtime::ResourcePolicyError>()
            .is_some());
    }
    let error = encode_cut_batch_with(&["one"], Ok(()), policy, NonZeroUsize::MIN, None, |_| {
        Err(anyhow::anyhow!("made-up forward failed"))
    })
    .unwrap_err();
    let crate::TesseraError::EncodingError { source, .. } = error else {
        panic!("inference failures must fail the whole call");
    };
    assert_eq!(source.to_string(), "made-up forward failed");
    let error = encode_cut_batch_with(&["one"], Ok(()), policy, NonZeroUsize::MIN, None, |_| {
        Ok(Vec::new())
    })
    .unwrap_err();
    assert!(error.to_string().contains("outcome count mismatch"));
}

#[test]
fn all_refused_cut_batch_needs_no_job_or_embedding_budget() {
    let policy = ResourcePolicy::new(5, 0, 0, usize::MAX)
        .with_max_input_bytes_per_sequence(1)
        .with_max_job_items(0)
        .with_max_job_input_bytes(0)
        .with_max_output_bytes(0);
    let tokenizer = cut_tokenizer_with_policy(policy);
    let outcomes = encode_cut_batch_with(
        &["", " \n\t ", "\u{2003}", "too"],
        tokenizer.validate_cut_configuration(),
        policy,
        NonZeroUsize::MIN,
        None,
        |_| panic!("refused texts must not be tokenized or embedded"),
    )
    .unwrap();
    assert_eq!(outcomes.len(), 4);
    assert!(outcomes
        .iter()
        .all(|outcome| matches!(outcome, CutEmbeddingOutcome::Refused(_))));
    let invalid = cut_tokenizer_with_policy(policy.with_max_sequence_tokens(0));
    let error = encode_cut_batch_with(
        &[""],
        invalid.validate_cut_configuration(),
        policy,
        NonZeroUsize::MIN,
        None,
        |_| panic!("invalid configuration must precede embedding"),
    )
    .unwrap_err();
    assert!(matches!(error, crate::TesseraError::ConfigError(_)));
    assert!(error.to_string().contains("InvalidCutConfiguration"));
}
