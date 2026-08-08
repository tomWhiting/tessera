use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

use super::{HfTokenizer, Tokenizer};
use crate::runtime::ResourcePolicy;

fn tokenizer(resource_policy: ResourcePolicy) -> Tokenizer {
    let vocabulary = [
        ("[UNK]".to_string(), 0),
        ("[PAD]".to_string(), 1),
        ("one".to_string(), 2),
        ("two".to_string(), 3),
        ("three".to_string(), 4),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".to_string())
        .build()
        .expect("in-memory tokenizer model should be valid");
    let mut inner = HfTokenizer::new(model);
    inner.with_pre_tokenizer(Some(Whitespace {}));

    Tokenizer {
        inner,
        resource_policy,
    }
}

#[test]
fn sequence_limit_rejects_instead_of_truncating() {
    let tokenizer = tokenizer(ResourcePolicy::new(2, 16, 2048, usize::MAX));

    let error = tokenizer
        .encode("one two three", false)
        .expect_err("three tokens must exceed a two-token limit");

    assert_eq!(
        error.to_string(),
        "Sequence token count 3 exceeds resource policy limit 2"
    );
}

#[test]
fn padded_batch_limit_is_checked_before_padding() {
    let tokenizer = tokenizer(ResourcePolicy::new(3, 2, 5, usize::MAX));

    let error = tokenizer
        .encode_batch(&["one", "one two three"], false)
        .expect_err("two sequences padded to three tokens require six token cells");

    assert_eq!(
        error.to_string(),
        "Padded batch token count 6 exceeds resource policy limit 5"
    );
}

#[test]
fn single_input_obeys_batch_token_limit() {
    let tokenizer = tokenizer(ResourcePolicy::new(3, 1, 2, usize::MAX));

    let error = tokenizer
        .encode("one two three", false)
        .expect_err("a single input still occupies one padded batch");

    assert_eq!(
        error.to_string(),
        "Padded batch token count 3 exceeds resource policy limit 2"
    );
}

#[test]
fn item_limit_is_checked_before_batch_tokenization() {
    let tokenizer = tokenizer(ResourcePolicy::new(3, 2, 6, usize::MAX));

    let error = tokenizer
        .encode_batch(&["one", "two", "three"], false)
        .expect_err("three items must exceed a two-item limit");

    assert_eq!(
        error.to_string(),
        "Batch item count 3 exceeds resource policy limit 2"
    );
}

#[test]
fn exact_batch_boundary_and_empty_batch_are_valid() {
    let bounded_tokenizer = tokenizer(ResourcePolicy::new(3, 2, 6, usize::MAX));
    let batch = bounded_tokenizer
        .encode_batch(&["one", "one two three"], false)
        .expect("exact policy boundary should be accepted");

    assert_eq!(batch.len(), 2);
    assert!(batch.iter().all(|(tokens, _)| tokens.len() == 3));

    let zero_policy_tokenizer = tokenizer(ResourcePolicy::new(0, 0, 0, 0));
    assert!(zero_policy_tokenizer
        .encode_batch(&[], false)
        .expect("empty batches should always be accepted")
        .is_empty());
}

#[test]
fn raw_input_bytes_are_rejected_before_tokenization() {
    let bounded_tokenizer = tokenizer(
        ResourcePolicy::default()
            .with_max_sequence_tokens(3)
            .with_max_input_bytes_per_sequence(3),
    );

    let error = bounded_tokenizer
        .encode("four", false)
        .expect_err("four UTF-8 bytes must exceed a three-byte input limit");

    assert_eq!(
        error.to_string(),
        "Input byte count 4 exceeds resource policy limit 3"
    );
}

#[test]
fn unregistered_tokenizers_are_rejected_before_network_access() {
    let error = Tokenizer::from_pretrained("example/unregistered-tokenizer")
        .err()
        .expect("unregistered tokenizer must fail");

    assert!(error.to_string().contains("not registered"));
}

#[test]
fn tokenizers_without_a_registry_pin_are_rejected_before_network_access() {
    let error = Tokenizer::from_pretrained("jinaai/jina-colbert-v2-96")
        .err()
        .expect("an unpinned tokenizer must fail");

    assert!(error
        .to_string()
        .contains("has no pinned HuggingFace revision"));
}
