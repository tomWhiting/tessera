use super::{probe_span, validate_dense_probe_counts};

fn tokenizer() -> tokenizers::Tokenizer {
    let vocab = [("[UNK]", 0), ("one", 1), ("two", 2), ("three", 3), ("é", 4)]
        .into_iter()
        .map(|(word, id)| (word.to_owned(), id))
        .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tokenizer.with_post_processor(Some(tokenizers::processors::bert::BertProcessing::new(
        ("[SEP]".to_owned(), 11),
        ("[CLS]".to_owned(), 10),
    )));
    tokenizer
}

#[test]
fn constructed_probe_includes_special_tokens_in_the_used_count() {
    let tokenizer = tokenizer();
    let text = probe_span("one two three one", 6, 4, &tokenizer).unwrap();
    assert_eq!(text, "one two");
    let total = tokenizer.encode(text, true).unwrap().len();
    assert!(validate_dense_probe_counts(6, Some(4), total).is_ok());
}

#[test]
fn constructed_probe_refuses_source_and_reencoded_count_mismatches() {
    let tokenizer = tokenizer();
    let error = probe_span("one two three one", 7, 4, &tokenizer)
        .unwrap_err()
        .to_string();
    assert!(error.contains("constructed_probe_source_token_mismatch"));
    let error = validate_dense_probe_counts(6, Some(4), 3)
        .unwrap_err()
        .to_string();
    assert!(error.contains("constructed_probe_token_mismatch"));
    assert!(error.contains("expected=4; observed=3"));
    assert!(probe_span("one two three one", 6, 2, &tokenizer).is_err());
    assert!(probe_span("one two three one", 6, 7, &tokenizer).is_err());
}

#[test]
fn whole_reference_keeps_its_existing_admission() {
    assert!(validate_dense_probe_counts(7, None, 7).is_ok());
    assert!(validate_dense_probe_counts(7, None, 6).is_err());
}

#[test]
fn constructed_probe_uses_utf8_byte_offsets() {
    let tokenizer = tokenizer();
    assert_eq!(
        probe_span("é two three one", 6, 4, &tokenizer).unwrap(),
        "é two"
    );
}
