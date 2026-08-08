use super::flatten_attention_mask;
use crate::backends::candle::encoder::role::PreparedInput;

fn input(attention_mask: Vec<u32>) -> PreparedInput {
    let length = attention_mask.len();
    PreparedInput {
        token_ids: vec![0; length],
        output_mask: attention_mask.clone(),
        attention_mask,
    }
}

#[test]
fn bert_attention_keeps_one_for_attended_tokens() {
    let inputs = [input(vec![1, 1, 0]), input(vec![1, 0, 0])];

    assert_eq!(flatten_attention_mask(&inputs, false), [1, 1, 0, 1, 0, 0]);
}

#[test]
fn distilbert_attention_is_inverted_for_masked_fill() {
    let inputs = [input(vec![1, 1, 0]), input(vec![1, 0, 0])];

    assert_eq!(flatten_attention_mask(&inputs, true), [0, 0, 1, 0, 1, 1]);
}
