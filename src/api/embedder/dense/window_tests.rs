use crate::core::tokenizer::tests::cut_tokenizer;
use crate::{ContextWindowConfig, TesseraDense};

#[test]
fn dense_window_path_keeps_every_content_token_and_source_span() {
    let tokenizer = cut_tokenizer(5);
    let text = "one two three one";
    let (total, windows) = tokenizer
        .encode_spanned_windows("", text, ContextWindowConfig::new(5, 1))
        .unwrap();
    assert_eq!(total, 4);
    assert_eq!(windows.len(), 2);
    assert_eq!(windows[0].byte_start, 0);
    assert_eq!(windows[0].byte_end, 13);
    assert_eq!(windows[1].byte_start, 8);
    assert_eq!(windows[1].byte_end, text.len());
    assert_eq!(windows[0].window.token_ids, [10, 2, 3, 4, 11]);
    assert_eq!(windows[1].window.token_ids, [10, 4, 2, 11]);
    std::hint::black_box(TesseraDense::encode_windows);
    std::hint::black_box(TesseraDense::encode_batch_windows);
}
