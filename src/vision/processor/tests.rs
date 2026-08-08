use super::*;

#[test]
fn image_layout_has_exact_placeholder_count_and_official_suffix() {
    let prompt = render_full_image_prompt(1_024);
    let prefix_bytes = IMAGE_TOKEN.len() * 1_024;

    assert_eq!(&prompt[..prefix_bytes], IMAGE_TOKEN.repeat(1_024));
    assert_eq!(&prompt[prefix_bytes..], "<bos>Describe the image.\n");
    assert_eq!(prompt.matches(IMAGE_TOKEN).count(), 1_024);
}

#[test]
fn candle_image_suffix_omits_placeholder_tokens() {
    let suffix = render_image_suffix();

    assert_eq!(suffix, "<bos>Describe the image.\n");
    assert!(!suffix.contains(IMAGE_TOKEN));
}

#[test]
fn query_layout_uses_ten_active_augmentation_tokens() {
    let prompt = render_query("How much is due?");

    assert_eq!(
        prompt,
        "<bos>Question: How much is due?<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>\n"
    );
    assert_eq!(prompt.matches(PAD_TOKEN).count(), 10);
}

#[test]
fn token_layout_rejects_masked_augmentation_tokens() {
    let error = validate_token_layout("query", &[1, 2, 3], &[1, 1, 0]).unwrap_err();

    assert!(error
        .to_string()
        .contains("requires every prompt and augmentation token to be active"));
}
