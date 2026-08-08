use super::*;

#[test]
fn maxsim_adapter_retains_prompt_suffix_vectors() {
    let document = VisionEmbedding::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]],
        2,
        2,
        Some("page.png".to_string()),
    )
    .unwrap();

    let adapted = document_token_embeddings(&document).unwrap();

    assert_eq!(document.num_patches(), 2);
    assert_eq!(document.num_vectors(), 3);
    assert_eq!(adapted.shape(), (3, 2));
    assert_eq!(adapted.text(), "page.png");
}
