use super::*;

#[test]
fn accepts_well_formed_sequence_embeddings() {
    let embedding = VisionEmbedding::new(
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        2,
        2,
        Some("page.png".to_string()),
    )
    .unwrap();

    assert_eq!(embedding.shape(), (2, 2));
    assert_eq!(embedding.num_patches(), 2);
    assert_eq!(embedding.num_vectors(), 2);
    assert_eq!(embedding.source(), Some("page.png"));
}

#[test]
fn retains_prompt_rows_separately_from_physical_patches() {
    let embedding =
        VisionEmbedding::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]], 2, 1, None).unwrap();

    assert_eq!(embedding.num_patches(), 2);
    assert_eq!(embedding.num_vectors(), 4);
    assert_eq!(embedding.shape(), (4, 1));
    assert_eq!(embedding.vectors().len(), 4);
}

#[test]
fn rejects_zero_or_inconsistent_dimensions() {
    assert!(VisionEmbedding::new(Vec::new(), 0, 2, None).is_err());
    assert!(VisionEmbedding::new(vec![Vec::new()], 1, 0, None).is_err());
    assert!(VisionEmbedding::new(vec![vec![1.0]], 2, 1, None).is_err());
    assert!(VisionEmbedding::new(vec![vec![1.0]], 1, 2, None).is_err());
}

#[test]
fn rejects_non_finite_values() {
    assert!(VisionEmbedding::new(vec![vec![f32::NAN]], 1, 1, None).is_err());
    assert!(VisionEmbedding::new(vec![vec![f32::INFINITY]], 1, 1, None).is_err());
}
