use super::*;

#[test]
fn accepts_well_formed_series_embeddings() {
    let embedding = TimeSeriesEmbedding::new(
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        2,
        2,
        Some(vec![32, 48]),
        Some("sensor".to_string()),
    )
    .unwrap();

    assert_eq!(embedding.shape(), (2, 2));
    assert_eq!(embedding.vectors(), &[vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert_eq!(embedding.original_lengths(), Some([32, 48].as_slice()));
    assert_eq!(embedding.source(), Some("sensor"));
}

#[test]
fn rejects_invalid_shape_and_length_metadata() {
    assert!(TimeSeriesEmbedding::new(Vec::new(), 0, 2, None, None).is_err());
    assert!(TimeSeriesEmbedding::new(vec![Vec::new()], 1, 0, None, None).is_err());
    assert!(TimeSeriesEmbedding::new(vec![vec![1.0]], 2, 1, None, None).is_err());
    assert!(TimeSeriesEmbedding::new(vec![vec![1.0]], 1, 2, None, None).is_err());
    assert!(TimeSeriesEmbedding::new(vec![vec![1.0]], 1, 1, Some(Vec::new()), None).is_err());
    assert!(TimeSeriesEmbedding::new(vec![vec![1.0]], 1, 1, Some(vec![0]), None).is_err());
}

#[test]
fn rejects_non_finite_values() {
    assert!(TimeSeriesEmbedding::new(vec![vec![f32::NEG_INFINITY]], 1, 1, None, None).is_err());
}
