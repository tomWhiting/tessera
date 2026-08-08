use super::*;

#[test]
fn derives_metadata_from_valid_vectors() {
    let embeddings = QuantizedEmbeddings::new(vec![
        BinaryVector::from_packed(vec![0b0000_0101], 3).unwrap(),
        BinaryVector::from_packed(vec![0b0000_0010], 3).unwrap(),
    ])
    .unwrap();

    assert_eq!(embeddings.num_tokens(), 2);
    assert_eq!(embeddings.original_dim(), 3);
    assert_eq!(embeddings.vectors().len(), 2);
    assert_eq!(embeddings.memory_bytes(), 2);
    assert!((embeddings.compression_ratio() - 12.0).abs() < f32::EPSILON);
}

#[test]
fn rejects_empty_or_mixed_dimension_collections() {
    assert!(QuantizedEmbeddings::new(Vec::new()).is_err());

    let mixed = vec![
        BinaryVector::from_packed(vec![0], 3).unwrap(),
        BinaryVector::from_packed(vec![0], 4).unwrap(),
    ];
    assert!(QuantizedEmbeddings::new(mixed).is_err());
}
