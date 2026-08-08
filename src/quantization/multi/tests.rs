use super::*;
use crate::quantization::{BinaryQuantization, Quantization};

#[test]
fn quantization_requires_nonempty_consistent_vectors() {
    let quantizer = BinaryQuantization::new();

    assert!(quantize_multi(&quantizer, &[]).is_err());
    assert!(quantize_multi(&quantizer, &[Vec::new()]).is_err());
    assert!(quantize_multi(&quantizer, &[vec![1.0], vec![1.0, 2.0]]).is_err());
}

#[test]
fn distance_computes_max_sim() {
    let quantizer = BinaryQuantization::new();
    let query = quantize_multi(&quantizer, &[vec![1.0, 1.0], vec![-1.0, 1.0]]).unwrap();
    let document = quantize_multi(&quantizer, &[vec![1.0, -1.0], vec![-1.0, 1.0]]).unwrap();

    let score = multi_vector_distance(&quantizer, &query, &document).unwrap();
    assert!((score - 3.0).abs() < f32::EPSILON);
}

#[test]
fn distance_rejects_empty_inputs() {
    let quantizer = BinaryQuantization::new();
    let vector = quantizer.quantize_vector(&[1.0]).unwrap();

    assert!(multi_vector_distance(&quantizer, &[], std::slice::from_ref(&vector)).is_err());
    assert!(multi_vector_distance(&quantizer, std::slice::from_ref(&vector), &[]).is_err());
}
