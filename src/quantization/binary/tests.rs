use super::*;

fn assert_close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < f32::EPSILON);
}

#[test]
fn quantizes_and_dequantizes_a_single_vector() {
    let quantizer = BinaryQuantization::new();
    let binary = quantizer.quantize_vector(&[0.5, -0.3, 0.8, -0.1]).unwrap();

    assert_eq!(binary.dim(), 4);
    assert_eq!(binary.packed(), &[0b0101]);
    assert_eq!(
        quantizer.dequantize_vector(&binary).unwrap(),
        vec![1.0, -1.0, 1.0, -1.0]
    );
}

#[test]
fn hamming_similarity_handles_identical_opposite_and_partial_vectors() {
    let quantizer = BinaryQuantization::new();
    let left = quantizer.quantize_vector(&[1.0, 1.0, -1.0, -1.0]).unwrap();
    let partial = quantizer.quantize_vector(&[1.0, -1.0, -1.0, 1.0]).unwrap();
    let opposite = quantizer.quantize_vector(&[-1.0, -1.0, 1.0, 1.0]).unwrap();

    assert_close(quantizer.distance(&left, &left).unwrap(), 4.0);
    assert_close(quantizer.distance(&left, &partial).unwrap(), 2.0);
    assert_close(quantizer.distance(&left, &opposite).unwrap(), 0.0);
}

#[test]
fn preserves_non_byte_aligned_dimensions() {
    let quantizer = BinaryQuantization::new();
    let binary = quantizer.quantize_vector(&[0.5, -0.3, 0.8]).unwrap();

    assert_eq!(binary.dim(), 3);
    assert_eq!(binary.memory_bytes(), 1);
    assert_eq!(
        quantizer.dequantize_vector(&binary).unwrap(),
        vec![1.0, -1.0, 1.0]
    );
}

#[test]
fn rejects_empty_and_non_finite_float_vectors() {
    let quantizer = BinaryQuantization::new();

    assert!(quantizer.quantize_vector(&[]).is_err());
    assert!(quantizer.quantize_vector(&[f32::NAN]).is_err());
    assert!(quantizer.quantize_vector(&[f32::INFINITY]).is_err());
}

#[test]
fn zero_values_use_the_non_positive_bit() {
    let quantizer = BinaryQuantization::new();
    let binary = quantizer.quantize_vector(&[0.0, -0.0]).unwrap();

    assert_eq!(binary.packed(), &[0]);
}

#[test]
fn packed_constructor_rejects_noncanonical_representations() {
    assert!(BinaryVector::from_packed(Vec::new(), 0).is_err());
    assert!(BinaryVector::from_packed(Vec::new(), 1).is_err());
    assert!(BinaryVector::from_packed(vec![0], 9).is_err());
    assert!(BinaryVector::from_packed(vec![0b1000_0000], 3).is_err());

    let valid = BinaryVector::from_packed(vec![0b0000_0101], 3).unwrap();
    assert_eq!(valid.dim(), 3);
}

#[test]
fn distance_rejects_dimension_mismatch() {
    let quantizer = BinaryQuantization::new();
    let two = quantizer.quantize_vector(&[1.0, -1.0]).unwrap();
    let three = quantizer.quantize_vector(&[1.0, -1.0, 1.0]).unwrap();

    assert!(quantizer.distance(&two, &three).is_err());
}
