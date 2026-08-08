use super::*;

#[test]
fn projected_shape_validation_reports_measured_and_expected() {
    let tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
    let error = tensor_to_finite_flat(&tensor, 2, 4, "query projection").unwrap_err();

    assert!(error.to_string().contains("shape [2, 3]"));
    assert!(error.to_string().contains("expected [2, 4]"));
}

#[test]
fn projected_values_must_be_finite() {
    let tensor = Tensor::from_vec(vec![0.0_f32, 1.0, f32::NAN, 3.0], (2, 2), &Device::Cpu).unwrap();
    let error = tensor_to_finite_flat(&tensor, 2, 2, "image projection").unwrap_err();

    assert!(error.to_string().contains("non-finite value NaN"));
    assert!(error.to_string().contains("row 1, column 0"));
}

#[test]
fn token_tensor_rejects_an_empty_layout() {
    let error = token_ids_tensor(&[], &Device::Cpu).unwrap_err();

    assert!(error.to_string().contains("token layout is empty"));
}

#[test]
fn normalization_rejects_zero_and_overflowing_norms() {
    let zero_error = normalize_rows(&mut [0.0_f32, 0.0], 1, 2, "projection").unwrap_err();
    assert!(zero_error.to_string().contains("invalid L2 norm 0"));

    let overflow_error = normalize_rows(&mut [f32::MAX, f32::MAX], 1, 2, "projection").unwrap_err();
    assert!(overflow_error
        .to_string()
        .contains("non-finite squared norm"));
}

#[test]
fn normalization_matches_upstream_rowwise_division() {
    let mut values = [3.0_f32, 4.0, 0.0, 2.0];
    normalize_rows(&mut values, 2, 2, "projection").unwrap();

    assert!((values[0] - 0.6).abs() < f32::EPSILON);
    assert!((values[1] - 0.8).abs() < f32::EPSILON);
    assert!(values[2].abs() < f32::EPSILON);
    assert!((values[3] - 1.0).abs() < f32::EPSILON);
}

#[test]
fn forward_preflight_rechecks_activation_for_long_queries() {
    let profile = TransformerProfile::new(2_048, 16_384, 8).unwrap();
    let image_tokens = 1_030;
    let query_tokens = 2_048;
    let image_activation_bytes =
        usize::try_from(profile.peak_bytes(1, image_tokens, ModelDType::F32)).unwrap();
    let policy = ResourcePolicy::new(query_tokens, 1, query_tokens, usize::MAX)
        .with_max_attention_cells(query_tokens * query_tokens)
        .with_max_activation_bytes(image_activation_bytes);

    validate_forward_resources(
        &policy,
        profile,
        ModelDType::F32,
        image_tokens,
        "ColPali image",
    )
    .unwrap();
    let error = validate_forward_resources(
        &policy,
        profile,
        ModelDType::F32,
        query_tokens,
        "ColPali query",
    )
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("ColPali query activation preflight failed"));
    assert!(error.to_string().contains("Estimated activation bytes"));
    assert!(error
        .to_string()
        .contains(&image_activation_bytes.to_string()));
}

#[test]
fn forward_preflight_rechecks_sequence_and_batch_limits() {
    let profile = TransformerProfile::new(8, 32, 2).unwrap();
    let sequence_policy = ResourcePolicy::new(3, 1, 8, usize::MAX);
    let sequence_error = validate_forward_resources(
        &sequence_policy,
        profile,
        ModelDType::F32,
        4,
        "ColPali query",
    )
    .unwrap_err();
    assert!(sequence_error
        .to_string()
        .contains("ColPali query sequence preflight failed"));

    let batch_policy = ResourcePolicy::new(8, 1, 3, usize::MAX);
    let batch_error =
        validate_forward_resources(&batch_policy, profile, ModelDType::F32, 4, "ColPali image")
            .unwrap_err();
    assert!(batch_error
        .to_string()
        .contains("ColPali image batch preflight failed"));
}
