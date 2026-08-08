use super::{validate_finite_matrices, validate_matrix_shapes};
use crate::runtime::ResourcePolicy;

#[test]
fn matrix_preflight_rejects_zero_width_and_dimension_mismatch() {
    let policy = ResourcePolicy::default();
    assert!(validate_matrix_shapes(&[2, 0], &[4, 128], 128, policy)
        .expect_err("zero-width query must fail")
        .contains("non-zero"));
    assert!(validate_matrix_shapes(&[2, 128], &[4, 64], 128, policy)
        .expect_err("mismatched matrices must fail")
        .contains("model dimension"));
}

#[test]
fn matrix_preflight_rejects_invalid_rank_without_indexing() {
    let policy = ResourcePolicy::default();
    assert!(validate_matrix_shapes(&[], &[4, 128], 128, policy)
        .expect_err("missing query axes must fail")
        .contains("two-dimensional"));
    assert!(validate_matrix_shapes(&[2, 128], &[4], 128, policy)
        .expect_err("missing document axis must fail")
        .contains("two-dimensional"));
}

#[test]
fn matrix_preflight_rejects_non_finite_values() {
    validate_finite_matrices(&[0.0, 1.0], &[2.0, 3.0]).expect("finite matrices should pass");
    assert!(validate_finite_matrices(&[f32::NAN], &[0.0]).is_err());
    assert!(validate_finite_matrices(&[0.0], &[f32::INFINITY]).is_err());
}

#[test]
fn matrix_preflight_bounds_rows_and_total_elements() {
    let policy = ResourcePolicy::default()
        .with_max_sequence_tokens(2)
        .with_max_job_items(4)
        .with_max_output_bytes(3_072);
    validate_matrix_shapes(&[2, 128], &[4, 128], 128, policy).expect("exact limits should pass");
    assert!(validate_matrix_shapes(&[3, 128], &[4, 128], 128, policy).is_err());
    assert!(validate_matrix_shapes(&[2, 128], &[5, 128], 128, policy).is_err());
    assert!(validate_matrix_shapes(
        &[2, 128],
        &[4, 128],
        128,
        policy.with_max_output_bytes(3_071)
    )
    .is_err());
}
