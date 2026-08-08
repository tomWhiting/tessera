use std::num::NonZeroUsize;

use super::{ResourcePolicy, ResourcePolicyError};

#[test]
fn conservative_defaults_are_stable() {
    let policy = ResourcePolicy::default();

    assert_eq!(policy.max_sequence_tokens(), 512);
    assert_eq!(policy.max_batch_items(), 16);
    assert_eq!(policy.max_batch_tokens(), 2048);
    assert_eq!(policy.max_model_bytes(), 2 * 1024 * 1024 * 1024);
    assert_eq!(policy.max_input_bytes_per_sequence(), 1024 * 1024);
    assert_eq!(policy.max_attention_cells(), 1_048_576);
}

#[test]
fn exact_limits_are_accepted() {
    let policy = ResourcePolicy::default();

    assert_eq!(policy.validate_sequence(512), Ok(()));
    assert_eq!(policy.validate_batch(16, 128), Ok(()));
    assert_eq!(policy.validate_model_context(512), Ok(()));
}

#[test]
fn conservative_batch_size_accounts_for_worst_case_padding() {
    let policy = ResourcePolicy::default();

    assert_eq!(
        policy.conservative_batch_size().map(NonZeroUsize::get),
        Some(4)
    );
    assert_eq!(
        policy.with_max_batch_tokens(511).conservative_batch_size(),
        None
    );
    assert_eq!(
        policy.with_max_sequence_tokens(0).conservative_batch_size(),
        None
    );
    assert_eq!(
        policy
            .with_max_attention_cells(262_143)
            .conservative_batch_size(),
        None
    );
}

#[test]
fn over_limit_errors_report_measured_and_allowed_counts() {
    let policy = ResourcePolicy::default();

    assert_eq!(
        policy.validate_sequence(513),
        Err(ResourcePolicyError::SequenceTokens {
            measured: 513,
            allowed: 512,
        })
    );
    assert_eq!(
        policy.validate_batch(17, 1),
        Err(ResourcePolicyError::BatchItems {
            measured: 17,
            allowed: 16,
        })
    );
    assert_eq!(
        policy.validate_batch(16, 129),
        Err(ResourcePolicyError::BatchTokens {
            measured: 2064,
            allowed: 2048,
        })
    );
    assert_eq!(
        policy.validate_input_bytes(1_048_577),
        Err(ResourcePolicyError::InputBytes {
            measured: 1_048_577,
            allowed: 1_048_576,
        })
    );
    assert_eq!(
        policy.with_max_batch_tokens(2560).validate_batch(5, 512),
        Err(ResourcePolicyError::AttentionCells {
            measured: 1_310_720,
            allowed: 1_048_576,
        })
    );
}

#[test]
fn empty_batches_are_valid_even_with_zero_limits() {
    let policy = ResourcePolicy::new(0, 0, 0, 0);

    assert_eq!(policy.validate_batch(0, 0), Ok(()));
}

#[test]
fn f32_model_estimate_is_checked_before_loading() {
    let policy = ResourcePolicy::default();

    assert_eq!(policy.validate_model_parameters("109M", 4), Ok(436_000_000));
    assert_eq!(
        policy.validate_model_parameters("3B", 4),
        Err(ResourcePolicyError::ModelBytes {
            measured: 12_000_000_000,
            allowed: 2 * 1024 * 1024 * 1024,
        })
    );

    let raised = policy.with_max_model_bytes(12_000_000_000);
    assert_eq!(
        raised.validate_model_parameters("3B", 4),
        Ok(12_000_000_000)
    );
    assert_eq!(
        raised.validate_model_parameters("1.05B", 4),
        Ok(4_200_000_000)
    );
}

#[test]
fn model_context_cannot_be_exceeded() {
    let policy = ResourcePolicy::default().with_max_sequence_tokens(513);

    assert_eq!(
        policy.validate_model_context(512),
        Err(ResourcePolicyError::ModelContext {
            measured: 513,
            allowed: 512,
        })
    );
}
