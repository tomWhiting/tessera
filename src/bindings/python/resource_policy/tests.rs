use super::{validate_batch_items, validate_text_pair_values, validate_text_value};
use crate::runtime::ResourcePolicy;

#[test]
fn text_preflight_rejects_empty_and_oversized_values() {
    let policy = ResourcePolicy::default().with_max_input_bytes_per_sequence(3);

    assert_eq!(
        validate_text_value("", "query", policy).expect_err("empty query must fail"),
        "query must not be empty"
    );
    assert!(validate_text_value("four", "query", policy)
        .expect_err("oversized query must fail")
        .contains("Input byte count 4 exceeds resource policy limit 3"));
    validate_text_value("yes", "query", policy).expect("exact byte limit should pass");
}

#[test]
fn batch_preflight_runs_before_python_vector_allocation() {
    let policy = ResourcePolicy::default()
        .with_max_batch_items(2)
        .with_max_job_items(3);

    assert!(validate_batch_items(0, policy)
        .expect_err("empty Python batch must fail")
        .contains("at least one"));
    validate_batch_items(2, policy).expect("batch limit should be inclusive");
    assert!(validate_batch_items(3, policy)
        .expect_err("batch item limit must fail")
        .contains("Batch item count 3 exceeds resource policy limit 2"));
}

#[test]
fn paired_scoring_preflight_enforces_aggregate_job_bytes() {
    let policy = ResourcePolicy::default()
        .with_max_input_bytes_per_sequence(4)
        .with_max_job_input_bytes(5);

    validate_text_pair_values("ab", "left", "cde", "right", policy)
        .expect("aggregate job limit should be inclusive");
    assert!(
        validate_text_pair_values("abc", "left", "def", "right", policy)
            .expect_err("aggregate scoring input must be bounded")
            .contains("Job input byte count 6 exceeds resource policy limit 5")
    );
}
