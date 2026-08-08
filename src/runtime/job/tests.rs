use super::{f32_output_bytes, JobTracker};
use crate::runtime::{ResourcePolicy, ResourcePolicyError};

#[test]
fn tracker_rejects_work_before_mutating_counters() {
    let policy = ResourcePolicy::default()
        .with_max_job_items(1)
        .with_max_job_input_bytes(4);
    let mut tracker = JobTracker::new(policy);

    tracker.admit_input(4).expect("exact limit should pass");
    assert!(matches!(
        tracker.admit_input(1),
        Err(ResourcePolicyError::JobItems { .. })
    ));
}

#[test]
fn collected_and_streamed_output_have_distinct_accounting() {
    let policy = ResourcePolicy::default().with_max_output_bytes(8);
    let mut tracker = JobTracker::new(policy);

    tracker.retain_output(4).expect("first result should fit");
    tracker.retain_output(4).expect("exact total should fit");
    assert!(tracker.retain_output(1).is_err());

    let stream_tracker = JobTracker::new(policy);
    stream_tracker
        .validate_streamed_output(8)
        .expect("one streamed item should fit");
    stream_tracker
        .validate_streamed_output(8)
        .expect("a consumed item is not retained");
}

#[test]
fn f32_byte_estimate_saturates_instead_of_wrapping() {
    assert_eq!(f32_output_bytes(2), 8);
    assert_eq!(f32_output_bytes(usize::MAX), usize::MAX);
}
