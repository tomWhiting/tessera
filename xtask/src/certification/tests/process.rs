use std::path::Path;

use crate::certification::evidence::ChildOutcome;

use super::{apply_launcher_result, outcome_path, rss_method, RSS_SAMPLE_INTERVAL_MS};

#[test]
fn launcher_paths_and_rss_labels_are_explicit() {
    assert_eq!(RSS_SAMPLE_INTERVAL_MS, 50);
    assert_eq!(rss_method(0), "unavailable");
    assert_eq!(rss_method(1), "sampled-process-rss-watchdog");
    assert_eq!(
        outcome_path(Path::new("/repo"), "model", 1, 42),
        Path::new("/repo/.tessera/cert-evidence/model/.42-run-1.outcome")
    );
}

#[test]
fn launcher_failure_overrides_a_child_written_pass() {
    let mut outcome = ChildOutcome {
        status: "passed".to_string(),
        error: None,
        verified_artifacts: Vec::new(),
        observation: None,
    };

    apply_launcher_result(
        &mut outcome,
        Some("sampled RSS exceeded limit"),
        false,
        "signal: 9",
    );

    assert_eq!(outcome.status, "failed");
    assert_eq!(outcome.error.as_deref(), Some("sampled RSS exceeded limit"));
}

#[test]
fn launcher_failure_preserves_both_monitor_and_child_errors() {
    let mut outcome = ChildOutcome {
        status: "failed".to_string(),
        error: Some("smoke contract failed".to_string()),
        verified_artifacts: Vec::new(),
        observation: None,
    };

    apply_launcher_result(
        &mut outcome,
        Some("child exceeded timeout"),
        false,
        "signal: 9",
    );

    assert_eq!(outcome.status, "failed");
    assert_eq!(
        outcome.error.as_deref(),
        Some("child exceeded timeout; child reported: smoke contract failed")
    );
}
