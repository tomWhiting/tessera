use crate::certification::artifacts::VerifiedArtifact;

use super::{eligible_current_cohort, EvidenceSummary, PeakRssSummary};

fn evidence(commit: &str) -> EvidenceSummary {
    EvidenceSummary {
        schema_version: 1,
        model_id: "model".to_string(),
        spec_sha256: "digest".to_string(),
        profile: "smoke".to_string(),
        device: "cpu".to_string(),
        status: "passed".to_string(),
        source_commit: commit.to_string(),
        source_dirty: false,
        peak_rss: PeakRssSummary {
            enforced: true,
            bytes: Some(10),
        },
        verified_artifacts: vec![VerifiedArtifact {
            path: "model.safetensors".to_string(),
            size_bytes: 10,
            sha256: "a".repeat(64),
        }],
    }
}

#[test]
fn dirty_history_does_not_poison_a_clean_cohort() {
    let mut dirty = evidence("old");
    dirty.source_dirty = true;
    let clean_one = evidence("current");
    let clean_two = evidence("current");
    let history = [&dirty, &clean_one, &clean_two];

    assert_eq!(
        eligible_current_cohort(&history, "current", true, true),
        (Some("current".to_string()), 2)
    );
}

#[test]
fn older_commit_history_cannot_certify_current_head() {
    let old_one = evidence("old");
    let old_two = evidence("old");
    let old_three = evidence("old");
    let current_one = evidence("current");
    let current_two = evidence("current");
    let history = [&old_one, &old_two, &old_three, &current_one, &current_two];
    assert_eq!(
        eligible_current_cohort(&history, "current", true, true),
        (Some("current".to_string()), 2)
    );

    let split = [&old_one, &old_two, &old_three, &current_one];
    assert_eq!(
        eligible_current_cohort(&split, "current", true, true).1,
        1,
        "a larger old cohort must not certify the current source commit"
    );
}
