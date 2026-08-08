use super::{common_source_commit, EvidenceSummary, PeakRssSummary};

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
    }
}

#[test]
fn readiness_requires_one_shared_source_commit() {
    assert_eq!(
        common_source_commit(&[evidence("abc"), evidence("abc")]),
        Some("abc".to_string())
    );
    assert_eq!(
        common_source_commit(&[evidence("abc"), evidence("def")]),
        None
    );
}
