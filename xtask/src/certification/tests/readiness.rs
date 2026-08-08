use crate::certification::artifacts::VerifiedArtifact;
use crate::certification::reference::{
    compare, LoadedReference, ReferenceComparison, ReferenceDocument, ReferencePointer,
};
use crate::certification::spec::{
    self, CapabilityScope, CertificationDevice, CertificationDtype, SemanticMode,
};

use super::{eligible_current_cohort, evidence_matches, EvidenceSummary, PeakRssSummary};

fn evidence(commit: &str) -> EvidenceSummary {
    EvidenceSummary {
        schema_version: 1,
        model_id: "model".to_string(),
        spec_sha256: "digest".to_string(),
        profile: "smoke".to_string(),
        device: "cpu".to_string(),
        capability: CapabilityScope {
            device: CertificationDevice::Cpu,
            dtype: CertificationDtype::F32,
            semantic_mode: SemanticMode::Query,
            max_sequence_tokens: 8,
            context_window_tokens: 8,
        },
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
        reference_comparison: ReferenceComparison::not_configured(),
    }
}

#[test]
fn readiness_requires_a_passed_comparison_to_the_exact_checked_reference() {
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let mut loaded = spec::load_model(repository, "bge-base-en-v1.5").unwrap();
    let pointer = ReferencePointer {
        path: "official.json".to_string(),
        sha256: "a".repeat(64),
    };
    let reference_document: ReferenceDocument = serde_json::from_slice(
        &std::fs::read(repository.join("certification/references/contract/dense.json")).unwrap(),
    )
    .unwrap();
    let reference = LoadedReference {
        path: pointer.path.clone(),
        sha256: pointer.sha256.clone(),
        document: reference_document,
    };
    loaded
        .spec
        .profiles
        .get_mut("smoke")
        .unwrap()
        .official_reference = Some(pointer.clone());
    let profile = loaded.spec.profile("smoke").unwrap();
    let mut entry = EvidenceSummary {
        schema_version: 2,
        model_id: loaded.spec.model.id.clone(),
        spec_sha256: loaded.sha256.clone(),
        profile: "smoke".to_string(),
        device: "cpu".to_string(),
        capability: profile.capability.clone(),
        status: "passed".to_string(),
        source_commit: "current".to_string(),
        source_dirty: false,
        peak_rss: PeakRssSummary {
            enforced: true,
            bytes: Some(10),
        },
        verified_artifacts: loaded
            .spec
            .artifacts
            .iter()
            .map(|artifact| VerifiedArtifact {
                path: artifact.path.clone(),
                size_bytes: artifact.size_bytes,
                sha256: artifact.sha256.clone(),
            })
            .collect(),
        reference_comparison: ReferenceComparison::not_configured(),
    };
    assert!(!evidence_matches(&entry, &loaded, "smoke", &reference));

    let expected = reference.document.expected.clone();
    entry.reference_comparison = compare(&reference, &expected).unwrap();
    entry.reference_comparison.reference_sha256 = Some("f".repeat(64));
    assert!(!evidence_matches(&entry, &loaded, "smoke", &reference));

    entry.reference_comparison = compare(&reference, &expected).unwrap();
    assert!(evidence_matches(&entry, &loaded, "smoke", &reference));
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
