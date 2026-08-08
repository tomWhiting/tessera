use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::{
    compare, comparison_is_complete, load_checked, resolve_image, validate_tolerance,
    ComparisonStatus, LoadedReference, NumericTolerance, ReferenceDocument, ReferenceOutput,
    ReferencePointer, ReferenceProbe,
};
use crate::certification::spec::{
    ArtifactSpec, CapabilityScope, CertificationDevice, CertificationDtype, CertificationSpec,
    ModelSpec, ProcessLimits, ProfileKind, ProfileSpec, PromotionSpec, Representation,
    ResourceLimits, RetrievalFixture, SemanticMode, SmokeSpec,
};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn load_fixture(name: &str) -> LoadedReference {
    let path = repository()
        .join("certification/references/contract")
        .join(format!("{name}.json"));
    let document: ReferenceDocument = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    LoadedReference {
        path: format!("contract/{name}.json"),
        sha256: "fixture".to_string(),
        document,
    }
}

fn dense_spec(pointer: ReferencePointer) -> CertificationSpec {
    let capability = CapabilityScope {
        device: CertificationDevice::Cpu,
        dtype: CertificationDtype::F32,
        semantic_mode: SemanticMode::Query,
        max_sequence_tokens: 8,
        context_window_tokens: 8,
    };
    let profile = ProfileSpec {
        kind: ProfileKind::Smoke,
        capability,
        resource_policy: ResourceLimits {
            max_sequence_tokens: 8,
            max_batch_items: 1,
            max_batch_tokens: 8,
            max_model_bytes: 1,
            max_input_bytes_per_sequence: 1024,
            max_attention_cells: 64,
            max_job_items: 1,
            max_job_input_bytes: 1024,
            max_output_bytes: 1024,
            max_activation_bytes: 1024,
        },
        process: ProcessLimits {
            cpu_threads: 1,
            timeout_seconds: 1,
            max_artifact_bytes: 1,
            min_free_disk_bytes: 0,
            max_peak_rss_bytes: 1,
        },
        official_reference: Some(pointer),
    };
    CertificationSpec {
        schema_version: 2,
        model: ModelSpec {
            id: "contract-dense".to_string(),
            repository: "example/contract-dense".to_string(),
            revision: "a".repeat(40),
            representation: Representation::Dense,
        },
        artifacts: vec![ArtifactSpec {
            path: "model.bin".to_string(),
            size_bytes: 1,
            sha256: "a".repeat(64),
        }],
        profiles: BTreeMap::from([("smoke".to_string(), profile)]),
        smoke: SmokeSpec {
            fixture: RetrievalFixture {
                query: "q".to_string(),
                positive: "p".to_string(),
                negative: "n".to_string(),
            },
            expected_dimension: 3,
            expected_vocabulary_size: None,
            normalized: true,
            minimum_score_margin: 0.0,
            repeat_similarity_minimum: 0.99,
        },
        promotion: PromotionSpec {
            minimum_successful_runs: 2,
            required_profiles: vec!["smoke".to_string()],
            require_clean_source: true,
            require_enforced_rss: true,
        },
    }
}

#[test]
fn checked_reference_requires_the_actual_file_digest() {
    let valid_pointer = ReferencePointer {
        path: "contract/dense.json".to_string(),
        sha256: "b981201785c1ff478745bf0414e2c6d00c8c540d851668b3677ba1c6b1b76039".to_string(),
    };
    let spec = dense_spec(valid_pointer.clone());
    let profile = spec.profile("smoke").unwrap();
    assert!(load_checked(&repository(), &spec, "smoke", profile, &valid_pointer).is_ok());

    let fake_pointer = ReferencePointer {
        path: valid_pointer.path,
        sha256: "f".repeat(64),
    };
    let fake_spec = dense_spec(fake_pointer.clone());
    let error = load_checked(
        &repository(),
        &fake_spec,
        "smoke",
        fake_spec.profile("smoke").unwrap(),
        &fake_pointer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("SHA-256 mismatch"));
}

#[test]
fn dense_comparison_is_tolerant_but_not_a_presence_gate() {
    let reference = load_fixture("dense");
    let close = ReferenceOutput::Dense {
        values: vec![0.600_4, 0.799_5, 0.0],
    };
    let mut comparison = compare(&reference, &close).unwrap();
    assert_eq!(comparison.status, ComparisonStatus::Passed);
    assert!(comparison_is_complete(&comparison, &reference));
    comparison.observed_output_sha256 = Some("not-a-digest".to_string());
    assert!(!comparison_is_complete(&comparison, &reference));

    let drifted = ReferenceOutput::Dense {
        values: vec![0.7, 0.7, 0.0],
    };
    assert_eq!(
        compare(&reference, &drifted).unwrap().status,
        ComparisonStatus::Failed
    );
}

#[test]
fn sparse_comparison_requires_the_same_sorted_coordinates() {
    let reference = load_fixture("sparse");
    let observed = ReferenceOutput::Sparse {
        vocabulary_size: 8,
        indices: vec![1, 5, 7],
        values: vec![0.5, 1.5, 2.0],
    };
    let comparison = compare(&reference, &observed).unwrap();
    assert_eq!(comparison.status, ComparisonStatus::Failed);
    assert!(comparison.detail.contains("sparse-indices-match=false"));
}

#[test]
fn multi_vector_comparison_checks_each_row() {
    let reference = load_fixture("multi-vector");
    let close = ReferenceOutput::MultiVector {
        rows: 2,
        columns: 2,
        values: vec![1.0, 0.000_2, 0.000_2, 1.0],
    };
    assert_eq!(
        compare(&reference, &close).unwrap().status,
        ComparisonStatus::Passed
    );

    let rotated_row = ReferenceOutput::MultiVector {
        rows: 2,
        columns: 2,
        values: vec![0.0, 1.0, 0.0, 1.0],
    };
    assert_eq!(
        compare(&reference, &rotated_row).unwrap().status,
        ComparisonStatus::Failed
    );
}

#[test]
fn vision_comparison_is_shape_and_patch_row_scoped() {
    let reference = load_fixture("vision");
    let close = ReferenceOutput::Vision {
        rows: 2,
        columns: 2,
        values: vec![0.800_2, 0.599_8, 0.0, 1.0],
    };
    let comparison = compare(&reference, &close).unwrap();
    assert_eq!(comparison.status, ComparisonStatus::Passed);
    assert_eq!(comparison.observed_shape, vec![2, 2]);
}

#[test]
fn image_probe_bytes_and_tolerance_ceilings_are_enforced() {
    let reference = load_fixture("vision");
    assert!(resolve_image(&repository(), &reference.document.probe).is_ok());
    let mut tampered_probe = reference.document.probe;
    let ReferenceProbe::Image { sha256, .. } = &mut tampered_probe else {
        panic!("vision fixture must use an image probe");
    };
    *sha256 = "f".repeat(64);
    assert!(resolve_image(&repository(), &tampered_probe).is_err());

    assert!(validate_tolerance(NumericTolerance {
        absolute: 0.001_1,
        relative: 0.001,
        minimum_cosine: 0.999,
    })
    .is_err());
    assert!(validate_tolerance(NumericTolerance {
        absolute: 0.001,
        relative: 0.001,
        minimum_cosine: 0.998,
    })
    .is_err());
}

#[test]
fn long_context_reference_must_be_a_near_limit_probe() {
    let valid_pointer = ReferencePointer {
        path: "contract/dense.json".to_string(),
        sha256: "b981201785c1ff478745bf0414e2c6d00c8c540d851668b3677ba1c6b1b76039".to_string(),
    };
    let mut spec = dense_spec(valid_pointer.clone());
    spec.profiles.get_mut("smoke").unwrap().kind = ProfileKind::LongContext;
    let error = load_checked(
        &repository(),
        &spec,
        "smoke",
        spec.profile("smoke").unwrap(),
        &valid_pointer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("at least 87.5%"));
}
