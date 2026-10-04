use super::{compare_vectors, record_path, write_record, MeasurementRecord};

#[test]
fn vector_comparison_distinguishes_signed_zero_bytes() {
    let equal = compare_vectors(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
    assert!(equal.byte_equal);
    let changed = compare_vectors(&[1.0, 0.0], &[1.0, -0.0]).unwrap();
    assert!(!changed.byte_equal);
    assert!(changed.max_absolute_difference.abs() < f64::EPSILON);
    assert!((changed.minimum_cosine - 1.0).abs() < f64::EPSILON);
}

#[test]
fn vector_comparison_reports_largest_difference_and_cosine() {
    let comparison = compare_vectors(&[1.0, 0.0], &[0.0, 2.0]).unwrap();
    assert!(!comparison.byte_equal);
    assert!((comparison.max_absolute_difference - 2.0).abs() < f64::EPSILON);
    assert!(comparison.minimum_cosine.abs() < f64::EPSILON);
}

#[test]
fn vector_comparison_refuses_invalid_inputs() {
    for (left, right) in [
        (vec![], vec![]),
        (vec![1.0], vec![1.0, 2.0]),
        (vec![f32::NAN], vec![1.0]),
        (vec![1.0], vec![f32::INFINITY]),
        (vec![0.0], vec![0.0]),
    ] {
        assert!(compare_vectors(&left, &right).is_err());
    }
}

#[test]
fn measurement_record_is_json_without_a_run_verdict_and_round_trips() {
    let root = std::env::temp_dir().join(format!(
        "tessera-measure-{}-{}",
        std::process::id(),
        crate::certification::evidence::now_unix_ms().unwrap()
    ));
    let path = record_path(&root, "model", 1);
    let record = MeasurementRecord {
        schema_version: 1,
        kind: "dense_measurement".into(),
        source_commit: "a".repeat(40),
        source_dirty: false,
        model_id: "model".into(),
        model_revision: "b".repeat(40),
        profile: "smoke".into(),
        spec_sha256: "c".repeat(64),
        installed_manifest_sha256: Some("d".repeat(64)),
        configured_threads: 2,
        comparison_threads: 1,
        batch: vec![],
        threads: compare_vectors(&[1.0], &[1.0]).unwrap(),
    };
    write_record(&path, &record).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let loaded: MeasurementRecord = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(loaded, record);
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(value.get("status").is_none());
    assert!(value.get("passed").is_none());
    assert!(write_record(&path, &record).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
