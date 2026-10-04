use super::{cosine, min_max, sparse_cosine, sparse_dot, validate_probe_token_count};

#[test]
fn compact_similarity_helpers_are_stable() {
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < f32::EPSILON);
    assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < f32::EPSILON);
    let mismatched = cosine(&[1.0], &[1.0, 0.0]);
    assert!(mismatched.is_infinite() && mismatched.is_sign_negative());

    let left = [(1, 2.0), (4, 3.0)];
    let right = [(0, 8.0), (4, 5.0)];
    assert!((sparse_dot(&left, &right) - 15.0).abs() < f32::EPSILON);
    assert!(sparse_cosine(&left, &left) > 0.9999);
}

#[test]
fn min_max_reports_observed_range() {
    assert_eq!(min_max(&[2.0, -1.0, 4.0]), (-1.0, 4.0));
}

#[test]
fn pinned_probe_token_count_must_match_the_local_tokenizer() {
    assert!(validate_probe_token_count(8_000, 8_000).is_ok());
    assert!(validate_probe_token_count(8_000, 12).is_err());
}

fn old_dense_record() -> crate::certification::evidence::EvidenceRecord {
    serde_json::from_str(include_str!("fixtures/dense-fe5b58b.json")).unwrap()
}

#[test]
fn dense_batch_size_respects_batch_item_limit() {
    let mut limits = old_dense_record().resource_policy;
    limits.max_batch_items = 1;
    limits.max_job_items = 2;
    let plan = crate::certification::evidence::DenseBatchPlan::for_limits(&limits);
    assert_eq!(plan.batch_size, 1);
}

#[test]
fn dense_batch_size_respects_job_item_limit() {
    let mut limits = old_dense_record().resource_policy;
    limits.max_batch_items = 2;
    limits.max_job_items = 1;
    let plan = crate::certification::evidence::DenseBatchPlan::for_limits(&limits);
    assert_eq!(plan.batch_size, 1);
}

#[test]
fn two_item_dense_plan_keeps_batch_checks_enabled() {
    let mut limits = old_dense_record().resource_policy;
    for (batch_items, job_items) in [(2, 2), (8, 4)] {
        limits.max_batch_items = batch_items;
        limits.max_job_items = job_items;
        let plan = crate::certification::evidence::DenseBatchPlan::for_limits(&limits);
        assert_eq!(plan.batch_size, 2);
        assert!(plan.not_run_checks().is_empty());
    }
}

fn limited_record(
    batch_items: usize,
    job_items: usize,
) -> crate::certification::evidence::EvidenceRecord {
    use crate::certification::evidence::{build_record, ChildOutcome, RecordInput};
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let mut loaded =
        crate::certification::spec::load_model(repository, "bge-base-en-v1.5").unwrap();
    let limits = &mut loaded
        .spec
        .profiles
        .get_mut("smoke")
        .unwrap()
        .resource_policy;
    limits.max_batch_items = batch_items;
    limits.max_job_items = job_items;
    let old = old_dense_record();
    let mut observation = old.observation.unwrap();
    observation.batch_shapes.clear();
    observation.checks.retain(|check| {
        !matches!(
            check.name.as_str(),
            "batch-shape" | "batch-sequential-parity"
        )
    });
    build_record(
        repository,
        RecordInput {
            loaded: &loaded,
            profile: "smoke",
            repetition: old.repetition,
            child_pid: old.process_id,
            started_unix_ms: old.started_unix_ms,
            completed_unix_ms: old.completed_unix_ms,
            peak_rss: old.peak_rss,
            outcome: ChildOutcome {
                status: old.status,
                error: old.error,
                verified_artifacts: old.verified_artifacts,
                observation: Some(observation),
                reference_comparison: old.reference_comparison,
                installed_manifest_sha256: old.installed_manifest_sha256,
            },
        },
    )
    .unwrap()
}

fn assert_limit_record(batch_items: usize, job_items: usize) {
    let record = limited_record(batch_items, job_items);
    assert_eq!(record.status, "passed");
    assert_eq!(record.observation.as_ref().unwrap().checks.len(), 5);
    assert_eq!(record.not_run_checks.len(), 2);
    let skipped = serde_json::to_value(&record.not_run_checks).unwrap();
    for (index, name) in ["batch-shape", "batch-sequential-parity"]
        .into_iter()
        .enumerate()
    {
        assert_eq!(skipped[index]["name"], name);
        assert!(skipped[index]["reason"]
            .as_str()
            .unwrap()
            .contains(&format!(
                "max_batch_items={batch_items}, max_job_items={job_items}"
            )));
        assert!(skipped[index].get("passed").is_none());
        assert!(skipped[index].get("failed").is_none());
    }
}

#[test]
fn batch_limit_one_record_names_checks_not_run() {
    assert_limit_record(1, 2);
}

#[test]
fn job_limit_one_record_names_checks_not_run() {
    assert_limit_record(2, 1);
}

#[test]
fn actual_old_evidence_loads_without_new_members() {
    let old_text = include_str!("fixtures/dense-fe5b58b.json");
    let old_value: serde_json::Value = serde_json::from_str(old_text).unwrap();
    let record = old_dense_record();
    assert!(record.not_run_checks.is_empty());
    let round_trip: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&record).unwrap()).unwrap();
    assert_eq!(round_trip, old_value);
}
