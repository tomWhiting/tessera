use std::sync::{Arc, Barrier};

use candle_core::{Device, DeviceLocation};

use super::{
    preflight_and_reserve_registered_model, ModelResidencyError, ModelResidencyKey,
    ModelResidencyLedger,
};
use crate::models::registry::ModelType;
use crate::runtime::{ModelDType, ResourcePolicy};

fn key(model_id: &'static str, revision: &'static str) -> ModelResidencyKey {
    ModelResidencyKey {
        model_id,
        revision,
        device: DeviceLocation::Cpu,
        dtype: ModelDType::F32,
    }
}

#[test]
fn duplicate_residency_is_rejected_until_the_permit_drops() {
    let ledger = ModelResidencyLedger::default();
    let model = key("model-a", "revision-a");
    let permit = ledger
        .try_reserve(model, 400, 1_000)
        .expect("first instance should reserve residency");

    let error = ledger
        .try_reserve(model, 400, 1_000)
        .expect_err("a second retained copy must be rejected");
    assert!(matches!(error, ModelResidencyError::Duplicate { .. }));
    assert!(error.to_string().contains("reuse the existing embedder"));

    drop(permit);
    let _permit = ledger
        .try_reserve(model, 400, 1_000)
        .expect("dropping the model must release residency");
}

#[test]
fn requesting_policy_is_the_prospective_aggregate_ceiling() {
    let ledger = ModelResidencyLedger::default();
    let _first = ledger
        .try_reserve(key("model-a", "revision-a"), 600, 1_000)
        .expect("first model should fit");

    let error = ledger
        .try_reserve(key("model-b", "revision-b"), 500, 1_000)
        .expect_err("combined residency should exceed the requesting policy");
    assert_eq!(
        error,
        ModelResidencyError::AggregateBytes {
            resident: 600,
            requested: 500,
            prospective: 1_100,
            allowed: 1_000,
        }
    );

    let _second = ledger
        .try_reserve(key("model-b", "revision-b"), 500, 1_200)
        .expect("a deliberate higher requesting policy should admit the aggregate");
}

#[test]
fn physical_device_location_is_part_of_the_duplicate_key() {
    let model = crate::models::registry::get_model("bge-base-en-v1.5")
        .expect("BGE registry entry should exist");
    let first = ModelResidencyKey::new(model, &Device::Cpu, ModelDType::F32)
        .expect("preflighted BGE metadata should have a revision");
    let second = ModelResidencyKey::new(model, &Device::Cpu, ModelDType::F32)
        .expect("preflighted BGE metadata should have a revision");

    assert_eq!(first, second);
    assert_eq!(first.device, DeviceLocation::Cpu);
}

#[test]
fn missing_revision_is_a_returned_error() {
    let model = crate::models::registry::get_model("jina-colbert-v2-96")
        .expect("unavailable catalog metadata should remain discoverable");
    let error = ModelResidencyKey::new(model, &Device::Cpu, ModelDType::F32)
        .expect_err("unpinned metadata cannot form a residency key");

    assert!(matches!(
        error,
        ModelResidencyError::MissingRevision {
            model_id: "jina-colbert-v2-96"
        }
    ));
}

#[test]
fn registered_preflight_reserves_before_artifact_access() {
    let policy = ResourcePolicy::default();
    let (_, permit) = preflight_and_reserve_registered_model(
        "BAAI/bge-base-en-v1.5",
        512,
        ModelType::Dense,
        &Device::Cpu,
        &policy,
    )
    .expect("first preflight should reserve the registered model");

    let error = preflight_and_reserve_registered_model(
        "BAAI/bge-base-en-v1.5",
        512,
        ModelType::Dense,
        &Device::Cpu,
        &policy,
    )
    .expect_err("duplicate residency should fail before any artifact access");
    assert!(error.to_string().contains("already resident"));

    drop(permit);
}

#[test]
fn concurrent_duplicate_admission_has_one_winner() {
    const THREADS: usize = 8;
    let ledger = Arc::new(ModelResidencyLedger::default());
    let start = Arc::new(Barrier::new(THREADS));
    let finish = Arc::new(Barrier::new(THREADS));
    let mut workers = Vec::new();

    for _ in 0..THREADS {
        let ledger = Arc::clone(&ledger);
        let start = Arc::clone(&start);
        let finish = Arc::clone(&finish);
        workers.push(std::thread::spawn(move || {
            start.wait();
            let permit = ledger.try_reserve(key("model-a", "revision-a"), 400, 1_000);
            finish.wait();
            permit.is_ok()
        }));
    }

    let admitted = workers
        .into_iter()
        .map(|worker| worker.join().expect("worker should not panic"))
        .filter(|admitted| *admitted)
        .count();
    assert_eq!(admitted, 1);
}
