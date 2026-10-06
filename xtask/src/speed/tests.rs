use super::cli::Route;
use super::{fixtures, record};

fn fixture() -> fixtures::Fixture {
    fixtures::Fixture {
        name: "short".to_string(),
        repetitions: 30,
        raw_bytes: 119,
        content_tokens: 30,
        complete_tokens: 32,
        sha256: "622346a0acb83057aa6f14da0a5b6fda9136e07e07def4ad6c04ae623c28a6d2".to_string(),
    }
}

#[test]
fn fixture_hash_tampering_is_refused() {
    let mut fixture = fixture();
    let text = fixtures::text(&fixture);
    fixture.sha256 = fixtures::digest(text.as_bytes());
    assert!(fixtures::check(&fixture, &text, 30, 32).is_ok());
    assert!(fixtures::check(&fixture, &text.replace("red", "blue"), 30, 32).is_err());
}

#[test]
fn token_assertion_mismatch_is_refused() {
    let mut fixture = fixture();
    let text = fixtures::text(&fixture);
    fixture.sha256 = fixtures::digest(text.as_bytes());
    assert!(fixtures::check(&fixture, &text, 29, 31).is_err());
    assert!(fixtures::check(&fixture, &text, 30, 33).is_err());
}

#[test]
fn unknown_fixture_manifest_members_are_refused() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("../../speed-fixtures.json")).unwrap();
    value["surprise"] = serde_json::json!(true);
    assert!(serde_json::from_value::<fixtures::Manifest>(value).is_err());
}

#[test]
fn mixed_tensor_batch_cost_includes_padding() {
    let lengths: Vec<usize> = (0..20).map(|i| [32, 32, 32, 128, 512][i % 5]).collect();
    let counts = record::counts(&lengths, Route::Batch);
    assert_eq!(counts.real_tokens, 2944);
    assert_eq!(counts.physical_token_rows_derived, 8704);
    assert_eq!(counts.real_squared_lengths, 1_126_400);
    assert_eq!(counts.physical_squared_lengths_derived, 4_259_840);
    assert_eq!(counts.forward_calls_derived, 5);
}

#[test]
fn outcome_cost_keeps_physical_rows_unpadded() {
    let counts = record::counts(&[32, 32, 32, 512], Route::Outcomes);
    assert_eq!(counts.real_tokens, 608);
    assert_eq!(counts.physical_token_rows_derived, 608);
    assert_eq!(counts.admission_token_rows_derived, 2048);
    assert_eq!(counts.forward_calls_derived, 4);
    assert_eq!(counts.raw_output_bytes, 4 * 768 * 4);
}

#[test]
fn final_tensor_chunk_counts_only_present_items() {
    let counts = record::counts(&[32, 32, 32, 32, 128], Route::Batch);
    assert_eq!(counts.forward_calls_derived, 2);
    assert_eq!(counts.physical_token_rows_derived, 256);
    assert_eq!(counts.raw_output_bytes, 5 * 768 * 4);
}

#[test]
fn installed_asset_digest_cannot_drift_from_the_pin() {
    let spec = serde_json::json!({"artifacts":[{"path":"model.safetensors","sha256":"pinned","size_bytes":12}]});
    let mut installed = serde_json::json!({"schema_version":1,"model":{"id":fixtures::MODEL,"repository":"BAAI/bge-base-en-v1.5","revision":fixtures::REVISION},"artifacts":spec["artifacts"]});
    assert!(fixtures::check_declarations(&installed, &spec).is_ok());
    installed["artifacts"][0]["sha256"] = serde_json::json!("changed");
    assert!(fixtures::check_declarations(&installed, &spec).is_err());
}

#[test]
fn absent_pinned_artifact_list_is_refused() {
    assert!(fixtures::check_declarations(&serde_json::json!({}), &serde_json::json!({})).is_err());
}
