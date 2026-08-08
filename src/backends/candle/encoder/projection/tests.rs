use super::validate_projection_contract;

#[test]
fn accepts_a_complete_projection_contract() {
    validate_projection_contract(true, Some(128), 128, 768, 768)
        .expect("matching registry and checkpoint dimensions should be accepted");
}

#[test]
fn rejects_missing_or_mismatched_projection_metadata() {
    let missing = validate_projection_contract(false, Some(128), 128, 768, 768).unwrap_err();
    assert!(missing.to_string().contains("must declare"));

    let output_mismatch = validate_projection_contract(true, Some(96), 128, 768, 768).unwrap_err();
    assert!(output_mismatch.to_string().contains("configured embedding"));

    let hidden_mismatch = validate_projection_contract(true, Some(128), 128, 384, 768).unwrap_err();
    assert!(hidden_mismatch.to_string().contains("checkpoint hidden"));
}
