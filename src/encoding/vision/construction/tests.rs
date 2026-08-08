use super::*;

#[test]
fn projection_is_mandatory() {
    let error = validate_projection_contract("model", false, None, 128, 2_048, 2_048).unwrap_err();

    assert!(error
        .to_string()
        .contains("no registered ColPali projection"));
}

#[test]
fn projection_dimensions_must_match_registry_and_checkpoint() {
    let requested =
        validate_projection_contract("model", true, Some(128), 64, 2_048, 2_048).unwrap_err();
    assert!(requested
        .to_string()
        .contains("Requested embedding dimension 64"));

    let hidden =
        validate_projection_contract("model", true, Some(128), 128, 1_024, 2_048).unwrap_err();
    assert!(hidden
        .to_string()
        .contains("Registered hidden dimension 1024"));
}

#[test]
fn valid_projection_contract_returns_runtime_dimensions() {
    let dimensions =
        validate_projection_contract("model", true, Some(128), 128, 2_048, 2_048).unwrap();

    assert_eq!(dimensions, (2_048, 128));
}
