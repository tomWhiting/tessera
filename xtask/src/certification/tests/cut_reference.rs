use super::validate_dense_probe_counts;

#[test]
fn cut_reference_accepts_full_and_used_counts_including_special_tokens() {
    assert!(validate_dense_probe_counts(3_000, Some(2_048), 3_000, 2_048, true).is_ok());
}

#[test]
fn cut_reference_refuses_every_count_or_cut_mismatch_with_both_counts() {
    for (total, used, cut) in [
        (2_999, 2_048, true),
        (3_000, 2_047, true),
        (3_000, 2_048, false),
    ] {
        let error = validate_dense_probe_counts(3_000, Some(2_048), total, used, cut)
            .unwrap_err()
            .to_string();
        assert!(error.contains("cut_reference_token_mismatch"), "{error}");
        assert!(
            error.contains(&format!("observed total={total}, used={used}")),
            "{error}"
        );
        assert!(error.contains("expected total=3000, used=2048"), "{error}");
    }
}

#[test]
fn uncut_reference_keeps_its_existing_admission() {
    assert!(validate_dense_probe_counts(7, None, 7, 7, false).is_ok());
    assert!(validate_dense_probe_counts(7, None, 6, 6, false).is_err());
    assert!(validate_dense_probe_counts(7, None, 7, 5, true).is_err());
}
