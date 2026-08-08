use super::TransformerProfile;
use crate::runtime::ModelDType;

#[test]
fn parses_bert_and_nomic_dimension_names() {
    let bert = TransformerProfile::from_config_json(
        r#"{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12}"#,
    )
    .expect("BERT profile should parse");
    let nomic =
        TransformerProfile::from_config_json(r#"{"n_embd":768,"n_inner":3072,"n_head":12}"#)
            .expect("Nomic profile should parse");
    assert_eq!(bert, nomic);
}

#[test]
fn estimate_scales_quadratically_and_respects_dtype_width() {
    let profile = TransformerProfile::from_config_json(
        r#"{"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12}"#,
    )
    .expect("profile should parse");
    let short = profile.peak_bytes(1, 512, ModelDType::F32);
    let long = profile.peak_bytes(1, 8192, ModelDType::F32);
    let long_f16 = profile.peak_bytes(1, 8192, ModelDType::F16);

    assert!(long > short.saturating_mul(100));
    assert_eq!(long_f16.saturating_mul(2), long);
}

#[test]
fn missing_heads_is_rejected() {
    let error = TransformerProfile::from_config_json(r#"{"hidden_size":768}"#)
        .expect_err("attention heads are required");
    assert!(error.to_string().contains("attention heads"));
}
