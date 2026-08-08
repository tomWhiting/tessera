use super::{model_dtype_name, parse_model_dtype_value};
use crate::runtime::ModelDType;

#[test]
fn parser_accepts_the_exact_public_dtype_spellings() {
    let cases = [
        ("f32", ModelDType::F32),
        ("f16", ModelDType::F16),
        ("bf16", ModelDType::BF16),
    ];

    for (value, expected) in cases {
        assert_eq!(parse_model_dtype_value(value), Ok(expected));
        assert_eq!(model_dtype_name(expected), value);
    }
}

#[test]
fn parser_rejects_implicit_coercions_and_unknown_values() {
    for value in ["", "F32", "float32", " f16", "bf16 ", "fp16"] {
        let error = parse_model_dtype_value(value).expect_err("invalid dtype must fail");
        assert!(error.contains(value));
        assert!(error.contains("f32, f16, bf16"));
    }
}

#[test]
fn constructors_expose_keyword_only_dtype_with_f32_default() {
    const SIGNATURE: &str =
        "#[pyo3(signature = (model_id, *, resource_policy=None, dtype=\"f32\"))]";
    let sources = [
        ("TesseraDense", include_str!("../dense.rs")),
        ("TesseraMultiVector", include_str!("../multivector.rs")),
        ("TesseraSparse", include_str!("../sparse.rs")),
        ("TesseraVision", include_str!("../vision.rs")),
    ];

    for (name, source) in sources {
        assert!(
            source.contains(SIGNATURE),
            "{name} must expose keyword-only dtype with an f32 default"
        );
    }
}
