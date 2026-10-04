use super::validate_registry;
use crate::schema::{ModelRegistry, SupportTier};

fn catalog_json() -> serde_json::Value {
    serde_json::from_str(include_str!("../models.json")).expect("models.json should parse as JSON")
}

#[test]
fn support_tiers_define_runnability() {
    assert!(SupportTier::Supported.is_runnable());
    assert!(SupportTier::Experimental.is_runnable());
    assert!(!SupportTier::CatalogOnly.is_runnable());
}

#[test]
#[should_panic(expected = "Unsupported model registry schema version")]
fn unsupported_registry_schema_versions_are_rejected() {
    let mut catalog = catalog_json();
    catalog["version"] = serde_json::Value::String("1.0".to_string());

    let registry = serde_json::from_value::<ModelRegistry>(catalog)
        .expect("modified catalog should deserialize");
    validate_registry(&registry);
}

#[test]
fn support_metadata_is_required() {
    let mut catalog = catalog_json();
    catalog["model_categories"]["multi_vector"]["models"][0]
        .as_object_mut()
        .expect("model should be an object")
        .remove("support");

    let result = serde_json::from_value::<ModelRegistry>(catalog);
    assert!(result.is_err(), "models without support metadata must fail");
}

#[test]
#[should_panic(expected = "must have a nonempty support note")]
fn blank_support_notes_are_rejected() {
    let mut catalog = catalog_json();
    catalog["model_categories"]["multi_vector"]["models"][0]["support"]["note"] =
        serde_json::Value::String("   ".to_string());

    let registry = serde_json::from_value::<ModelRegistry>(catalog)
        .expect("modified catalog should deserialize");
    validate_registry(&registry);
}

#[test]
#[should_panic(expected = "must pin a HuggingFace revision")]
fn runnable_models_require_a_revision() {
    let mut catalog = catalog_json();
    catalog["model_categories"]["multi_vector"]["models"][1]["revision"] = serde_json::Value::Null;

    let registry = serde_json::from_value::<ModelRegistry>(catalog)
        .expect("modified catalog should deserialize");
    validate_registry(&registry);
}

#[test]
#[should_panic(expected = "exact lowercase 40-hex commit SHA")]
fn floating_revisions_are_rejected() {
    let mut catalog = catalog_json();
    catalog["model_categories"]["multi_vector"]["models"][1]["revision"] =
        serde_json::Value::String("main".to_string());

    let registry = serde_json::from_value::<ModelRegistry>(catalog)
        .expect("modified catalog should deserialize");
    validate_registry(&registry);
}

#[test]
#[should_panic(expected = "exact lowercase 40-hex commit SHA")]
fn uppercase_commit_shas_are_rejected() {
    let mut catalog = catalog_json();
    catalog["model_categories"]["multi_vector"]["models"][1]["revision"] =
        serde_json::Value::String("C72AA89BC61AFDD85373643F3A1A75B2AAD6E0FE".to_string());

    let registry = serde_json::from_value::<ModelRegistry>(catalog)
        .expect("modified catalog should deserialize");
    validate_registry(&registry);
}

#[test]
fn audited_weight_metadata_preserves_absent_and_sharded_safetensors() {
    let registry = serde_json::from_value::<ModelRegistry>(catalog_json())
        .expect("catalog should deserialize");

    let splade = registry
        .models()
        .find(|model| model.id == "splade-pp-en-v1")
        .expect("SPLADE v1 metadata");
    assert!(splade.files.weights.safetensors.is_none());

    let colpali = registry
        .models()
        .find(|model| model.id == "colpali-v1.2")
        .expect("ColPali metadata");
    assert_eq!(
        colpali.files.weights.safetensors.as_deref(),
        Some("model.safetensors.index.json")
    );

    let snowflake = registry
        .models()
        .find(|model| model.id == "snowflake-arctic-l")
        .expect("Snowflake metadata");
    assert_eq!(snowflake.specs.parameters, "568M");
}

fn validate_with_dense_change(
    change: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) {
    let mut catalog = catalog_json();
    change(
        catalog["model_categories"]["dense"]["models"][0]
            .as_object_mut()
            .expect("model should be an object"),
    );
    let registry = serde_json::from_value::<ModelRegistry>(catalog)
        .expect("modified catalog should deserialize");
    validate_registry(&registry);
}

#[test]
fn safetensors_only_weights_are_accepted() {
    validate_with_dense_change(|model| {
        model["files"]["weights"] = serde_json::json!({"safetensors": "model.safetensors"});
    });
}

#[test]
fn legacy_only_weights_are_accepted() {
    validate_with_dense_change(|model| {
        model["files"]["weights"] = serde_json::json!({"pytorch": "pytorch_model.bin"});
    });
}

#[test]
fn onnx_only_weight_metadata_is_accepted() {
    validate_with_dense_change(|model| {
        model["files"]["weights"] = serde_json::json!({"onnx": "onnx/model.onnx"});
    });
}

#[test]
#[should_panic(expected = "Model bge-base-en-v1.5 must declare at least one weight artifact")]
fn no_weight_file_is_refused_by_model_name() {
    validate_with_dense_change(|model| {
        model["files"]["weights"] = serde_json::json!({});
    });
}

#[test]
#[should_panic(expected = "Model bge-base-en-v1.5 has an empty PyTorch artifact path")]
fn blank_legacy_weight_file_is_refused() {
    validate_with_dense_change(|model| {
        model["files"]["weights"]["pytorch"] = serde_json::json!("   ");
    });
}

#[test]
fn generated_weight_metadata_preserves_each_format_and_absence() {
    for (weights, safetensors, pytorch, onnx) in [
        (
            serde_json::json!({"safetensors": "model.safetensors"}),
            "Some(\"model.safetensors\")",
            "None",
            "None",
        ),
        (
            serde_json::json!({"pytorch": "pytorch_model.bin"}),
            "None",
            "Some(\"pytorch_model.bin\")",
            "None",
        ),
        (
            serde_json::json!({"onnx": "onnx/model.onnx"}),
            "None",
            "None",
            "Some(\"onnx/model.onnx\")",
        ),
    ] {
        let mut catalog = catalog_json();
        catalog["model_categories"]["dense"]["models"][0]["files"]["weights"] = weights;
        let registry =
            serde_json::from_value::<ModelRegistry>(catalog).expect("catalog should deserialize");
        let model = registry
            .models()
            .find(|model| model.id == "bge-base-en-v1.5")
            .expect("model metadata");
        let generated = crate::model_constant::generate_model_constant(model);
        assert!(generated.contains(&format!("safetensors_file: {safetensors},")));
        assert!(generated.contains(&format!("pytorch_file: {pytorch},")));
        assert!(generated.contains(&format!("onnx_file: {onnx},")));
    }
}

#[test]
#[should_panic(expected = "must declare prompts")]
fn dense_entries_require_prompts() {
    validate_with_dense_change(|model| {
        model.remove("prompts");
    });
}

#[test]
#[should_panic(expected = "must declare a distance")]
fn dense_entries_require_a_distance() {
    validate_with_dense_change(|model| {
        model.remove("distance");
    });
}

#[test]
#[should_panic(expected = "has invalid distance 'angular'")]
fn unknown_distance_words_are_rejected() {
    validate_with_dense_change(|model| {
        model.insert("distance".to_string(), serde_json::json!("angular"));
    });
}

#[test]
#[should_panic(expected = "card comparison URL must be at its pinned revision")]
fn card_comparison_must_cite_the_pinned_revision() {
    validate_with_dense_change(|model| {
        model["card_comparison"]["url"] =
            serde_json::json!("https://huggingface.co/BAAI/bge-base-en-v1.5/blob/main/README.md");
    });
}
