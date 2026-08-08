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
