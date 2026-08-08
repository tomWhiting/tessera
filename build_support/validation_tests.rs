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
