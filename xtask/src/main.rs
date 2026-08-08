use std::error::Error;
use std::path::PathBuf;

// The build script consumes the complete schema; the policy command only needs
// the validation subset, so some shared fields are intentionally unused here.
#[allow(dead_code)]
#[path = "../../build_support/schema.rs"]
mod schema;
#[path = "../../build_support/validation.rs"]
mod validation;

mod policy;

fn main() -> Result<(), Box<dyn Error>> {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be located under the repository root")
        .to_path_buf();

    match std::env::args().nth(1).as_deref() {
        Some("all") => {
            policy::check_file_sizes(&repository)?;
            check_registry(&repository)?;
        }
        Some("file-size") => policy::check_file_sizes(&repository)?,
        Some("registry") => check_registry(&repository)?,
        _ => {
            return Err("usage: cargo run -p tessera-xtask -- <all|file-size|registry>".into());
        }
    }

    Ok(())
}

fn check_registry(repository: &std::path::Path) -> Result<(), Box<dyn Error>> {
    let source = std::fs::read_to_string(repository.join("models.json"))?;
    let registry: schema::ModelRegistry = serde_json::from_str(&source)?;
    validation::validate_registry(&registry);
    let runnable_models = registry
        .models()
        .filter(|model| model.support.tier.is_runnable())
        .count();
    println!(
        "registry policy passed: {} models ({} runnable) across {} categories",
        registry.models().count(),
        runnable_models,
        registry.model_categories.len()
    );
    Ok(())
}
