//! Model registry with compile-time generated model metadata.
//!
//! This module provides a type-safe catalog generated at compile time from
//! models.json. The catalog includes models retained for discovery even when
//! Tessera does not currently expose a runtime adapter for them.
//!
//! # Example
//!
//! ```no_run
//! use tessera::model_registry::{get_model, runnable_models, COLBERT_V2};
//!
//! // Access specific model constant
//! println!("Model: {}", COLBERT_V2.name);
//! println!("Dimensions: {}", COLBERT_V2.embedding_dim);
//!
//! // Lookup by ID
//! let model = get_model("colbert-small").expect("Model not found");
//! println!("Found: {}", model.name);
//!
//! // List models with a runtime path
//! for model in runnable_models() {
//!     println!("{}: {} dims, {}K context",
//!         model.name,
//!         model.embedding_dim,
//!         model.context_length / 1000
//!     );
//! }
//! ```

// Generated at build time from the checked-in registry description.
include!(concat!(env!("OUT_DIR"), "/model_registry.rs"));

#[cfg(test)]
mod tests;
