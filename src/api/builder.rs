//! Builder pattern for configuring Tessera embedders.
//!
//! Provides fluent, type-specific builders for dense, multi-vector, sparse,
//! and vision-language models.
//!
//! # Example
//!
//! ```ignore
//! use tessera::api::{QuantizationConfig, TesseraDenseBuilder, TesseraMultiVectorBuilder};
//!
//! let mv_embedder = TesseraMultiVectorBuilder::new()
//!     .model("colbert-v2")
//!     .quantization(QuantizationConfig::Binary)
//!     .build()?;
//!
//! let dense_embedder = TesseraDenseBuilder::new()
//!     .model("nomic-embed-v1.5")
//!     .dimension(256)
//!     .build()?;
//! ```

mod dense;
mod multi_vector;
mod quantization;
mod sparse;
mod vision;

pub use dense::TesseraDenseBuilder;
pub use multi_vector::TesseraMultiVectorBuilder;
pub use quantization::QuantizationConfig;
pub use sparse::TesseraSparseBuilder;
pub use vision::TesseraVisionBuilder;

use crate::error::{Result, TesseraError};
use crate::models::ModelInfo;

/// Rejects catalog metadata that does not have a runtime adapter.
pub(crate) fn ensure_runnable_model(model: &ModelInfo) -> Result<()> {
    if model.is_runnable() {
        return Ok(());
    }

    Err(TesseraError::ConfigError(format!(
        "Model '{}' is catalog-only and cannot be loaded: {}",
        model.id, model.support_note
    )))
}
