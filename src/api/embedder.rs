//! Main Tessera embedder interface.
//!
//! Provides user-facing dense, multi-vector, sparse, vision-language, and
//! optionally time-series embedders, plus the model-type-detecting [`Tessera`]
//! factory.
//!
//! # Example
//!
//! ```ignore
//! use tessera::{Tessera, TesseraDense, TesseraMultiVector};
//!
//! let embedder = Tessera::new("colbert-v2")?;
//! let mv_embedder = TesseraMultiVector::new("colbert-v2")?;
//! let dense_embedder = TesseraDense::new("bge-base-en-v1.5")?;
//! ```

mod dense;
mod factory;
mod multi_vector;
mod quantized;
mod sparse;
mod vision;

pub use dense::TesseraDense;
pub use factory::Tessera;
pub use multi_vector::TesseraMultiVector;
pub use quantized::QuantizedEmbeddings;
pub use sparse::TesseraSparse;
pub use vision::TesseraVision;
