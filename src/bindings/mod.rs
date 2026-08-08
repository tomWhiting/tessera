//! Language bindings for Tessera.
//!
//! Provides foreign function interfaces (FFI) and language-specific
//! bindings for using Tessera from other programming languages:
//!
//! - **Python** (requires `python` feature): typed PyO3 facades for dense,
//!   sparse, multi-vector, and vision-language encoders.
//!
//! Bindings expose the high-level API from [`crate::api`] with
//! language-appropriate idioms and error handling.
//!
//! # Python Bindings
//!
//! Feature-gated behind `python` feature flag. Provides:
//! - `TesseraDense`, `TesseraMultiVector`, `TesseraSparse`, and `TesseraVision`
//! - `NumPy` array interop for embeddings
//! - An immutable `ResourcePolicy` shared with the Rust builders
//!
//! Example:
//! ```python
//! from tessera import TesseraMultiVector
//!
//! embedder = TesseraMultiVector("colbert-v2")
//! embeddings = embedder.encode("What is ML?")
//! ```
//!
//! # Building Bindings
//!
//! Python:
//! ```bash
//! cargo build --release --features python
//! maturin develop
//! ```

#[cfg(feature = "python")]
pub mod python;
