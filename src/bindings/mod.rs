//! Language bindings for Tessera.
//!
//! Provides foreign function interfaces (FFI) and language-specific
//! bindings for using Tessera from other programming languages:
//!
//! - **Python** (requires `python` feature): typed PyO3 facades for dense,
//!   sparse, multi-vector, and vision-language encoders.
//! - **WebAssembly** (requires `wasm` feature): a reserved placeholder; no
//!   JavaScript runtime API is implemented yet.
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
//! # WebAssembly Bindings
//!
//! The `wasm` feature currently compiles a placeholder only. It deliberately
//! makes no npm package or browser-runtime claim.
//!
//! # Building Bindings
//!
//! Python:
//! ```bash
//! cargo build --release --features python
//! maturin develop
//! ```
//!
//! WebAssembly bindings need a separate implementation before they are usable.

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;
