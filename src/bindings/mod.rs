//! Language bindings for Tessera.
//!
//! Provides foreign function interfaces (FFI) and language-specific
//! bindings for using Tessera from other programming languages:
//!
//! - [`python`]: PyO3-based Python bindings (`pip install tessera`)
//! - [`wasm`]: wasm-bindgen for TypeScript/JavaScript (`npm install tessera-wasm`)
//!
//! Bindings expose the high-level API from [`crate::api`] with
//! language-appropriate idioms and error handling.
//!
//! # Python Bindings
//!
//! Feature-gated behind `python` feature flag. Provides:
//! - `Tessera` class matching Rust API
//! - `NumPy` array interop for embeddings
//! - Pythonic error messages
//! - Type hints for IDE support
//!
//! Example:
//! ```python
//! from tessera import Tessera
//!
//! embedder = Tessera("colbert-v2")
//! embeddings = embedder.encode("What is ML?")
//! ```
//!
//! # WebAssembly Bindings
//!
//! Feature-gated behind `wasm` feature flag. Provides:
//! - `Tessera` class for TypeScript/JavaScript
//! - Async API with Promise support
//! - `Float32Array` interop for embeddings
//! - Browser and Node.js compatibility
//!
//! Example:
//! ```typescript
//! import { Tessera } from 'tessera-wasm';
//!
//! const embedder = await Tessera.new("colbert-v2");
//! const embeddings = await embedder.encode("What is ML?");
//! ```
//!
//! # Building Bindings
//!
//! Python:
//! ```bash
//! cargo build --release --features python
//! maturin develop
//! ```
//!
//! WASM:
//! ```bash
//! wasm-pack build --target web --features wasm
//! ```

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;
