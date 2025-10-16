//! WebAssembly bindings for Tessera using wasm-bindgen.
//!
//! Provides JavaScript/TypeScript bindings for running Tessera in
//! web browsers and Node.js environments. Uses wasm-bindgen for
//! interop and supports async operations.
//!
//! # Installation
//!
//! ```bash
//! npm install tessera-wasm
//! ```
//!
//! # Usage (Browser)
//!
//! ```typescript
//! import init, { Tessera } from 'tessera-wasm';
//!
//! // Initialize WASM module
//! await init();
//!
//! // Create embedder
//! const embedder = await Tessera.new("jina-colbert-v2");
//!
//! // Encode text
//! const embeddings = await embedder.encode("What is machine learning?");
//! console.log(embeddings.length);  // num_tokens
//! console.log(embeddings[0].length);  // embedding_dim
//!
//! // Compute similarity
//! const score = await embedder.similarity("query", "document");
//! console.log(`Similarity: ${score}`);
//! ```
//!
//! # Usage (Node.js)
//!
//! ```javascript
//! const { Tessera } = require('tessera-wasm');
//!
//! (async () => {
//!   const embedder = await Tessera.new("colbert-v2");
//!   const embeddings = await embedder.encode("Hello world");
//! })();
//! ```
//!
//! # Building
//!
//! ```bash
//! wasm-pack build --target web --features wasm
//! ```

// This module is only compiled when the "wasm" feature is enabled
// TODO: Implement wasm-bindgen bindings
//
// Required dependencies (add to Cargo.toml with wasm feature):
// wasm-bindgen = "0.2"
// wasm-bindgen-futures = "0.4"
// js-sys = "0.3"
// web-sys = "0.3"
//
// Implementation outline:
// 1. Create WasmTessera struct wrapping crate::api::Tessera
// 2. Use #[wasm_bindgen] attribute for JS interop
// 3. Implement async new() for initialization
// 4. Implement async encode() returning JsValue (Float32Array)
// 5. Implement async encode_batch() for batch processing
// 6. Implement similarity() for convenience
// 7. Add proper error conversion (anyhow::Error → JsValue)
// 8. Handle memory management (wasm-bindgen handles most)

/// Placeholder for WebAssembly bindings.
///
/// This module will contain wasm-bindgen-based JavaScript bindings
/// when the `wasm` feature is enabled.
pub struct WasmTessera {
    // TODO: Wrap crate::api::Tessera
}

// Example implementation structure (not compiled without wasm-bindgen):
//
// #[wasm_bindgen]
// pub struct WasmTessera {
//     inner: crate::api::Tessera,
// }
//
// #[wasm_bindgen]
// impl WasmTessera {
//     #[wasm_bindgen(constructor)]
//     pub async fn new(model: String) -> Result<WasmTessera, JsValue> {
//         let inner = crate::api::Tessera::new(&model)
//             .map_err(|e| JsValue::from_str(&e.to_string()))?;
//         Ok(Self { inner })
//     }
//
//     #[wasm_bindgen]
//     pub async fn encode(&self, text: String) -> Result<JsValue, JsValue> {
//         let embeddings = self.inner.encode(&text)
//             .map_err(|e| JsValue::from_str(&e.to_string()))?;
//         // Convert Vec<Vec<f32>> to nested Float32Array
//         todo!("Convert to JS arrays")
//     }
// }
