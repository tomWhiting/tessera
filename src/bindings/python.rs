#![allow(missing_docs)]
//! Python bindings for Tessera using `PyO3`.
//!
//! The implementation is split by public Python class so registration, array
//! conversion, and each embedding paradigm can evolve independently.

mod conversion;
mod dense;
mod multivector;
mod resource_policy;
mod sparse;
mod vision;

pub use dense::PyTesseraDense;
pub use multivector::PyTesseraMultiVector;
pub use resource_policy::PyResourcePolicy;
pub use sparse::PyTesseraSparse;
pub use vision::PyTesseraVision;

use pyo3::prelude::{pymodule, Bound, PyModule, PyResult, Python};
use pyo3::types::PyModuleMethods;

#[pymodule]
fn tessera(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResourcePolicy>()?;
    module.add_class::<PyTesseraDense>()?;
    module.add_class::<PyTesseraMultiVector>()?;
    module.add_class::<PyTesseraSparse>()?;
    module.add_class::<PyTesseraVision>()?;
    let doc = "Tessera: Multi-paradigm embedding library for Rust\n\n\
        Provides typed embedding APIs with explicit resource limits and optional GPU acceleration.\n\n\
        Supports:\n\
        - Dense embeddings (TesseraDense): Single-vector sentence embeddings\n\
        - Multi-vector embeddings (TesseraMultiVector): ColBERT token-level embeddings\n\
        - Sparse embeddings (TesseraSparse): SPLADE vocabulary-space embeddings\n\
        - Vision-language (TesseraVision): ColPali document retrieval\n\n\
        Examples:\n\
        >>> from tessera import TesseraDense, TesseraMultiVector, TesseraSparse\n\
        >>> # Dense embeddings\n\
        >>> dense = TesseraDense('bge-base-en-v1.5')\n\
        >>> emb = dense.encode('What is machine learning?')\n\
        >>> print(emb.shape)  # (768,)\n\n\
        >>> # Multi-vector embeddings\n\
        >>> colbert = TesseraMultiVector('colbert-v2')\n\
        >>> embs = colbert.encode('What is machine learning?')\n\
        >>> print(embs.shape)  # (num_tokens, 128)\n\n\
        >>> # Sparse embeddings\n\
        >>> sparse = TesseraSparse('splade-pp-en-v1')\n\
        >>> indices, values = sparse.encode('machine learning')\n\
        >>> print(f'Non-zero dims: {len(indices)}')  # ~100-200\n\
        >>> # ResourcePolicy is immutable; explicitly raise only the limits you need\n\
        >>> from tessera import ResourcePolicy, TesseraVision\n\
        >>> vision_policy = ResourcePolicy(max_model_bytes=12_000_000_000)\n\
        >>> vision = TesseraVision('colpali-v1.2', resource_policy=vision_policy)\n\
    ";

    module.add("__doc__", doc)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
