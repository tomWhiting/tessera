#![allow(missing_docs)]
// PyO3's `#[pymethods]` expansion converts `PyErr` through `Into<PyErr>`.
#![allow(clippy::useless_conversion)]

use super::conversion::{sparse_embedding_to_pyarrays, tessera_error_to_pyerr};
use super::resource_policy::{effective_policy_for_model, extract_text_batch, PyResourcePolicy};
use crate::api::TesseraSparseBuilder;
use crate::runtime::ResourcePolicy;
use numpy::PyArray1;
use pyo3::prelude::{pyclass, pymethods, Bound, Py, PyRef, PyResult, Python};
use pyo3::types::PySequence;

type SparseArrays = (Py<PyArray1<i32>>, Py<PyArray1<f32>>);

#[pyclass(name = "TesseraSparse")]
pub struct PyTesseraSparse {
    inner: crate::api::TesseraSparse,
    resource_policy: ResourcePolicy,
}

#[pymethods]
impl PyTesseraSparse {
    /// Create a sparse embedder for a registered model.
    #[new]
    #[pyo3(signature = (model_id, *, resource_policy=None))]
    fn new(model_id: &str, resource_policy: Option<PyRef<'_, PyResourcePolicy>>) -> PyResult<Self> {
        let resource_policy = resource_policy.map(|policy| policy.inner());
        let mut builder = TesseraSparseBuilder::new().model(model_id);
        if let Some(policy) = resource_policy {
            builder = builder.resource_policy(policy);
        }
        let inner = builder.build().map_err(tessera_error_to_pyerr)?;
        Ok(Self {
            inner,
            resource_policy: effective_policy_for_model(model_id, resource_policy),
        })
    }

    /// Encode one string as `(indices, values)` NumPy arrays.
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<SparseArrays> {
        let embedding = py
            .allow_threads(|| self.inner.encode(text))
            .map_err(tessera_error_to_pyerr)?;
        sparse_embedding_to_pyarrays(py, &embedding)
    }

    /// Encode multiple strings as sparse array pairs.
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: &Bound<'_, PySequence>,
    ) -> PyResult<Vec<SparseArrays>> {
        let texts = extract_text_batch(texts, self.resource_policy)?;
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embeddings = py
            .allow_threads(|| self.inner.encode_batch(&text_refs))
            .map_err(tessera_error_to_pyerr)?;
        embeddings
            .iter()
            .map(|embedding| sparse_embedding_to_pyarrays(py, embedding))
            .collect()
    }

    /// Compute sparse dot-product similarity between two strings.
    fn similarity(&self, text_a: &str, text_b: &str) -> PyResult<f32> {
        self.inner
            .similarity(text_a, text_b)
            .map_err(tessera_error_to_pyerr)
    }

    /// Return the vocabulary dimension.
    fn vocab_size(&self) -> PyResult<usize> {
        Ok(self.inner.vocab_size())
    }

    /// Return the registered model identifier.
    fn model(&self) -> PyResult<String> {
        Ok(self.inner.model().to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TesseraSparse(model='{}', vocab_size={})",
            self.inner.model(),
            self.inner.vocab_size()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}
