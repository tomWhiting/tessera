#![allow(missing_docs)]
// PyO3's `#[pymethods]` expansion converts `PyErr` through `Into<PyErr>`.
#![allow(clippy::useless_conversion)]

use super::conversion::{tessera_error_to_pyerr, token_embeddings_to_pyarray};
use super::resource_policy::{effective_policy_for_model, extract_text_batch, PyResourcePolicy};
use crate::api::TesseraMultiVectorBuilder;
use crate::runtime::ResourcePolicy;
use numpy::PyArray2;
use pyo3::prelude::{pyclass, pymethods, Bound, Py, PyRef, PyResult, Python};
use pyo3::types::PySequence;

#[pyclass(name = "TesseraMultiVector")]
pub struct PyTesseraMultiVector {
    inner: crate::api::TesseraMultiVector,
    resource_policy: ResourcePolicy,
}

#[pymethods]
impl PyTesseraMultiVector {
    /// Create a multi-vector embedder for a registered model.
    #[new]
    #[pyo3(signature = (model_id, *, resource_policy=None))]
    fn new(model_id: &str, resource_policy: Option<PyRef<'_, PyResourcePolicy>>) -> PyResult<Self> {
        let resource_policy = resource_policy.map(|policy| policy.inner());
        let mut builder = TesseraMultiVectorBuilder::new().model(model_id);
        if let Some(policy) = resource_policy {
            builder = builder.resource_policy(policy);
        }
        let inner = builder.build().map_err(tessera_error_to_pyerr)?;
        Ok(Self {
            inner,
            resource_policy: effective_policy_for_model(model_id, resource_policy),
        })
    }

    /// Encode one string as token-level embeddings.
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Py<PyArray2<f32>>> {
        let embeddings = py
            .allow_threads(|| self.inner.encode(text))
            .map_err(tessera_error_to_pyerr)?;
        Ok(token_embeddings_to_pyarray(py, embeddings))
    }

    /// Encode multiple strings and return one token matrix per input.
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: &Bound<'_, PySequence>,
    ) -> PyResult<Vec<Py<PyArray2<f32>>>> {
        let texts = extract_text_batch(texts, self.resource_policy)?;
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embeddings = py
            .allow_threads(|| self.inner.encode_batch(&text_refs))
            .map_err(tessera_error_to_pyerr)?;
        Ok(embeddings
            .into_iter()
            .map(|embedding| token_embeddings_to_pyarray(py, embedding))
            .collect())
    }

    /// Compute MaxSim similarity between two strings.
    fn similarity(&self, text_a: &str, text_b: &str) -> PyResult<f32> {
        self.inner
            .similarity(text_a, text_b)
            .map_err(tessera_error_to_pyerr)
    }

    /// Return the per-token embedding dimension.
    fn dimension(&self) -> PyResult<usize> {
        Ok(self.inner.dimension())
    }

    /// Return the registered model identifier.
    fn model(&self) -> PyResult<String> {
        Ok(self.inner.model().to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TesseraMultiVector(model='{}', dimension={})",
            self.inner.model(),
            self.inner.dimension()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}
