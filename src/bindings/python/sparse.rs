#![allow(missing_docs)]
// PyO3's `#[pymethods]` expansion converts `PyErr` through `Into<PyErr>`.
#![allow(clippy::useless_conversion)]

use super::conversion::{sparse_embedding_to_pyarrays, tessera_error_to_pyerr};
use super::dtype::{model_dtype_name, parse_model_dtype};
use super::resource_policy::{
    effective_policy_for_model, extract_text_batch, validated_text_input, validated_text_pair,
    PyResourcePolicy,
};
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
    #[pyo3(signature = (model_id, *, resource_policy=None, dtype="f32"))]
    fn new(
        py: Python<'_>,
        model_id: &str,
        resource_policy: Option<PyRef<'_, PyResourcePolicy>>,
        dtype: &str,
    ) -> PyResult<Self> {
        let dtype = parse_model_dtype(dtype)?;
        let resource_policy = resource_policy.map(|policy| policy.inner());
        let mut builder = TesseraSparseBuilder::new().model(model_id).dtype(dtype);
        if let Some(policy) = resource_policy {
            builder = builder.resource_policy(policy);
        }
        let inner = py
            .allow_threads(move || builder.build())
            .map_err(tessera_error_to_pyerr)?;
        Ok(Self {
            inner,
            resource_policy: effective_policy_for_model(model_id, resource_policy),
        })
    }

    /// Encode one string as `(indices, values)` NumPy arrays.
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<SparseArrays> {
        let text = validated_text_input(text, "text", self.resource_policy)?;
        let embedding = py
            .allow_threads(|| self.inner.encode(&text))
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
    fn similarity(&self, py: Python<'_>, text_a: &str, text_b: &str) -> PyResult<f32> {
        let (text_a, text_b) =
            validated_text_pair(text_a, "text_a", text_b, "text_b", self.resource_policy)?;
        py.allow_threads(|| self.inner.similarity(&text_a, &text_b))
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

    /// Return the parameter dtype selected for this loaded model.
    fn dtype(&self) -> &'static str {
        model_dtype_name(self.inner.model_dtype())
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
