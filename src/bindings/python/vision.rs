#![allow(missing_docs)]
// PyO3's `#[pymethods]` expansion converts `PyErr` through `Into<PyErr>`.
#![allow(clippy::useless_conversion)]

use super::conversion::{
    tessera_error_to_pyerr, token_embeddings_to_pyarray, vision_embedding_to_pyarray,
};
use super::resource_policy::PyResourcePolicy;
use crate::api::TesseraVisionBuilder;
use crate::core::{TokenEmbeddings, VisionEmbedding};
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::{pyclass, pymethods, Bound, Py, PyRef, PyResult, Python};

#[pyclass(name = "TesseraVision")]
pub struct PyTesseraVision {
    inner: crate::api::TesseraVision,
}

#[pymethods]
impl PyTesseraVision {
    /// Create a vision embedder for a registered model.
    #[new]
    #[pyo3(signature = (model_id, *, resource_policy=None))]
    fn new(model_id: &str, resource_policy: Option<PyRef<'_, PyResourcePolicy>>) -> PyResult<Self> {
        let mut builder = TesseraVisionBuilder::new().model(model_id);
        if let Some(policy) = resource_policy {
            builder = builder.resource_policy(policy.inner());
        }
        let inner = builder.build().map_err(tessera_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Encode an image or document path as patch embeddings.
    fn encode_document(&self, py: Python<'_>, path: &str) -> PyResult<Py<PyArray2<f32>>> {
        let embedding = py
            .allow_threads(|| self.inner.encode_document(path))
            .map_err(tessera_error_to_pyerr)?;
        vision_embedding_to_pyarray(py, &embedding)
    }

    /// Encode text as query token embeddings.
    fn encode_query(&self, py: Python<'_>, text: &str) -> PyResult<Py<PyArray2<f32>>> {
        let embedding = py
            .allow_threads(|| self.inner.encode_query(text))
            .map_err(tessera_error_to_pyerr)?;
        Ok(token_embeddings_to_pyarray(py, embedding))
    }

    /// Score previously encoded query and document matrices.
    fn search(
        &self,
        query: &Bound<'_, PyArray2<f32>>,
        document: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<f32> {
        let query_shape = query.shape();
        let document_shape = document.shape();
        let query_readonly = query.readonly();
        let document_readonly = document.readonly();
        let query_values = query_readonly.as_slice().map_err(|err| {
            PyValueError::new_err(format!("Failed to get query array slice: {err}"))
        })?;
        let document_values = document_readonly.as_slice().map_err(|err| {
            PyValueError::new_err(format!("Failed to get document array slice: {err}"))
        })?;

        let query_array = ndarray::Array2::from_shape_vec(
            (query_shape[0], query_shape[1]),
            query_values.to_vec(),
        )
        .map_err(|err| PyValueError::new_err(format!("Failed to create query array: {err}")))?;
        let query_embedding = TokenEmbeddings {
            embeddings: query_array,
            num_tokens: query_shape[0],
            embedding_dim: query_shape[1],
            text: String::new(),
        };
        let document_embedding = VisionEmbedding {
            embeddings: document_values
                .chunks(document_shape[1])
                .map(<[f32]>::to_vec)
                .collect(),
            num_patches: document_shape[0],
            embedding_dim: document_shape[1],
            source: None,
        };

        self.inner
            .search(&query_embedding, &document_embedding)
            .map_err(tessera_error_to_pyerr)
    }

    /// Encode and score a text query against a document path.
    fn search_document(&self, query_text: &str, document_path: &str) -> PyResult<f32> {
        self.inner
            .search_document(query_text, document_path)
            .map_err(tessera_error_to_pyerr)
    }

    /// Return the patch embedding dimension.
    fn embedding_dim(&self) -> PyResult<usize> {
        Ok(self.inner.embedding_dim())
    }

    /// Return the expected number of patches.
    fn num_patches(&self) -> PyResult<usize> {
        Ok(self.inner.num_patches())
    }

    /// Return the registered model identifier.
    fn model(&self) -> PyResult<String> {
        Ok(self.inner.model().to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TesseraVision(model='{}', patches={}, dim={})",
            self.inner.model(),
            self.inner.num_patches(),
            self.inner.embedding_dim()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}
