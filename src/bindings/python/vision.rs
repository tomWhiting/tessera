#![allow(missing_docs)]
// PyO3's `#[pymethods]` expansion converts `PyErr` through `Into<PyErr>`.
#![allow(clippy::useless_conversion)]

use super::conversion::{
    tessera_error_to_pyerr, token_embeddings_to_pyarray, vision_embedding_to_pyarray,
};
use super::dtype::{model_dtype_name, parse_model_dtype};
use super::resource_policy::{
    effective_policy_for_model, validated_text_input, validated_text_pair, PyResourcePolicy,
};
use crate::api::TesseraVisionBuilder;
use crate::core::{TokenEmbeddings, VisionEmbedding};
use crate::runtime::ResourcePolicy;
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::{pyclass, pymethods, Bound, Py, PyRef, PyResult, Python};

#[cfg(test)]
#[path = "vision/tests.rs"]
mod tests;

#[pyclass(name = "TesseraVision")]
pub struct PyTesseraVision {
    inner: crate::api::TesseraVision,
    resource_policy: ResourcePolicy,
}

#[pymethods]
impl PyTesseraVision {
    /// Create a vision embedder for a registered model.
    #[new]
    #[pyo3(signature = (model_id, *, resource_policy=None, dtype="f32"))]
    fn new(
        py: Python<'_>,
        model_id: &str,
        resource_policy: Option<PyRef<'_, PyResourcePolicy>>,
        dtype: &str,
    ) -> PyResult<Self> {
        let dtype = parse_model_dtype(dtype)?;
        let mut builder = TesseraVisionBuilder::new().model(model_id);
        let resource_policy = resource_policy.map(|policy| policy.inner());
        if let Some(policy) = resource_policy {
            builder = builder.resource_policy(policy);
        }
        builder = builder.dtype(dtype);
        let inner = py
            .allow_threads(move || builder.build())
            .map_err(tessera_error_to_pyerr)?;
        Ok(Self {
            inner,
            resource_policy: effective_policy_for_model(model_id, resource_policy),
        })
    }

    /// Encode an image or document path as patch embeddings.
    fn encode_document(&self, py: Python<'_>, path: &str) -> PyResult<Py<PyArray2<f32>>> {
        let path = validated_text_input(path, "path", self.resource_policy)?;
        let embedding = py
            .allow_threads(|| self.inner.encode_document(&path))
            .map_err(tessera_error_to_pyerr)?;
        vision_embedding_to_pyarray(py, &embedding)
    }

    /// Encode text as query token embeddings.
    fn encode_query(&self, py: Python<'_>, text: &str) -> PyResult<Py<PyArray2<f32>>> {
        let text = validated_text_input(text, "text", self.resource_policy)?;
        let embedding = py
            .allow_threads(|| self.inner.encode_query(&text))
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
        validate_matrix_shapes(
            query_shape,
            document_shape,
            self.inner.embedding_dim(),
            self.resource_policy,
        )
        .map_err(PyValueError::new_err)?;
        let query_readonly = query.readonly();
        let document_readonly = document.readonly();
        let query_values = query_readonly.as_slice().map_err(|err| {
            PyValueError::new_err(format!("Failed to get query array slice: {err}"))
        })?;
        let document_values = document_readonly.as_slice().map_err(|err| {
            PyValueError::new_err(format!("Failed to get document array slice: {err}"))
        })?;
        validate_finite_matrices(query_values, document_values).map_err(PyValueError::new_err)?;

        let query_array = ndarray::Array2::from_shape_vec(
            (query_shape[0], query_shape[1]),
            query_values.to_vec(),
        )
        .map_err(|err| PyValueError::new_err(format!("Failed to create query array: {err}")))?;
        let query_embedding = TokenEmbeddings::new(query_array, String::new())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let document_embedding = VisionEmbedding::new(
            document_values
                .chunks(document_shape[1])
                .map(<[f32]>::to_vec)
                .collect(),
            document_shape[0],
            document_shape[1],
            None,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

        query
            .py()
            .allow_threads(|| self.inner.search(&query_embedding, &document_embedding))
            .map_err(tessera_error_to_pyerr)
    }

    /// Encode and score a text query against a document path.
    fn search_document(
        &self,
        py: Python<'_>,
        query_text: &str,
        document_path: &str,
    ) -> PyResult<f32> {
        let (query_text, document_path) = validated_text_pair(
            query_text,
            "query_text",
            document_path,
            "document_path",
            self.resource_policy,
        )?;
        py.allow_threads(|| self.inner.search_document(&query_text, &document_path))
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

    /// Return the parameter dtype selected for this loaded model.
    fn dtype(&self) -> &'static str {
        model_dtype_name(self.inner.model_dtype())
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

fn validate_matrix_shapes(
    query: &[usize],
    document: &[usize],
    expected_dim: usize,
    policy: ResourcePolicy,
) -> Result<(), String> {
    let [query_rows, query_dim] = query else {
        return Err(format!(
            "Query matrix must be two-dimensional; got shape with {} axes",
            query.len()
        ));
    };
    let [document_rows, document_dim] = document else {
        return Err(format!(
            "Document matrix must be two-dimensional; got shape with {} axes",
            document.len()
        ));
    };
    let (query_rows, query_dim) = (*query_rows, *query_dim);
    let (document_rows, document_dim) = (*document_rows, *document_dim);
    if query_rows == 0 || document_rows == 0 || query_dim == 0 || document_dim == 0 {
        return Err("Query and document matrices must have non-zero rows and columns".to_string());
    }
    if query_dim != document_dim || query_dim != expected_dim {
        return Err(format!(
            "Matrix dimensions must both equal model dimension {expected_dim}; got query {query_dim}, document {document_dim}"
        ));
    }
    policy
        .validate_sequence(query_rows)
        .map_err(|error| error.to_string())?;
    policy
        .validate_job(document_rows, 0)
        .map_err(|error| error.to_string())?;
    let elements = query_rows
        .saturating_mul(query_dim)
        .saturating_add(document_rows.saturating_mul(document_dim));
    policy
        .validate_output_bytes(elements.saturating_mul(std::mem::size_of::<f32>()))
        .map_err(|error| error.to_string())
}

fn validate_finite_matrices(query: &[f32], document: &[f32]) -> Result<(), String> {
    if query.iter().chain(document).all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err("Query and document matrices must contain only finite values".to_string())
    }
}
