#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::collection_is_never_read,
    clippy::type_complexity,
    clippy::unnecessary_wraps
)]

use crate::core::{DenseEmbedding, SparseEmbedding, TokenEmbeddings, VisionEmbedding};
use crate::error::TesseraError;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::{Py, PyErr, PyResult, Python};

pub(super) fn tessera_error_to_pyerr(err: TesseraError) -> PyErr {
    match err {
        TesseraError::ModelNotFound { model_id } => {
            PyRuntimeError::new_err(format!("Model '{model_id}' not found in registry"))
        }
        TesseraError::ModelLoadError { model_id, source } => {
            PyRuntimeError::new_err(format!("Failed to load model '{model_id}': {source}"))
        }
        TesseraError::EncodingError { context, source } => {
            PyRuntimeError::new_err(format!("Encoding failed: {context} - {source}"))
        }
        TesseraError::UnsupportedDimension {
            model_id,
            requested,
            supported,
        } => PyValueError::new_err(format!(
            "Unsupported dimension {requested} for model '{model_id}'. Supported: {supported:?}"
        )),
        TesseraError::DeviceError(msg) => PyRuntimeError::new_err(format!("Device error: {msg}")),
        TesseraError::QuantizationError(msg) => {
            PyValueError::new_err(format!("Quantization error: {msg}"))
        }
        TesseraError::DimensionMismatch { expected, actual } => PyValueError::new_err(format!(
            "Dimension mismatch: expected {expected}, got {actual}"
        )),
        TesseraError::TokenizationError(err) => {
            PyRuntimeError::new_err(format!("Tokenization error: {err}"))
        }
        TesseraError::ConfigError(msg) => {
            PyValueError::new_err(format!("Configuration error: {msg}"))
        }
        TesseraError::MatryoshkaError(msg) => {
            PyValueError::new_err(format!("Matryoshka truncation error: {msg}"))
        }
        TesseraError::IoError(err) => PyIOError::new_err(format!("IO error: {err}")),
        TesseraError::TensorError(err) => {
            PyRuntimeError::new_err(format!("Tensor operation error: {err}"))
        }
        TesseraError::Other(err) => PyRuntimeError::new_err(format!("Error: {err}")),
    }
}

pub(super) fn token_embeddings_to_pyarray(
    py: Python<'_>,
    embeddings: TokenEmbeddings,
) -> Py<PyArray2<f32>> {
    embeddings.into_matrix().into_pyarray_bound(py).unbind()
}

pub(super) fn dense_embedding_to_pyarray(
    py: Python<'_>,
    embedding: DenseEmbedding,
) -> Py<PyArray1<f32>> {
    embedding.into_values().into_pyarray_bound(py).unbind()
}

pub(super) fn sparse_embedding_to_pyarrays(
    py: Python<'_>,
    embedding: &SparseEmbedding,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
    let indices = embedding
        .entries()
        .iter()
        .map(|(index, _)| {
            i32::try_from(*index).map_err(|_| {
                PyValueError::new_err(format!(
                    "Sparse vocabulary index {index} cannot be represented as NumPy int32"
                ))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let values = embedding
        .entries()
        .iter()
        .map(|(_, value)| *value)
        .collect();
    Ok((
        PyArray1::from_vec_bound(py, indices).unbind(),
        PyArray1::from_vec_bound(py, values).unbind(),
    ))
}

pub(super) fn vision_embedding_to_pyarray(
    py: Python<'_>,
    embedding: &VisionEmbedding,
) -> PyResult<Py<PyArray2<f32>>> {
    Ok(PyArray2::from_vec2_bound(py, embedding.vectors())?.unbind())
}
