use crate::core::{DenseEmbedding, SparseEmbedding, TokenEmbeddings, VisionEmbedding};
use crate::error::TesseraError;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::{Py, PyErr, PyResult, Python};

#[cfg(test)]
#[path = "conversion/tests.rs"]
mod tests;

#[derive(Debug, PartialEq, Eq)]
enum ExceptionKind {
    Runtime,
    Value,
    Io,
}

fn exception_kind(err: &TesseraError) -> ExceptionKind {
    match err {
        TesseraError::UnsupportedWeightsFormat { .. }
        | TesseraError::UnsupportedDimension { .. }
        | TesseraError::QuantizationError(_)
        | TesseraError::DimensionMismatch { .. }
        | TesseraError::ConfigError(_)
        | TesseraError::MatryoshkaError(_) => ExceptionKind::Value,
        TesseraError::IoError(_) => ExceptionKind::Io,
        TesseraError::FetchingNotBuiltIn { .. }
        | TesseraError::ModelNotFound { .. }
        | TesseraError::ModelLoadError { .. }
        | TesseraError::EncodingError { .. }
        | TesseraError::DeviceError(_)
        | TesseraError::TokenizationError(_)
        | TesseraError::TensorError(_)
        | TesseraError::Other(_) => ExceptionKind::Runtime,
    }
}

pub(super) fn tessera_error_to_pyerr(err: TesseraError) -> PyErr {
    let message = err.to_string();
    match exception_kind(&err) {
        ExceptionKind::Runtime => PyRuntimeError::new_err(message),
        ExceptionKind::Value => PyValueError::new_err(message),
        ExceptionKind::Io => PyIOError::new_err(message),
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
