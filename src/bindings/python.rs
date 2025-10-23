#![allow(
    clippy::doc_markdown,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_pass_by_value,
    clippy::unnecessary_wraps,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_link_with_quotes,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::wildcard_imports,
    clippy::useless_conversion,
    clippy::type_complexity,
    clippy::collection_is_never_read,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::trivially_copy_pass_by_ref,
    clippy::unused_self,
    clippy::if_same_then_else,
    clippy::derive_partial_eq_without_eq,
    clippy::match_wildcard_for_single_variants,
    clippy::no_effect_underscore_binding,
    missing_docs
)]
//! Python bindings for Tessera using `PyO3`.
//!
//! Provides a Pythonic interface to Tessera's embedding capabilities.
//! Integrates with `NumPy` for efficient array handling and follows
//! Python conventions for error handling and API design.
//!
//! # Installation
//!
//! ```bash
//! pip install tessera
//! ```
//!
//! # Usage
//!
//! ```python
//! from tessera import TesseraMultiVector
//! import numpy as np
//!
//! # Create embedder
//! embedder = TesseraMultiVector("colbert-v2")
//!
//! # Encode single text
//! embeddings = embedder.encode("What is machine learning?")
//! print(embeddings.shape)  # (num_tokens, embedding_dim)
//!
//! # Encode batch
//! texts = ["First text", "Second text"]
//! batch_embeddings = embedder.encode_batch(texts)
//!
//! # Compute similarity
//! score = embedder.similarity("query", "document")
//! print(f"Similarity: {score}")
//! ```

#[cfg(feature = "python")]
use crate::core::{DenseEmbedding, SparseEmbedding, TokenEmbeddings, VisionEmbedding};
#[cfg(feature = "python")]
use crate::error::TesseraError;
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::{pyclass, pymethods, pymodule, Bound, Py, PyErr, PyModule, PyResult, Python};
#[cfg(feature = "python")]
use pyo3::types::PyModuleMethods;

// ============================================================================
// Error Conversion
// ============================================================================

#[cfg(feature = "python")]
fn tessera_error_to_pyerr(err: TesseraError) -> PyErr {
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
        TesseraError::TokenizationError(e) => {
            PyRuntimeError::new_err(format!("Tokenization error: {e}"))
        }
        TesseraError::ConfigError(msg) => {
            PyValueError::new_err(format!("Configuration error: {msg}"))
        }
        TesseraError::MatryoshkaError(msg) => {
            PyValueError::new_err(format!("Matryoshka truncation error: {msg}"))
        }
        TesseraError::IoError(e) => PyIOError::new_err(format!("IO error: {e}")),
        TesseraError::TensorError(e) => {
            PyRuntimeError::new_err(format!("Tensor operation error: {e}"))
        }
        TesseraError::Other(e) => PyRuntimeError::new_err(format!("Error: {e}")),
    }
}

// ============================================================================
// NumPy Conversion Helpers
// ============================================================================

#[cfg(feature = "python")]
fn token_embeddings_to_pyarray(
    py: Python<'_>,
    embeddings: &TokenEmbeddings,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert ndarray::Array2 to PyArray2
    // The Array2 has shape (num_tokens, embedding_dim)
    // Collect rows into Vec<Vec<f32>>
    let rows: Vec<Vec<f32>> = (0..embeddings.num_tokens)
        .map(|i| {
            embeddings
                .embeddings
                .row(i)
                .iter()
                .copied()
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<_>>();

    // Create PyArray2 from the row vectors using PyO3 0.22 API
    let array_bound = PyArray2::from_vec2_bound(py, &rows)?;

    // Return as Py<PyArray2> (owned Python object)
    Ok(array_bound.unbind())
}

/// Convert a `DenseEmbedding` (Vec<f32>) to `PyArray1`<f32>.
#[cfg(feature = "python")]
#[allow(clippy::unnecessary_wraps)]
fn dense_embedding_to_pyarray(
    py: Python<'_>,
    embedding: &DenseEmbedding,
) -> PyResult<Py<PyArray1<f32>>> {
    // Convert Array1 to Vec
    let vec = embedding.embedding.to_vec();

    // Create PyArray1 from vec using PyO3 0.22 API
    let array_bound = PyArray1::from_vec_bound(py, vec);

    // Return as Py<PyArray1> (owned Python object)
    Ok(array_bound.unbind())
}

/// Convert a `SparseEmbedding` to (indices, values) tuple.
#[cfg(feature = "python")]
#[allow(clippy::unnecessary_wraps, clippy::type_complexity)]
fn sparse_embedding_to_pyarrays(
    py: Python<'_>,
    embedding: &SparseEmbedding,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
    // Extract indices and values separately
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let indices: Vec<i32> = embedding
        .weights
        .iter()
        .map(|(idx, _)| *idx as i32)
        .collect();
    let values: Vec<f32> = embedding.weights.iter().map(|(_, val)| *val).collect();

    // Create PyArray1s
    let indices_array = PyArray1::from_vec_bound(py, indices).unbind();
    let values_array = PyArray1::from_vec_bound(py, values).unbind();

    Ok((indices_array, values_array))
}

/// Convert a VisionEmbedding to PyArray2<f32>.
#[cfg(feature = "python")]
fn vision_embedding_to_pyarray(
    py: Python<'_>,
    embedding: &VisionEmbedding,
) -> PyResult<Py<PyArray2<f32>>> {
    // VisionEmbedding has Vec<Vec<f32>> structure
    // Convert to PyArray2 using from_vec2
    let array_bound = PyArray2::from_vec2_bound(py, &embedding.embeddings)?;

    Ok(array_bound.unbind())
}

/// Convert a candle Tensor to PyArray2<f32> (for time series).
#[cfg(feature = "python")]
fn tensor_to_pyarray2(py: Python<'_>, tensor: &candle_core::Tensor) -> PyResult<Py<PyArray2<f32>>> {
    use candle_core::Device;

    // Get tensor on CPU
    let tensor_cpu = tensor
        .to_device(&Device::Cpu)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to move tensor to CPU: {}", e)))?;

    // Get dimensions
    let dims = tensor_cpu.dims();
    if dims.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Expected 2D tensor, got {}D",
            dims.len()
        )));
    }

    // Flatten to Vec<f32>
    let data = tensor_cpu
        .flatten_all()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to flatten tensor: {}", e)))?;

    let vec = data
        .to_vec1::<f32>()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert tensor to vec: {}", e)))?;

    // Create rows for PyArray2
    let rows: Vec<Vec<f32>> = vec.chunks(dims[1]).map(|chunk| chunk.to_vec()).collect();

    let array_bound = PyArray2::from_vec2_bound(py, &rows)?;
    Ok(array_bound.unbind())
}

/// Convert a candle Tensor to PyArray3<f32> (for time series quantiles).
#[cfg(feature = "python")]
fn tensor_to_pyarray3(py: Python<'_>, tensor: &candle_core::Tensor) -> PyResult<Py<PyArray3<f32>>> {
    use candle_core::Device;

    // Get tensor on CPU
    let tensor_cpu = tensor
        .to_device(&Device::Cpu)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to move tensor to CPU: {}", e)))?;

    // Get dimensions
    let dims = tensor_cpu.dims();
    if dims.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "Expected 3D tensor, got {}D",
            dims.len()
        )));
    }

    // Flatten to Vec<f32>
    let data = tensor_cpu
        .flatten_all()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to flatten tensor: {}", e)))?;

    let vec = data
        .to_vec1::<f32>()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert tensor to vec: {}", e)))?;

    // Reshape to 3D: [batch, prediction_length, num_quantiles]
    let batch_size = dims[0];
    let pred_len = dims[1];
    let num_quantiles = dims[2];

    // Build the 3D structure
    let mut data_3d = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let mut batch_data = Vec::with_capacity(pred_len);
        for p in 0..pred_len {
            let mut quantile_data = Vec::with_capacity(num_quantiles);
            for q in 0..num_quantiles {
                let idx = b * (pred_len * num_quantiles) + p * num_quantiles + q;
                quantile_data.push(vec[idx]);
            }
            batch_data.push(quantile_data);
        }
        data_3d.push(batch_data);
    }

    // Convert to PyArray3
    // PyO3 0.22 doesn't have from_vec3, so we need to use from_vec and reshape
    let array_bound = PyArray1::from_vec_bound(py, vec)
        .reshape([batch_size, pred_len, num_quantiles])
        .map_err(|e| PyValueError::new_err(format!("Failed to reshape to 3D: {}", e)))?;

    Ok(array_bound.unbind())
}

/// Convert PyArray2<f32> to candle Tensor.
#[cfg(feature = "python")]
fn pyarray2_to_tensor(array: &Bound<'_, PyArray2<f32>>) -> PyResult<candle_core::Tensor> {
    use candle_core::{Device, Tensor};

    // Get shape
    let shape = array.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Expected 2D array, got {}D",
            shape.len()
        )));
    }

    // Get readonly array and convert to Vec
    let readonly = array.readonly();
    let slice = readonly
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to get array slice: {}", e)))?;

    // Create tensor
    Tensor::from_slice(slice, (shape[0], shape[1]), &Device::Cpu)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))
}

// ============================================================================
// PyTesseraMultiVector - ColBERT Multi-Vector Embedder
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "TesseraMultiVector")]
pub struct PyTesseraMultiVector {
    inner: crate::api::TesseraMultiVector,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTesseraMultiVector {
    /// Create a new multi-vector embedder.
    ///
    /// Args:
    ///     model_id: Model identifier from the registry (e.g., "colbert-v2", "jina-colbert-v2")
    ///
    /// Returns:
    ///     Initialized embedder ready for use
    ///
    /// Raises:
    ///     RuntimeError: If model is not found or cannot be loaded
    ///
    /// Example:
    ///     >>> embedder = TesseraMultiVector("colbert-v2")
    ///     >>> embeddings = embedder.encode("What is machine learning?")
    #[new]
    fn new(model_id: &str) -> PyResult<Self> {
        let inner =
            crate::api::TesseraMultiVector::new(model_id).map_err(tessera_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Encode a single text into token-level embeddings.
    ///
    /// Args:
    ///     text: Text to encode
    ///
    /// Returns:
    ///     NumPy array of shape (num_tokens, embedding_dim) with dtype float32
    ///
    /// Raises:
    ///     RuntimeError: If encoding fails
    ///
    /// Example:
    ///     >>> embeddings = embedder.encode("What is machine learning?")
    ///     >>> print(embeddings.shape)
    ///     (8, 128)
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Py<PyArray2<f32>>> {
        let embeddings = self.inner.encode(text).map_err(tessera_error_to_pyerr)?;
        token_embeddings_to_pyarray(py, &embeddings)
    }

    /// Encode multiple texts in batch.
    ///
    /// More efficient than calling encode() repeatedly due to batched GPU inference.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of NumPy arrays, one per input text
    ///
    /// Raises:
    ///     RuntimeError: If batch encoding fails
    ///
    /// Example:
    ///     >>> texts = ["First document", "Second document", "Third document"]
    ///     >>> batch_embs = embedder.encode_batch(texts)
    ///     >>> len(batch_embs)
    ///     3
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Py<PyArray2<f32>>>> {
        // Convert Vec<String> to Vec<&str> for the Rust API
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        let embeddings_vec = self
            .inner
            .encode_batch(&text_refs)
            .map_err(tessera_error_to_pyerr)?;

        embeddings_vec
            .iter()
            .map(|emb| token_embeddings_to_pyarray(py, emb))
            .collect()
    }

    /// Compute MaxSim similarity between two texts.
    ///
    /// Convenience method that encodes both texts and computes MaxSim similarity.
    ///
    /// Args:
    ///     text_a: First text
    ///     text_b: Second text
    ///
    /// Returns:
    ///     Similarity score (higher = more similar)
    ///
    /// Raises:
    ///     RuntimeError: If encoding or similarity computation fails
    ///
    /// Example:
    ///     >>> score = embedder.similarity(
    ///     ...     "What is machine learning?",
    ///     ...     "Machine learning is a subset of AI"
    ///     ... )
    ///     >>> print(f"Similarity: {score:.4f}")
    ///     Similarity: 0.8523
    fn similarity(&self, text_a: &str, text_b: &str) -> PyResult<f32> {
        self.inner
            .similarity(text_a, text_b)
            .map_err(tessera_error_to_pyerr)
    }

    /// Get the embedding dimension.
    ///
    /// Returns:
    ///     Dimensionality of each token's embedding vector
    ///
    /// Example:
    ///     >>> dim = embedder.dimension()
    ///     >>> print(f"Dimension: {dim}")
    ///     Dimension: 128
    fn dimension(&self) -> PyResult<usize> {
        Ok(self.inner.dimension())
    }

    /// Get the model identifier.
    ///
    /// Returns:
    ///     Model ID from the registry
    ///
    /// Example:
    ///     >>> model = embedder.model()
    ///     >>> print(f"Using model: {model}")
    ///     Using model: colbert-v2
    fn model(&self) -> PyResult<String> {
        Ok(self.inner.model().to_string())
    }

    /// String representation for debugging.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TesseraMultiVector(model='{}', dimension={})",
            self.inner.model(),
            self.inner.dimension()
        ))
    }

    /// String representation for display.
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

// ============================================================================
// PyTesseraDense - Dense Single-Vector Embedder
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "TesseraDense")]
pub struct PyTesseraDense {
    inner: crate::api::TesseraDense,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTesseraDense {
    /// Create a new dense embedder.
    ///
    /// Args:
    ///     model_id: Model identifier from the registry (e.g., "bge-base-en-v1.5", "nomic-embed-text-v1")
    ///
    /// Returns:
    ///     Initialized embedder ready for use
    ///
    /// Raises:
    ///     RuntimeError: If model is not found or cannot be loaded
    ///
    /// Example:
    ///     >>> embedder = TesseraDense("bge-base-en-v1.5")
    ///     >>> embedding = embedder.encode("What is machine learning?")
    #[new]
    fn new(model_id: &str) -> PyResult<Self> {
        let inner = crate::api::TesseraDense::new(model_id).map_err(tessera_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Encode a single text into a dense embedding.
    ///
    /// Args:
    ///     text: Text to encode
    ///
    /// Returns:
    ///     NumPy array of shape (embedding_dim,) with dtype float32
    ///
    /// Raises:
    ///     RuntimeError: If encoding fails
    ///
    /// Example:
    ///     >>> embedding = embedder.encode("What is machine learning?")
    ///     >>> print(embedding.shape)
    ///     (768,)
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Py<PyArray1<f32>>> {
        let embedding = self.inner.encode(text).map_err(tessera_error_to_pyerr)?;
        dense_embedding_to_pyarray(py, &embedding)
    }

    /// Encode multiple texts in batch.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of NumPy arrays, one per input text
    ///
    /// Raises:
    ///     RuntimeError: If batch encoding fails
    ///
    /// Example:
    ///     >>> texts = ["First document", "Second document"]
    ///     >>> batch_embs = embedder.encode_batch(texts)
    ///     >>> len(batch_embs)
    ///     2
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Py<PyArray1<f32>>>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings_vec = self
            .inner
            .encode_batch(&text_refs)
            .map_err(tessera_error_to_pyerr)?;

        embeddings_vec
            .iter()
            .map(|emb| dense_embedding_to_pyarray(py, emb))
            .collect()
    }

    /// Compute cosine similarity between two texts.
    ///
    /// Args:
    ///     text_a: First text
    ///     text_b: Second text
    ///
    /// Returns:
    ///     Similarity score (higher = more similar)
    ///
    /// Example:
    ///     >>> score = embedder.similarity(
    ///     ...     "What is machine learning?",
    ///     ...     "Machine learning is a subset of AI"
    ///     ... )
    ///     >>> print(f"Similarity: {score:.4f}")
    fn similarity(&self, text_a: &str, text_b: &str) -> PyResult<f32> {
        self.inner
            .similarity(text_a, text_b)
            .map_err(tessera_error_to_pyerr)
    }

    /// Get the embedding dimension.
    ///
    /// Returns:
    ///     Dimensionality of the embedding vector
    ///
    /// Example:
    ///     >>> dim = embedder.dimension()
    ///     >>> print(f"Dimension: {dim}")
    fn dimension(&self) -> PyResult<usize> {
        Ok(self.inner.dimension())
    }

    /// Get the model identifier.
    ///
    /// Returns:
    ///     Model ID from the registry
    ///
    /// Example:
    ///     >>> model = embedder.model()
    ///     >>> print(f"Using model: {model}")
    fn model(&self) -> PyResult<String> {
        Ok(self.inner.model().to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TesseraDense(model='{}', dimension={})",
            self.inner.model(),
            self.inner.dimension()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

// ============================================================================
// PyTesseraSparse - Sparse SPLADE Embedder
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "TesseraSparse")]
pub struct PyTesseraSparse {
    inner: crate::api::TesseraSparse,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTesseraSparse {
    /// Create a new sparse embedder.
    ///
    /// Args:
    ///     model_id: Model identifier from the registry (e.g., "splade-cocondenser")
    ///
    /// Returns:
    ///     Initialized embedder ready for use
    ///
    /// Raises:
    ///     RuntimeError: If model is not found or cannot be loaded
    ///
    /// Example:
    ///     >>> embedder = TesseraSparse("splade-cocondenser")
    ///     >>> indices, values = embedder.encode("What is machine learning?")
    #[new]
    fn new(model_id: &str) -> PyResult<Self> {
        let inner = crate::api::TesseraSparse::new(model_id).map_err(tessera_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Encode a single text into sparse representation.
    ///
    /// Args:
    ///     text: Text to encode
    ///
    /// Returns:
    ///     Tuple of (indices, values) as NumPy arrays for sparse representation
    ///
    /// Raises:
    ///     RuntimeError: If encoding fails
    ///
    /// Example:
    ///     >>> indices, values = embedder.encode("machine learning")
    ///     >>> print(f"Non-zero dims: {len(indices)}")
    #[allow(clippy::elidable_lifetime_names)]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        text: &str,
    ) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
        let embedding = self.inner.encode(text).map_err(tessera_error_to_pyerr)?;
        sparse_embedding_to_pyarrays(py, &embedding)
    }

    /// Encode multiple texts in batch.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of (indices, values) tuples
    ///
    /// Raises:
    ///     RuntimeError: If batch encoding fails
    ///
    /// Example:
    ///     >>> texts = ["First document", "Second document"]
    ///     >>> batch_embs = embedder.encode_batch(texts)
    ///     >>> len(batch_embs)
    ///     2
    #[allow(clippy::elidable_lifetime_names)]
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Vec<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings_vec = self
            .inner
            .encode_batch(&text_refs)
            .map_err(tessera_error_to_pyerr)?;

        embeddings_vec
            .iter()
            .map(|emb| sparse_embedding_to_pyarrays(py, emb))
            .collect()
    }

    /// Compute sparse dot product similarity between two texts.
    ///
    /// Args:
    ///     text_a: First text
    ///     text_b: Second text
    ///
    /// Returns:
    ///     Similarity score (higher = more similar)
    ///
    /// Example:
    ///     >>> score = embedder.similarity("machine learning", "deep learning")
    ///     >>> print(f"Similarity: {score:.4f}")
    fn similarity(&self, text_a: &str, text_b: &str) -> PyResult<f32> {
        self.inner
            .similarity(text_a, text_b)
            .map_err(tessera_error_to_pyerr)
    }

    /// Get the vocabulary size.
    ///
    /// Returns:
    ///     Full vocabulary dimension (typically 30522 for BERT)
    ///
    /// Example:
    ///     >>> vocab_size = embedder.vocab_size()
    ///     >>> print(f"Vocab size: {vocab_size}")
    fn vocab_size(&self) -> PyResult<usize> {
        Ok(self.inner.vocab_size())
    }

    /// Get the model identifier.
    ///
    /// Returns:
    ///     Model ID from the registry
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

// ============================================================================
// PyTesseraVision - Vision-Language ColPali Embedder
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "TesseraVision")]
pub struct PyTesseraVision {
    inner: crate::api::TesseraVision,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTesseraVision {
    /// Create a new vision-language embedder.
    ///
    /// Args:
    ///     model_id: Model identifier from the registry (e.g., "colpali-v1.3-hf")
    ///
    /// Returns:
    ///     Initialized embedder ready for use
    ///
    /// Raises:
    ///     RuntimeError: If model is not found or cannot be loaded
    ///
    /// Example:
    ///     >>> embedder = TesseraVision("colpali-v1.3-hf")
    ///     >>> doc_emb = embedder.encode_document("invoice.jpg")
    #[new]
    fn new(model_id: &str) -> PyResult<Self> {
        let inner = crate::api::TesseraVision::new(model_id).map_err(tessera_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Encode a document image into patch embeddings.
    ///
    /// Args:
    ///     path: Path to document image (PNG, JPEG, etc.) or PDF
    ///
    /// Returns:
    ///     NumPy array of shape (num_patches, embedding_dim)
    ///
    /// Raises:
    ///     RuntimeError: If image loading or encoding fails
    ///
    /// Example:
    ///     >>> doc_emb = embedder.encode_document("invoice.jpg")
    ///     >>> print(doc_emb.shape)
    ///     (1024, 128)
    #[allow(clippy::elidable_lifetime_names)]
    fn encode_document<'py>(&self, py: Python<'py>, path: &str) -> PyResult<Py<PyArray2<f32>>> {
        let embedding = self
            .inner
            .encode_document(path)
            .map_err(tessera_error_to_pyerr)?;
        vision_embedding_to_pyarray(py, &embedding)
    }

    /// Encode a text query into token embeddings.
    ///
    /// Args:
    ///     text: Query text
    ///
    /// Returns:
    ///     NumPy array of shape (num_tokens, embedding_dim)
    ///
    /// Raises:
    ///     RuntimeError: If encoding fails
    ///
    /// Example:
    ///     >>> query_emb = embedder.encode_query("What is the total amount?")
    ///     >>> print(query_emb.shape)
    ///     (8, 128)
    #[allow(clippy::elidable_lifetime_names)]
    fn encode_query<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Py<PyArray2<f32>>> {
        let embedding = self
            .inner
            .encode_query(text)
            .map_err(tessera_error_to_pyerr)?;
        token_embeddings_to_pyarray(py, &embedding)
    }

    /// Compute late interaction score between query and document.
    ///
    /// Args:
    ///     query: Query token embeddings (from encode_query)
    ///     document: Document patch embeddings (from encode_document)
    ///
    /// Returns:
    ///     Similarity score (higher = more similar)
    ///
    /// Example:
    ///     >>> query_emb = embedder.encode_query("total amount")
    ///     >>> doc_emb = embedder.encode_document("invoice.jpg")
    ///     >>> score = embedder.search(query_emb, doc_emb)
    fn search(
        &self,
        query: &Bound<'_, PyArray2<f32>>,
        document: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<f32> {
        // Convert PyArray2 to TokenEmbeddings and VisionEmbedding structures
        let query_shape = query.shape();
        let doc_shape = document.shape();

        // Get readonly arrays
        let query_ro = query.readonly();
        let doc_ro = document.readonly();

        let query_slice = query_ro.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Failed to get query array slice: {}", e))
        })?;

        let doc_slice = doc_ro.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Failed to get document array slice: {}", e))
        })?;

        // Convert to TokenEmbeddings for query
        let query_array =
            ndarray::Array2::from_shape_vec((query_shape[0], query_shape[1]), query_slice.to_vec())
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to create query array: {}", e))
                })?;

        let query_emb = TokenEmbeddings {
            embeddings: query_array,
            num_tokens: query_shape[0],
            embedding_dim: query_shape[1],
            text: String::new(),
        };

        // Convert to VisionEmbedding for document
        let doc_embeddings: Vec<Vec<f32>> = doc_slice
            .chunks(doc_shape[1])
            .map(|chunk| chunk.to_vec())
            .collect();

        let doc_emb = VisionEmbedding {
            embeddings: doc_embeddings,
            num_patches: doc_shape[0],
            embedding_dim: doc_shape[1],
            source: None,
        };

        // Compute similarity
        self.inner
            .search(&query_emb, &doc_emb)
            .map_err(tessera_error_to_pyerr)
    }

    /// Convenience method: search with text query and image path.
    ///
    /// Args:
    ///     query_text: Query text
    ///     document_path: Path to document image
    ///
    /// Returns:
    ///     Similarity score
    ///
    /// Example:
    ///     >>> score = embedder.search_document("total amount", "invoice.jpg")
    ///     >>> print(f"Score: {score:.4f}")
    fn search_document(&self, query_text: &str, document_path: &str) -> PyResult<f32> {
        self.inner
            .search_document(query_text, document_path)
            .map_err(tessera_error_to_pyerr)
    }

    /// Get the embedding dimension.
    ///
    /// Returns:
    ///     Dimensionality of each patch embedding
    fn embedding_dim(&self) -> PyResult<usize> {
        Ok(self.inner.embedding_dim())
    }

    /// Get the number of patches per image.
    ///
    /// Returns:
    ///     Number of patches (typically 1024 for ColPali)
    fn num_patches(&self) -> PyResult<usize> {
        Ok(self.inner.num_patches())
    }

    /// Get the model identifier.
    ///
    /// Returns:
    ///     Model ID from the registry
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

// ============================================================================
// PyTesseraTimeSeries - Time Series Forecasting
// ============================================================================

#[cfg(all(feature = "python", feature = "timeseries"))]
#[pyclass(name = "TesseraTimeSeries")]
pub struct PyTesseraTimeSeries {
    inner: crate::api::TesseraTimeSeries,
}

#[cfg(all(feature = "python", feature = "timeseries"))]
#[pymethods]
impl PyTesseraTimeSeries {
    /// Create a new time series forecaster.
    ///
    /// Args:
    ///     model_id: Model identifier from the registry (e.g., "chronos-bolt-small")
    ///
    /// Returns:
    ///     Initialized forecaster ready for use
    ///
    /// Raises:
    ///     RuntimeError: If model is not found or cannot be loaded
    ///
    /// Example:
    ///     >>> forecaster = TesseraTimeSeries("chronos-bolt-small")
    ///     >>> forecast = forecaster.forecast(data)
    #[new]
    fn new(model_id: &str) -> PyResult<Self> {
        let inner = crate::api::TesseraTimeSeries::new(model_id).map_err(tessera_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Generate point forecast (median prediction).
    ///
    /// Args:
    ///     context: Historical time series data as NumPy array [batch, context_length]
    ///
    /// Returns:
    ///     Forecasted values as NumPy array [batch, prediction_length]
    ///
    /// Raises:
    ///     RuntimeError: If forecasting fails
    ///
    /// Example:
    ///     >>> import numpy as np
    ///     >>> data = np.random.randn(1, 2048).astype(np.float32)
    ///     >>> forecast = forecaster.forecast(data)
    ///     >>> print(forecast.shape)
    ///     (1, 64)
    #[allow(clippy::elidable_lifetime_names)]
    fn forecast<'py>(
        &mut self,
        py: Python<'py>,
        context: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<Py<PyArray2<f32>>> {
        // Convert PyArray2 to Tensor
        let tensor = pyarray2_to_tensor(context)?;

        // Generate forecast
        let forecast_tensor = self
            .inner
            .forecast(&tensor)
            .map_err(tessera_error_to_pyerr)?;

        // Convert back to PyArray2
        tensor_to_pyarray2(py, &forecast_tensor)
    }

    /// Generate probabilistic forecast with all quantiles.
    ///
    /// Args:
    ///     context: Historical time series data as NumPy array [batch, context_length]
    ///
    /// Returns:
    ///     Quantile predictions as NumPy array [batch, prediction_length, num_quantiles]
    ///
    /// Raises:
    ///     RuntimeError: If forecasting fails
    ///
    /// Example:
    ///     >>> data = np.random.randn(1, 2048).astype(np.float32)
    ///     >>> quantiles = forecaster.forecast_quantiles(data)
    ///     >>> print(quantiles.shape)
    ///     (1, 64, 9)
    #[allow(clippy::elidable_lifetime_names)]
    fn forecast_quantiles<'py>(
        &mut self,
        py: Python<'py>,
        context: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<Py<PyArray3<f32>>> {
        // Convert PyArray2 to Tensor
        let tensor = pyarray2_to_tensor(context)?;

        // Generate quantile forecasts
        let quantiles_tensor = self
            .inner
            .forecast_quantiles(&tensor)
            .map_err(tessera_error_to_pyerr)?;

        // Convert back to PyArray3
        tensor_to_pyarray3(py, &quantiles_tensor)
    }

    /// Get the prediction horizon length.
    ///
    /// Returns:
    ///     Number of timesteps forecasted
    fn prediction_length(&self) -> PyResult<usize> {
        Ok(self.inner.prediction_length())
    }

    /// Get the context length.
    ///
    /// Returns:
    ///     Required input sequence length
    fn context_length(&self) -> PyResult<usize> {
        Ok(self.inner.context_length())
    }

    /// Get the quantile levels.
    ///
    /// Returns:
    ///     List of quantiles predicted by the model
    fn quantiles(&self) -> PyResult<Vec<f32>> {
        Ok(self.inner.quantiles().to_vec())
    }

    /// Get the model identifier.
    ///
    /// Returns:
    ///     Model ID from the registry
    fn model(&self) -> PyResult<String> {
        Ok(self.inner.model().to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TesseraTimeSeries(model='{}', context_len={}, pred_len={})",
            self.inner.model(),
            self.inner.context_length(),
            self.inner.prediction_length()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

// ============================================================================
// Module Registration
// ============================================================================

#[cfg(feature = "python")]
#[pymodule]
fn tessera(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all embedder classes
    m.add_class::<PyTesseraDense>()?;
    m.add_class::<PyTesseraMultiVector>()?;
    m.add_class::<PyTesseraSparse>()?;
    m.add_class::<PyTesseraVision>()?;
    #[cfg(feature = "timeseries")]
    m.add_class::<PyTesseraTimeSeries>()?;

    // Add module docstring
    #[cfg(feature = "timeseries")]
    let doc = "Tessera: Multi-paradigm embedding library for Rust\n\n\
        Provides production-ready embeddings with GPU acceleration.\n\n\
        Supports:\n\
        - Dense embeddings (TesseraDense): Single-vector sentence embeddings\n\
        - Multi-vector embeddings (TesseraMultiVector): ColBERT token-level embeddings\n\
        - Sparse embeddings (TesseraSparse): SPLADE vocabulary-space embeddings\n\
        - Vision-language (TesseraVision): ColPali document retrieval\n\
        - Time series (TesseraTimeSeries): Chronos Bolt forecasting\n\n\
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
        >>> sparse = TesseraSparse('splade-cocondenser')\n\
        >>> indices, values = sparse.encode('machine learning')\n\
        >>> print(f'Non-zero dims: {len(indices)}')  # ~100-200\n\
    ";

    #[cfg(not(feature = "timeseries"))]
    let doc = "Tessera: Multi-paradigm embedding library for Rust\n\n\
        Provides production-ready embeddings with GPU acceleration.\n\n\
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
        >>> sparse = TesseraSparse('splade-cocondenser')\n\
        >>> indices, values = sparse.encode('machine learning')\n\
        >>> print(f'Non-zero dims: {len(indices)}')  # ~100-200\n\
    ";

    m.add("__doc__", doc)?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

// ============================================================================
// Stub for non-Python builds
// ============================================================================

#[cfg(not(feature = "python"))]
/// Placeholder for Python bindings.
///
/// This module will contain PyO3-based Python bindings when the
/// `python` feature is enabled.
pub struct PyTessera {
    // Placeholder
}
