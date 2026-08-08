#![allow(missing_docs)]

use super::conversion::tessera_error_to_pyerr;
use candle_core::{Device, Tensor};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::{pyclass, pymethods, Bound, Py, PyResult, Python};

fn tensor_values(tensor: &Tensor) -> PyResult<(Vec<usize>, Vec<f32>)> {
    let tensor = tensor
        .to_device(&Device::Cpu)
        .map_err(|err| PyRuntimeError::new_err(format!("Failed to move tensor to CPU: {err}")))?;
    let dims = tensor.dims().to_vec();
    let values = tensor
        .flatten_all()
        .map_err(|err| PyRuntimeError::new_err(format!("Failed to flatten tensor: {err}")))?
        .to_vec1::<f32>()
        .map_err(|err| {
            PyRuntimeError::new_err(format!("Failed to convert tensor to vec: {err}"))
        })?;
    Ok((dims, values))
}

fn tensor_to_pyarray2(py: Python<'_>, tensor: &Tensor) -> PyResult<Py<PyArray2<f32>>> {
    let (dims, values) = tensor_values(tensor)?;
    if dims.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Expected 2D tensor, got {}D",
            dims.len()
        )));
    }
    let rows: Vec<Vec<f32>> = values.chunks(dims[1]).map(<[f32]>::to_vec).collect();
    Ok(PyArray2::from_vec2_bound(py, &rows)?.unbind())
}

fn tensor_to_pyarray3(py: Python<'_>, tensor: &Tensor) -> PyResult<Py<PyArray3<f32>>> {
    let (dims, values) = tensor_values(tensor)?;
    if dims.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "Expected 3D tensor, got {}D",
            dims.len()
        )));
    }
    let array = PyArray1::from_vec_bound(py, values)
        .reshape([dims[0], dims[1], dims[2]])
        .map_err(|err| PyValueError::new_err(format!("Failed to reshape to 3D: {err}")))?;
    Ok(array.unbind())
}

fn pyarray2_to_tensor(array: &Bound<'_, PyArray2<f32>>) -> PyResult<Tensor> {
    let shape = array.shape();
    let readonly = array.readonly();
    let values = readonly
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("Failed to get array slice: {err}")))?;
    Tensor::from_slice(values, (shape[0], shape[1]), &Device::Cpu)
        .map_err(|err| PyRuntimeError::new_err(format!("Failed to create tensor: {err}")))
}

#[pyclass(name = "TesseraTimeSeries")]
pub struct PyTesseraTimeSeries {
    inner: crate::api::TesseraTimeSeries,
}

#[pymethods]
impl PyTesseraTimeSeries {
    /// Create a time-series forecaster for a registered model.
    #[new]
    fn new(model_id: &str) -> PyResult<Self> {
        let inner = crate::api::TesseraTimeSeries::new(model_id).map_err(tessera_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Generate the median forecast for a two-dimensional context array.
    fn forecast(
        &mut self,
        py: Python<'_>,
        context: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let context = pyarray2_to_tensor(context)?;
        let forecast = py
            .allow_threads(|| self.inner.forecast(&context))
            .map_err(tessera_error_to_pyerr)?;
        tensor_to_pyarray2(py, &forecast)
    }

    /// Generate all forecast quantiles for a context array.
    fn forecast_quantiles(
        &mut self,
        py: Python<'_>,
        context: &Bound<'_, PyArray2<f32>>,
    ) -> PyResult<Py<PyArray3<f32>>> {
        let context = pyarray2_to_tensor(context)?;
        let quantiles = py
            .allow_threads(|| self.inner.forecast_quantiles(&context))
            .map_err(tessera_error_to_pyerr)?;
        tensor_to_pyarray3(py, &quantiles)
    }

    /// Return the prediction horizon.
    fn prediction_length(&self) -> PyResult<usize> {
        Ok(self.inner.prediction_length())
    }

    /// Return the required context length.
    fn context_length(&self) -> PyResult<usize> {
        Ok(self.inner.context_length())
    }

    /// Return the model's forecast quantiles.
    fn quantiles(&self) -> PyResult<Vec<f32>> {
        Ok(self.inner.quantiles().to_vec())
    }

    /// Return the registered model identifier.
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
