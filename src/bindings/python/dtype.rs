use crate::runtime::ModelDType;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[cfg(test)]
#[path = "dtype/tests.rs"]
mod tests;

pub(super) fn parse_model_dtype(value: &str) -> PyResult<ModelDType> {
    parse_model_dtype_value(value).map_err(PyValueError::new_err)
}

pub(super) const fn model_dtype_name(dtype: ModelDType) -> &'static str {
    match dtype {
        ModelDType::F32 => "f32",
        ModelDType::F16 => "f16",
        ModelDType::BF16 => "bf16",
    }
}

fn parse_model_dtype_value(value: &str) -> Result<ModelDType, String> {
    match value {
        "f32" => Ok(ModelDType::F32),
        "f16" => Ok(ModelDType::F16),
        "bf16" => Ok(ModelDType::BF16),
        _ => Err(format!(
            "Unsupported dtype '{value}'; expected exactly one of: f32, f16, bf16"
        )),
    }
}
