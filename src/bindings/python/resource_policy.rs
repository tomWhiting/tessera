#![allow(missing_docs)]
// PyO3's `#[pymethods]` expansion converts `PyErr` through `Into<PyErr>`.
#![allow(clippy::useless_conversion)]

use crate::runtime::ResourcePolicy;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::{pyclass, pymethods, Bound, PyErr, PyResult};
use pyo3::types::{PyAnyMethods, PySequence, PySequenceMethods, PyString, PyStringMethods};

#[cfg(test)]
#[path = "resource_policy/tests.rs"]
mod tests;

const DEFAULT_MAX_SEQUENCE_TOKENS: usize = 512;
const DEFAULT_MAX_BATCH_ITEMS: usize = 16;
const DEFAULT_MAX_BATCH_TOKENS: usize = 2_048;
const DEFAULT_MAX_MODEL_BYTES: usize = 2 * 1_024 * 1_024 * 1_024;
const DEFAULT_MAX_INPUT_BYTES_PER_SEQUENCE: usize = 1_024 * 1_024;
const DEFAULT_MAX_ATTENTION_CELLS: usize = 1_048_576;
const DEFAULT_MAX_JOB_ITEMS: usize = 1_024;
const DEFAULT_MAX_JOB_INPUT_BYTES: usize = 64 * 1_024 * 1_024;
const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1_024 * 1_024;
const DEFAULT_MAX_ACTIVATION_BYTES: usize = 512 * 1_024 * 1_024;

/// Immutable Python wrapper for Tessera's pre-allocation resource limits.
#[derive(Clone, Copy)]
#[pyclass(name = "ResourcePolicy", frozen)]
pub struct PyResourcePolicy {
    inner: ResourcePolicy,
}

impl PyResourcePolicy {
    pub(super) const fn inner(&self) -> ResourcePolicy {
        self.inner
    }
}

pub(super) fn effective_policy_for_model(
    model_id: &str,
    override_policy: Option<ResourcePolicy>,
) -> ResourcePolicy {
    override_policy.unwrap_or_else(|| {
        crate::models::registry::get_model(model_id).map_or_else(ResourcePolicy::default, |model| {
            ResourcePolicy::for_model_context(model.context_length)
        })
    })
}

pub(super) fn extract_text_batch(
    texts: &Bound<'_, PySequence>,
    resource_policy: ResourcePolicy,
) -> PyResult<Vec<String>> {
    if texts.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "texts must be a sequence of strings, not a single string",
        ));
    }
    let batch_items = texts.len()?;
    validate_batch_items(batch_items, resource_policy).map_err(PyValueError::new_err)?;
    let mut extracted = Vec::with_capacity(batch_items);
    let mut total_input_bytes = 0_usize;
    for index in 0..batch_items {
        let item = texts.get_item(index)?;
        let text = item
            .downcast::<PyString>()
            .map_err(|_| PyTypeError::new_err(format!("texts[{index}] must be a string")))?;
        let text = text.to_str()?;
        validate_text_value(text, &format!("texts[{index}]"), resource_policy)
            .map_err(PyValueError::new_err)?;
        total_input_bytes = total_input_bytes.saturating_add(text.len());
        resource_policy
            .validate_job(index.saturating_add(1), total_input_bytes)
            .map_err(resource_policy_error)?;
        extracted.push(text.to_owned());
    }
    Ok(extracted)
}

pub(super) fn validated_text_input(
    text: &str,
    label: &str,
    resource_policy: ResourcePolicy,
) -> PyResult<String> {
    validate_text_value(text, label, resource_policy).map_err(PyValueError::new_err)?;
    resource_policy
        .validate_job(1, text.len())
        .map_err(resource_policy_error)?;
    Ok(text.to_owned())
}

pub(super) fn validated_text_pair(
    first: &str,
    first_label: &str,
    second: &str,
    second_label: &str,
    resource_policy: ResourcePolicy,
) -> PyResult<(String, String)> {
    validate_text_pair_values(first, first_label, second, second_label, resource_policy)
        .map_err(PyValueError::new_err)?;
    Ok((first.to_owned(), second.to_owned()))
}

fn validate_batch_items(batch_items: usize, resource_policy: ResourcePolicy) -> Result<(), String> {
    if batch_items == 0 {
        return Err("texts must contain at least one string".to_string());
    }
    resource_policy
        .validate_batch(batch_items, 0)
        .map_err(|error| format!("Resource policy rejected Python batch: {error}"))?;
    resource_policy
        .validate_job(batch_items, 0)
        .map_err(|error| format!("Resource policy rejected Python batch: {error}"))
}

fn validate_text_value(
    text: &str,
    label: &str,
    resource_policy: ResourcePolicy,
) -> Result<(), String> {
    if text.is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    resource_policy
        .validate_input_bytes(text.len())
        .map_err(|error| format!("Resource policy rejected {label}: {error}"))
}

fn validate_text_pair_values(
    first: &str,
    first_label: &str,
    second: &str,
    second_label: &str,
    resource_policy: ResourcePolicy,
) -> Result<(), String> {
    validate_text_value(first, first_label, resource_policy)?;
    validate_text_value(second, second_label, resource_policy)?;
    resource_policy
        .validate_job(2, first.len().saturating_add(second.len()))
        .map_err(|error| format!("Resource policy rejected paired inputs: {error}"))
}

fn resource_policy_error(error: crate::runtime::ResourcePolicyError) -> PyErr {
    PyValueError::new_err(format!("Resource policy rejected Python batch: {error}"))
}

#[pymethods]
impl PyResourcePolicy {
    /// Create an immutable resource policy with conservative defaults.
    #[new]
    #[pyo3(signature = (
        max_sequence_tokens=DEFAULT_MAX_SEQUENCE_TOKENS,
        max_batch_items=DEFAULT_MAX_BATCH_ITEMS,
        max_batch_tokens=DEFAULT_MAX_BATCH_TOKENS,
        max_model_bytes=DEFAULT_MAX_MODEL_BYTES,
        max_input_bytes_per_sequence=DEFAULT_MAX_INPUT_BYTES_PER_SEQUENCE,
        max_attention_cells=DEFAULT_MAX_ATTENTION_CELLS,
        max_job_items=DEFAULT_MAX_JOB_ITEMS,
        max_job_input_bytes=DEFAULT_MAX_JOB_INPUT_BYTES,
        max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES,
        max_activation_bytes=DEFAULT_MAX_ACTIVATION_BYTES,
    ))]
    #[allow(clippy::too_many_arguments)] // Mirrors the explicit Python keyword-only policy surface.
    const fn new(
        max_sequence_tokens: usize,
        max_batch_items: usize,
        max_batch_tokens: usize,
        max_model_bytes: usize,
        max_input_bytes_per_sequence: usize,
        max_attention_cells: usize,
        max_job_items: usize,
        max_job_input_bytes: usize,
        max_output_bytes: usize,
        max_activation_bytes: usize,
    ) -> Self {
        Self {
            inner: ResourcePolicy::new(
                max_sequence_tokens,
                max_batch_items,
                max_batch_tokens,
                max_model_bytes,
            )
            .with_max_input_bytes_per_sequence(max_input_bytes_per_sequence)
            .with_max_attention_cells(max_attention_cells)
            .with_max_job_items(max_job_items)
            .with_max_job_input_bytes(max_job_input_bytes)
            .with_max_output_bytes(max_output_bytes)
            .with_max_activation_bytes(max_activation_bytes),
        }
    }

    #[getter]
    const fn max_sequence_tokens(&self) -> usize {
        self.inner.max_sequence_tokens()
    }

    #[getter]
    const fn max_batch_items(&self) -> usize {
        self.inner.max_batch_items()
    }

    #[getter]
    const fn max_batch_tokens(&self) -> usize {
        self.inner.max_batch_tokens()
    }

    #[getter]
    const fn max_model_bytes(&self) -> usize {
        self.inner.max_model_bytes()
    }

    #[getter]
    const fn max_input_bytes_per_sequence(&self) -> usize {
        self.inner.max_input_bytes_per_sequence()
    }

    #[getter]
    const fn max_attention_cells(&self) -> usize {
        self.inner.max_attention_cells()
    }

    #[getter]
    const fn max_job_items(&self) -> usize {
        self.inner.max_job_items()
    }

    #[getter]
    const fn max_job_input_bytes(&self) -> usize {
        self.inner.max_job_input_bytes()
    }

    #[getter]
    const fn max_output_bytes(&self) -> usize {
        self.inner.max_output_bytes()
    }

    #[getter]
    const fn max_activation_bytes(&self) -> usize {
        self.inner.max_activation_bytes()
    }

    const fn with_max_sequence_tokens(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_sequence_tokens(value),
        }
    }

    const fn with_max_batch_items(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_batch_items(value),
        }
    }

    const fn with_max_batch_tokens(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_batch_tokens(value),
        }
    }

    const fn with_max_model_bytes(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_model_bytes(value),
        }
    }

    const fn with_max_input_bytes_per_sequence(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_input_bytes_per_sequence(value),
        }
    }

    const fn with_max_attention_cells(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_attention_cells(value),
        }
    }

    const fn with_max_job_items(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_job_items(value),
        }
    }

    const fn with_max_job_input_bytes(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_job_input_bytes(value),
        }
    }

    const fn with_max_output_bytes(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_output_bytes(value),
        }
    }

    const fn with_max_activation_bytes(&self, value: usize) -> Self {
        Self {
            inner: self.inner.with_max_activation_bytes(value),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ResourcePolicy(max_sequence_tokens={}, max_batch_items={}, max_batch_tokens={}, max_model_bytes={}, max_input_bytes_per_sequence={}, max_attention_cells={}, max_job_items={}, max_job_input_bytes={}, max_output_bytes={}, max_activation_bytes={})",
            self.max_sequence_tokens(),
            self.max_batch_items(),
            self.max_batch_tokens(),
            self.max_model_bytes(),
            self.max_input_bytes_per_sequence(),
            self.max_attention_cells(),
            self.max_job_items(),
            self.max_job_input_bytes(),
            self.max_output_bytes(),
            self.max_activation_bytes(),
        )
    }
}
