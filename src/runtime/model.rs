use anyhow::{bail, Result};
use candle_core::Device;

use super::{configure_cpu_threads, ResourcePolicy};
use crate::models::registry::{self, ModelInfo, ModelType};

#[cfg(test)]
mod tests;

/// Validates a known model and CPU runtime before any Hub or tokenizer access.
pub fn preflight_registered_model(
    model_name: &str,
    model_context_tokens: usize,
    expected_model_type: ModelType,
    device: &Device,
    resource_policy: &ResourcePolicy,
) -> Result<&'static ModelInfo> {
    resource_policy
        .validate_model_context(model_context_tokens)
        .map_err(|error| anyhow::anyhow!("Invalid configured model context policy: {error}"))?;

    let model_info = registry::get_model_by_hf_id(model_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Model '{model_name}' is not registered and cannot be safely resource-preflighted"
        )
    })?;
    if !model_info.is_runnable() {
        bail!(
            "Model '{}' is catalog-only and cannot be loaded: {}",
            model_info.id,
            model_info.support_note
        );
    }
    if model_info.model_type != expected_model_type {
        bail!(
            "Model '{}' has registry type '{:?}', but this encoder requires '{:?}'",
            model_info.id,
            model_info.model_type,
            expected_model_type
        );
    }
    model_info.revision.ok_or_else(|| {
        anyhow::anyhow!(
            "Model '{}' has no pinned HuggingFace revision and cannot be loaded",
            model_info.id
        )
    })?;
    resource_policy
        .validate_model_context(model_info.context_length)
        .map_err(|error| anyhow::anyhow!("Invalid registered model context policy: {error}"))?;
    resource_policy
        .validate_model_parameters(model_info.parameters, 4)
        .map_err(|error| anyhow::anyhow!("Invalid model parameter policy: {error}"))?;

    if matches!(device, Device::Cpu) {
        configure_cpu_threads(2)
            .map_err(|error| anyhow::anyhow!("Failed to configure Candle CPU threads: {error}"))?;
    }
    Ok(model_info)
}
