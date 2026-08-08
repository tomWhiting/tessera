use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard, OnceLock};

use candle_core::{Device, DeviceLocation};
use thiserror::Error;

#[cfg(test)]
use super::preflight_registered_model;
use super::{preflight_registered_model_with_dtype, ModelDType, ResourcePolicy};
use crate::models::registry::{ModelInfo, ModelType};

#[cfg(test)]
mod tests;

static PROCESS_MODEL_RESIDENCY: OnceLock<ModelResidencyLedger> = OnceLock::new();

fn process_model_residency() -> &'static ModelResidencyLedger {
    PROCESS_MODEL_RESIDENCY.get_or_init(ModelResidencyLedger::default)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ModelResidencyKey {
    model_id: &'static str,
    revision: &'static str,
    device: DeviceLocation,
    dtype: ModelDType,
}

impl ModelResidencyKey {
    fn new(
        model: &'static ModelInfo,
        device: &Device,
        dtype: ModelDType,
    ) -> Result<Self, ModelResidencyError> {
        let revision = model
            .revision
            .ok_or(ModelResidencyError::MissingRevision { model_id: model.id })?;
        Ok(Self {
            model_id: model.id,
            revision,
            device: device.location(),
            dtype,
        })
    }
}

#[derive(Debug, Default)]
struct ResidencyState {
    resident_parameter_bytes: u128,
    reservations: HashSet<ModelResidencyKey>,
}

#[derive(Debug, Default)]
struct ModelResidencyLedger {
    state: Mutex<ResidencyState>,
}

impl ModelResidencyLedger {
    fn try_reserve(
        &self,
        key: ModelResidencyKey,
        estimated_parameter_bytes: u128,
        aggregate_limit: usize,
    ) -> Result<ModelResidencyPermit<'_>, ModelResidencyError> {
        let mut state = self.lock_state()?;
        if !state.reservations.insert(key) {
            return Err(ModelResidencyError::Duplicate {
                model_id: key.model_id,
                revision: key.revision,
                device: key.device,
                dtype: key.dtype,
            });
        }

        let Some(prospective_bytes) = state
            .resident_parameter_bytes
            .checked_add(estimated_parameter_bytes)
        else {
            state.reservations.remove(&key);
            return Err(ModelResidencyError::ByteCountOverflow);
        };
        if prospective_bytes > aggregate_limit as u128 {
            state.reservations.remove(&key);
            return Err(ModelResidencyError::AggregateBytes {
                resident: state.resident_parameter_bytes,
                requested: estimated_parameter_bytes,
                prospective: prospective_bytes,
                allowed: aggregate_limit,
            });
        }

        state.resident_parameter_bytes = prospective_bytes;
        drop(state);
        Ok(ModelResidencyPermit {
            ledger: self,
            key,
            estimated_parameter_bytes,
        })
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, ResidencyState>, ModelResidencyError> {
        self.state.lock().map_err(|_| ModelResidencyError::Poisoned)
    }

    fn release(&self, key: ModelResidencyKey, estimated_parameter_bytes: u128) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.reservations.remove(&key) {
            state.resident_parameter_bytes = state
                .resident_parameter_bytes
                .saturating_sub(estimated_parameter_bytes);
        }
    }
}

/// Process-wide reservation retained for the lifetime of one model instance.
#[must_use = "dropping the permit releases model residency admission"]
#[derive(Debug)]
pub struct ModelResidencyPermit<'a> {
    ledger: &'a ModelResidencyLedger,
    key: ModelResidencyKey,
    estimated_parameter_bytes: u128,
}

impl Drop for ModelResidencyPermit<'_> {
    fn drop(&mut self) {
        self.ledger
            .release(self.key, self.estimated_parameter_bytes);
    }
}

/// Preflights and reserves one F32 model before any artifact or tokenizer I/O.
///
/// The requesting policy's model-byte limit is also the prospective aggregate
/// residency ceiling for this admission decision. Raising it is therefore an
/// explicit opt-in to the requested model plus every model already resident.
#[cfg(test)]
pub fn preflight_and_reserve_registered_model(
    model_name: &str,
    model_context_tokens: usize,
    expected_model_type: ModelType,
    device: &Device,
    resource_policy: &ResourcePolicy,
) -> anyhow::Result<(&'static ModelInfo, ModelResidencyPermit<'static>)> {
    let model = preflight_registered_model(
        model_name,
        model_context_tokens,
        expected_model_type,
        device,
        resource_policy,
    )?;
    reserve_preflighted_model(model, device, ModelDType::F32, resource_policy)
}

/// Preflights and reserves one explicitly typed model before any model I/O.
pub fn preflight_and_reserve_registered_model_with_dtype(
    model_name: &str,
    model_context_tokens: usize,
    expected_model_type: ModelType,
    device: &Device,
    dtype: ModelDType,
    resource_policy: &ResourcePolicy,
) -> anyhow::Result<(&'static ModelInfo, ModelResidencyPermit<'static>)> {
    let model = preflight_registered_model_with_dtype(
        model_name,
        model_context_tokens,
        expected_model_type,
        device,
        dtype,
        resource_policy,
    )?;
    reserve_preflighted_model(model, device, dtype, resource_policy)
}

fn reserve_preflighted_model(
    model: &'static ModelInfo,
    device: &Device,
    dtype: ModelDType,
    resource_policy: &ResourcePolicy,
) -> anyhow::Result<(&'static ModelInfo, ModelResidencyPermit<'static>)> {
    let estimated_parameter_bytes = resource_policy
        .validate_model_parameters(model.parameters, dtype.bytes_per_parameter())
        .map_err(|error| anyhow::anyhow!("Invalid model parameter policy: {error}"))?;
    let key = ModelResidencyKey::new(model, device, dtype)?;
    let permit = process_model_residency().try_reserve(
        key,
        estimated_parameter_bytes,
        resource_policy.max_model_bytes(),
    )?;
    Ok((model, permit))
}

/// Failure to admit another retained model instance in this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
enum ModelResidencyError {
    #[error(
        "model '{model_id}' at revision {revision} is already resident on {device:?} with {dtype:?}; Tessera does not share model tensors, so reuse the existing embedder or drop it before loading another copy"
    )]
    Duplicate {
        model_id: &'static str,
        revision: &'static str,
        device: DeviceLocation,
        dtype: ModelDType,
    },
    #[error(
        "prospective resident model parameter bytes {prospective} ({resident} resident + {requested} requested) exceed the requesting resource policy limit {allowed}"
    )]
    AggregateBytes {
        resident: u128,
        requested: u128,
        prospective: u128,
        allowed: usize,
    },
    #[error("resident model byte accounting overflowed")]
    ByteCountOverflow,
    #[error("model '{model_id}' has no immutable revision for residency admission")]
    MissingRevision { model_id: &'static str },
    #[error("process-wide model residency state is poisoned")]
    Poisoned,
}
