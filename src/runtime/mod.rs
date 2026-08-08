//! Runtime resource limits plus process-wide model and inference admission.

mod estimate;
mod inference;
mod job;
mod load;
mod model;
mod policy;
mod residency;
mod threading;
mod window;

pub(crate) use estimate::TransformerProfile;
pub(crate) use inference::acquire_inference_permit;
pub use inference::{
    configure_inference_gate, try_acquire_inference, InferenceGateConfig, InferenceGateConfigError,
    InferenceGateError,
};
pub(crate) use job::{f32_output_bytes, JobTracker};
pub use load::{ModelDType, ModelDTypeError};
#[cfg(test)]
pub(crate) use model::preflight_registered_model;
pub(crate) use model::preflight_registered_model_with_dtype;
pub(crate) use policy::resolve_registry_policy_with_dtype;
pub use policy::{ResourcePolicy, ResourcePolicyError};
pub(crate) use residency::{
    preflight_and_reserve_registered_model_with_dtype, ModelResidencyPermit,
};
pub use threading::{configure_cpu_threads, CpuThreadConfig, CpuThreadConfigError};
pub(crate) use window::{plan_token_windows, TokenWindow};
pub use window::{ContextWindowConfig, ContextWindowError};
