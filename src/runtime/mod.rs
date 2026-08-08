//! Runtime resource limits and process-wide inference admission.

mod inference;
mod model;
mod policy;
mod threading;

pub(crate) use inference::acquire_inference_permit;
pub(crate) use model::preflight_registered_model;
pub(crate) use policy::resolve_registry_policy;
pub use policy::{ResourcePolicy, ResourcePolicyError};
pub use threading::{configure_cpu_threads, CpuThreadConfig, CpuThreadConfigError};
