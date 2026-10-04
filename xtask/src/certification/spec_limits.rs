use super::{CertResult, CertificationSpec, ProfileSpec};

pub(super) fn validate_memory(spec: &CertificationSpec, profile: &ProfileSpec) -> CertResult<()> {
    let resource = &profile.resource_policy;
    let process = &profile.process;
    let declared_live_bytes = resource
        .max_model_bytes
        .checked_add(resource.max_activation_bytes)
        .ok_or("profile live-memory requirement overflowed")?;
    let declared_live_bytes = u64::try_from(declared_live_bytes)
        .map_err(|_| "profile live-memory requirement does not fit u64")?;
    if declared_live_bytes > process.max_peak_rss_bytes {
        return Err(format!(
            "model '{}' model-plus-activation budget exceeds its RSS watchdog",
            spec.model.id
        )
        .into());
    }
    let artifact_bytes = spec.expected_artifact_bytes()?;
    if artifact_bytes > process.max_artifact_bytes {
        return Err(format!(
            "model '{}' artifacts exceed its artifact-byte cap",
            spec.model.id
        )
        .into());
    }
    Ok(())
}
