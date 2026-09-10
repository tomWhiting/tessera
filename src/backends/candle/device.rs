//! Device management for Candle backend (CPU/Metal).

#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "cuda"))]
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;

#[cfg(all(target_os = "macos", feature = "metal"))]
fn validate_metal_ordinal(ordinal: usize, device_count: usize) -> Result<()> {
    anyhow::ensure!(
        ordinal < device_count,
        "Metal device ordinal {ordinal} is unavailable; detected {device_count} Metal device(s)"
    );
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_device_at(ordinal: usize) -> Result<Device> {
    // Candle 0.11 calls `swap_remove(ordinal)` without checking this list first,
    // which panics when Metal is unavailable rather than returning an error.
    let device_count = candle_metal_kernels::metal::Device::all().len();
    validate_metal_ordinal(ordinal, device_count)?;
    Device::new_metal(ordinal)
        .with_context(|| format!("Failed to create Metal device at ordinal {ordinal}"))
}

/// Selects the best available device for computation.
///
/// Selection order is fixed and feature-gated at compile time:
///
/// 1. Metal ordinal 0, when the target is macOS **and** the `metal` feature is
///    enabled. The ordinal is validated against `Device::all()` first because
///    Candle 0.11 panics rather than erring on an empty Metal device list.
/// 2. CUDA ordinal 0, when the `cuda` feature is enabled.
/// 3. CPU.
///
/// Each accelerator step is a *fallback*, not a guarantee: if the accelerator
/// device cannot be created, the failure is discarded and selection continues
/// to the next step, so a machine with a broken driver silently returns
/// [`Device::Cpu`]. Callers that must fail loudly on a missing accelerator
/// should construct the device themselves — [`metal_device`] or
/// [`cuda_device`], or `Device::new_metal`/`Device::new_cuda` for a non-zero
/// ordinal — and pass it to a builder's `device` selector.
///
/// # Returns
///
/// The selected device.
///
/// # Errors
///
/// Returns an error only if a future selection step becomes fallible; the
/// current steps all fall through to [`Device::Cpu`].
pub fn get_device() -> Result<Device> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        // Try Metal first on macOS
        if let Ok(device) = metal_device() {
            return Ok(device);
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = cuda_device() {
            return Ok(device);
        }
    }

    // Default to CPU
    Ok(Device::Cpu)
}

/// Attempts to create a Metal device.
///
/// # Returns
/// Metal device if available, otherwise returns an error
///
/// # Errors
///
/// Returns an error if Metal device creation fails.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn metal_device() -> Result<Device> {
    metal_device_at(0)
}

/// Attempts to create the first CUDA device.
///
/// # Errors
///
/// Returns an error when CUDA support is unavailable or device creation fails.
#[cfg(feature = "cuda")]
pub fn cuda_device() -> Result<Device> {
    Device::new_cuda(0).context("Failed to create CUDA device")
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;

    #[test]
    fn metal_ordinal_guard_rejects_an_empty_device_list() {
        let error = validate_metal_ordinal(0, 0).expect_err("zero devices must be rejected");
        assert!(error.to_string().contains("detected 0 Metal device"));
    }

    #[test]
    fn metal_ordinal_guard_rejects_an_out_of_range_device() {
        let error = validate_metal_ordinal(1, 1).expect_err("ordinal one must be rejected");
        assert!(error.to_string().contains("ordinal 1 is unavailable"));
    }

    #[test]
    fn metal_ordinal_guard_accepts_an_available_device() {
        validate_metal_ordinal(0, 1).expect("ordinal zero must be accepted");
    }

    #[test]
    fn metal_feature_uses_a_real_metal_device() {
        let device = metal_device().expect("Metal feature requires a working Metal device");
        assert!(matches!(device, Device::Metal(_)));
    }
}
