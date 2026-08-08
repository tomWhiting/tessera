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
/// On macOS with Apple Silicon, this will attempt to use Metal if available.
/// Otherwise, it falls back to CPU.
///
/// # Returns
/// The selected device
///
/// # Errors
///
/// This function currently does not return errors, but returns a Result for API consistency.
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
        if let Ok(device) = Device::new_cuda(0) {
            return Ok(device);
        }
    }

    // Default to CPU
    Ok(Device::Cpu)
}

/// Creates a CPU device explicitly.
///
/// # Errors
///
/// This function never returns errors.
pub const fn cpu_device() -> Result<Device> {
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

/// Returns a string describing the device.
#[must_use]
pub fn device_description(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        #[cfg(feature = "metal")]
        Device::Metal(_) => "Metal".to_string(),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => "CUDA".to_string(),
        _ => "Unknown".to_string(),
    }
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
