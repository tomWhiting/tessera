//! Device management for Candle backend (CPU/Metal).

#[cfg(target_os = "macos")]
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;

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
    #[cfg(target_os = "macos")]
    {
        // Try Metal first on macOS
        match Device::new_metal(0) {
            Ok(device) => {
                println!("Using Metal device for acceleration");
                return Ok(device);
            }
            Err(_) => {
                println!("Metal not available, falling back to CPU");
            }
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
#[cfg(target_os = "macos")]
pub fn metal_device() -> Result<Device> {
    Device::new_metal(0).context("Failed to create Metal device")
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
