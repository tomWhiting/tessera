//! Device management for Candle backend (CPU/Metal).

use anyhow::{Context, Result};
use candle_core::Device;

/// Selects the best available device for computation.
///
/// On macOS with Apple Silicon, this will attempt to use Metal if available.
/// Otherwise, it falls back to CPU.
///
/// # Returns
/// The selected device
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
pub fn cpu_device() -> Result<Device> {
    Ok(Device::Cpu)
}

/// Attempts to create a Metal device.
///
/// # Returns
/// Metal device if available, otherwise returns an error
#[cfg(target_os = "macos")]
pub fn metal_device() -> Result<Device> {
    Device::new_metal(0).context("Failed to create Metal device")
}

/// Returns a string describing the device.
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
