//! Backend selection for Burn framework.

use burn_ndarray::{NdArray, NdArrayDevice};

/// Type alias for the CPU backend using NdArray.
pub type CpuBackend = NdArray<f32>;

/// Returns a CPU device for Burn.
pub fn cpu_device() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

/// Returns a string describing the backend.
pub fn backend_description() -> String {
    "Burn NdArray (CPU)".to_string()
}

// Note: Burn's WGPU backend would be used for GPU acceleration
// For this prototype, we focus on CPU backend for simplicity
