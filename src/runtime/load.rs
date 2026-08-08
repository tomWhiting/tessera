use candle_core::{DType, Device};
use thiserror::Error;

#[cfg(test)]
mod tests;

/// Parameter dtype used when loading a registered model.
///
/// Tessera defaults to [`ModelDType::F32`]. Lower precision is explicit because
/// support depends on the selected model, device, and backend. A successful
/// load does not imply that a different device/dtype combination is certified.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelDType {
    /// 32-bit IEEE floating point.
    #[default]
    F32,
    /// 16-bit IEEE floating point.
    F16,
    /// 16-bit brain floating point.
    BF16,
}

impl ModelDType {
    /// Approximate parameter bytes used for registry resource preflight.
    #[must_use]
    pub const fn bytes_per_parameter(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    /// Validates the dtype against Tessera's conservative device contract.
    ///
    /// CPU inference is deliberately restricted to F32. Accelerator support
    /// is still evidence-scoped and may fail cleanly for an unsupported model
    /// or operation.
    pub fn validate_device(self, device: &Device) -> Result<(), ModelDTypeError> {
        if matches!(device, Device::Cpu) && self != Self::F32 {
            return Err(ModelDTypeError::CpuRequiresF32 { requested: self });
        }
        Ok(())
    }

    pub(crate) const fn candle_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
        }
    }
}

/// Invalid model dtype/device combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ModelDTypeError {
    /// Candle CPU model execution is supported only with F32 parameters.
    #[error("CPU model loading requires F32 parameters; requested {requested:?}")]
    CpuRequiresF32 {
        /// Rejected dtype.
        requested: ModelDType,
    },
}
