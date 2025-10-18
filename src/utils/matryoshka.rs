//! Matryoshka representation learning utilities.
//!
//! Handles dimension truncation for models supporting Matryoshka embeddings,
//! where embedding dimensions can be reduced without retraining by truncating
//! the first N dimensions of the learned representations.
//!
//! # Truncation Strategies
//!
//! Different models require truncation at different points in the pipeline:
//!
//! - **`TruncateHidden`**: Truncate BERT hidden states BEFORE projection layer
//!   (for `ColBERT` v2 models with projection)
//! - **`TruncateOutput`**: Truncate final output embeddings AFTER projection
//!   (for models without projection like Jina-ColBERT)
//! - **`TruncatePooled`**: Truncate after pooling for dense models
//!   (for BERT-style dense encoders like BGE, Nomic)
//!
//! # References
//!
//! - "Matryoshka Representation Learning" (Kusupati et al., 2022)
//! - Jina AI `ColBERT` v2 documentation

use crate::error::{Result, TesseraError};
use candle_core::Tensor;

/// Matryoshka truncation strategy.
///
/// Specifies at which point in the encoding pipeline to truncate
/// the embedding dimensions for Matryoshka representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatryoshkaStrategy {
    /// Truncate BERT hidden states before projection layer.
    ///
    /// Used for models with projection layers (e.g., `ColBERT` v2)
    /// where the hidden states are projected to a lower dimension.
    /// Truncation happens before the projection.
    TruncateHidden,

    /// Truncate final output embeddings after projection.
    ///
    /// Used for models without projection layers (e.g., Jina-ColBERT)
    /// where the final embeddings are truncated directly.
    TruncateOutput,

    /// Truncate after pooling for dense models.
    ///
    /// Used for dense encoders (e.g., BGE, Nomic) where a single
    /// pooled vector is produced and then truncated.
    TruncatePooled,
}

impl MatryoshkaStrategy {
    /// Parse strategy from string.
    #[must_use] pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "truncate_hidden" => Some(Self::TruncateHidden),
            "truncate_output" => Some(Self::TruncateOutput),
            "truncate_pooled" => Some(Self::TruncatePooled),
            _ => None,
        }
    }

    /// Convert strategy to string.
    #[must_use] pub const fn as_str(&self) -> &'static str {
        match self {
            Self::TruncateHidden => "truncate_hidden",
            Self::TruncateOutput => "truncate_output",
            Self::TruncatePooled => "truncate_pooled",
        }
    }
}

/// Apply Matryoshka dimension truncation to a tensor.
///
/// Truncates the last dimension of the tensor to the target dimension size.
/// This operation is typically applied to embedding tensors.
///
/// # Arguments
///
/// * `tensor` - Input tensor to truncate (any shape, last dim will be truncated)
/// * `target_dim` - Target dimension size (must be <= current last dimension)
/// * `strategy` - Truncation strategy (informational, doesn't affect operation)
///
/// # Returns
///
/// Truncated tensor with last dimension reduced to `target_dim`
///
/// # Errors
///
/// Returns `TesseraError::MatryoshkaError` if:
/// - Target dimension exceeds current dimension
/// - Tensor operations fail
///
/// # Example
///
/// ```no_run
/// use candle_core::{Device, Tensor};
/// use tessera::utils::{apply_matryoshka, MatryoshkaStrategy};
///
/// # fn example() -> tessera::error::Result<()> {
/// let device = Device::Cpu;
/// let tensor = Tensor::zeros((10, 768), candle_core::DType::F32, &device)?;
/// let truncated = apply_matryoshka(&tensor, 128, MatryoshkaStrategy::TruncateOutput)?;
/// assert_eq!(truncated.dims(), &[10, 128]);
/// # Ok(())
/// # }
/// ```
pub fn apply_matryoshka(
    tensor: &Tensor,
    target_dim: usize,
    _strategy: MatryoshkaStrategy,
) -> Result<Tensor> {
    let shape = tensor.dims();
    let current_dim = shape[shape.len() - 1];

    if target_dim > current_dim {
        return Err(TesseraError::MatryoshkaError(format!(
            "Target dimension {target_dim} exceeds current dimension {current_dim}"
        )));
    }

    if target_dim == current_dim {
        // No truncation needed
        return Ok(tensor.clone());
    }

    // Truncate last dimension: narrow(dim, start, len)
    tensor
        .narrow(shape.len() - 1, 0, target_dim)
        .map_err(|e| TesseraError::MatryoshkaError(format!("Failed to truncate tensor: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_strategy_from_str() {
        assert_eq!(
            MatryoshkaStrategy::from_str("truncate_hidden"),
            Some(MatryoshkaStrategy::TruncateHidden)
        );
        assert_eq!(
            MatryoshkaStrategy::from_str("truncate_output"),
            Some(MatryoshkaStrategy::TruncateOutput)
        );
        assert_eq!(
            MatryoshkaStrategy::from_str("truncate_pooled"),
            Some(MatryoshkaStrategy::TruncatePooled)
        );
        assert_eq!(MatryoshkaStrategy::from_str("invalid"), None);
    }

    #[test]
    fn test_strategy_as_str() {
        assert_eq!(
            MatryoshkaStrategy::TruncateHidden.as_str(),
            "truncate_hidden"
        );
        assert_eq!(
            MatryoshkaStrategy::TruncateOutput.as_str(),
            "truncate_output"
        );
        assert_eq!(
            MatryoshkaStrategy::TruncatePooled.as_str(),
            "truncate_pooled"
        );
    }

    #[test]
    fn test_apply_matryoshka_2d() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((10, 768), DType::F32, &device)?;

        let truncated = apply_matryoshka(&tensor, 128, MatryoshkaStrategy::TruncateOutput)?;
        assert_eq!(truncated.dims(), &[10, 128]);

        Ok(())
    }

    #[test]
    fn test_apply_matryoshka_3d() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((2, 10, 768), DType::F32, &device)?;

        let truncated = apply_matryoshka(&tensor, 128, MatryoshkaStrategy::TruncateHidden)?;
        assert_eq!(truncated.dims(), &[2, 10, 128]);

        Ok(())
    }

    #[test]
    fn test_apply_matryoshka_no_truncation() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((10, 128), DType::F32, &device)?;

        // Same dimension - should return clone
        let truncated = apply_matryoshka(&tensor, 128, MatryoshkaStrategy::TruncateOutput)?;
        assert_eq!(truncated.dims(), &[10, 128]);

        Ok(())
    }

    #[test]
    fn test_apply_matryoshka_invalid_target() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((10, 128), DType::F32, &device).unwrap();

        // Target dimension larger than current
        let result = apply_matryoshka(&tensor, 256, MatryoshkaStrategy::TruncateOutput);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TesseraError::MatryoshkaError(_)
        ));
    }

    #[test]
    fn test_matryoshka_preserves_values() -> Result<()> {
        let device = Device::Cpu;

        // Create tensor with known values
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, (3, 10), &device)?;

        // Truncate to 5 dimensions
        let truncated = apply_matryoshka(&tensor, 5, MatryoshkaStrategy::TruncateOutput)?;
        assert_eq!(truncated.dims(), &[3, 5]);

        // Verify first row values are preserved
        let first_row = truncated.get(0)?;
        let values = first_row.to_vec1::<f32>()?;
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }
}
