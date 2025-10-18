//! Preprocessing utilities for Chronos Bolt time series model.
//!
//! Provides scaling, patching, and quantization functions required by
//! the Chronos Bolt architecture for converting continuous time series
//! into discrete tokens for the T5 encoder-decoder.

use candle_core::{DType, Result, Tensor};

/// Scale time series by absolute mean.
///
/// Divides each series by its absolute mean to normalize scale.
/// This is required for Chronos to handle time series of different magnitudes.
///
/// # Arguments
/// * `input` - Time series tensor of shape [batch, length]
///
/// # Returns
/// Tuple of (scaled_tensor, scale_factors) where:
/// * `scaled_tensor` - Normalized time series [batch, length]
/// * `scale_factors` - Absolute mean of each series [batch, 1]
///
/// # Errors
/// Returns error if tensor operations fail or shapes are invalid
///
/// # Example
/// ```ignore
/// let input = Tensor::randn(0.0, 1.0, (4, 512), &device)?;
/// let (scaled, scale) = scale_by_mean(&input)?;
/// ```
pub fn scale_by_mean(input: &Tensor) -> Result<(Tensor, Tensor)> {
    // input: [batch, length]

    // Compute absolute mean per series
    let abs_input = input.abs()?;
    let scale = abs_input.mean_keepdim(1)?; // [batch, 1]

    // Add epsilon to prevent division by zero
    let epsilon = 1e-10;
    let scale_safe = (scale + epsilon)?;

    // Scale
    let scaled = input.broadcast_div(&scale_safe)?;

    Ok((scaled, scale_safe))
}

/// Create non-overlapping patches from time series.
///
/// Reshapes a flat time series into non-overlapping patches.
/// This is the first step in converting continuous data into
/// discrete tokens for the T5 model.
///
/// # Arguments
/// * `input` - Time series tensor of shape [batch, length]
/// * `patch_size` - Number of timesteps per patch (typically 16)
///
/// # Returns
/// Patched tensor of shape [batch, num_patches, patch_size]
/// where num_patches = length / patch_size
///
/// # Errors
/// Returns error if length is not divisible by patch_size
///
/// # Example
/// ```ignore
/// let input = Tensor::randn(0.0, 1.0, (4, 512), &device)?;
/// let patches = create_patches(&input, 16)?; // [4, 32, 16]
/// ```
pub fn create_patches(input: &Tensor, patch_size: usize) -> Result<Tensor> {
    // input: [batch, length]
    // output: [batch, num_patches, patch_size]

    let (batch, length) = input.dims2()?;

    if length % patch_size != 0 {
        candle_core::bail!(
            "Length {} is not divisible by patch_size {}",
            length,
            patch_size
        );
    }

    let num_patches = length / patch_size;

    // Reshape: [batch, length] -> [batch, num_patches, patch_size]
    input.reshape((batch, num_patches, patch_size))
}

/// Quantize continuous values into discrete bins (0 to num_bins-1).
///
/// Converts continuous time series values into discrete token IDs
/// for the T5 vocabulary. Uses uniform binning from -10 to 10.
///
/// # Arguments
/// * `input` - Time series tensor of shape [batch, num_patches, patch_size]
/// * `num_bins` - Number of discrete bins (typically 4096 for Chronos)
///
/// # Returns
/// Quantized tensor of shape [batch, num_patches, patch_size] with i64 dtype
/// containing bin indices in range [0, num_bins-1]
///
/// # Errors
/// Returns error if tensor operations fail
///
/// # Example
/// ```ignore
/// let patches = Tensor::randn(0.0, 1.0, (4, 32, 16), &device)?;
/// let tokens = quantize_to_bins(&patches, 4096)?;
/// ```
pub fn quantize_to_bins(input: &Tensor, num_bins: usize) -> Result<Tensor> {
    // input: [batch, num_patches, patch_size]
    // output: [batch, num_patches, patch_size] as i64 token IDs

    // Chronos uses uniform bins from -10 to 10
    let min_val = -10.0_f32;
    let max_val = 10.0_f32;
    let bin_width = (max_val - min_val) / (num_bins as f32);

    // Clip to range
    let clipped = input.clamp(min_val as f64, max_val as f64)?;

    // Compute bin indices
    let shifted = ((clipped - min_val as f64)? / bin_width as f64)?;
    let bins = shifted.floor()?.to_dtype(DType::I64)?;

    // Clamp to [0, num_bins-1] to ensure valid indices
    let max_bin = (num_bins - 1) as i64;
    let bins = bins.clamp(0i64, max_bin)?;

    Ok(bins)
}

/// Dequantize bin indices back to continuous values.
///
/// Converts discrete token IDs back into continuous time series values
/// by mapping each bin to its center value.
///
/// # Arguments
/// * `bins` - Tensor of bin indices [batch, length] as i64
/// * `num_bins` - Number of discrete bins used in quantization
///
/// # Returns
/// Continuous values tensor [batch, length] as f32
///
/// # Errors
/// Returns error if tensor operations fail
///
/// # Example
/// ```ignore
/// let bins = Tensor::new(&[0i64, 100, 200], &device)?;
/// let values = dequantize_from_bins(&bins, 4096)?;
/// ```
pub fn dequantize_from_bins(bins: &Tensor, num_bins: usize) -> Result<Tensor> {
    // bins: [batch, length] as i64
    // output: [batch, length] as f32

    let min_val = -10.0_f32;
    let max_val = 10.0_f32;
    let bin_width = (max_val - min_val) / (num_bins as f32);

    // Convert to f32 and compute bin centers
    let bins_f32 = bins.to_dtype(DType::F32)?;
    let bin_center_offset = min_val + bin_width / 2.0;
    let values = ((bins_f32 * bin_width as f64)? + bin_center_offset as f64)?;

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_scale_by_mean() {
        let device = Device::Cpu;
        let input = Tensor::new(&[[1.0f32, 2.0, 3.0], [10.0, 20.0, 30.0]], &device).unwrap();

        let (scaled, scale) = scale_by_mean(&input).unwrap();

        // Check shapes
        assert_eq!(scaled.dims(), &[2, 3]);
        assert_eq!(scale.dims(), &[2, 1]);

        // First series: mean = 2.0, scaled values should be around [0.5, 1.0, 1.5]
        // Second series: mean = 20.0, scaled values should be around [0.5, 1.0, 1.5]
        let scaled_data = scaled.to_vec2::<f32>().unwrap();
        assert!((scaled_data[0][0] - 0.5).abs() < 0.1);
        assert!((scaled_data[1][0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_create_patches() {
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (4, 512), &device).unwrap();

        let patches = create_patches(&input, 16).unwrap();

        // Check shape: [batch, num_patches, patch_size]
        assert_eq!(patches.dims(), &[4, 32, 16]);
    }

    #[test]
    fn test_create_patches_invalid_size() {
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (4, 513), &device).unwrap();

        // Should fail because 513 is not divisible by 16
        let result = create_patches(&input, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_to_bins() {
        let device = Device::Cpu;
        let input = Tensor::new(&[[[0.0f32, 5.0, -5.0, 10.0, -10.0]]], &device).unwrap();

        let bins = quantize_to_bins(&input, 4096).unwrap();

        // Check shape
        assert_eq!(bins.dims(), &[1, 1, 5]);

        // Check dtype
        assert_eq!(bins.dtype(), DType::I64);

        // Check values are in valid range
        let bins_data = bins.to_vec3::<i64>().unwrap();
        for row in &bins_data {
            for patch in row {
                for &bin in patch {
                    assert!(bin >= 0 && bin < 4096);
                }
            }
        }
    }

    #[test]
    fn test_dequantize_from_bins() {
        let device = Device::Cpu;
        let bins = Tensor::new(&[[0i64, 2048, 4095]], &device).unwrap();

        let values = dequantize_from_bins(&bins, 4096).unwrap();

        // Check shape and dtype
        assert_eq!(values.dims(), &[1, 3]);
        assert_eq!(values.dtype(), DType::F32);

        // Check approximate values
        let values_data = values.to_vec2::<f32>().unwrap();
        // Bin 0 should be near -10, bin 2048 near 0, bin 4095 near 10
        assert!(values_data[0][0] < -9.0);
        assert!(values_data[0][1].abs() < 1.0);
        assert!(values_data[0][2] > 9.0);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let device = Device::Cpu;
        let input = Tensor::new(&[[[0.0f32, 1.0, -1.0, 5.0, -5.0]]], &device).unwrap();

        // Quantize
        let bins = quantize_to_bins(&input, 4096).unwrap();

        // Dequantize
        let (batch, patches, patch_size) = bins.dims3().unwrap();
        let bins_flat = bins.reshape((batch, patches * patch_size)).unwrap();
        let reconstructed = dequantize_from_bins(&bins_flat, 4096).unwrap();

        // Values should be approximately the same (within bin width tolerance)
        let input_flat = input.reshape((batch, patches * patch_size)).unwrap();
        let input_data = input_flat.to_vec2::<f32>().unwrap();
        let reconstructed_data = reconstructed.to_vec2::<f32>().unwrap();

        for (orig, recon) in input_data[0].iter().zip(&reconstructed_data[0]) {
            // Tolerance should be within one bin width (20/4096 ≈ 0.005)
            assert!((orig - recon).abs() < 0.1);
        }
    }
}
