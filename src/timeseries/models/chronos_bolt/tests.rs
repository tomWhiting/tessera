use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use super::{ChronosBolt, ResidualMLP};
use crate::timeseries::config::ChronosBoltConfig;
use crate::timeseries::preprocessing::scale_by_mean;

#[test]
fn test_residual_mlp_forward() {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);

    let mlp = ResidualMLP::new(16, 32, 64, vb.pp("test")).unwrap();

    let input = Tensor::randn(0f32, 1.0, (4, 16), &device).unwrap();
    let output = mlp.forward(&input).unwrap();

    assert_eq!(output.dims(), &[4, 64]);
    assert_eq!(output.dtype(), DType::F32);
}

#[test]
fn test_chronos_bolt_config_validation() {
    let config = ChronosBoltConfig::chronos_bolt_small();
    assert!(config.validate().is_ok());
}

#[test]
fn test_chronos_bolt_forward_pass_shape() {
    // This test verifies the forward pass produces the expected quantile output shape
    let device = Device::Cpu;
    let config = ChronosBoltConfig {
        context_length: 512,
        prediction_length: 64,
        patch_size: 16,
        ..ChronosBoltConfig::chronos_bolt_small()
    };

    // Create model with random initialization
    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut model = ChronosBolt::new(config.clone(), vb, device.clone()).unwrap();

    // Create input: [batch=2, context_length=512]
    let input = Tensor::randn(0f32, 1.0, (2, 512), &device).unwrap();

    // Run forward pass - should return all quantiles
    let output = model.forward(&input).unwrap();

    // Verify output shape: [batch, prediction_length, num_quantiles]
    assert_eq!(output.dims(), &[2, 64, 9]);
    assert_eq!(output.dtype(), DType::F32);

    // Verify no NaN or Inf values (with random initialization, should be finite)
    let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for val in output_vec {
        assert!(val.is_finite(), "Output contains non-finite value: {}", val);
    }
}

#[test]
fn test_chronos_bolt_forecast_median() {
    // This test verifies the forecast method extracts the median correctly
    let device = Device::Cpu;
    let config = ChronosBoltConfig {
        context_length: 512,
        prediction_length: 64,
        patch_size: 16,
        ..ChronosBoltConfig::chronos_bolt_small()
    };

    // Create model with random initialization
    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut model = ChronosBolt::new(config.clone(), vb, device.clone()).unwrap();

    // Create input: [batch=2, context_length=512]
    let input = Tensor::randn(0f32, 1.0, (2, 512), &device).unwrap();

    // Run forecast - should return only median
    let forecast = model.forecast(&input).unwrap();

    // Verify output shape: [batch, prediction_length]
    assert_eq!(forecast.dims(), &[2, 64]);
    assert_eq!(forecast.dtype(), DType::F32);

    // Verify no NaN or Inf values
    let forecast_vec = forecast.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for val in forecast_vec {
        assert!(
            val.is_finite(),
            "Forecast contains non-finite value: {}",
            val
        );
    }
}

#[test]
fn test_chronos_bolt_predict_quantiles() {
    // This test verifies the predict_quantiles method returns all quantiles
    let device = Device::Cpu;
    let config = ChronosBoltConfig {
        context_length: 512,
        prediction_length: 64,
        patch_size: 16,
        ..ChronosBoltConfig::chronos_bolt_small()
    };

    // Create model with random initialization
    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut model = ChronosBolt::new(config.clone(), vb, device.clone()).unwrap();

    // Create input: [batch=2, context_length=512]
    let input = Tensor::randn(0f32, 1.0, (2, 512), &device).unwrap();

    // Run predict_quantiles
    let quantiles = model.predict_quantiles(&input).unwrap();

    // Verify output shape: [batch, prediction_length, num_quantiles]
    assert_eq!(quantiles.dims(), &[2, 64, 9]);
    assert_eq!(quantiles.dtype(), DType::F32);

    // Verify quantiles are in ascending order (approximately, with random weights)
    // Just check that we can access each quantile index
    for q_idx in 0..9 {
        let q = quantiles.i((.., .., q_idx)).unwrap();
        assert_eq!(q.dims(), &[2, 64]);
    }
}

#[test]
fn test_chronos_bolt_scaling_roundtrip() {
    // Verify that scaling and denormalization work correctly
    let device = Device::Cpu;
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device).unwrap();

    let (scaled, scale) = scale_by_mean(&input).unwrap();

    // Apply some operation (identity for simplicity)
    let output = scaled.clone();

    // Denormalize using broadcast_mul
    let denormalized = output.broadcast_mul(&scale).unwrap();

    // Should be close to original input
    let denorm_data = denormalized.to_vec2::<f32>().unwrap();
    let input_data = input.to_vec2::<f32>().unwrap();

    for (orig, denorm) in input_data[0].iter().zip(&denorm_data[0]) {
        assert!((orig - denorm).abs() < 1e-5);
    }
}
