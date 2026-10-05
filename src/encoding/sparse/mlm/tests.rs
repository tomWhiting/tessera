use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear};

use super::MlmHead;

#[test]
fn head_uses_exact_gelu_for_fixed_inputs() -> Result<()> {
    let expected = [
        -0.004_049_694_f32,
        -0.158_655_26,
        0.0,
        0.841_344_8,
        2.995_950_2,
    ];
    let mean = expected.iter().sum::<f32>() / 5.0;
    let variance = expected
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / 5.0;
    let scale = (variance + 1e-12).sqrt();
    let identity = Tensor::eye(5, DType::F32, &Device::Cpu)?;
    let head = MlmHead {
        transform_dense: Linear::new(identity.clone(), None),
        // Cancel normalization for the exact activation values.
        transform_layer_norm: LayerNorm::new(
            Tensor::full(scale, 5, &Device::Cpu)?,
            Tensor::full(mean, 5, &Device::Cpu)?,
            1e-12,
        ),
        decoder: Linear::new(identity, None),
    };
    let input = Tensor::new(&[[-3.0_f32, -1.0, 0.0, 1.0, 3.0]], &Device::Cpu)?;
    let actual = head.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-6,
            "activation {index}: expected {expected}, observed {actual}"
        );
    }
    Ok(())
}
