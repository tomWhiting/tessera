use candle_core::Device;

use super::ModelDType;

#[test]
fn parameter_widths_are_exact() {
    assert_eq!(ModelDType::F32.bytes_per_parameter(), 4);
    assert_eq!(ModelDType::F16.bytes_per_parameter(), 2);
    assert_eq!(ModelDType::BF16.bytes_per_parameter(), 2);
}

#[test]
fn cpu_rejects_lower_precision_before_model_loading() {
    ModelDType::F32
        .validate_device(&Device::Cpu)
        .expect("F32 CPU is supported");

    for dtype in [ModelDType::F16, ModelDType::BF16] {
        let error = dtype
            .validate_device(&Device::Cpu)
            .expect_err("lower-precision CPU load must be rejected");
        assert!(error.to_string().contains("requires F32"));
    }
}
