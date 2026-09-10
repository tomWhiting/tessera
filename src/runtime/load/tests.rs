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

#[test]
fn accelerator_dtype_contract_is_not_restricted_to_f32() {
    // `validate_device` is the only dtype gate Tessera applies before loading.
    // It narrows CPU to F32 and leaves every accelerator dtype to the backend,
    // so a Metal or CUDA load of F16/BF16 fails (if it fails) inside Candle
    // with a kernel or safetensors error, not here.
    let cpu_only_rejects = matches!(
        ModelDType::F16.validate_device(&Device::Cpu),
        Err(super::ModelDTypeError::CpuRequiresF32 { .. })
    );
    assert!(cpu_only_rejects, "the CPU gate must be the F32 gate");

    // Enumerated so a future dtype variant forces a decision here.
    for dtype in [ModelDType::F32, ModelDType::F16, ModelDType::BF16] {
        assert!(
            matches!(dtype.candle_dtype(), candle_core::DType::F32)
                || matches!(
                    dtype.candle_dtype(),
                    candle_core::DType::F16 | candle_core::DType::BF16
                ),
            "every Tessera dtype must map to a float Candle dtype"
        );
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_accepts_every_declared_dtype() {
    let device = crate::backends::candle::device::metal_device()
        .expect("the metal feature requires a working Metal device");
    assert!(matches!(device, Device::Metal(_)));

    for dtype in [ModelDType::F32, ModelDType::F16, ModelDType::BF16] {
        dtype
            .validate_device(&device)
            .expect("Tessera does not gate any dtype on Metal");
    }
}
