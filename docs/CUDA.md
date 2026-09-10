# CUDA

**Status: unproven.** Nothing in this document has been executed against an
NVIDIA GPU. It was written on a macOS machine with no NVIDIA hardware, where the
CUDA build cannot start (see [What stops a CUDA build off a GPU
box](#what-stops-a-cuda-build-off-a-gpu-box)). The build path, the smoke script,
and the tolerance below are all first-run material, not a gate.

## What the `cuda` feature pulls

`Cargo.toml` defines it as a pure pass-through to Candle:

```toml
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
```

Tessera has no CUDA source of its own. The entire `cfg(feature = "cuda")`
surface in `src/` is three items in `src/backends/candle/device.rs`: an
`anyhow::Context` import, the CUDA branch inside `get_device`, and the
`cuda_device()` constructor.

From `Cargo.lock`, enabling it brings in:

| Crate | Version | Needs `nvcc` at build time |
| --- | --- | --- |
| `cudarc` | 0.19.8 (via `candle-core`) | **Yes** — `build.rs` runs `nvcc --version` to detect the CUDA version |
| `cudarc` | 0.17.8 (via `ug-cuda`, via `candle-ug`) | Yes, same detection |
| `candle-kernels` | 0.11.0 | **Yes** — compiles the `.cu` kernels through `cudaforge` 0.1.6 |
| `cudaforge` | 0.1.6 | Yes — it *is* the nvcc driver |
| `ug-cuda` | 0.5.0 | Through `cudarc` 0.17.8 |

`cudarc` is loaded dynamically at runtime (`libloading`), so the CUDA *runtime*
libraries do not have to be linked statically — but the *toolkit* is a hard
build-time requirement for both `cudarc` and `candle-kernels`.

## Prerequisites

- Linux. (Candle 0.11's CUDA path is not supported on macOS, and no Apple
  machine has NVIDIA hardware.)
- An NVIDIA GPU with a working driver — `nvidia-smi` must run.
- The CUDA toolkit, with `nvcc` on `PATH`. `cudarc` also honours
  `CUDA_HOME`, `CUDA_PATH`, `CUDA_ROOT`, `CUDA_TOOLKIT_ROOT_DIR`, `CUDNN_LIB`
  and `CUDARC_CUDA_VERSION`.
- The toolchain pinned in `rust-toolchain.toml`.
- Network access on the first run, so the checkpoint can be fetched.

## The exact commands

```bash
# from the repository root
cargo check --locked --no-default-features --features cuda
cargo build --locked --no-default-features --features cuda --release --example device_parity
./scripts/cuda-smoke                       # defaults: bge-base-en-v1.5, ordinal 0, tolerance 0.999
./scripts/cuda-smoke --model bge-base-en-v1.5 --ordinal 1 --tolerance 0.999
./scripts/cuda-smoke --help
```

## What the smoke checks

`scripts/cuda-smoke` refuses to run on anything but Linux, then requires
`cargo`, `nvcc` and `nvidia-smi`, prints the nvcc version and the GPU table, and
checks the requested ordinal against the number of visible GPUs. It then builds
`--features cuda` and runs the `device_parity` example.

`examples/device_parity.rs` does the actual measurement:

1. Builds `TesseraDense` on an **explicitly named** CUDA device — not through
   `tessera::get_device`, which silently falls back to CPU and would make a
   CPU-vs-CPU comparison pass while proving nothing.
2. Encodes three fixed sentences and records the wall time of the encode pass.
3. Drops that embedder, so only one copy of the parameters is ever resident and
   the process-wide residency ledger cannot reject the second load.
4. Repeats on `Device::Cpu`.
5. Compares each sentence's CUDA vector against its CPU vector with an
   f64 cosine that does not assume either side is normalized, and prints the
   CPU-to-accelerator wall-time ratio.

It exits non-zero if any sentence falls under the tolerance, if the two devices
disagree on dimension, or if any component is non-finite.

The 0.999 default tolerance is a **guess**, not a measurement. CUDA and CPU
kernels reduce in different orders, so exact equality is not expected; the right
number can only be set after a first real run.

## What stops a CUDA build off a GPU box

Run on macOS 25.6 (Apple silicon, no NVIDIA hardware), 2026-09-10:

```
cargo check --locked --no-default-features --features cuda
```

fails with two build-script errors, and never reaches any Tessera code:

```
error: failed to run custom build command for `cudarc v0.19.8`
  thread 'main' panicked at .../cudarc-0.19.8/build.rs:168:13:
  `nvcc --version` failed.
  Err(Os { code: 2, kind: NotFound, message: "No such file or directory" })

error: failed to run custom build command for `candle-kernels v0.11.0`
  Error: NvccNotFound("No nvcc found in PATH or standard locations")
```

Because those build scripts run before compilation, **no CUDA-gated Tessera code
is type-checked on a machine without the toolkit.** The three items in
`src/backends/candle/device.rs` were reviewed by hand instead.

## Device and dtype behaviour on CUDA

- **Selection.** `get_device()` (`src/backends/candle/device.rs:53`) tries
  Metal-on-macOS first, then CUDA ordinal 0, then CPU. Each accelerator step
  discards its error and falls through, so a broken driver yields
  `Device::Cpu` with no diagnostic. To fail loudly, construct the device
  yourself — `tessera::Device::new_cuda(n)` or `cuda_device()` — and hand it to
  a builder's `.device(...)`.
- **Ordinal.** `cuda_device()` is hardwired to ordinal 0. Multi-GPU selection
  goes through `Device::new_cuda(n)` and a builder.
- **Residency.** The process-wide ledger keys on
  `DeviceLocation`, so a model resident on `cuda:0` does not block the same
  model on `cuda:1` or on CPU (`src/runtime/residency.rs:21-45`).
- **dtype.** `ModelDType::validate_device` (`src/runtime/load.rs:39`) rejects
  F16/BF16 **on CPU only**. On CUDA, F16 and BF16 are passed straight to
  Candle, so an unsupported model or kernel fails inside Candle at load or at
  the forward pass, not in Tessera's preflight. F32 remains the default
  (`src/runtime/load.rs:17`). No accelerator dtype has been certified.
- **Weight loading is device-agnostic.** All four
  `VarBuilder::from_mmaped_safetensors` sites take the target `&device` and the
  selected `dtype` and are otherwise identical across devices:
  `src/backends/candle/encoder.rs:129`, `src/encoding/dense/loading.rs:99`,
  `src/encoding/sparse.rs:220`, `src/encoding/vision/construction.rs:134`.
  (A fifth, `src/timeseries/models/chronos_bolt/model.rs:163`, sits behind the
  quarantined `timeseries` feature.) The `unsafe` is the mmap contract — the
  file must not be mutated while mapped — and carries no device assumption.

## What remains unproven

- The CUDA build has never completed anywhere.
- No Tessera CUDA-gated line has ever been type-checked.
- No embedding has ever been produced on a CUDA device.
- The 0.999 cosine tolerance is unmeasured.
- F16 and BF16 on CUDA are ungated by Tessera and untested.
