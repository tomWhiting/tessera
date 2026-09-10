# Metal

The contract a Metal run can rely on: how the device is chosen, which dtypes
Tessera itself will accept, and what it does not promise.

## Enabling it

```toml
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal", "dep:candle-metal-kernels"]
```

```bash
cargo test  --locked --no-default-features --features metal --lib
cargo clippy --locked --no-default-features --features metal --workspace --all-targets -- -D warnings
```

Both were run on macOS 25.6 / Apple silicon on 2026-09-10 and exited 0. Note
that `scripts/check` does **not** exercise the `metal` feature — its clippy and
test legs cover the default, `pdf`, `timeseries` and `python` feature sets — so
a Metal change has to be checked with the two commands above by hand.

## How the device is selected

`get_device()` — `src/backends/candle/device.rs:53`, re-exported as
`tessera::get_device` — has a fixed, compile-time-gated order:

1. **Metal ordinal 0**, only when the target is macOS *and* the `metal` feature
   is on. The ordinal is validated against
   `candle_metal_kernels::metal::Device::all().len()` before the device is
   created (`device.rs:9-25`), because Candle 0.11 calls `swap_remove(ordinal)`
   on that list without checking it first and therefore *panics* on a machine
   with no Metal device rather than returning an error.
2. **CUDA ordinal 0**, only when the `cuda` feature is on.
3. **CPU.**

Each accelerator step is a fallback: if the device cannot be created the error
is discarded and selection continues. A Mac whose Metal device is unavailable
therefore gets `Device::Cpu` back with no diagnostic. Anything that must fail
loudly should build the device itself and pass it to a builder:

```rust
use tessera::{metal_device, TesseraDense};

let embedder = TesseraDense::builder()
    .model("bge-base-en-v1.5")
    .device(metal_device()?)     // guarded: errors instead of panicking
    .build()?;
```

`tessera::metal_device()` (macOS + `metal` only) is the guarded ordinal-0
constructor. `candle_core::Device::new_metal(n)` — re-exported as
`tessera::Device` — is the only route to a non-zero ordinal, and it **skips the
guard**, so it inherits Candle's panic on an empty device list.

## Which dtypes it accepts

`ModelDType` (`src/runtime/load.rs:14`) is `F32` (the default), `F16`, `BF16`.
The only dtype gate Tessera applies is
`ModelDType::validate_device` (`src/runtime/load.rs:39`), and it is a *CPU*
gate:

| Device | F32 | F16 | BF16 |
| --- | --- | --- | --- |
| CPU | accepted | rejected — `CpuRequiresF32` | rejected — `CpuRequiresF32` |
| Metal | accepted | accepted by Tessera | accepted by Tessera |
| CUDA | accepted | accepted by Tessera | accepted by Tessera |

"Accepted by Tessera" means passed straight through: the dtype reaches
`VarBuilder::from_mmaped_safetensors` and then the model, so an unsupported
Metal kernel or a checkpoint that has no such tensors fails inside Candle at
load or at the forward pass — not in preflight, and not with a Tessera error
type. **No accelerator dtype is certified.** Set one with
`.dtype(ModelDType::F16)` on any builder and treat the first run as the
experiment.

This table is enforced by two tests in `src/runtime/load/tests.rs`:
`accelerator_dtype_contract_is_not_restricted_to_f32` (model-free, runs
everywhere) and `metal_accepts_every_declared_dtype` (gated on macOS + `metal`,
creates a real Metal device and asserts all three dtypes pass the gate).

Note that dtype also changes admission arithmetic: `bytes_per_parameter()` is 4
for F32 and 2 for F16/BF16 (`src/runtime/load.rs:27`), and that number feeds the
residency ledger's aggregate byte ceiling.

## Residency on Metal

The process-wide ledger keys on `(model_id, revision, DeviceLocation, dtype)`
(`src/runtime/residency.rs:21-45`). Consequences on a Mac:

- The same model at the same revision, dtype and Metal ordinal cannot be loaded
  twice — the second load fails with `Duplicate`. Share the embedder instead;
  it is `Send + Sync` (see `docs/THREADING.md`).
- The same model on Metal and on CPU *are* different keys and can coexist, but
  both then count against the aggregate byte ceiling of the requesting policy.
- A permit is released on drop (`residency.rs:127-132`), so dropping an embedder
  frees its accounting.

## Weight loading

All four `VarBuilder::from_mmaped_safetensors` sites take the target `&device`
and the selected dtype and are otherwise device-agnostic:
`src/backends/candle/encoder.rs:129`, `src/encoding/dense/loading.rs:99`,
`src/encoding/sparse.rs:220`, `src/encoding/vision/construction.rs:134`. The
`unsafe` is the mmap contract — the file must not be mutated while mapped — and
says nothing about the device.

## Not promised

- No Metal numerical parity with CPU has been measured. `examples/device_parity.rs`
  will do it (`--device metal`), but no run is recorded.
- No Metal dtype other than F32 has been exercised.
- Metal ordinals above 0 are unguarded.
- `get_device()`'s CPU fallback is silent by design and unchanged.
