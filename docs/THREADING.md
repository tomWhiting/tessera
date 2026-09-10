# Threading

What a multi-threaded server gets from Tessera today, with the code that decides
it. Nothing here is aspirational: it is a reading of the current tree, backed by
compile-time assertions that fail the build if it stops being true.

## The four facades are `Send + Sync`

All four public embedders, and the four backend encoders they own by value, are
both `Send` and `Sync`:

| Facade | Encoder it owns | Verdict |
| --- | --- | --- |
| `TesseraDense` (`src/api/embedder/dense.rs:17`) | `CandleDenseEncoder` (`src/encoding/dense.rs:111`) | `Send + Sync` |
| `TesseraSparse` (`src/api/embedder/sparse.rs:14`) | `CandleSparseEncoder` (`src/encoding/sparse.rs:137`) | `Send + Sync` |
| `TesseraMultiVector` (`src/api/embedder/multi_vector.rs:20`) | `CandleBertEncoder` (`src/backends/candle/encoder.rs:42`) | `Send + Sync` |
| `TesseraVision` (`src/api/embedder/vision.rs:19`) | `ColPaliEncoder` (`src/encoding/vision.rs:70`) | `Send + Sync` |

Nothing blocks the bounds. There is no `Rc`, no `RefCell`, and no raw pointer in
any of these types. The fields that could have been a problem are not:

- `candle_core::Device` and `candle_core::Tensor` are `Send + Sync` on every
  backend Candle 0.11 compiles.
- `ModelResidencyPermit<'static>` (`src/runtime/residency.rs:121`) holds
  `&'static ModelResidencyLedger`, whose only state is a `Mutex`.
- `ColPaliEncoder` reaches `Sync` through `Arc<Mutex<PaliGemmaModel>>`
  (`src/encoding/vision.rs:72`), which is also why the vision forward pass
  serializes against itself independently of the gate below.

This is enforced, not asserted in prose. `tests/thread_safety_test.rs` covers
the public facades and `src/encoding/thread_safety.rs` covers the private
encoders and the admission permits; both use `const fn assert_send_sync<T: Send
+ Sync>()` discharged in `const _: () = ...` items, so `cargo test
--no-default-features` fails to *compile* if a future field breaks the contract.
No model is downloaded and no device is created by either file.

So `Arc<TesseraDense>` shared across request threads is sound today.

## But there is exactly one forward pass in the process

Sharing the embedder does not buy concurrency. Every forward pass takes a
process-wide exclusive permit first:

- `InferenceGate` is a single `static` with one `occupied: bool`
  (`src/runtime/inference.rs:14`, `:144-155`). Admission is FIFO by ticket
  (`:189-217`).
- Six call sites take it, immediately before `model.forward(...)`, and drop it
  explicitly once the result has been copied back to CPU:
  `src/encoding/dense/inference.rs:206`/`:229` (single) and `:335`/`:355`
  (batch), `src/encoding/sparse.rs:372`/`:388`,
  `src/backends/candle/encoder/inference.rs:64`/`:83`, and
  `src/encoding/vision/inference.rs:64`/`:92` and `:111`/`:138`.
- The permit releases on drop, including on the early `?` returns between
  acquire and drop (`src/runtime/inference.rs:102-112`), so a failed forward
  pass does not wedge the gate.

The gate is process-wide, not per-model and not per-device. Two embedders on two
different GPUs still serialize against each other.

Defaults, from `src/runtime/inference.rs:11-12`: **16** waiting callers and a
**30 s** wait. A 17th concurrent caller does not queue — it fails immediately
with `InferenceGateError::QueueFull`. A caller that waits longer than 30 s fails
with `TimedOut`. Both are errors returned to the request, not backpressure.

Change them once, before the first forward pass, with
`configure_inference_gate(InferenceGateConfig::new(waiters, timeout))`
(`src/runtime/inference.rs:57`). The first configuration wins for the life of
the process; a different one afterwards is rejected with `AlreadyConfigured`.
`try_acquire_inference()` is the non-blocking probe, and it returns
`InferencePermit`, which is now re-exported at the crate root so a caller can
actually name it.

## What that adds up to

For a server, today:

- **One `Arc<Embedder>` per model, shared across all threads.** Sound.
- **One forward pass at a time, process-wide.** Throughput is one model
  execution, serialized, regardless of thread count or device count.
- **Permits are released on drop**, so failures and panics unwind cleanly; the
  bookkeeping mutex has explicit poison recovery
  (`src/runtime/inference.rs:251-261`).
- **No pool.** There is no worker pool, no per-device queue, no batching across
  callers. `encode_batch` takes the gate once per chunk, and with no
  `batch_size` configured the chunk size is 1 (`src/api/embedder/dense.rs:227`),
  so a 64-text batch takes and releases the gate 64 times and interleaves with
  every other thread between chunks.
- **Loading is also serialized by the ledger's `Duplicate` rule**: the same
  model at the same revision/device/dtype cannot be loaded twice in one process
  (`src/runtime/residency.rs:66-73`). Load once, share the handle.
- **CPU threads are capped at 2 by default.** `preflight` calls
  `configure_cpu_threads(2)` (`src/runtime/model.rs:82`) before the first CPU
  model, which sets `RAYON_NUM_THREADS` and `CANDLE_NUM_THREADS`. Raise it by
  calling `configure_cpu_threads(n)` yourself during single-threaded startup —
  first call wins (`src/runtime/threading.rs:65`).

A pool is not in this tree and is not implied by any of the above.
