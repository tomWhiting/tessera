//! Compile-time thread-safety contract for the public embedder facades.
//!
//! A multi-threaded server shares one embedder across request threads. That is
//! only sound when the facade is both [`Send`] (it can be moved into a worker
//! thread) and [`Sync`] (`&Embedder` can be handed to several threads at once,
//! which is what `Arc<Embedder>` requires).
//!
//! These assertions are model-free: they instantiate generic functions whose
//! bounds are discharged by the type checker, so no weights are downloaded and
//! no device is created. If a future field breaks the contract — a `Rc`, a
//! `RefCell`, a raw pointer, a non-`Sync` handle — this test file stops
//! compiling and `cargo test --no-default-features` fails.

use tessera::{TesseraDense, TesseraMultiVector, TesseraSparse, TesseraVision};

/// Discharges `T: Send + Sync` at type-check time.
const fn assert_send_sync<T: Send + Sync>() {}

/// Discharges the bound `Arc<T>: Send + Sync` needs to cross thread boundaries.
const fn assert_shareable_across_threads<T: Send + Sync + 'static>() {}

// Enforced in every build of this test crate, not only when a test body runs.
const _: () = assert_send_sync::<TesseraDense>();
const _: () = assert_send_sync::<TesseraSparse>();
const _: () = assert_send_sync::<TesseraMultiVector>();
const _: () = assert_send_sync::<TesseraVision>();

const _: () = assert_shareable_across_threads::<TesseraDense>();
const _: () = assert_shareable_across_threads::<TesseraSparse>();
const _: () = assert_shareable_across_threads::<TesseraMultiVector>();
const _: () = assert_shareable_across_threads::<TesseraVision>();

#[test]
fn dense_facade_is_send_and_sync() {
    assert_send_sync::<TesseraDense>();
    assert_shareable_across_threads::<TesseraDense>();
}

#[test]
fn sparse_facade_is_send_and_sync() {
    assert_send_sync::<TesseraSparse>();
    assert_shareable_across_threads::<TesseraSparse>();
}

#[test]
fn multi_vector_facade_is_send_and_sync() {
    assert_send_sync::<TesseraMultiVector>();
    assert_shareable_across_threads::<TesseraMultiVector>();
}

#[test]
fn vision_facade_is_send_and_sync() {
    assert_send_sync::<TesseraVision>();
    assert_shareable_across_threads::<TesseraVision>();
}

/// The process-wide admission types a threaded server touches on every request
/// must themselves cross threads, otherwise the facades above could not.
#[test]
fn runtime_admission_types_cross_threads() {
    assert_send_sync::<tessera::ResourcePolicy>();
    assert_send_sync::<tessera::InferenceGateConfig>();
    assert_send_sync::<tessera::ModelDType>();
}
