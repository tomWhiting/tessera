//! In-crate thread-safety contract for the four backend encoders.
//!
//! The public facades in `crate::api::embedder` own these encoders by value, so
//! a facade can only be `Send + Sync` while every encoder below is. Asserting
//! the encoders separately means a regression names the layer that broke rather
//! than only the facade that surfaced it.
//!
//! Every assertion is discharged by the type checker. Nothing here downloads a
//! checkpoint, creates a device, or allocates a tensor, so the module is safe
//! under `cargo test --no-default-features`.

use crate::backends::CandleBertEncoder;
use crate::encoding::dense::CandleDenseEncoder;
use crate::encoding::sparse::CandleSparseEncoder;
use crate::encoding::vision::ColPaliEncoder;
use crate::runtime::{InferencePermit, ModelResidencyPermit};

const fn assert_send_sync<T: Send + Sync>() {}

// Backend encoders held by value inside the public facades.
const _: () = assert_send_sync::<CandleDenseEncoder>();
const _: () = assert_send_sync::<CandleSparseEncoder>();
const _: () = assert_send_sync::<CandleBertEncoder>();
const _: () = assert_send_sync::<ColPaliEncoder>();

// Process-wide admission handles a threaded caller holds across a forward pass.
const _: () = assert_send_sync::<ModelResidencyPermit<'static>>();
const _: () = assert_send_sync::<InferencePermit<'static>>();

// Candle types every encoder embeds. Listed explicitly so a Candle upgrade that
// drops one of these bounds fails here, next to the reason it matters.
const _: () = assert_send_sync::<candle_core::Device>();
const _: () = assert_send_sync::<candle_core::Tensor>();
const _: () = assert_send_sync::<candle_nn::Linear>();
const _: () = assert_send_sync::<crate::core::Tokenizer>();

#[test]
fn dense_encoder_is_send_and_sync() {
    assert_send_sync::<CandleDenseEncoder>();
}

#[test]
fn sparse_encoder_is_send_and_sync() {
    assert_send_sync::<CandleSparseEncoder>();
}

#[test]
fn multi_vector_encoder_is_send_and_sync() {
    assert_send_sync::<CandleBertEncoder>();
}

#[test]
fn vision_encoder_is_send_and_sync() {
    assert_send_sync::<ColPaliEncoder>();
}

#[test]
fn admission_permits_are_send_and_sync() {
    assert_send_sync::<ModelResidencyPermit<'static>>();
    assert_send_sync::<InferencePermit<'static>>();
}
