//! Encoding strategies for different embedding paradigms.
//!
//! This module provides the core encoding implementations that convert
//! input data (text, images, time series) into embedding representations.
//! Each submodule handles a specific embedding paradigm:
//!
//! - [`dense`]: Single-vector embeddings via pooling strategies (BERT, BGE, Nomic, GTE)
//! - [`sparse`]: Sparse vocabulary-space embeddings for interpretable search (SPLADE)
//! - [`vision`]: Image and visual document encoding (`ColPali`)
//!
//! The encoding layer sits between the Candle backend and the high-level API,
//! implementing paradigm-specific logic.

pub mod dense;
pub mod sparse;
pub mod vision;
