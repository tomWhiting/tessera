//! miniCOIL sparse vectors from encoder token outputs.
//!
//! This reproduces the post-encoder half of Qdrant's FastEmbed 0.8.1 miniCOIL
//! model (`fastembed/sparse/minicoil.py` and `fastembed/sparse/utils/` at commit
//! `8de28b8f2d4525167c7ebe988aa06b740a59e0c0`): WordPiece pieces are joined into
//! words, each word is resolved against the miniCOIL vocabulary, known words are
//! projected to four values per vocabulary id, unknown words fall back to a
//! hashed BM25 term, and the result is laid out as one sparse vector.
//!
//! The conversion is pure: the model tables and the projection rows are loaded
//! beforehand by [`tables::load_vocabulary`], [`tables::load_stopwords`] and
//! [`projection::ProjectionFile`].
//! IDF is not applied; FastEmbed leaves it to the search engine.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the miniCOIL conversion is wired into the sparse embedder by a later change"
    )
)]

pub mod error;
mod murmur;
pub mod projection;
pub mod resolve;
pub mod tables;
pub mod vector;

#[cfg(test)]
mod tests;

pub use error::MinicoilError;
pub use projection::ProjectionRows;
pub use tables::MinicoilTables;
