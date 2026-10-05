use std::path::PathBuf;

/// Why a miniCOIL table could not be read or a vector could not be formed.
#[derive(Debug, thiserror::Error)]
pub enum MinicoilError {
    /// A model file could not be opened or read.
    #[error("cannot read miniCOIL file {path}: {source}")]
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The projection file does not start with the NumPy magic and a known version.
    #[error("projection file {path} is not a NumPy .npy file of version 1, 2 or 3")]
    ProjectionMagic {
        /// The projection file.
        path: PathBuf,
    },
    /// The projection header is not the dictionary NumPy writes.
    #[error("projection file {path} has an unreadable header: {detail}")]
    ProjectionHeader {
        /// The projection file.
        path: PathBuf,
        /// What could not be read.
        detail: String,
    },
    /// The projection values are not little-endian 32-bit floats.
    #[error("projection file {path} stores {descr}, not little-endian float32 ('<f4')")]
    ProjectionType {
        /// The projection file.
        path: PathBuf,
        /// The stored NumPy type descriptor.
        descr: String,
    },
    /// The projection values are stored in Fortran order.
    #[error("projection file {path} is in Fortran order, not C order")]
    ProjectionOrder {
        /// The projection file.
        path: PathBuf,
    },
    /// The projection shape is not (vocabulary, 512, 4).
    #[error("projection file {path} has shape {shape:?}, not (vocabulary, 512, 4)")]
    ProjectionShape {
        /// The projection file.
        path: PathBuf,
        /// The stored shape.
        shape: Vec<usize>,
    },
    /// The file length does not match the header's shape.
    #[error("projection file {path} holds {actual} data bytes; its header needs {expected}")]
    ProjectionLength {
        /// The projection file.
        path: PathBuf,
        /// Bytes the shape requires.
        expected: u64,
        /// Bytes present after the header.
        actual: u64,
    },
    /// A projection row has the wrong number of values.
    #[error("projection row {vocab_id} has {values} values; expected {expected}")]
    ProjectionRowValues {
        /// The vocabulary id.
        vocab_id: u32,
        /// Number of values given.
        values: usize,
        /// Number of values expected.
        expected: usize,
    },
    /// A vocabulary id has no projection row.
    #[error("vocabulary id {vocab_id} has no projection row")]
    MissingRow {
        /// The vocabulary id.
        vocab_id: u32,
    },
    /// The vocabulary JSON is malformed.
    #[error("vocabulary file {path} is malformed: {detail}")]
    Vocabulary {
        /// The vocabulary file.
        path: PathBuf,
        /// What was wrong.
        detail: String,
    },
    /// The token vectors do not match the tokens.
    #[error("{tokens} tokens were given with {values} vector values; expected {expected}")]
    TokenVectors {
        /// Number of tokens.
        tokens: usize,
        /// Number of vector values given.
        values: usize,
        /// Number of values expected (tokens × 512).
        expected: usize,
    },
}
