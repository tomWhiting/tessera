use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::MinicoilError;

/// Width of an encoder token vector, the projection's input.
pub const PROJECTION_INPUT: usize = 512;
/// Values per vocabulary word, the projection's output.
pub const PROJECTION_OUTPUT: usize = 4;

const ROW_VALUES: usize = PROJECTION_INPUT * PROJECTION_OUTPUT;
const ROW_BYTES: usize = ROW_VALUES * 4;
const MAGIC: &[u8; 6] = b"\x93NUMPY";
const MAX_HEADER_BYTES: usize = 64 * 1024;

/// Projection matrices for the vocabulary ids one conversion needs.
///
/// Each row is a C-order 512 × 4 matrix: value `[i * 4 + o]` multiplies input
/// `i` into output `o`.
#[derive(Debug, Clone, Default)]
pub struct ProjectionRows {
    rows: HashMap<u32, Vec<f32>>,
}

impl ProjectionRows {
    /// Adds the 2048 values of one vocabulary id's matrix.
    ///
    /// # Panics
    ///
    /// Panics if `values` does not hold exactly 512 × 4 values.
    pub fn insert(&mut self, vocab_id: u32, values: Vec<f32>) {
        assert_eq!(
            values.len(),
            ROW_VALUES,
            "a projection row holds 512 x 4 values"
        );
        self.rows.insert(vocab_id, values);
    }

    /// The matrix for `vocab_id`, if present.
    #[must_use]
    pub fn get(&self, vocab_id: u32) -> Option<&[f32]> {
        self.rows.get(&vocab_id).map(Vec::as_slice)
    }
}

/// An opened `minicoil.triplet.model.npy`, read one vocabulary row at a time.
///
/// Only the header is read on open; [`ProjectionFile::read_rows`] reads the
/// 8 KiB rows asked for, so the whole file is never held in memory.
#[derive(Debug)]
pub struct ProjectionFile {
    path: PathBuf,
    file: File,
    data_offset: u64,
    vocab_rows: usize,
}

impl ProjectionFile {
    /// Opens the file and checks its header: `<f4`, C order, shape (n, 512, 4),
    /// and a data length matching that shape.
    ///
    /// # Errors
    ///
    /// Returns a named [`MinicoilError`] for an unreadable file, a missing magic,
    /// an unreadable header, another number type, Fortran order, another shape,
    /// or a data length that disagrees with the shape.
    pub fn open(path: &Path) -> Result<Self, MinicoilError> {
        let io = |source| MinicoilError::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut file = File::open(path).map_err(io)?;
        let mut preamble = [0_u8; 8];
        file.read_exact(&mut preamble).map_err(io)?;
        if &preamble[..6] != MAGIC || !matches!(preamble[6], 1..=3) {
            return Err(MinicoilError::ProjectionMagic {
                path: path.to_path_buf(),
            });
        }
        let header_len = if preamble[6] == 1 {
            let mut length = [0_u8; 2];
            file.read_exact(&mut length).map_err(io)?;
            usize::from(u16::from_le_bytes(length))
        } else {
            let mut length = [0_u8; 4];
            file.read_exact(&mut length).map_err(io)?;
            usize::try_from(u32::from_le_bytes(length)).unwrap_or(usize::MAX)
        };
        if header_len > MAX_HEADER_BYTES {
            return Err(header_error(path, "header is longer than 64 KiB"));
        }
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header).map_err(io)?;
        let header = String::from_utf8(header)
            .map_err(|_| header_error(path, "header is not UTF-8 text"))?;
        let vocab_rows = check_header(path, &header)?;

        let data_offset = file.stream_position().map_err(io)?;
        let actual = file
            .metadata()
            .map_err(io)?
            .len()
            .saturating_sub(data_offset);
        let expected = u64::try_from(vocab_rows)
            .ok()
            .and_then(|rows| rows.checked_mul(ROW_BYTES as u64))
            .unwrap_or(u64::MAX);
        if actual != expected {
            return Err(MinicoilError::ProjectionLength {
                path: path.to_path_buf(),
                expected,
                actual,
            });
        }
        Ok(Self {
            path: path.to_path_buf(),
            file,
            data_offset,
            vocab_rows,
        })
    }

    /// Number of vocabulary rows, the unknown id 0 included.
    #[must_use]
    pub const fn vocab_rows(&self) -> usize {
        self.vocab_rows
    }

    /// Reads the matrices for `vocab_ids`, each once.
    ///
    /// # Errors
    ///
    /// Returns [`MinicoilError::MissingRow`] for an id beyond the file and
    /// [`MinicoilError::Io`] when a read fails.
    pub fn read_rows(&mut self, vocab_ids: &[u32]) -> Result<ProjectionRows, MinicoilError> {
        let mut rows = ProjectionRows::default();
        let mut bytes = vec![0_u8; ROW_BYTES];
        for &vocab_id in vocab_ids {
            if rows.get(vocab_id).is_some() {
                continue;
            }
            let index = usize::try_from(vocab_id).unwrap_or(usize::MAX);
            if index >= self.vocab_rows {
                return Err(MinicoilError::MissingRow { vocab_id });
            }
            let offset = self.data_offset + u64::from(vocab_id) * ROW_BYTES as u64;
            let io = |source| MinicoilError::Io {
                path: self.path.clone(),
                source,
            };
            self.file.seek(SeekFrom::Start(offset)).map_err(io)?;
            self.file.read_exact(&mut bytes).map_err(io)?;
            let values = bytes
                .chunks_exact(4)
                .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
                .collect();
            rows.insert(vocab_id, values);
        }
        Ok(rows)
    }
}

fn header_error(path: &Path, detail: &str) -> MinicoilError {
    MinicoilError::ProjectionHeader {
        path: path.to_path_buf(),
        detail: detail.to_string(),
    }
}

/// Checks the header dictionary and returns the number of vocabulary rows.
fn check_header(path: &Path, header: &str) -> Result<usize, MinicoilError> {
    let descr =
        quoted_value(header, "descr").ok_or_else(|| header_error(path, "no 'descr' entry"))?;
    if descr != "<f4" {
        return Err(MinicoilError::ProjectionType {
            path: path.to_path_buf(),
            descr: descr.to_string(),
        });
    }
    let fortran = raw_value(header, "fortran_order")
        .ok_or_else(|| header_error(path, "no 'fortran_order' entry"))?;
    match fortran {
        "False" => {}
        "True" => {
            return Err(MinicoilError::ProjectionOrder {
                path: path.to_path_buf(),
            })
        }
        other => return Err(header_error(path, &format!("fortran_order is {other}"))),
    }
    let shape = shape_value(header).ok_or_else(|| header_error(path, "no readable 'shape'"))?;
    match shape.as_slice() {
        [rows, PROJECTION_INPUT, PROJECTION_OUTPUT] if *rows > 0 => Ok(*rows),
        _ => Err(MinicoilError::ProjectionShape {
            path: path.to_path_buf(),
            shape,
        }),
    }
}

fn raw_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let start = header.find(&format!("'{key}':"))? + key.len() + 3;
    let rest = header[start..].trim_start();
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Some(rest[..end].trim())
}

fn quoted_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    raw_value(header, key)?
        .strip_prefix('\'')?
        .strip_suffix('\'')
}

fn shape_value(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape':")? + "'shape':".len();
    let rest = header[start..].trim_start().strip_prefix('(')?;
    let inner = &rest[..rest.find(')')?];
    inner
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().ok())
        .collect()
}
