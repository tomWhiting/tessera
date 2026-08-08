//! Lightweight model-weight metadata helpers.

use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};

const SAFETENSORS_LENGTH_BYTES: u64 = 8;
// Tensor metadata for the catalog checkpoints is far below this ceiling. Keep
// malformed or hostile headers from turning lightweight detection into a large
// allocation before model construction.
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 16 * 1024 * 1024;

/// Reads tensor names from a safetensors JSON header without loading its data.
pub fn safetensors_tensor_names(path: &Path) -> Result<Vec<String>> {
    let mut file =
        File::open(path).with_context(|| format!("Opening safetensors file {}", path.display()))?;
    let file_len = file
        .seek(std::io::SeekFrom::End(0))
        .with_context(|| format!("Reading safetensors size for {}", path.display()))?;
    file.rewind()
        .with_context(|| format!("Rewinding safetensors file {}", path.display()))?;

    read_safetensors_tensor_names(&mut file, file_len)
        .with_context(|| format!("Reading safetensors header from {}", path.display()))
}

fn read_safetensors_tensor_names(reader: &mut impl Read, file_len: u64) -> Result<Vec<String>> {
    anyhow::ensure!(
        file_len >= SAFETENSORS_LENGTH_BYTES,
        "Safetensors file is shorter than its 8-byte header length"
    );

    let mut length_bytes = [0_u8; SAFETENSORS_LENGTH_BYTES as usize];
    reader
        .read_exact(&mut length_bytes)
        .context("Reading safetensors header length")?;
    let header_len = u64::from_le_bytes(length_bytes);

    anyhow::ensure!(header_len > 0, "Safetensors JSON header is empty");
    anyhow::ensure!(
        header_len <= MAX_SAFETENSORS_HEADER_BYTES,
        "Safetensors JSON header is unexpectedly large ({header_len} bytes)"
    );
    anyhow::ensure!(
        header_len <= file_len - SAFETENSORS_LENGTH_BYTES,
        "Safetensors JSON header length exceeds the file size"
    );

    let header_len =
        usize::try_from(header_len).context("Safetensors header does not fit usize")?;
    let mut header = vec![0_u8; header_len];
    reader
        .read_exact(&mut header)
        .context("Reading safetensors JSON header")?;

    let entries: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&header).context("Parsing safetensors JSON header")?;

    Ok(entries
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| name != "__metadata__")
        .collect())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{read_safetensors_tensor_names, MAX_SAFETENSORS_HEADER_BYTES};

    fn file_with_header(header: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut file = Vec::with_capacity(8 + header.len() + payload.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header);
        file.extend_from_slice(payload);
        file
    }

    #[test]
    fn reads_only_names_from_json_header() {
        let bytes = file_with_header(
            br#"{"__metadata__":{"format":"pt"},"encoder.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#,
            &[0xff; 16],
        );
        let mut reader = Cursor::new(&bytes);

        let names = read_safetensors_tensor_names(&mut reader, bytes.len() as u64).unwrap();

        assert_eq!(names, vec!["encoder.weight".to_string()]);
        assert_eq!(reader.position(), (bytes.len() - 16) as u64);
    }

    #[test]
    fn rejects_a_header_length_beyond_the_file() {
        let mut bytes = 128_u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let mut reader = Cursor::new(&bytes);

        let error = read_safetensors_tensor_names(&mut reader, bytes.len() as u64).unwrap_err();

        assert!(error.to_string().contains("exceeds the file size"));
    }

    #[test]
    fn rejects_an_oversized_header_before_allocating_it() {
        let declared = MAX_SAFETENSORS_HEADER_BYTES + 1;
        let bytes = declared.to_le_bytes();
        let mut reader = Cursor::new(bytes);

        let error = read_safetensors_tensor_names(&mut reader, declared + 8).unwrap_err();

        assert!(error.to_string().contains("unexpectedly large"));
        assert_eq!(reader.position(), 8);
    }
}
