use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    validate_collection_page_count, validate_render_dpi, PdfRenderer, MAX_COLLECTED_PAGES,
    MAX_PDF_BYTES, MAX_RENDER_DPI,
};

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn oversized_pdf() -> Self {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tessera-pdf-limit-{}-{sequence}.pdf",
            std::process::id()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("temporary PDF path should be unique");
        file.seek(SeekFrom::Start(MAX_PDF_BYTES))
            .expect("temporary file should be seekable");
        file.write_all(&[0])
            .expect("temporary sparse file should be writable");
        Self { path }
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[test]
fn dpi_validation_accepts_boundary_and_rejects_unsafe_values() {
    assert!(validate_render_dpi(1).is_ok());
    assert!(validate_render_dpi(MAX_RENDER_DPI).is_ok());
    assert_eq!(
        validate_render_dpi(0).unwrap_err().to_string(),
        "PDF render DPI must be greater than zero"
    );
    assert_eq!(
        validate_render_dpi(MAX_RENDER_DPI + 1)
            .unwrap_err()
            .to_string(),
        format!(
            "PDF render DPI {} exceeds limit {MAX_RENDER_DPI}",
            MAX_RENDER_DPI + 1
        )
    );
}

#[test]
fn collecting_page_validation_rejects_before_rendering() {
    assert!(validate_collection_page_count(MAX_COLLECTED_PAGES).is_ok());
    assert_eq!(
        validate_collection_page_count(MAX_COLLECTED_PAGES + 1)
            .unwrap_err()
            .to_string(),
        format!(
            "PDF page count {} exceeds collecting limit {MAX_COLLECTED_PAGES}; render pages individually",
            MAX_COLLECTED_PAGES + 1
        )
    );
}

#[test]
fn oversized_file_is_rejected_before_pdf_parsing() {
    let file = TempFile::oversized_pdf();
    let renderer = PdfRenderer::new().expect("default PDF policy should be valid");

    let error = renderer
        .open(&file.path)
        .err()
        .expect("oversized input must fail before Poppler is invoked");

    assert_eq!(
        error.to_string(),
        format!(
            "PDF file size {} bytes exceeds limit {MAX_PDF_BYTES} bytes",
            MAX_PDF_BYTES + 1
        )
    );
}
