//! PDF rendering utilities for vision-language models.
//!
//! Provides page-by-page rendering of PDF documents to images
//! for processing with vision models like `ColPali`.
//!
//! Uses the opt-in `pdf2image` wrapper around a system Poppler installation.
//! Supported formats and platform behavior therefore depend on that external
//! installation.

use anyhow::{Context, Result};
use image::DynamicImage;
#[cfg(feature = "pdf")]
use pdf2image::{RenderOptionsBuilder, PDF};
#[cfg(feature = "pdf")]
use std::fs::File;
#[cfg(feature = "pdf")]
use std::io::Read;
use std::path::Path;

#[cfg(all(test, feature = "pdf"))]
#[path = "pdf/tests.rs"]
mod tests;

/// Maximum PDF bytes retained in memory by the Poppler wrapper.
#[cfg(feature = "pdf")]
const MAX_PDF_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum pages returned by a collecting API.
#[cfg(feature = "pdf")]
const MAX_COLLECTED_PAGES: usize = 16;
/// Maximum accepted rasterization resolution.
#[cfg(feature = "pdf")]
const MAX_RENDER_DPI: u32 = 300;
/// Maximum width or height produced by Poppler before model preprocessing.
#[cfg(feature = "pdf")]
const MAX_RENDER_EDGE: u32 = 2048;

#[cfg(feature = "pdf")]
/// PDF rendering utility for ColPali document processing.
///
/// Uses Poppler via pdf2image to render PDF pages as images.
pub struct PdfRenderer;

#[cfg(feature = "pdf")]
/// An opened PDF whose bytes and page metadata are reused across page renders.
pub struct PdfDocument {
    pdf: PDF,
}

#[cfg(feature = "pdf")]
impl PdfDocument {
    /// Return the number of pages in the document.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pdf.page_count() as usize
    }

    /// Reject a collecting operation before any pages are rendered.
    pub(crate) fn validate_collection_page_count(&self) -> Result<()> {
        validate_collection_page_count(self.page_count())
    }

    /// Render one zero-based page from the already-open document.
    pub fn render_page(&self, page_index: usize, dpi: u32) -> Result<DynamicImage> {
        validate_render_dpi(dpi)?;
        anyhow::ensure!(
            page_index < self.page_count(),
            "PDF page index {page_index} is outside the document's {} pages",
            self.page_count()
        );

        let render_options = RenderOptionsBuilder::default()
            .resolution(pdf2image::DPI::Uniform(dpi))
            .scale(pdf2image::Scale::Uniform(MAX_RENDER_EDGE))
            .build()
            .context("Failed to build render options")?;

        let page_num = u32::try_from(page_index)
            .context("PDF page index exceeds supported range")?
            .checked_add(1)
            .context("PDF page index exceeds supported range")?;
        let pages = self
            .pdf
            .render(pdf2image::Pages::Single(page_num), render_options)
            .context("Failed to render PDF page")?;

        pages
            .into_iter()
            .next()
            .with_context(|| format!("Page {page_index} not found in PDF"))
    }
}

#[cfg(feature = "pdf")]
impl PdfRenderer {
    /// Create a new PDF renderer.
    ///
    /// Note: Requires Poppler to be installed on the system.
    /// - macOS: `brew install poppler`
    /// - Ubuntu: `apt-get install poppler-utils`
    pub const fn new() -> Result<Self> {
        Ok(Self)
    }

    /// Open a PDF once so multiple pages can be rendered without rereading it.
    pub fn open(&self, pdf_path: &Path) -> Result<PdfDocument> {
        let pdf_bytes = read_pdf_bounded(pdf_path)?;
        let pdf = PDF::from_bytes(pdf_bytes).context("Failed to load PDF document")?;
        Ok(PdfDocument { pdf })
    }

    /// Render a specific page from a PDF file.
    ///
    /// # Arguments
    /// * `pdf_path` - Path to PDF file
    /// * `page_index` - Zero-based page index
    /// * `dpi` - Render resolution (default: 200 DPI works well for `ColPali`)
    ///
    /// # Returns
    /// `DynamicImage` containing the rendered page
    pub fn render_page(
        &self,
        pdf_path: &Path,
        page_index: usize,
        dpi: u32,
    ) -> Result<DynamicImage> {
        self.open(pdf_path)?.render_page(page_index, dpi)
    }

    /// Get the number of pages in a PDF.
    pub fn page_count(&self, pdf_path: &Path) -> Result<usize> {
        Ok(self.open(pdf_path)?.page_count())
    }

    /// Render all pages from a PDF file.
    ///
    /// Returns a vector of rendered pages.
    /// For large PDFs, consider using page-by-page rendering instead.
    pub fn render_all_pages(&self, pdf_path: &Path, dpi: u32) -> Result<Vec<DynamicImage>> {
        let document = self.open(pdf_path)?;
        document.validate_collection_page_count()?;
        let count = document.page_count();
        (0..count).map(|i| document.render_page(i, dpi)).collect()
    }
}

#[cfg(feature = "pdf")]
fn read_pdf_bounded(pdf_path: &Path) -> Result<Vec<u8>> {
    let file = File::open(pdf_path)
        .with_context(|| format!("Failed to open PDF document {}", pdf_path.display()))?;
    let measured_bytes = file
        .metadata()
        .with_context(|| format!("Failed to inspect PDF document {}", pdf_path.display()))?
        .len();
    anyhow::ensure!(
        measured_bytes <= MAX_PDF_BYTES,
        "PDF file size {measured_bytes} bytes exceeds limit {MAX_PDF_BYTES} bytes"
    );

    let initial_capacity = usize::try_from(measured_bytes)
        .context("PDF file size does not fit this platform's address space")?;
    let mut pdf_bytes = Vec::with_capacity(initial_capacity);
    file.take(MAX_PDF_BYTES + 1)
        .read_to_end(&mut pdf_bytes)
        .with_context(|| format!("Failed to read PDF document {}", pdf_path.display()))?;
    anyhow::ensure!(
        pdf_bytes.len() as u64 <= MAX_PDF_BYTES,
        "PDF file grew beyond limit {MAX_PDF_BYTES} bytes while being read"
    );
    Ok(pdf_bytes)
}

#[cfg(feature = "pdf")]
fn validate_render_dpi(dpi: u32) -> Result<()> {
    anyhow::ensure!(dpi > 0, "PDF render DPI must be greater than zero");
    anyhow::ensure!(
        dpi <= MAX_RENDER_DPI,
        "PDF render DPI {dpi} exceeds limit {MAX_RENDER_DPI}"
    );
    Ok(())
}

#[cfg(feature = "pdf")]
fn validate_collection_page_count(page_count: usize) -> Result<()> {
    anyhow::ensure!(
        page_count <= MAX_COLLECTED_PAGES,
        "PDF page count {page_count} exceeds collecting limit {MAX_COLLECTED_PAGES}; render pages individually"
    );
    Ok(())
}

#[cfg(not(feature = "pdf"))]
/// PDF rendering utility for ColPali document processing.
///
/// Uses Poppler via pdf2image to render PDF pages as images.
/// This is a stub implementation when PDF feature is not enabled.
pub struct PdfRenderer;

#[cfg(not(feature = "pdf"))]
impl PdfRenderer {
    pub fn new() -> Result<Self> {
        anyhow::bail!("PDF support not enabled. Compile with --features pdf")
    }
}
