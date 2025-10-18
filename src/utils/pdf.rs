//! PDF rendering utilities for vision-language models.
//!
//! Provides page-by-page rendering of PDF documents to images
//! for processing with vision models like `ColPali`.
//!
//! Uses pdf2image (Poppler-based) for reliable cross-platform PDF rendering.

use anyhow::{Context, Result};
use image::DynamicImage;
#[cfg(feature = "pdf")]
use pdf2image::{RenderOptionsBuilder, PDF};
use std::path::Path;

#[cfg(feature = "pdf")]
pub struct PdfRenderer;

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
        let pdf = PDF::from_file(pdf_path).context("Failed to load PDF document")?;

        let render_options = RenderOptionsBuilder::default()
            .resolution(pdf2image::DPI::Uniform(dpi))
            .build()
            .context("Failed to build render options")?;

        // pdf2image uses 1-based indexing
        let page_num = (page_index + 1) as u32;
        let pages = pdf
            .render(pdf2image::Pages::Range(page_num..=page_num), render_options)
            .context("Failed to render PDF page")?;

        let img = pages
            .into_iter()
            .next()
            .with_context(|| format!("Page {page_index} not found in PDF"))?;

        Ok(img)
    }

    /// Get the number of pages in a PDF.
    pub fn page_count(&self, pdf_path: &Path) -> Result<usize> {
        let pdf = PDF::from_file(pdf_path).context("Failed to load PDF document")?;
        Ok(pdf.page_count() as usize)
    }

    /// Render all pages from a PDF file.
    ///
    /// Returns a vector of rendered pages.
    /// For large PDFs, consider using page-by-page rendering instead.
    pub fn render_all_pages(&self, pdf_path: &Path, dpi: u32) -> Result<Vec<DynamicImage>> {
        let count = self.page_count(pdf_path)?;
        (0..count)
            .map(|i| self.render_page(pdf_path, i, dpi))
            .collect()
    }
}

#[cfg(not(feature = "pdf"))]
pub struct PdfRenderer;

#[cfg(not(feature = "pdf"))]
impl PdfRenderer {
    pub fn new() -> Result<Self> {
        anyhow::bail!("PDF support not enabled. Compile with --features pdf")
    }
}
