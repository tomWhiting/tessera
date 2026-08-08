use super::ColPaliEncoder;
use crate::core::VisionEmbedding;
use anyhow::{Context, Result};
use std::path::Path;

impl ColPaliEncoder {
    /// Encode a specific page from a PDF file.
    ///
    /// # Arguments
    /// * `pdf_path` - Path to PDF file
    /// * `page_index` - Zero-based page index
    ///
    /// # Returns
    /// `VisionEmbedding` for the specified PDF page
    ///
    /// # Errors
    /// Returns error if:
    /// - PDF rendering fails
    /// - Image encoding fails
    pub fn encode_pdf_page(&self, pdf_path: &Path, page_index: usize) -> Result<VisionEmbedding> {
        use crate::utils::PdfRenderer;

        // Render and encode directly from memory. This avoids a lossy PNG write,
        // a second image decode, temp-file collisions, and cleanup races.
        let renderer = PdfRenderer::new().context("Failed to create PDF renderer")?;
        let image = renderer
            .render_page(pdf_path, page_index, 200)
            .with_context(|| format!("Failed to render page {page_index} from PDF"))?;
        let source = format!("{}#page={page_index}", pdf_path.display());
        self.encode_dynamic_image(&image, Some(source))
    }

    /// Encode all pages from a PDF document.
    ///
    /// Processes each page sequentially for memory efficiency.
    ///
    /// # Arguments
    /// * `pdf_path` - Path to PDF file
    ///
    /// # Returns
    /// Vector of `VisionEmbeddings`, one per page
    ///
    /// # Errors
    /// Returns error if:
    /// - PDF cannot be opened
    /// - Any page rendering fails
    pub fn encode_pdf_document(&self, pdf_path: &Path) -> Result<Vec<VisionEmbedding>> {
        use crate::utils::PdfRenderer;

        let renderer = PdfRenderer::new().context("Failed to create PDF renderer")?;
        let document = renderer.open(pdf_path)?;
        document.validate_collection_page_count()?;
        let page_count = document.page_count();

        (0..page_count)
            .map(|i| {
                let image = document
                    .render_page(i, 200)
                    .with_context(|| format!("Failed to render page {i} from PDF"))?;
                let source = format!("{}#page={i}", pdf_path.display());
                self.encode_dynamic_image(&image, Some(source))
            })
            .collect()
    }
}
