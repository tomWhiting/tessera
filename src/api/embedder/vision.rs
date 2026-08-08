use crate::api::TesseraVisionBuilder;
use crate::core::{TokenEmbeddings, VisionEmbedding};
use crate::encoding::vision::ColPaliEncoder;
use crate::error::{Result, TesseraError};
use crate::runtime::{f32_output_bytes, JobTracker, ModelDType, ResourcePolicy};
use crate::utils::similarity::max_sim;
use std::path::Path;

#[cfg(test)]
mod tests;

/// Vision-language embedder for `ColPali` document retrieval.
///
/// Encodes page images as full ColPali visual-sequence embeddings and enables
/// text queries to search visually through documents without OCR.
///
/// Thread-safe and can be shared across threads (except for encoding operations
/// which require exclusive access due to interior mutability).
pub struct TesseraVision {
    /// Backend encoder (`ColPali` encoder)
    encoder: ColPaliEncoder,
    /// Model identifier from registry
    model_id: String,
    resource_policy: ResourcePolicy,
}

impl TesseraVision {
    /// Create a new vision-language embedder with default configuration.
    ///
    /// This convenience constructor automatically:
    /// - Looks up the model in the registry
    /// - Selects the best available device (Metal > CUDA > CPU)
    /// - Applies the conservative default resource policy before downloading
    /// - Initializes the `PaliGemma` vision-language model
    ///
    /// The current 3B F32 adapter is larger than the default model-memory
    /// budget, so use the builder with an explicit policy after checking the
    /// target machine's capacity.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry (currently "colpali-v1.2")
    ///
    /// # Returns
    ///
    /// Initialized embedder when the selected model fits the default policy.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - Model is not a vision-language model type
    /// - Model exceeds the default resource policy
    /// - Model cannot be downloaded or loaded
    /// - Device initialization fails
    ///
    /// # Explicit resource example
    ///
    /// ```ignore
    /// use tessera::{ResourcePolicy, TesseraVision};
    ///
    /// // ColPali combines 1,024 visual positions with prompt tokens. These
    /// // illustrative F32 ceilings are an explicit high-memory opt-in.
    /// let policy = ResourcePolicy::default()
    ///     .with_max_sequence_tokens(2_048)
    ///     .with_max_batch_items(1)
    ///     .with_max_batch_tokens(2_048)
    ///     .with_max_attention_cells(4_194_304)
    ///     .with_max_activation_bytes(1024 * 1024 * 1024)
    ///     .with_max_model_bytes(12 * 1024 * 1024 * 1024);
    /// let embedder = TesseraVision::builder()
    ///     .model("colpali-v1.2")
    ///     .resource_policy(policy)
    ///     .build()?;
    /// let doc_emb = embedder.encode_document("invoice.jpg")?;
    /// let query_emb = embedder.encode_query("What is the total amount?")?;
    /// let score = embedder.search(&query_emb, &doc_emb)?;
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        TesseraVisionBuilder::new().model(model_id).build()
    }

    /// Create a builder for advanced configuration.
    #[must_use]
    pub const fn builder() -> TesseraVisionBuilder {
        TesseraVisionBuilder::new()
    }

    /// Internal constructor used by builder.
    pub(crate) const fn from_encoder(
        encoder: ColPaliEncoder,
        model_id: String,
        resource_policy: ResourcePolicy,
    ) -> Self {
        Self {
            encoder,
            model_id,
            resource_policy,
        }
    }

    /// Encode a document image into patch embeddings.
    ///
    /// Returns the upstream ColPali multi-vector representation: one vector for
    /// each image patch followed by the conditioned document-prompt vectors.
    /// A 448×448 image contributes 1024 physical patch positions; the exact
    /// total vector count also includes the tokenizer-defined prompt suffix.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to document image (PNG, JPEG, etc.)
    ///
    /// # Returns
    ///
    /// `VisionEmbedding` containing the complete document sequence.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Image cannot be loaded
    /// - Image preprocessing fails
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let doc_emb = embedder.encode_document("invoice.jpg")?;
    /// println!("Patches: {}, vectors: {}", doc_emb.num_patches(), doc_emb.num_vectors());
    /// ```
    pub fn encode_document(&self, image_path: &str) -> Result<VisionEmbedding> {
        let path = Path::new(image_path);
        let embedding =
            self.encoder
                .encode_image(path)
                .map_err(|e| TesseraError::EncodingError {
                    context: format!("Failed to encode document image: '{image_path}'"),
                    source: e,
                })?;
        self.resource_policy
            .validate_output_bytes(f32_output_bytes(
                embedding
                    .num_vectors()
                    .saturating_mul(embedding.embedding_dim()),
            ))
            .map_err(|error| resource_error("Vision output exceeds collection limit", error))?;
        Ok(embedding)
    }

    /// Encode one zero-based PDF page as a ColPali document sequence.
    ///
    /// This is available with the opt-in `pdf` feature and requires Poppler on
    /// the host. PDF bytes, render resolution, raster dimensions, and output
    /// embeddings remain subject to the configured safety limits.
    ///
    /// # Errors
    ///
    /// Returns an error if the PDF cannot be read, the page is out of range,
    /// rendering or inference fails, or the output exceeds the resource policy.
    #[cfg(feature = "pdf")]
    pub fn encode_pdf_page(&self, pdf_path: &str, page_index: usize) -> Result<VisionEmbedding> {
        let embedding = self
            .encoder
            .encode_pdf_page(Path::new(pdf_path), page_index)
            .map_err(|error| TesseraError::EncodingError {
                context: format!("Failed to encode PDF page {page_index} from '{pdf_path}'"),
                source: error,
            })?;
        self.resource_policy
            .validate_output_bytes(vision_output_bytes(&embedding))
            .map_err(|error| resource_error("PDF page output exceeds limit", error))?;
        Ok(embedding)
    }

    /// Encode every page in a bounded PDF as ColPali document sequences.
    ///
    /// Pages are rendered and inferred sequentially. The PDF renderer caps the
    /// number of collected pages, while the resource policy caps cumulative
    /// retained embedding bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the PDF violates its input/page/render limits, any
    /// page fails, or the collected embeddings exceed the resource policy.
    #[cfg(feature = "pdf")]
    pub fn encode_pdf_document(&self, pdf_path: &str) -> Result<Vec<VisionEmbedding>> {
        let embeddings = self
            .encoder
            .encode_pdf_document(Path::new(pdf_path))
            .map_err(|error| TesseraError::EncodingError {
                context: format!("Failed to encode PDF document '{pdf_path}'"),
                source: error,
            })?;
        let mut tracker = JobTracker::new(self.resource_policy);
        for embedding in &embeddings {
            tracker
                .retain_output(vision_output_bytes(embedding))
                .map_err(|error| {
                    resource_error("PDF document output exceeds collection limit", error)
                })?;
        }
        Ok(embeddings)
    }

    /// Encode a text query into token embeddings.
    ///
    /// Returns multi-vector representation where each vector corresponds to
    /// a query token. Compatible with late interaction (`MaxSim`) scoring
    /// against document sequence embeddings.
    ///
    /// # Arguments
    ///
    /// * `text` - Query text
    ///
    /// # Returns
    ///
    /// `TokenEmbeddings` containing query token embeddings.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tokenization fails
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query_emb = embedder.encode_query("What is the total amount?")?;
    /// println!("Query tokens: {}", query_emb.num_tokens());
    /// ```
    pub fn encode_query(&self, text: &str) -> Result<TokenEmbeddings> {
        let mut tracker = JobTracker::new(self.resource_policy);
        tracker
            .admit_input(text.len())
            .map_err(|error| resource_error("Vision query exceeds job limits", error))?;
        let embedding =
            self.encoder
                .encode_text(text)
                .map_err(|e| TesseraError::EncodingError {
                    context: format!("Failed to encode query text ({} UTF-8 bytes)", text.len()),
                    source: e,
                })?;
        tracker
            .retain_output(f32_output_bytes(
                embedding
                    .num_tokens()
                    .saturating_mul(embedding.embedding_dim()),
            ))
            .map_err(|error| resource_error("Vision query output exceeds limit", error))?;
        Ok(embedding)
    }

    /// Compute late interaction score between query and document.
    ///
    /// Uses `MaxSim` scoring: for each query token, find maximum similarity
    /// across every retained document vector, then sum across query tokens.
    ///
    /// # Arguments
    ///
    /// * `query` - Query token embeddings
    /// * `document` - Document patch embeddings
    ///
    /// # Returns
    ///
    /// Similarity score (higher = more similar).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query_emb = embedder.encode_query("total amount")?;
    /// let doc_emb = embedder.encode_document("invoice.jpg")?;
    /// let score = embedder.search(&query_emb, &doc_emb)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the embeddings cannot be processed or if dimensions are mismatched.
    pub fn search(&self, query: &TokenEmbeddings, document: &VisionEmbedding) -> Result<f32> {
        let doc_embeddings =
            document_token_embeddings(document).map_err(|e| TesseraError::EncodingError {
                context: "Failed to validate document embeddings for MaxSim".to_string(),
                source: e,
            })?;

        max_sim(query, &doc_embeddings).map_err(|e| TesseraError::EncodingError {
            context: "Failed to compute MaxSim score".to_string(),
            source: e,
        })
    }

    /// Convenience method: search with text query and image path.
    ///
    /// Encodes both query and document, then computes similarity.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails or the image cannot be read.
    pub fn search_document(&self, query_text: &str, image_path: &str) -> Result<f32> {
        let query_emb = self.encode_query(query_text)?;
        let doc_emb = self.encode_document(image_path)?;
        self.search(&query_emb, &doc_emb)
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        use crate::core::VisionEncoder;
        self.encoder.embedding_dim()
    }

    /// Get the number of physical image patches (excluding prompt vectors).
    pub fn num_patches(&self) -> usize {
        use crate::core::VisionEncoder;
        self.encoder.num_patches()
    }

    /// Get the model identifier.
    pub fn model(&self) -> &str {
        &self.model_id
    }

    /// Parameter dtype selected for this model instance.
    #[must_use]
    pub const fn model_dtype(&self) -> ModelDType {
        self.encoder.model_dtype()
    }
}

fn resource_error(context: &str, error: crate::runtime::ResourcePolicyError) -> TesseraError {
    TesseraError::EncodingError {
        context: context.to_string(),
        source: anyhow::Error::new(error),
    }
}

#[cfg(feature = "pdf")]
fn vision_output_bytes(embedding: &VisionEmbedding) -> usize {
    f32_output_bytes(
        embedding
            .num_vectors()
            .saturating_mul(embedding.embedding_dim()),
    )
}

fn document_token_embeddings(document: &VisionEmbedding) -> anyhow::Result<TokenEmbeddings> {
    let (num_vectors, embedding_dim) = document.shape();
    let array = ndarray::Array2::from_shape_vec(
        (num_vectors, embedding_dim),
        document.vectors().iter().flatten().copied().collect(),
    )?;
    TokenEmbeddings::new(array, document.source().unwrap_or_default().to_string())
}
