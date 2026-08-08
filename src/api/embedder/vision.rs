use crate::api::TesseraVisionBuilder;
use crate::core::{TokenEmbeddings, VisionEmbedding};
use crate::encoding::vision::ColPaliEncoder;
use crate::error::{Result, TesseraError};
use crate::utils::similarity::max_sim;
use std::path::Path;

/// Vision-language embedder for `ColPali` document retrieval.
///
/// Encodes document page images as multi-vector patch embeddings and enables
/// text queries to search visually through documents without OCR.
///
/// Thread-safe and can be shared across threads (except for encoding operations
/// which require exclusive access due to interior mutability).
pub struct TesseraVision {
    /// Backend encoder (`ColPali` encoder)
    encoder: ColPaliEncoder,
    /// Model identifier from registry
    model_id: String,
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
    /// // The current 3B F32 adapter exceeds the conservative 2 GiB default.
    /// // Raise this only after checking the target machine's memory budget.
    /// let policy = ResourcePolicy::default()
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
    pub(crate) const fn from_encoder(encoder: ColPaliEncoder, model_id: String) -> Self {
        Self { encoder, model_id }
    }

    /// Encode a document image into patch embeddings.
    ///
    /// Returns multi-vector representation where each vector corresponds to
    /// an image patch (14×14 pixels). Typically produces 1024 patch embeddings
    /// for 448×448 images.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to document image (PNG, JPEG, etc.)
    ///
    /// # Returns
    ///
    /// `VisionEmbedding` containing patch embeddings (shape: [1024, 128]).
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
    /// println!("Patches: {}, Dim: {}", doc_emb.num_patches(), doc_emb.embedding_dim());
    /// ```
    pub fn encode_document(&self, image_path: &str) -> Result<VisionEmbedding> {
        let path = Path::new(image_path);
        self.encoder
            .encode_image(path)
            .map_err(|e| TesseraError::EncodingError {
                context: format!("Failed to encode document image: '{image_path}'"),
                source: e,
            })
    }

    /// Encode a text query into token embeddings.
    ///
    /// Returns multi-vector representation where each vector corresponds to
    /// a query token. Compatible with late interaction (`MaxSim`) scoring
    /// against document patch embeddings.
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
    /// println!("Query tokens: {}", query_emb.num_tokens);
    /// ```
    pub fn encode_query(&self, text: &str) -> Result<TokenEmbeddings> {
        self.encoder
            .encode_text(text)
            .map_err(|e| TesseraError::EncodingError {
                context: format!("Failed to encode query text ({} UTF-8 bytes)", text.len()),
                source: e,
            })
    }

    /// Compute late interaction score between query and document.
    ///
    /// Uses `MaxSim` scoring: for each query token, find maximum similarity
    /// across all document patches, then sum across query tokens.
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
        // Convert VisionEmbedding to format compatible with max_sim
        // max_sim expects (&TokenEmbeddings, &TokenEmbeddings) but we can adapt it

        // Create a TokenEmbeddings-like structure from VisionEmbedding
        // We need to convert Vec<Vec<f32>> to Array2<f32>
        let doc_array = ndarray::Array2::from_shape_vec(
            (document.num_patches, document.embedding_dim),
            document.embeddings.iter().flatten().copied().collect(),
        )
        .map_err(|e| TesseraError::EncodingError {
            context: "Failed to convert document embeddings to array".to_string(),
            source: e.into(),
        })?;

        let doc_embeddings = TokenEmbeddings {
            embeddings: doc_array,
            num_tokens: document.num_patches,
            embedding_dim: document.embedding_dim,
            text: document.source.clone().unwrap_or_default(),
        };

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

    /// Get the number of patches per image.
    pub fn num_patches(&self) -> usize {
        use crate::core::VisionEncoder;
        self.encoder.num_patches()
    }

    /// Get the model identifier.
    pub fn model(&self) -> &str {
        &self.model_id
    }
}
