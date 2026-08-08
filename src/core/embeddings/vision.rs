use super::Encoder;

/// Vision embedding representation for ColPali-style vision-language models.
///
/// Represents an image as a collection of patch embeddings, similar to how
/// `TokenEmbeddings` represents text as token embeddings. Each patch corresponds
/// to a spatial region of the input image (e.g., 448×448 → 32×32 patches = 1024 total).
///
/// Used for vision-language retrieval where queries are text and documents
/// are page images (PDFs, scans, screenshots). Compatible with late interaction
/// scoring (`MaxSim`) similar to `ColBERT`.
///
/// # Example Models
/// - wanghaofan/colpali-v1.2
/// - vidore/colpali-v1
#[derive(Debug, Clone)]
pub struct VisionEmbedding {
    /// Patch embeddings: shape [`num_patches`, `embedding_dim`]
    ///
    /// Typically 1024 patches for 448×448 images (32×32 grid).
    /// Each patch embedding is a vector of dimension `embedding_dim`.
    pub embeddings: Vec<Vec<f32>>,

    /// Number of patches in the image grid.
    ///
    /// For `ColPali` with 448×448 input, this is 32×32 = 1024 patches.
    pub num_patches: usize,

    /// Embedding dimension per patch.
    ///
    /// Typically 128 for `ColPali` (matches `ColBERT` dimension for compatibility).
    pub embedding_dim: usize,

    /// Optional: Source image path or identifier.
    ///
    /// Used for tracking the origin of the image embedding.
    pub source: Option<String>,
}

impl VisionEmbedding {
    /// Create a new vision embedding.
    ///
    /// # Arguments
    /// * `embeddings` - The patch embeddings (shape [`num_patches`, `embedding_dim`])
    /// * `num_patches` - Number of patches (typically 1024 for `ColPali`)
    /// * `embedding_dim` - Dimension per patch (typically 128 for `ColPali`)
    /// * `source` - Optional source image path/identifier
    ///
    /// # Returns
    /// A new `VisionEmbedding` instance
    #[must_use]
    pub const fn new(
        embeddings: Vec<Vec<f32>>,
        num_patches: usize,
        embedding_dim: usize,
        source: Option<String>,
    ) -> Self {
        Self {
            embeddings,
            num_patches,
            embedding_dim,
            source,
        }
    }

    /// Get the number of patches.
    ///
    /// # Returns
    /// Number of patches in this image embedding
    #[must_use]
    pub const fn num_patches(&self) -> usize {
        self.num_patches
    }

    /// Get the embedding dimension per patch.
    ///
    /// # Returns
    /// Dimensionality of each patch embedding vector
    #[must_use]
    pub const fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the source image path/identifier if available.
    ///
    /// # Returns
    /// Optional reference to the source identifier
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Get the shape of the embedding matrix as (`num_patches`, `embedding_dim`).
    ///
    /// # Returns
    /// Tuple of (number of patches, embedding dimension)
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.num_patches, self.embedding_dim)
    }
}

/// Vision encoder producing patch-level embeddings (ColPali-style).
///
/// Encodes images into multi-vector representations where each vector
/// corresponds to a spatial patch. This enables late interaction scoring
/// with text queries for vision-language retrieval, similar to `ColBERT`'s
/// token-level interactions.
///
/// # Characteristics
/// - Variable-length output (depends on image size)
/// - Patch-level granularity (typically 32×32 = 1024 patches per image)
/// - Designed for late interaction scoring with text queries
/// - Typically 128-384 dimensions per patch
/// - Compatible with `MaxSim` scoring used in `ColBERT`
///
/// # Example Models
/// - wanghaofan/colpali-v1.2
/// - vidore/colpali-v1
pub trait VisionEncoder: Encoder<Output = VisionEmbedding> {
    /// Get the number of patches per image.
    ///
    /// # Returns
    /// Number of patches the encoder produces per image.
    /// Typically 1024 for `ColPali` (32×32 grid of 14×14 pixel patches from 448×448 images).
    fn num_patches(&self) -> usize;

    /// Get the embedding dimension per patch.
    ///
    /// # Returns
    /// Dimensionality of each patch embedding vector.
    /// Typically 128 for `ColPali` (matches `ColBERT` dimension for compatibility).
    fn embedding_dim(&self) -> usize;

    /// Get the input image resolution.
    ///
    /// # Returns
    /// Tuple of (width, height) in pixels that the encoder expects.
    /// Typically (448, 448) for `ColPali`.
    fn image_resolution(&self) -> (u32, u32);
}
