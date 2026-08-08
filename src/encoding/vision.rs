//! Vision-language embedding encoder (`ColPali`).
//!
//! Implements vision-language retrieval using multi-vector patch embeddings
//! and late interaction scoring with the `ColPali` architecture.
//!
//! `ColPali` extends `ColBERT`'s late interaction approach to visual documents:
//! - Images are divided into patches (typically 32×32 grid from 448×448 images)
//! - Each patch gets embedded into a vector (multi-vector representation)
//! - Text queries use token-level embeddings
//! - `MaxSim` is computed between query tokens and image patches
//! - Enables fine-grained visual question answering and document retrieval
//!
//! # Registry status
//!
//! The current PaliGemma adapter is registered as the experimental
//! `colpali-v1.2` entry. Other `ColPali` metadata may remain catalog-only when
//! checkpoint namespaces do not match this adapter.
//!
//! # Architecture
//!
//! `ColPali` uses `PaliGemma` which combines:
//! - **Vision Encoder**: SigLIP-So400m for image understanding
//! - **Language Model**: Gemma-2B for text processing
//! - **Late Interaction**: `MaxSim` scoring for retrieval
//!
//! # Example
//!
//! ```no_run
//! use tessera::encoding::ColPaliEncoder;
//! use tessera::models::ModelConfig;
//! use tessera::ResourcePolicy;
//! use tessera::core::{Encoder, VisionEncoder};
//! use candle_core::Device;
//!
//! let config = ModelConfig::from_registry("colpali-v1.2").unwrap();
//! let device = Device::Cpu;
//! let policy = ResourcePolicy::default()
//!     .with_max_model_bytes(12 * 1024 * 1024 * 1024);
//! let encoder = ColPaliEncoder::new_with_resource_policy(config, device, policy).unwrap();
//!
//! // Encode image
//! let image_embedding = encoder.encode("path/to/image.jpg").unwrap();
//!
//! // Encode text query (for retrieval)
//! let query_embedding = encoder.encode_text("What is shown in this document?").unwrap();
//! ```

mod construction;
mod inference;
#[cfg(feature = "pdf")]
mod pdf;

use crate::core::{Encoder, Tokenizer, VisionEmbedding, VisionEncoder};
use crate::vision::ImageProcessor;
use anyhow::Result;
use candle_core::Device;
use candle_nn::Linear;
use candle_transformers::models::paligemma::Model as PaliGemmaModel;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Vision-language encoder using `ColPali` architecture (PaliGemma-based).
///
/// This encoder supports image-to-embedding and text-to-embedding operations
/// for vision-language retrieval using late interaction (`MaxSim` scoring).
pub struct ColPaliEncoder {
    /// `PaliGemma` model for vision-language processing (wrapped in `Arc<Mutex>` for thread-safe sharing)
    model: Arc<Mutex<PaliGemmaModel>>,

    /// Tokenizer for text encoding
    tokenizer: Tokenizer,

    /// Image preprocessor
    image_processor: ImageProcessor,

    /// Device for tensor operations
    device: Device,

    /// Embedding dimension per patch (typically 128)
    embedding_dim: usize,

    /// Number of patches per image (typically 1024 for 448×448)
    num_patches: usize,

    /// Image resolution (width, height)
    image_resolution: (u32, u32),

    /// Custom text projection layer (2048 -> 128)
    /// Projects text embeddings from `PaliGemma`'s hidden size to `ColPali`'s embedding dimension
    custom_text_projection: Linear,
}

impl Encoder for ColPaliEncoder {
    type Output = VisionEmbedding;

    fn encode(&self, input: &str) -> Result<Self::Output> {
        // For vision encoder, input is interpreted as image path
        let path = Path::new(input);
        self.encode_image(path)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Self::Output>> {
        // Encode each image path in batch
        inputs.iter().map(|&path| self.encode(path)).collect()
    }
}

impl VisionEncoder for ColPaliEncoder {
    fn num_patches(&self) -> usize {
        self.num_patches
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn image_resolution(&self) -> (u32, u32) {
        self.image_resolution
    }
}
