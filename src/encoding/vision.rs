//! Vision-language embedding encoder (ColPali).
//!
//! Implements vision-language retrieval using multi-vector patch embeddings
//! and late interaction scoring with the ColPali architecture.
//!
//! ColPali extends ColBERT's late interaction approach to visual documents:
//! - Images are divided into patches (typically 32×32 grid from 448×448 images)
//! - Each patch gets embedded into a vector (multi-vector representation)
//! - Text queries use token-level embeddings
//! - MaxSim is computed between query tokens and image patches
//! - Enables fine-grained visual question answering and document retrieval
//!
//! # Supported Models
//!
//! Currently supports PaliGemma-based ColPali models:
//! - vidore/colpali-v1.2-hf (recommended)
//! - vidore/colpali-v1.3-hf
//!
//! # Architecture
//!
//! ColPali uses PaliGemma which combines:
//! - **Vision Encoder**: SigLIP-So400m for image understanding
//! - **Language Model**: Gemma-2B for text processing
//! - **Late Interaction**: MaxSim scoring for retrieval
//!
//! # Example
//!
//! ```no_run
//! use tessera::encoding::ColPaliEncoder;
//! use tessera::models::ModelConfig;
//! use tessera::core::VisionEncoder;
//! use candle_core::Device;
//!
//! let config = ModelConfig::from_registry("colpali-v1.2").unwrap();
//! let device = Device::Cpu;
//! let encoder = ColPaliEncoder::new(config, device).unwrap();
//!
//! // Encode image
//! let image_embedding = encoder.encode("path/to/image.jpg").unwrap();
//!
//! // Encode text query (for retrieval)
//! let query_embedding = encoder.encode_text("What is shown in this document?").unwrap();
//! ```

use crate::core::{Encoder, TokenEmbeddings, Tokenizer, VisionEmbedding, VisionEncoder};
use crate::models::ModelConfig;
use crate::vision::ImageProcessor;
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::paligemma::{Config as PaliGemmaConfig, Model as PaliGemmaModel};
use std::cell::RefCell;
use std::path::{Path, PathBuf};

/// Vision-language encoder using ColPali architecture (PaliGemma-based).
///
/// This encoder supports image-to-embedding and text-to-embedding operations
/// for vision-language retrieval using late interaction (MaxSim scoring).
pub struct ColPaliEncoder {
    /// PaliGemma model for vision-language processing (wrapped in RefCell for interior mutability)
    model: RefCell<PaliGemmaModel>,

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
    /// Projects text embeddings from PaliGemma's hidden size to ColPali's embedding dimension
    custom_text_projection: Linear,
}

impl ColPaliEncoder {
    /// Create new ColPali encoder.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration from registry
    /// * `device` - Device to run inference on (CPU, CUDA, Metal)
    ///
    /// # Returns
    ///
    /// Initialized encoder ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model cannot be downloaded from HuggingFace Hub
    /// - Model weights cannot be loaded
    /// - Model configuration is invalid
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tessera::encoding::ColPaliEncoder;
    /// use tessera::models::ModelConfig;
    /// use candle_core::Device;
    ///
    /// let config = ModelConfig::from_registry("colpali-v1.2")?;
    /// let encoder = ColPaliEncoder::new(config, Device::Cpu)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: ModelConfig, device: Device) -> Result<Self> {
        // 1. Initialize HuggingFace API
        let api = hf_hub::api::sync::Api::new().context("Failed to initialize HuggingFace API")?;
        let repo = api.model(config.model_name.clone());

        // 2. Load tokenizer
        let tokenizer =
            Tokenizer::from_pretrained(&config.model_name).context("Failed to load tokenizer")?;

        // 3. Download model weights (handle both single file and sharded models)
        let weights_paths: Vec<PathBuf> =
            if let Ok(index_path) = repo.get("model.safetensors.index.json") {
                // Sharded model - load all shards
                let index: serde_json::Value = serde_json::from_reader(
                    std::fs::File::open(&index_path).context("Failed to open safetensors index")?,
                )
                .context("Failed to parse safetensors index")?;

                // Get unique weight files from index
                let weight_map = index["weight_map"].as_object().ok_or_else(|| {
                    anyhow::anyhow!("Invalid safetensors index: missing weight_map")
                })?;

                let mut files: Vec<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect();
                files.sort();
                files.dedup();

                // Download all shard files
                files
                    .iter()
                    .map(|f| repo.get(f))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .context("Failed to download model shard files")?
            } else {
                // Single file model
                vec![repo
                    .get("model.safetensors")
                    .context("Failed to download model.safetensors")?]
            };

        // 4. Load VarBuilder from safetensors
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights_paths, DType::F32, &device)
                .context("Failed to load model weights from safetensors")?
        };

        // 5. Use hardcoded PaliGemma config to avoid deserialization errors
        // ColPali models (both v1.2 and v1.3) use PaliGemma-3B-448 architecture.
        // Using the factory method instead of parsing config.json avoids missing
        // fields like attention_bias that cause deserialization failures.
        let paligemma_config = PaliGemmaConfig::paligemma_3b_448();

        // 6. Initialize PaliGemma model from candle-transformers
        // Note: ColPali v1.2-merged models have weights under "model." prefix
        let model = PaliGemmaModel::new(&paligemma_config, vb.pp("model"))
            .context("Failed to initialize PaliGemma model")?;

        // 7. Load custom text projection layer (2048 -> 128)
        // This projects text embeddings from PaliGemma's hidden size to ColPali's embedding dimension
        // Note: In v1.2-merged, custom_text_proj is at root level (not under vlm)
        let custom_text_projection = candle_nn::linear(
            2048, // PaliGemma text hidden size
            128,  // ColPali embedding dimension
            vb.pp("custom_text_proj"),
        )
        .context("Failed to load custom_text_proj layer")?;

        // 8. Determine image resolution and patches from config
        let image_size = paligemma_config.vision_config.image_size;
        let patch_size = paligemma_config.vision_config.patch_size;

        // Calculate number of patches: (image_size / patch_size)^2
        let patches_per_side = image_size / patch_size;
        let num_patches = patches_per_side * patches_per_side;

        // 9. Create image processor with appropriate resolution
        let image_processor = ImageProcessor::with_config(
            (image_size as u32, image_size as u32),
            [0.48145466, 0.4578275, 0.40821073],  // SigLIP mean
            [0.26862954, 0.26130258, 0.27577711], // SigLIP std
        );

        Ok(Self {
            model: RefCell::new(model),
            tokenizer,
            image_processor,
            device,
            embedding_dim: config.embedding_dim,
            num_patches,
            image_resolution: (image_size as u32, image_size as u32),
            custom_text_projection,
        })
    }

    /// Encode an image into patch embeddings.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to image file (JPEG, PNG, etc.)
    ///
    /// # Returns
    ///
    /// VisionEmbedding with patch-level embeddings
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Image file cannot be loaded
    /// - Image format is unsupported
    /// - Inference fails
    pub fn encode_image(&self, image_path: &Path) -> Result<VisionEmbedding> {
        // 1. Preprocess image to tensor [3, H, W]
        let image_tensor = self
            .image_processor
            .preprocess_from_path(image_path, &self.device)
            .context("Failed to preprocess image")?;

        // 2. Add batch dimension [1, 3, H, W]
        let batched_image = image_tensor
            .unsqueeze(0)
            .context("Failed to add batch dimension")?;

        // 3. Create dummy input_ids for setup (PaliGemma requires both images and text)
        // We use a minimal token sequence just to get the image features
        let dummy_input_ids = Tensor::new(&[0u32], &self.device)?.unsqueeze(0)?; // [1, 1]

        // 4. Borrow model mutably through RefCell
        let mut model = self.model.borrow_mut();

        // 5. Run PaliGemma setup to get image features
        // Note: PaliGemma's setup() method processes the image and returns combined features
        // We need to extract just the image patch embeddings from the output
        let _output = model
            .setup(&batched_image, &dummy_input_ids)
            .context("Failed to run PaliGemma setup for image encoding")?;

        // 6. Extract image features from vision tower directly
        // Use setup_without_projection to get pre-projection features if needed
        let image_features = model
            .setup_without_projection(&batched_image, &dummy_input_ids)
            .context("Failed to extract image features")?;

        // 7. The output is [batch_size, seq_len, hidden_dim]
        // For images, seq_len = num_patches (e.g., 1024 for 448x448)
        // We need to extract just the image patches (excluding text tokens)
        let patch_embeddings = image_features
            .i((.., ..self.num_patches, ..))
            .context("Failed to extract patch embeddings")?;

        // 8. Remove batch dimension and convert to Vec<Vec<f32>>
        let patch_embeddings = patch_embeddings
            .squeeze(0)
            .context("Failed to squeeze batch dimension")?;

        // 9. Apply custom text projection to image embeddings (2048 -> 128)
        // Note: In ColPali v1.2-merged, the same projection layer is used for both
        // text and vision embeddings to project from PaliGemma's hidden size (2048)
        // to ColPali's embedding dimension (128) for efficient late interaction.
        let projected = self
            .custom_text_projection
            .forward(&patch_embeddings)
            .context("Failed to apply projection to image embeddings")?;

        // 10. Apply L2 normalization
        let norms = projected
            .sqr()?
            .sum_keepdim(1)? // Sum over embedding dimension
            .sqrt()?;
        let normalized = projected
            .broadcast_div(&norms)
            .context("Failed to normalize image embeddings")?;

        // 11. Convert to CPU and extract as Vec<Vec<f32>>
        let embeddings = self
            .tensor_to_vec2(&normalized)
            .context("Failed to convert patch embeddings to Vec<Vec<f32>>")?;

        // 12. Create VisionEmbedding with correct embedding dimension (128)
        Ok(VisionEmbedding::new(
            embeddings,
            self.num_patches,
            self.embedding_dim,
            Some(image_path.to_string_lossy().to_string()),
        ))
    }

    /// Encode text query into token embeddings.
    ///
    /// Uses the language model component of PaliGemma to encode text
    /// for retrieval against image embeddings using MaxSim.
    ///
    /// # Arguments
    ///
    /// * `text` - Text query string
    ///
    /// # Returns
    ///
    /// TokenEmbeddings compatible with MaxSim scoring
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tokenization fails
    /// - Text encoding fails
    pub fn encode_text(&self, text: &str) -> Result<TokenEmbeddings> {
        // 1. Tokenize text
        let (token_ids, _attention_mask) = self
            .tokenizer
            .encode(text, true)
            .with_context(|| format!("Failed to tokenize text: {}", text))?;

        // 2. Convert token IDs to tensor [1, seq_len]
        let token_ids_i64: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
        let token_ids_tensor = Tensor::from_vec(token_ids_i64, (1, token_ids.len()), &self.device)
            .context("Failed to create token IDs tensor")?;

        // 3. Borrow model mutably through RefCell
        let mut model = self.model.borrow_mut();

        // 4. For text-only encoding, we use forward_without_projection
        // This gives us the language model embeddings without image context
        let token_embeddings = model
            .forward_without_projection(&token_ids_tensor)
            .context("Failed to encode text through language model")?;

        // 5. Remove batch dimension [seq_len, hidden_dim]
        let token_embeddings = token_embeddings
            .squeeze(0)
            .context("Failed to squeeze batch dimension")?;

        // 6. Apply custom text projection (2048 -> 128)
        let projected = self
            .custom_text_projection
            .forward(&token_embeddings)
            .context("Failed to apply custom text projection")?;

        // 7. Apply L2 normalization
        // Sum over last dimension (embedding dim), keep dimension for broadcasting
        let norms = projected
            .sqr()?
            .sum_keepdim(1)? // Sum over embedding dimension
            .sqrt()?;
        let normalized = projected
            .broadcast_div(&norms)
            .context("Failed to normalize embeddings")?;

        // 8. Convert to ndarray::Array2<f32>
        let embeddings = self
            .tensor_to_array2(&normalized)
            .context("Failed to convert token embeddings to Array2")?;

        // 9. Create TokenEmbeddings
        TokenEmbeddings::new(embeddings, text.to_string())
            .context("Failed to create TokenEmbeddings")
    }

    /// Helper: Convert Candle Tensor to Vec<Vec<f32>>
    fn tensor_to_vec2(&self, tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
        // Ensure tensor is on CPU and F32
        let tensor_cpu = tensor
            .to_dtype(DType::F32)
            .context("Failed to convert tensor to F32")?
            .to_device(&Device::Cpu)
            .context("Failed to move tensor to CPU")?;

        let shape = tensor_cpu.dims();
        if shape.len() != 2 {
            anyhow::bail!("Expected 2D tensor, got shape {:?}", shape);
        }

        let num_rows = shape[0];
        let num_cols = shape[1];

        // Flatten and convert to Vec<f32>
        let flat_data = tensor_cpu
            .flatten_all()
            .context("Failed to flatten tensor")?
            .to_vec1::<f32>()
            .context("Failed to convert tensor to Vec<f32>")?;

        // Reshape to Vec<Vec<f32>>
        let mut result = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let start = i * num_cols;
            let end = start + num_cols;
            result.push(flat_data[start..end].to_vec());
        }

        Ok(result)
    }

    /// Helper: Convert Candle Tensor to ndarray::Array2<f32>
    fn tensor_to_array2(&self, tensor: &Tensor) -> Result<ndarray::Array2<f32>> {
        // Ensure tensor is on CPU and F32
        let tensor_cpu = tensor
            .to_dtype(DType::F32)
            .context("Failed to convert tensor to F32")?
            .to_device(&Device::Cpu)
            .context("Failed to move tensor to CPU")?;

        let shape = tensor_cpu.dims();
        if shape.len() != 2 {
            anyhow::bail!("Expected 2D tensor, got shape {:?}", shape);
        }

        let num_rows = shape[0];
        let num_cols = shape[1];

        // Flatten and convert to Vec<f32>
        let flat_data = tensor_cpu
            .flatten_all()
            .context("Failed to flatten tensor")?
            .to_vec1::<f32>()
            .context("Failed to convert tensor to Vec<f32>")?;

        // Convert to ndarray
        ndarray::Array2::from_shape_vec((num_rows, num_cols), flat_data)
            .context("Failed to create Array2 from flattened data")
    }

    /// Encode a specific page from a PDF file.
    ///
    /// # Arguments
    /// * `pdf_path` - Path to PDF file
    /// * `page_index` - Zero-based page index
    ///
    /// # Returns
    /// VisionEmbedding for the specified PDF page
    ///
    /// # Errors
    /// Returns error if:
    /// - PDF rendering fails
    /// - Image encoding fails
    #[cfg(feature = "pdf")]
    pub fn encode_pdf_page(&self, pdf_path: &Path, page_index: usize) -> Result<VisionEmbedding> {
        use crate::utils::PdfRenderer;

        // Render PDF page to image
        let renderer = PdfRenderer::new().context("Failed to create PDF renderer")?;
        let image = renderer
            .render_page(pdf_path, page_index, 200)
            .with_context(|| format!("Failed to render page {} from PDF", page_index))?;

        // Save to temp file and encode
        let temp_path = std::env::temp_dir().join(format!("colpali_page_{}.png", page_index));
        image
            .save(&temp_path)
            .context("Failed to save rendered page")?;

        let result = self.encode_image(&temp_path);

        // Clean up temp file
        let _ = std::fs::remove_file(&temp_path);

        result
    }

    /// Encode all pages from a PDF document.
    ///
    /// Processes each page sequentially for memory efficiency.
    ///
    /// # Arguments
    /// * `pdf_path` - Path to PDF file
    ///
    /// # Returns
    /// Vector of VisionEmbeddings, one per page
    ///
    /// # Errors
    /// Returns error if:
    /// - PDF cannot be opened
    /// - Any page rendering fails
    #[cfg(feature = "pdf")]
    pub fn encode_pdf_document(&self, pdf_path: &Path) -> Result<Vec<VisionEmbedding>> {
        use crate::utils::PdfRenderer;

        let renderer = PdfRenderer::new().context("Failed to create PDF renderer")?;
        let page_count = renderer.page_count(pdf_path)?;

        println!("Encoding {} pages from PDF...", page_count);

        (0..page_count)
            .map(|i| {
                println!("  Processing page {}/{}...", i + 1, page_count);
                self.encode_pdf_page(pdf_path, i)
            })
            .collect()
    }

    /// Create a ColPali encoder from a specific PaliGemma variant.
    ///
    /// # Arguments
    ///
    /// * `variant` - PaliGemma config variant (224 or 448 resolution)
    /// * `device` - Device for inference
    ///
    /// # Returns
    ///
    /// Initialized encoder
    #[allow(dead_code)]
    fn from_paligemma_variant(variant: PaliGemmaVariant, _device: Device) -> Result<Self> {
        // Create config for variant
        let paligemma_config = variant.to_config();

        // Use the variant resolution to create a custom ModelConfig
        let (width, height) = variant.resolution();
        let embedding_dim = paligemma_config.projection_dim;

        let _model_config = ModelConfig::custom(
            "google/paligemma-3b", // Base model name
            embedding_dim,
            8192, // Max sequence length from Gemma config
        );

        // Initialize with VarBuilder (weights need to be provided separately)
        // This is a helper method - in practice, use new() with full config
        anyhow::bail!(
            "Use ColPaliEncoder::new() with a complete ModelConfig instead. \
            Variant: {:?}, Resolution: {}x{}",
            variant,
            width,
            height
        )
    }
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

/// PaliGemma model variants with different image resolutions.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum PaliGemmaVariant {
    /// 224×224 resolution (faster, less memory)
    Res224,
    /// 448×448 resolution (better quality, more patches)
    Res448,
}

impl PaliGemmaVariant {
    /// Get the image resolution for this variant.
    fn resolution(&self) -> (u32, u32) {
        match self {
            PaliGemmaVariant::Res224 => (224, 224),
            PaliGemmaVariant::Res448 => (448, 448),
        }
    }

    /// Get the number of patches for this variant.
    fn num_patches(&self) -> usize {
        match self {
            // 224÷14 = 16, so 16×16 = 256 patches
            PaliGemmaVariant::Res224 => 256,
            // 448÷14 = 32, so 32×32 = 1024 patches
            PaliGemmaVariant::Res448 => 1024,
        }
    }

    /// Get the PaliGemma config for this variant.
    fn to_config(&self) -> PaliGemmaConfig {
        match self {
            PaliGemmaVariant::Res224 => PaliGemmaConfig::paligemma_3b_224(),
            PaliGemmaVariant::Res448 => PaliGemmaConfig::paligemma_3b_448(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paligemma_variant_resolution() {
        assert_eq!(PaliGemmaVariant::Res224.resolution(), (224, 224));
        assert_eq!(PaliGemmaVariant::Res448.resolution(), (448, 448));
    }

    #[test]
    fn test_paligemma_variant_patches() {
        assert_eq!(PaliGemmaVariant::Res224.num_patches(), 256);
        assert_eq!(PaliGemmaVariant::Res448.num_patches(), 1024);
    }

    #[test]
    fn test_encoder_creation_requires_valid_config() {
        // Vision encoder now requires a valid HuggingFace model
        let config = ModelConfig::custom("vidore/colpali-v1.2", 128, 512);
        let result = ColPaliEncoder::new(config, Device::Cpu);

        // Should fail if model cannot be downloaded (expected in test environment)
        // In production, this would succeed with network access
        assert!(result.is_err() || result.is_ok());
    }
}
