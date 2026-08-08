use super::ColPaliEncoder;
use crate::core::{TokenEmbeddings, VisionEmbedding};
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
#[cfg(feature = "pdf")]
use image::DynamicImage;
use std::path::Path;

impl ColPaliEncoder {
    /// Encode an image into patch embeddings.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to image file (JPEG, PNG, etc.)
    ///
    /// # Returns
    ///
    /// `VisionEmbedding` with patch-level embeddings
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Image file cannot be loaded
    /// - Image format is unsupported
    /// - Inference fails
    pub fn encode_image(&self, image_path: &Path) -> Result<VisionEmbedding> {
        // Preprocess image to tensor [3, H, W]. The tensor-based helper also
        // serves rendered PDF pages without a temporary PNG round trip.
        let image_tensor = self
            .image_processor
            .preprocess_from_path(image_path, &self.device)
            .context("Failed to preprocess image")?;

        self.encode_image_tensor(
            image_tensor,
            Some(image_path.to_string_lossy().into_owned()),
        )
    }

    #[cfg(feature = "pdf")]
    pub(super) fn encode_dynamic_image(
        &self,
        image: &DynamicImage,
        source: Option<String>,
    ) -> Result<VisionEmbedding> {
        let image_tensor = self
            .image_processor
            .preprocess_image(image, &self.device)
            .context("Failed to preprocess image")?;

        self.encode_image_tensor(image_tensor, source)
    }

    fn encode_image_tensor(
        &self,
        image_tensor: Tensor,
        source: Option<String>,
    ) -> Result<VisionEmbedding> {
        // 2. Add batch dimension [1, 3, H, W]
        let batched_image = image_tensor
            .unsqueeze(0)
            .context("Failed to add batch dimension")?;

        // 3. Create dummy input_ids for setup (PaliGemma requires both images and text)
        // We use a minimal token sequence just to get the image features
        let dummy_input_ids = Tensor::new(&[0u32], &self.device)?.unsqueeze(0)?; // [1, 1]

        // 4. Acquire global inference admission before the model-specific lock.
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let mut model = self
            .model
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire model lock: {}", e))?;

        // 5. Extract the image features in one pass. Calling `setup` first would
        // run the complete vision tower and language model twice while discarding
        // the first result.
        let image_features = model
            .setup_without_projection(&batched_image, &dummy_input_ids)
            .context("Failed to extract image features")?;
        drop(model);

        // 6. The output is [batch_size, seq_len, hidden_dim]
        // For images, seq_len = num_patches (e.g., 1024 for 448x448)
        // We need to extract just the image patches (excluding text tokens)
        let patch_embeddings = image_features
            .i((.., ..self.num_patches, ..))
            .context("Failed to extract patch embeddings")?;

        // 7. Remove batch dimension and convert to Vec<Vec<f32>>
        let patch_embeddings = patch_embeddings
            .squeeze(0)
            .context("Failed to squeeze batch dimension")?;

        // 8. Apply custom text projection to image embeddings (2048 -> 128)
        // Note: In ColPali v1.2-merged, the same projection layer is used for both
        // text and vision embeddings to project from PaliGemma's hidden size (2048)
        // to ColPali's embedding dimension (128) for efficient late interaction.
        let projected = self
            .custom_text_projection
            .forward(&patch_embeddings)
            .context("Failed to apply projection to image embeddings")?;

        // 9. Apply L2 normalization
        let norms = projected
            .sqr()?
            .sum_keepdim(1)? // Sum over embedding dimension
            .sqrt()?;
        let normalized = projected
            .broadcast_div(&norms)
            .context("Failed to normalize image embeddings")?;

        // 10. Convert to CPU and extract as Vec<Vec<f32>>
        let embeddings = self
            .tensor_to_vec2(&normalized)
            .context("Failed to convert patch embeddings to Vec<Vec<f32>>")?;
        drop(inference_permit);

        // 11. Create VisionEmbedding with correct embedding dimension (128)
        Ok(VisionEmbedding::new(
            embeddings,
            self.num_patches,
            self.embedding_dim,
            source,
        ))
    }

    /// Encode text query into token embeddings.
    ///
    /// Uses the language model component of `PaliGemma` to encode text
    /// for retrieval against image embeddings using `MaxSim`.
    ///
    /// # Arguments
    ///
    /// * `text` - Text query string
    ///
    /// # Returns
    ///
    /// `TokenEmbeddings` compatible with `MaxSim` scoring
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
            .with_context(|| format!("Failed to tokenize text ({} UTF-8 bytes)", text.len()))?;

        // 2. Convert token IDs to tensor [1, seq_len]
        let token_ids_i64: Vec<i64> = token_ids.iter().map(|&id| i64::from(id)).collect();
        let token_ids_tensor = Tensor::from_vec(token_ids_i64, (1, token_ids.len()), &self.device)
            .context("Failed to create token IDs tensor")?;

        // 3. Acquire global inference admission before the model-specific lock.
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let mut model = self
            .model
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire model lock: {}", e))?;

        // 4. For text-only encoding, we use forward_without_projection
        // This gives us the language model embeddings without image context
        let token_embeddings = model
            .forward_without_projection(&token_ids_tensor)
            .context("Failed to encode text through language model")?;
        drop(model);

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
        drop(inference_permit);

        // 9. Create TokenEmbeddings
        TokenEmbeddings::new(embeddings, text.to_string())
            .context("Failed to create TokenEmbeddings")
    }

    /// Helper: Convert Candle Tensor to Vec<Vec<f32>>
    pub(super) fn tensor_to_vec2(&self, tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
        // Ensure tensor is on CPU and F32
        let tensor_cpu = tensor
            .to_dtype(DType::F32)
            .context("Failed to convert tensor to F32")?
            .to_device(&Device::Cpu)
            .context("Failed to move tensor to CPU")?;

        let shape = tensor_cpu.dims();
        if shape.len() != 2 {
            anyhow::bail!("Expected 2D tensor, got shape {shape:?}");
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

    /// Helper: Convert Candle Tensor to `ndarray::Array2`<f32>
    pub(super) fn tensor_to_array2(&self, tensor: &Tensor) -> Result<ndarray::Array2<f32>> {
        // Ensure tensor is on CPU and F32
        let tensor_cpu = tensor
            .to_dtype(DType::F32)
            .context("Failed to convert tensor to F32")?
            .to_device(&Device::Cpu)
            .context("Failed to move tensor to CPU")?;

        let shape = tensor_cpu.dims();
        if shape.len() != 2 {
            anyhow::bail!("Expected 2D tensor, got shape {shape:?}");
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
}
