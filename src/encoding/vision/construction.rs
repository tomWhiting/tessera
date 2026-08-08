use super::ColPaliEncoder;
use crate::core::Tokenizer;
use crate::models::{registry::ModelType, ModelConfig};
use crate::runtime::{preflight_registered_model, ResourcePolicy};
use crate::vision::ImageProcessor;
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::paligemma::{Config as PaliGemmaConfig, Model as PaliGemmaModel};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

impl ColPaliEncoder {
    /// Create new `ColPali` encoder.
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
    /// - Model cannot be downloaded from `HuggingFace` Hub
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
        let resource_policy = ResourcePolicy::for_model_context(config.max_seq_length);
        Self::new_with_resource_policy(config, device, resource_policy)
    }

    /// Creates a `ColPali` encoder with explicit resource limits.
    pub fn new_with_resource_policy(
        config: ModelConfig,
        device: Device,
        resource_policy: ResourcePolicy,
    ) -> Result<Self> {
        preflight_registered_model(
            &config.model_name,
            config.max_seq_length,
            ModelType::VisionLanguage,
            &device,
            &resource_policy,
        )?;

        // 1. Initialize HuggingFace API
        let api = hf_hub::api::sync::Api::new().context("Failed to initialize HuggingFace API")?;
        let repo = api.model(config.model_name.clone());

        // 2. Load tokenizer
        let tokenizer = Tokenizer::from_pretrained_with_policy(&config.model_name, resource_policy)
            .context("Failed to load tokenizer")?;

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
                    .map(std::string::ToString::to_string)
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
            [0.481_454_66, 0.457_827_5, 0.408_210_73], // SigLIP mean
            [0.268_629_54, 0.261_302_6, 0.275_777_1],  // SigLIP std
        );

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer,
            image_processor,
            device,
            embedding_dim: config.embedding_dim,
            num_patches,
            image_resolution: (image_size as u32, image_size as u32),
            custom_text_projection,
        })
    }
}
