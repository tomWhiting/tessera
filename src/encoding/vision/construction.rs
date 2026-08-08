use super::ColPaliEncoder;
use crate::core::Tokenizer;
use crate::models::loader::ModelFileResolver;
use crate::models::{registry::ModelType, ModelConfig};
use crate::runtime::{
    preflight_and_reserve_registered_model_with_dtype, ModelDType, ResourcePolicy,
    TransformerProfile,
};
use crate::vision::{ColPaliPreprocessorConfig, ColPaliProcessor};
use anyhow::{Context, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::paligemma::{Config as PaliGemmaConfig, Model as PaliGemmaModel};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

impl ColPaliEncoder {
    /// Creates a `ColPali` encoder with explicit dtype and resource limits.
    pub fn new_with_dtype_and_resource_policy(
        config: ModelConfig,
        device: Device,
        dtype: ModelDType,
        resource_policy: ResourcePolicy,
    ) -> Result<Self> {
        let (model_info, residency) = preflight_and_reserve_registered_model_with_dtype(
            &config.model_name,
            config.max_seq_length,
            ModelType::VisionLanguage,
            &device,
            dtype,
            &resource_policy,
        )?;

        // Candle's PaliGemma adapter currently exposes the audited 3B/448
        // architecture as a factory rather than accepting the upstream config.
        let paligemma_config = PaliGemmaConfig::paligemma_3b_448();
        let (hidden_dim, embedding_dim) = validate_projection_contract(
            model_info.id,
            model_info.has_projection,
            model_info.projection_dims,
            config.embedding_dim,
            model_info.hidden_dim,
            paligemma_config.text_config.hidden_size,
        )?;
        validate_architecture_contract(model_info, &paligemma_config)?;

        // 1. Initialize the pinned model artifact resolver.
        let files = ModelFileResolver::new(model_info)?;

        // 2. Resolve and validate the image processor from the same immutable revision.
        let preprocessor_path = files
            .get("preprocessor_config.json")
            .context("Failed to resolve pinned ColPali preprocessor config")?;
        let preprocessor_config = ColPaliPreprocessorConfig::from_path(&preprocessor_path)?;

        // 3. Load the tokenizer and build the model-specific prompt processor.
        let tokenizer = Tokenizer::from_model_files_with_policy(&files, resource_policy)
            .context("Failed to load tokenizer")?;
        let processor = ColPaliProcessor::new(&preprocessor_config, &tokenizer)
            .context("Failed to initialize ColPali processor")?;

        let (num_patches, image_resolution) = validate_image_contract(
            &preprocessor_config,
            paligemma_config.vision_config.image_size,
            paligemma_config.vision_config.patch_size,
        )?;
        let combined_image_tokens = processor
            .image_seq_length()
            .checked_add(processor.image_prompt_token_ids().len())
            .context("ColPali image sequence length overflowed")?;
        anyhow::ensure!(
            combined_image_tokens <= paligemma_config.text_config.max_position_embeddings,
            "ColPali image sequence requires {combined_image_tokens} positions; PaliGemma allows {}",
            paligemma_config.text_config.max_position_embeddings
        );
        resource_policy
            .validate_sequence(combined_image_tokens)
            .map_err(|error| anyhow::anyhow!("ColPali image sequence preflight failed: {error}"))?;
        resource_policy
            .validate_batch(1, combined_image_tokens)
            .map_err(|error| anyhow::anyhow!("ColPali image batch preflight failed: {error}"))?;
        let profile = TransformerProfile::new(
            paligemma_config.text_config.hidden_size,
            paligemma_config.text_config.intermediate_size,
            paligemma_config.text_config.num_attention_heads,
        )
        .context("Building ColPali activation profile")?;
        resource_policy
            .validate_transformer_activations(profile, 1, combined_image_tokens, dtype)
            .map_err(|error| anyhow::anyhow!("ColPali activation preflight failed: {error}"))?;

        // 4. Download model weights (handle both single file and sharded models)
        let safetensors_file = model_info.safetensors_file.ok_or_else(|| {
            anyhow::anyhow!(
                "Vision model '{}' has no registered safetensors artifact",
                model_info.id
            )
        })?;
        let weights_paths: Vec<PathBuf> = if safetensors_file.ends_with(".safetensors.index.json") {
            let index_path = files.get(safetensors_file)?;
            // Sharded model - load all shards
            let index: serde_json::Value = serde_json::from_reader(
                std::fs::File::open(&index_path).context("Failed to open safetensors index")?,
            )
            .context("Failed to parse safetensors index")?;

            // Get unique weight files from index
            let weight_map = index["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("Invalid safetensors index: missing weight_map"))?;

            let mut weight_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str())
                .map(std::string::ToString::to_string)
                .collect();
            weight_files.sort();
            weight_files.dedup();

            // Download all shard files
            weight_files
                .iter()
                .map(|f| files.get(f))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("Failed to download model shard files")?
        } else {
            // Single file model
            vec![files
                .get(safetensors_file)
                .context("Failed to download safetensors weights")?]
        };

        // 5. Load VarBuilder from safetensors
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights_paths, dtype.candle_dtype(), &device)
                .context("Failed to load model weights from safetensors")?
        };

        // 6. Initialize PaliGemma model from candle-transformers.
        // Note: ColPali v1.2-merged models have weights under "model." prefix
        let model = PaliGemmaModel::new(&paligemma_config, vb.pp("model"))
            .context("Failed to initialize PaliGemma model")?;

        // 7. Load the mandatory ColPali projection from its checkpoint namespace.
        let custom_text_projection =
            candle_nn::linear(hidden_dim, embedding_dim, vb.pp("custom_text_proj"))
                .context("Failed to load mandatory custom_text_proj layer")?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer,
            processor,
            device,
            embedding_dim,
            hidden_dim,
            num_patches,
            image_resolution,
            custom_text_projection,
            dtype,
            resource_policy,
            transformer_profile: profile,
            _residency: residency,
        })
    }
}

fn validate_projection_contract(
    model_id: &str,
    has_projection: bool,
    projection_dims: Option<usize>,
    requested_dim: usize,
    registered_hidden_dim: usize,
    checkpoint_hidden_dim: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(
        has_projection,
        "Vision model '{model_id}' has no registered ColPali projection"
    );
    let projection_dim = projection_dims.with_context(|| {
        format!("Vision model '{model_id}' marks projection enabled but declares no output size")
    })?;
    anyhow::ensure!(
        projection_dim > 0,
        "Vision model '{model_id}' declares a zero-dimensional projection"
    );
    anyhow::ensure!(
        requested_dim == projection_dim,
        "Requested embedding dimension {requested_dim} does not match '{model_id}' projection dimension {projection_dim}"
    );
    anyhow::ensure!(
        registered_hidden_dim == checkpoint_hidden_dim,
        "Registered hidden dimension {registered_hidden_dim} does not match PaliGemma hidden dimension {checkpoint_hidden_dim}"
    );
    Ok((checkpoint_hidden_dim, projection_dim))
}

fn validate_architecture_contract(
    model: &crate::models::registry::ModelInfo,
    config: &PaliGemmaConfig,
) -> Result<()> {
    anyhow::ensure!(
        model.architecture_type == "paligemma",
        "Vision model '{}' architecture is {:?}; expected \"paligemma\"",
        model.id,
        model.architecture_type
    );
    anyhow::ensure!(
        model.vocab_size == config.text_config.vocab_size,
        "Registered vocabulary size {} does not match PaliGemma vocabulary size {}",
        model.vocab_size,
        config.text_config.vocab_size
    );
    Ok(())
}

fn validate_image_contract(
    preprocessor: &ColPaliPreprocessorConfig,
    model_image_size: usize,
    patch_size: usize,
) -> Result<(usize, (u32, u32))> {
    anyhow::ensure!(patch_size > 0, "PaliGemma patch size must be non-zero");
    anyhow::ensure!(
        model_image_size.is_multiple_of(patch_size),
        "PaliGemma image size {model_image_size} is not divisible by patch size {patch_size}"
    );
    let image_size =
        u32::try_from(model_image_size).context("PaliGemma image size does not fit in a u32")?;
    let resolution = (image_size, image_size);
    anyhow::ensure!(
        preprocessor.target_size() == resolution,
        "Preprocessor target is {:?}; PaliGemma expects {resolution:?}",
        preprocessor.target_size()
    );
    let patches_per_side = model_image_size / patch_size;
    let num_patches = patches_per_side
        .checked_mul(patches_per_side)
        .context("PaliGemma patch count overflowed")?;
    anyhow::ensure!(
        preprocessor.image_seq_length() == num_patches,
        "Preprocessor declares {} image tokens; PaliGemma produces {num_patches} patches",
        preprocessor.image_seq_length()
    );
    Ok((num_patches, resolution))
}

#[cfg(test)]
mod tests;
