use super::ColPaliEncoder;
use crate::core::{TokenEmbeddings, VisionEmbedding};
use crate::runtime::{ModelDType, ResourcePolicy, TransformerProfile};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
#[cfg(feature = "pdf")]
use image::DynamicImage;
use std::path::Path;

impl ColPaliEncoder {
    /// Encodes an image as one normalized vector per visual patch.
    pub fn encode_image(&self, image_path: &Path) -> Result<VisionEmbedding> {
        let image_tensor = self
            .processor
            .image_processor()
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
            .processor
            .image_processor()
            .preprocess_image(image, &self.device)
            .context("Failed to preprocess image")?;

        self.encode_image_tensor(image_tensor, source)
    }

    fn encode_image_tensor(
        &self,
        image_tensor: Tensor,
        source: Option<String>,
    ) -> Result<VisionEmbedding> {
        let prompt_ids = self.processor.image_prompt_token_ids();
        let expected_sequence = self
            .num_patches
            .checked_add(prompt_ids.len())
            .context("ColPali image sequence length overflowed")?;
        validate_forward_resources(
            &self.resource_policy,
            self.transformer_profile,
            self.dtype,
            expected_sequence,
            "ColPali image",
        )?;

        let batched_image = image_tensor
            .unsqueeze(0)
            .context("Failed to add image batch dimension")?;
        let prompt_tensor = token_ids_tensor(prompt_ids, &self.device)
            .context("Failed to create ColPali image prompt tensor")?;

        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let mut model = self
            .model
            .lock()
            .map_err(|error| anyhow::anyhow!("Failed to acquire model lock: {error}"))?;
        let image_features = model
            .setup_without_projection(&batched_image, &prompt_tensor)
            .context("Failed to encode image and ColPali document prompt")?;
        drop(model);

        validate_tensor_shape(
            &image_features,
            &[1, expected_sequence, self.hidden_dim],
            "PaliGemma image output",
        )?;
        let document_states = image_features
            .squeeze(0)
            .context("Failed to remove image batch dimension")?;
        let normalized = self.project_and_normalize(
            &document_states,
            expected_sequence,
            "image document projection",
        )?;
        let embeddings = normalized
            .chunks_exact(self.embedding_dim)
            .map(<[f32]>::to_vec)
            .collect();
        drop(inference_permit);

        VisionEmbedding::new(embeddings, self.num_patches, self.embedding_dim, source)
            .context("Failed to validate projected image embedding")
    }

    /// Encodes a query using ColPali's retrieval prompt and augmentation tokens.
    pub fn encode_text(&self, text: &str) -> Result<TokenEmbeddings> {
        let token_ids = self.processor.tokenize_query(text, &self.tokenizer)?;
        validate_forward_resources(
            &self.resource_policy,
            self.transformer_profile,
            self.dtype,
            token_ids.len(),
            "ColPali query",
        )?;
        let token_ids_tensor = token_ids_tensor(&token_ids, &self.device)
            .context("Failed to create ColPali query token tensor")?;

        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let mut model = self
            .model
            .lock()
            .map_err(|error| anyhow::anyhow!("Failed to acquire model lock: {error}"))?;
        let token_embeddings = model
            .forward_without_projection(&token_ids_tensor)
            .context("Failed to encode the ColPali query prompt")?;
        drop(model);

        validate_tensor_shape(
            &token_embeddings,
            &[1, token_ids.len(), self.hidden_dim],
            "PaliGemma query output",
        )?;
        let token_embeddings = token_embeddings
            .squeeze(0)
            .context("Failed to remove query batch dimension")?;
        let normalized = self.project_and_normalize(
            &token_embeddings,
            token_ids.len(),
            "query token projection",
        )?;
        let embeddings =
            ndarray::Array2::from_shape_vec((token_ids.len(), self.embedding_dim), normalized)
                .context("Failed to create query embedding matrix")?;
        drop(inference_permit);

        TokenEmbeddings::new(embeddings, text.to_string())
            .context("Failed to validate projected query embedding")
    }

    fn project_and_normalize(
        &self,
        hidden_states: &Tensor,
        expected_rows: usize,
        kind: &str,
    ) -> Result<Vec<f32>> {
        validate_tensor_shape(
            hidden_states,
            &[expected_rows, self.hidden_dim],
            &format!("{kind} input"),
        )?;
        let projected = self
            .custom_text_projection
            .forward(hidden_states)
            .with_context(|| format!("Failed to apply {kind}"))?;
        validate_tensor_shape(&projected, &[expected_rows, self.embedding_dim], kind)?;

        let mut values =
            tensor_to_finite_flat(&projected, expected_rows, self.embedding_dim, kind)?;
        normalize_rows(&mut values, expected_rows, self.embedding_dim, kind)?;
        Ok(values)
    }
}

fn validate_forward_resources(
    policy: &ResourcePolicy,
    profile: TransformerProfile,
    dtype: ModelDType,
    sequence_tokens: usize,
    kind: &str,
) -> Result<()> {
    policy
        .validate_sequence(sequence_tokens)
        .map_err(|error| anyhow::anyhow!("{kind} sequence preflight failed: {error}"))?;
    policy
        .validate_batch(1, sequence_tokens)
        .map_err(|error| anyhow::anyhow!("{kind} batch preflight failed: {error}"))?;
    policy
        .validate_transformer_activations(profile, 1, sequence_tokens, dtype)
        .map_err(|error| anyhow::anyhow!("{kind} activation preflight failed: {error}"))?;
    Ok(())
}

fn token_ids_tensor(token_ids: &[u32], device: &Device) -> Result<Tensor> {
    anyhow::ensure!(!token_ids.is_empty(), "ColPali token layout is empty");
    let token_ids: Vec<i64> = token_ids.iter().copied().map(i64::from).collect();
    let sequence_length = token_ids.len();
    Tensor::from_vec(token_ids, (1, sequence_length), device).map_err(Into::into)
}

fn validate_tensor_shape(tensor: &Tensor, expected: &[usize], kind: &str) -> Result<()> {
    let measured = tensor.dims();
    anyhow::ensure!(
        measured == expected,
        "{kind} has shape {measured:?}; expected {expected:?}"
    );
    Ok(())
}

fn tensor_to_finite_flat(
    tensor: &Tensor,
    expected_rows: usize,
    expected_columns: usize,
    kind: &str,
) -> Result<Vec<f32>> {
    validate_tensor_shape(tensor, &[expected_rows, expected_columns], kind)?;
    let values = tensor
        .to_dtype(DType::F32)
        .with_context(|| format!("Failed to convert {kind} to F32"))?
        .to_device(&Device::Cpu)
        .with_context(|| format!("Failed to move {kind} to CPU"))?
        .flatten_all()
        .with_context(|| format!("Failed to flatten {kind}"))?
        .to_vec1::<f32>()
        .with_context(|| format!("Failed to extract {kind}"))?;
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        let row = index / expected_columns;
        let column = index % expected_columns;
        anyhow::bail!("{kind} contains non-finite value {value} at row {row}, column {column}");
    }
    Ok(values)
}

fn normalize_rows(
    values: &mut [f32],
    expected_rows: usize,
    expected_columns: usize,
    kind: &str,
) -> Result<()> {
    let expected_values = expected_rows
        .checked_mul(expected_columns)
        .context("Projected embedding element count overflowed")?;
    anyhow::ensure!(
        values.len() == expected_values,
        "{kind} contains {} values; expected {expected_values}",
        values.len()
    );
    for (row_index, row) in values.chunks_exact_mut(expected_columns).enumerate() {
        let squared_norm = row.iter().try_fold(0.0_f32, |sum, &value| {
            let next = value.mul_add(value, sum);
            anyhow::ensure!(
                next.is_finite(),
                "{kind} row {row_index} has a non-finite squared norm"
            );
            Ok::<f32, anyhow::Error>(next)
        })?;
        let norm = squared_norm.sqrt();
        anyhow::ensure!(
            norm.is_finite() && norm > 0.0,
            "{kind} row {row_index} has invalid L2 norm {norm}"
        );
        for (column_index, value) in row.iter_mut().enumerate() {
            *value /= norm;
            anyhow::ensure!(
                value.is_finite(),
                "{kind} produced a non-finite normalized value at row {row_index}, column {column_index}"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
