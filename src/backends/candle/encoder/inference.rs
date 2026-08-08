use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ndarray::Array2;

use super::model::BertVariant;
use super::role::PreparedInput;
use super::{tensor, CandleBertEncoder};

#[cfg(test)]
mod tests;

impl CandleBertEncoder {
    pub(super) fn infer_one(
        &self,
        input: PreparedInput,
        text: &str,
    ) -> Result<crate::core::TokenEmbeddings> {
        let mut outputs = self.infer_batch(&[input], &[text])?;
        outputs
            .pop()
            .context("ColBERT single-item inference returned no output")
    }

    pub(super) fn infer_batch(
        &self,
        inputs: &[PreparedInput],
        texts: &[&str],
    ) -> Result<Vec<crate::core::TokenEmbeddings>> {
        anyhow::ensure!(
            inputs.len() == texts.len(),
            "Prepared ColBERT input count must match source text count"
        );
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = inputs.len();
        let sequence_length = inputs[0].token_ids.len();
        anyhow::ensure!(
            inputs.iter().all(|input| {
                input.token_ids.len() == sequence_length
                    && input.attention_mask.len() == sequence_length
                    && input.output_mask.len() == sequence_length
            }),
            "Prepared ColBERT batch must have uniform token and mask lengths"
        );
        self.resource_policy
            .validate_transformer_activations(
                self.transformer_profile,
                batch_size,
                sequence_length,
                self.dtype,
            )
            .map_err(|error| anyhow::anyhow!("ColBERT activation preflight failed: {error}"))?;

        let token_ids = inputs
            .iter()
            .flat_map(|input| input.token_ids.iter().copied().map(i64::from))
            .collect::<Vec<_>>();
        let token_ids = Tensor::from_vec(token_ids, (batch_size, sequence_length), &self.device)
            .context("Creating ColBERT batch token tensor")?;
        let attention_mask = self.attention_mask_tensor(inputs, sequence_length)?;

        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let output = self
            .model
            .forward(&token_ids, &attention_mask)
            .context("ColBERT model forward pass")?;
        let output = self.project(output)?;
        let output = tensor::l2_normalize_tokens(&output)
            .context("L2-normalizing projected ColBERT token embeddings")?;

        let results = inputs
            .iter()
            .zip(texts)
            .enumerate()
            .map(|(index, (input, text))| {
                extract_item(&output, index, &input.output_mask, text)
                    .with_context(|| format!("Extracting ColBERT batch item {index}"))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(inference_permit);
        Ok(results)
    }

    fn attention_mask_tensor(
        &self,
        inputs: &[PreparedInput],
        sequence_length: usize,
    ) -> Result<Tensor> {
        let batch_size = inputs.len();
        match &self.model {
            BertVariant::DistilBert(_) => {
                let mask = flatten_attention_mask(inputs, true);
                Tensor::from_vec(mask, (batch_size, 1, 1, sequence_length), &self.device)
                    .context("Creating broadcastable DistilBERT padding mask")
            }
            BertVariant::Bert(_) | BertVariant::JinaBert(_) => {
                let mask = flatten_attention_mask(inputs, false);
                Tensor::from_vec(mask, (batch_size, sequence_length), &self.device)
                    .context("Creating BERT attention mask")
            }
        }
    }

    fn project(&self, mut output: Tensor) -> Result<Tensor> {
        use crate::utils::MatryoshkaStrategy;

        let configured = self.config.target_dimension.zip(self.matryoshka_strategy);
        if let Some((target, strategy @ MatryoshkaStrategy::TruncateHidden)) = configured {
            output = crate::utils::apply_matryoshka(&output, target, strategy)
                .context("Applying Matryoshka truncation to ColBERT hidden states")?;
        }

        output = output
            .broadcast_matmul(&self.projection.t()?)
            .context("Applying mandatory ColBERT projection")?;

        if let Some((target, strategy)) = configured {
            if !matches!(strategy, MatryoshkaStrategy::TruncateHidden) {
                output = crate::utils::apply_matryoshka(&output, target, strategy)
                    .context("Applying Matryoshka truncation to ColBERT output")?;
            }
        }
        Ok(output)
    }
}

fn flatten_attention_mask(inputs: &[PreparedInput], inverted: bool) -> Vec<i64> {
    inputs
        .iter()
        .flat_map(|input| {
            input.attention_mask.iter().map(move |value| {
                if inverted {
                    i64::from(*value != 1)
                } else {
                    i64::from(*value)
                }
            })
        })
        .collect()
}

fn extract_item(
    output: &Tensor,
    index: usize,
    output_mask: &[u32],
    text: &str,
) -> Result<crate::core::TokenEmbeddings> {
    let item = output
        .get(index)
        .context("Selecting item from ColBERT batch")?
        .to_dtype(DType::F32)
        .context("Converting ColBERT output to F32")?
        .to_device(&Device::Cpu)
        .context("Moving ColBERT output to CPU")?;
    let (sequence_length, embedding_dim) =
        item.dims2().context("Reading ColBERT output dimensions")?;
    let values = item
        .flatten_all()
        .context("Flattening ColBERT output")?
        .to_vec1::<f32>()
        .context("Converting ColBERT output to values")?;
    let embeddings = Array2::from_shape_vec((sequence_length, embedding_dim), values)
        .context("Converting ColBERT output to ndarray")?;
    let embeddings = tensor::filter_by_attention_mask(&embeddings, output_mask)
        .context("Applying ColBERT output-selection mask")?;
    crate::core::TokenEmbeddings::new(embeddings, text.to_string())
        .context("Creating ColBERT token embeddings")
}
