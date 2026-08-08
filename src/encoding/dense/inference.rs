use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ndarray::Array1;

use super::{BertVariant, CandleDenseEncoder};
use crate::core::{DenseEmbedding, PoolingStrategy};

impl CandleDenseEncoder {
    /// Converts token IDs to a Candle tensor.
    fn tokens_to_tensor(&self, token_ids: &[u32], batch_size: usize) -> Result<Tensor> {
        let token_ids_i64: Vec<i64> = token_ids.iter().map(|&x| i64::from(x)).collect();
        let seq_len = token_ids.len() / batch_size;

        Tensor::from_vec(token_ids_i64, (batch_size, seq_len), &self.device)
            .context("Creating token ID tensor")
    }

    /// Applies pooling strategy to token embeddings.
    ///
    /// # Arguments
    /// * `token_embeddings` - Token embedding matrix (`seq_len` × `hidden_dim`)
    /// * `attention_mask` - Attention mask (1 = valid token, 0 = padding)
    ///
    /// # Returns
    /// Pooled embedding vector (`hidden_dim`)
    ///
    /// # Errors
    /// Returns an error if the token embeddings cannot be reshaped (shape mismatch)
    fn apply_pooling(
        &self,
        token_embeddings: &Array1<f32>,
        attention_mask: &[i64],
    ) -> Result<Array1<f32>> {
        // Convert flattened array back to 2D for pooling functions
        let seq_len = attention_mask.len();

        anyhow::ensure!(seq_len > 0, "Attention mask cannot be empty");

        let total_elements = token_embeddings.len();
        anyhow::ensure!(
            total_elements % seq_len == 0,
            "Token embeddings length ({total_elements}) must be divisible by sequence length ({seq_len}). \
             This indicates a shape mismatch between model output and attention mask."
        );

        let hidden_dim = total_elements / seq_len;
        let embeddings_2d =
            ndarray::Array2::from_shape_vec((seq_len, hidden_dim), token_embeddings.to_vec())
                .context("Failed to reshape token embeddings: ndarray shape mismatch")?;

        let pooled = match self.pooling_strategy {
            PoolingStrategy::Cls => {
                crate::utils::pooling::cls_pooling(&embeddings_2d, attention_mask)
            }
            PoolingStrategy::Mean => {
                crate::utils::pooling::mean_pooling(&embeddings_2d, attention_mask)
            }
            PoolingStrategy::Max => {
                crate::utils::pooling::max_pooling(&embeddings_2d, attention_mask)
            }
            PoolingStrategy::LastToken => {
                crate::utils::pooling::last_token_pooling(&embeddings_2d, attention_mask)
            }
        };

        Ok(pooled)
    }

    /// Processes output embeddings: applies Matryoshka truncation and normalization.
    ///
    /// # Arguments
    /// * `embedding` - Input embedding vector
    ///
    /// # Returns
    /// Processed embedding (truncated if configured, normalized if configured)
    ///
    /// # Errors
    /// Returns an error if target dimension is invalid
    fn process_output(&self, mut embedding: Array1<f32>) -> Result<Array1<f32>> {
        // Apply Matryoshka truncation if configured
        if let Some(target_dim) = self.config.target_dimension {
            anyhow::ensure!(
                target_dim > 0,
                "Target dimension must be greater than 0, got {target_dim}"
            );
            anyhow::ensure!(
                target_dim <= embedding.len(),
                "Target dimension ({}) cannot exceed embedding dimension ({})",
                target_dim,
                embedding.len()
            );

            embedding = embedding.slice(ndarray::s![..target_dim]).to_owned();
        }

        // Apply L2 normalization if configured
        if self.normalize {
            embedding = crate::utils::normalization::l2_normalize(&embedding);
        }

        Ok(embedding)
    }

    /// Encodes a single text input to a dense embedding.
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    ///
    /// # Returns
    /// Dense embedding for the input text
    pub fn encode(&self, text: &str) -> Result<DenseEmbedding> {
        // Tokenize input
        let (token_ids, attention_mask) = self
            .tokenizer
            .encode(text, true)
            .with_context(|| format!("Tokenizing text ({} UTF-8 bytes)", text.len()))?;

        // Convert to tensors
        let token_ids_tensor = self.tokens_to_tensor(&token_ids, 1)?;

        // Handle attention mask - DistilBERT in Candle uses inverted convention
        // Standard tokenizer: 1=attend, 0=pad
        // DistilBERT model: 0=attend, 1=pad
        // See: candle_transformers::models::distilbert::DistilBertModel::forward
        let attention_mask_processed: Vec<i64> = match &self.model {
            BertVariant::DistilBert(_) => {
                // Invert mask for DistilBERT
                attention_mask.iter().map(|&x| i64::from(x != 1)).collect()
            }
            _ => {
                // Standard BERT convention (no inversion needed)
                attention_mask.iter().map(|&x| i64::from(x)).collect()
            }
        };

        let attention_mask_tensor = Tensor::from_vec(
            attention_mask_processed.clone(),
            (1, attention_mask.len()),
            &self.device,
        )
        .context("Creating attention mask tensor")?;

        // Run model forward pass
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let output = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor)
            .context("Model forward pass")?;

        // Output shape: [1, seq_len, hidden_dim]
        // Squeeze batch dimension
        let embeddings = output.squeeze(0).context("Squeezing batch dimension")?;

        // Convert to CPU and flatten
        let embeddings_cpu = embeddings
            .to_dtype(DType::F32)
            .context("Converting to F32")?
            .to_device(&Device::Cpu)
            .context("Moving tensor to CPU")?;

        let embeddings_vec = embeddings_cpu
            .flatten_all()
            .context("Flattening tensor")?
            .to_vec1::<f32>()
            .context("Converting tensor to Vec<f32>")?;
        drop(inference_permit);

        let embeddings_array = Array1::from_vec(embeddings_vec);

        // Apply pooling
        let pooled = self.apply_pooling(&embeddings_array, &attention_mask_processed)?;

        // Process output (Matryoshka + normalization)
        let final_embedding = self.process_output(pooled)?;

        DenseEmbedding::new(final_embedding, text.to_string())
    }

    /// Encodes multiple text inputs in batch.
    ///
    /// # Arguments
    /// * `texts` - Slice of text inputs to encode
    ///
    /// # Returns
    /// Vector of dense embeddings, one per input
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Special case: single input
        if texts.len() == 1 {
            return Ok(vec![self.encode(texts[0])?]);
        }

        // Batch tokenization with padding
        let batch_tokenized = self
            .tokenizer
            .encode_batch(texts, true)
            .context("Batch tokenization")?;

        // Candle 0.11's JinaBERT forward pass does not accept an attention mask,
        // so padded keys and values would change the valid token representations.
        // Batch tokenization above still enforces the aggregate resource policy;
        // inference then uses unpadded inputs to preserve sequential parity.
        if !self.supports_padded_batch {
            drop(batch_tokenized);
            return texts.iter().map(|&text| self.encode(text)).collect();
        }

        let batch_size = batch_tokenized.len();
        let max_seq_len = batch_tokenized[0].0.len();

        // Convert token IDs to 2D tensor: [batch_size, max_seq_len]
        let mut all_token_ids = Vec::with_capacity(batch_size * max_seq_len);
        for (token_ids, _) in &batch_tokenized {
            for &token_id in token_ids {
                all_token_ids.push(i64::from(token_id));
            }
        }

        let token_ids_tensor =
            Tensor::from_vec(all_token_ids, (batch_size, max_seq_len), &self.device)
                .context("Creating batch token IDs tensor")?;

        // Convert attention masks - handle DistilBERT's inverted mask convention
        // We maintain two versions:
        // 1. all_attention_masks: For the model forward pass (inverted for DistilBERT)
        // 2. attention_masks_for_pooling: For pooling logic (always standard: 1=valid, 0=pad)
        let mut all_attention_masks = Vec::with_capacity(batch_size * max_seq_len);
        let mut attention_masks_for_pooling = Vec::with_capacity(batch_size);

        for (_, attention_mask) in &batch_tokenized {
            let mut mask_for_pooling = Vec::with_capacity(max_seq_len);

            for &mask_val in attention_mask {
                // Apply inversion for DistilBERT model input
                let processed_val = match &self.model {
                    BertVariant::DistilBert(_) => {
                        // DistilBERT expects: 0=attend, 1=pad
                        i64::from(mask_val != 1)
                    }
                    _ => {
                        // Standard BERT: 1=attend, 0=pad
                        i64::from(mask_val)
                    }
                };
                all_attention_masks.push(processed_val);

                // For pooling, we always use standard convention (1=valid, 0=padding)
                mask_for_pooling.push(i64::from(mask_val));
            }

            attention_masks_for_pooling.push(mask_for_pooling);
        }

        let attention_mask_tensor =
            Tensor::from_vec(all_attention_masks, (batch_size, max_seq_len), &self.device)
                .context("Creating batch attention mask tensor")?;

        // Single forward pass for entire batch
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let batch_output = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor)
            .context("Batch forward pass")?;

        // batch_output shape: [batch_size, max_seq_len, hidden_dim]
        let mut results = Vec::with_capacity(batch_size);

        // PERFORMANCE FIX: Move entire batch to CPU once (critical optimization)
        // Previously moved each sample individually inside the loop (50-100x slower)
        let batch_output_cpu = batch_output
            .to_dtype(DType::F32)
            .context("Converting batch to F32")?
            .to_device(&Device::Cpu)
            .context("Moving batch to CPU")?;

        // Drop the GPU tensor explicitly to free GPU memory immediately
        drop(batch_output);
        drop(inference_permit);

        for i in 0..batch_size {
            // Extract embeddings for this sample from CPU tensor
            let sample_output = batch_output_cpu
                .get(i)
                .context("Extracting sample from batch")?;

            let embeddings_vec = sample_output
                .flatten_all()
                .context("Flattening tensor")?
                .to_vec1::<f32>()
                .context("Converting tensor to Vec<f32>")?;

            let embeddings_array = Array1::from_vec(embeddings_vec);

            // Apply pooling using the standard attention mask
            let pooled = self.apply_pooling(&embeddings_array, &attention_masks_for_pooling[i])?;

            // Process output (Matryoshka + normalization)
            let final_embedding = self.process_output(pooled)?;

            results.push(DenseEmbedding::new(final_embedding, texts[i].to_string())?);
        }

        Ok(results)
    }
}
