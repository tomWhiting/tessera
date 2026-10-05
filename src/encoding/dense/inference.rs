use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ndarray::Array1;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use super::{BertVariant, CandleDenseEncoder};
use crate::core::embeddings::{CountedDenseEmbedding, Role};
use crate::core::tokenizer::WholeTokenizedInput;
use crate::core::{DenseEmbedding, PoolingStrategy};
use crate::models::registry::Prompts;
use crate::runtime::ContextWindowConfig;

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
            total_elements.is_multiple_of(seq_len),
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
        let (token_ids, attention_mask) = self
            .tokenizer
            .encode(text, true)
            .with_context(|| format!("Tokenizing text ({} UTF-8 bytes)", text.len()))?;

        let final_embedding = self.encode_tokenized(&token_ids, &attention_mask)?;
        DenseEmbedding::new(final_embedding, text.to_string())
    }

    pub(crate) fn validate_cut_configuration(&self, role: Option<Role>) -> Result<()> {
        self.tokenizer
            .validate_cut_configuration_with(&prompts_to_hold(self.prompts, role))
    }

    pub(crate) fn input_refusal(
        &self,
        text: &str,
        role: Option<Role>,
    ) -> Result<Option<crate::EmbeddingRefusal>> {
        match self
            .tokenizer
            .encode_with_prompt(prompt_for(self.prompts, role), text)
        {
            Ok(_) => Ok(None),
            Err(error) => error
                .downcast_ref::<crate::EmbeddingRefusal>()
                .copied()
                .map_or_else(|| Err(error), |refusal| Ok(Some(refusal))),
        }
    }

    pub(crate) fn encode_outcome(
        &self,
        text: &str,
        role: Option<Role>,
    ) -> Result<CountedDenseEmbedding> {
        let prompt = prompt_for(self.prompts, role);
        self.encode_whole_input(0, self.tokenizer.encode_with_prompt(prompt, text)?)
    }

    pub(crate) fn encode_batch_outcomes(
        &self,
        texts: &[&str],
        role: Option<Role>,
    ) -> Result<Vec<CountedDenseEmbedding>> {
        let inputs = self
            .tokenizer
            .encode_batch_with_prompt(prompt_for(self.prompts, role), texts)?;
        let Some(longest) = inputs.iter().map(|input| input.token_ids.len()).max() else {
            return Ok(Vec::new());
        };
        // The forwards of one batch run side by side, so the activation budget is
        // checked for all of them at the longest input before any starts.
        self.resource_policy
            .validate_transformer_activations(
                self.transformer_profile,
                inputs.len(),
                longest,
                self.dtype,
            )
            .map_err(|error| anyhow::anyhow!("Dense activation preflight failed: {error}"))?;
        // One permit admits the whole batch. Its texts run on the bounded Rayon
        // pool, so the thread ceiling still decides how many run at once.
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let embeddings = inputs
            .into_par_iter()
            .enumerate()
            .map(|(index, input)| self.encode_whole_input_admitted(index, input))
            .collect();
        drop(inference_permit);
        embeddings
    }

    /// Embeds one complete input; `index` is its position in the encoded slice.
    fn encode_whole_input(
        &self,
        index: usize,
        input: WholeTokenizedInput,
    ) -> Result<CountedDenseEmbedding> {
        let embedding = self.encode_tokenized(&input.token_ids, &input.attention_mask)?;
        Self::counted_embedding(index, &input, embedding)
    }

    /// As [`Self::encode_whole_input`], for a caller that holds the inference permit.
    fn encode_whole_input_admitted(
        &self,
        index: usize,
        input: WholeTokenizedInput,
    ) -> Result<CountedDenseEmbedding> {
        let embedding = self.encode_tokenized_admitted(&input.token_ids, &input.attention_mask)?;
        Self::counted_embedding(index, &input, embedding)
    }

    fn counted_embedding(
        index: usize,
        input: &WholeTokenizedInput,
        embedding: Array1<f32>,
    ) -> Result<CountedDenseEmbedding> {
        if !embedding.iter().all(|value| value.is_finite()) {
            return Err(anyhow::Error::new(
                crate::api::embedder::EmbedFailure::OutputInvalid {
                    index,
                    reason: "vector contains NaN or Inf values".to_string(),
                },
            ));
        }
        CountedDenseEmbedding::new(embedding, input.tokens_total)
    }

    /// Encodes a long input as bounded overlapping windows and returns their
    /// center-owned weighted mean.
    pub fn encode_windowed(
        &self,
        text: &str,
        config: ContextWindowConfig,
    ) -> Result<DenseEmbedding> {
        let windows = self
            .tokenizer
            .encode_windows(text, config)
            .with_context(|| format!("Planning windows for {} UTF-8 bytes", text.len()))?;
        let mut aggregate: Option<Array1<f32>> = None;
        let mut total_weight = 0_f32;

        for window in windows {
            let embedding = self.encode_tokenized(&window.token_ids, &window.attention_mask)?;
            let weight = window.owned_len().max(1) as f32;
            if let Some(sum) = aggregate.as_mut() {
                anyhow::ensure!(
                    sum.len() == embedding.len(),
                    "Dense window dimensions changed within one input"
                );
                sum.zip_mut_with(&embedding, |left, right| {
                    *left = right.mul_add(weight, *left);
                });
            } else {
                aggregate = Some(embedding.mapv(|value| value * weight));
            }
            total_weight += weight;
        }

        let mut aggregate = aggregate.context("Window planner returned no dense inputs")?;
        anyhow::ensure!(
            total_weight.is_finite() && total_weight > 0.0,
            "Invalid window weight"
        );
        aggregate.mapv_inplace(|value| value / total_weight);
        if self.normalize {
            aggregate = crate::utils::normalization::l2_normalize(&aggregate);
        }
        DenseEmbedding::new(aggregate, text.to_string())
    }

    pub(crate) fn encode_unpooled(&self, text: &str) -> Result<(Vec<String>, Vec<f32>)> {
        let (ids, mask) = self.tokenizer.encode(text, true)?;
        let values = self.forward_tokenized(&ids, &mask)?;
        let width = values
            .len()
            .checked_div(ids.len())
            .context("Empty token output")?;
        anyhow::ensure!(
            width > 0 && values.len() == ids.len() * width,
            "Token output shape does not match tokenizer positions"
        );
        anyhow::ensure!(
            values.iter().all(|value| value.is_finite()),
            "Token output contains non-finite values"
        );
        let mut tokens = Vec::new();
        let mut vectors = Vec::new();
        let values = values
            .as_slice()
            .context("Token output is not contiguous")?;
        for ((&id, &attend), row) in ids.iter().zip(&mask).zip(values.chunks_exact(width)) {
            if attend != 0 {
                tokens.push(self.tokenizer.token_string(id)?);
                vectors.extend_from_slice(row);
            }
        }
        Ok((tokens, vectors))
    }

    fn forward_tokenized(&self, token_ids: &[u32], attention_mask: &[u32]) -> Result<Array1<f32>> {
        self.resource_policy
            .validate_transformer_activations(
                self.transformer_profile,
                1,
                token_ids.len(),
                self.dtype,
            )
            .map_err(|error| anyhow::anyhow!("Dense activation preflight failed: {error}"))?;
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let embedding = self.forward_tokenized_admitted(token_ids, attention_mask);
        drop(inference_permit);
        embedding
    }

    // Pooling stays with the caller so token positions remain available.
    fn forward_tokenized_admitted(
        &self,
        token_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Array1<f32>> {
        anyhow::ensure!(!token_ids.is_empty(), "Tokenized input cannot be empty");
        anyhow::ensure!(
            token_ids.len() == attention_mask.len(),
            "Token ID and attention-mask lengths differ"
        );
        // Convert to tensors
        let token_ids_tensor = self.tokens_to_tensor(token_ids, 1)?;

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
            attention_mask_processed,
            (1, attention_mask.len()),
            &self.device,
        )
        .context("Creating attention mask tensor")?;

        // Run model forward pass
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

        Ok(Array1::from_vec(embeddings_vec))
    }

    fn encode_tokenized(&self, token_ids: &[u32], attention_mask: &[u32]) -> Result<Array1<f32>> {
        self.resource_policy
            .validate_transformer_activations(
                self.transformer_profile,
                1,
                token_ids.len(),
                self.dtype,
            )
            .map_err(|error| anyhow::anyhow!("Dense activation preflight failed: {error}"))?;
        let inference_permit = crate::runtime::acquire_inference_permit()
            .map_err(|error| anyhow::anyhow!("Failed to acquire inference admission: {error}"))?;
        let embedding = self.encode_tokenized_admitted(token_ids, attention_mask);
        drop(inference_permit);
        embedding
    }

    /// Runs one forward pass. The caller holds the process-wide inference
    /// permit and has checked the activation budget for everything it runs
    /// under that permit.
    fn encode_tokenized_admitted(
        &self,
        token_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Array1<f32>> {
        let embeddings_array = self.forward_tokenized_admitted(token_ids, attention_mask)?;

        // Apply pooling
        let pooling_mask = attention_mask
            .iter()
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>();
        let pooled = self.apply_pooling(&embeddings_array, &pooling_mask)?;

        // Process output (Matryoshka + normalization)
        self.process_output(pooled)
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
        self.resource_policy
            .validate_transformer_activations(
                self.transformer_profile,
                batch_size,
                max_seq_len,
                self.dtype,
            )
            .map_err(|error| anyhow::anyhow!("Dense batch activation preflight failed: {error}"))?;

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

/// The model's text joined before a caller's text for `role`; none without a role.
pub(super) const fn prompt_for(prompts: Prompts, role: Option<Role>) -> &'static str {
    match role {
        Some(Role::Query) => prompts.query,
        Some(Role::Document) => prompts.document,
        None => "",
    }
}

/// The prompts a sequence limit must hold: both with a role, none without one.
pub(super) const fn prompts_to_hold(prompts: Prompts, role: Option<Role>) -> [&'static str; 2] {
    match role {
        Some(_) => [prompts.query, prompts.document],
        None => ["", ""],
    }
}
