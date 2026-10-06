use crate::api::embedder::{EmbedFailure, ModelIdentity};
use crate::api::TesseraDenseBuilder;
use crate::core::embeddings::{CountedDenseEmbedding, EmbeddingOutcome, EmbeddingRefusal, Role};
use crate::core::embeddings::{
    DenseWindowEmbedding, WindowEmbeddingOutcome, WindowedDenseEmbedding,
};
use crate::core::tokenizer::SpannedTokenWindow;
use crate::core::{DenseEmbedding, DenseEncoder, Encoder};
use crate::encoding::dense::CandleDenseEncoder;
use crate::error::{Result, TesseraError};
use crate::runtime::{
    f32_output_bytes, ContextWindowConfig, JobTracker, ModelDType, ResourcePolicy, WindowExtent,
};
use std::num::NonZeroUsize;

#[cfg(test)]
mod outcome_tests;

/// Dense single-vector embedder for traditional sentence embeddings.
///
/// Produces a single pooled vector per input text via strategies like
/// CLS token, mean pooling, or max pooling. Suitable for semantic search
/// and classification tasks.
///
/// Thread-safe and can be shared across threads.
pub struct TesseraDense {
    /// Backend encoder (Candle dense encoder)
    encoder: CandleDenseEncoder,
    /// Model identifier from registry
    model_id: String,
    /// Maximum batch size for encode_batch (None = no limit)
    batch_size: Option<NonZeroUsize>,
    /// Milliseconds to sleep between batches
    yield_ms: Option<u64>,
    /// Whole-job and collected-output limits.
    resource_policy: ResourcePolicy,
    identity: ModelIdentity,
}

impl TesseraDense {
    /// Default per-window input length and content overlap for this model.
    #[must_use]
    pub const fn default_window_config(&self) -> ContextWindowConfig {
        ContextWindowConfig::new(self.identity.max_tokens, 64)
    }

    /// Returns one vector and original byte span per window, in source order.
    ///
    /// Role prefix and text are tokenized together once. Tokens reaching the text
    /// are content; wholly prefix tokens and specials are repeated per window
    /// and count toward its input limit. `None` uses the model's
    /// default configuration. Empty and byte-oversized texts are refused.
    ///
    /// # Errors
    /// Returns an error for invalid configuration, offsets, job budgets or inference.
    pub fn encode_windows(
        &self,
        text: &str,
        role: Option<Role>,
        config: Option<ContextWindowConfig>,
    ) -> Result<WindowEmbeddingOutcome> {
        self.encode_batch_windows(&[text], role, config)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                window_error(
                    "Single windowed input produced no outcome",
                    anyhow::anyhow!("Missing input outcome"),
                )
            })
    }

    /// Returns one ordered window result or named refusal per text.
    ///
    /// All input plans and collected-output limits are checked before inference.
    /// Forward groups obey both the configured batch size and the resource policy.
    ///
    /// # Errors
    /// Returns an error for invalid configuration, offsets, job budgets or inference.
    pub fn encode_batch_windows(
        &self,
        texts: &[&str],
        role: Option<Role>,
        config: Option<ContextWindowConfig>,
    ) -> Result<Vec<WindowEmbeddingOutcome>> {
        let config = config.unwrap_or_else(|| self.default_window_config());
        let batch_size = self
            .batch_size
            .map_or_else(|| self.resource_policy.max_batch_items(), NonZeroUsize::get)
            .min(self.resource_policy.max_batch_items());
        encode_windows_batch_with(
            texts,
            self.resource_policy,
            self.identity.dimensions,
            batch_size,
            |text| self.encoder.plan_spanned_windows(text, role, config),
            |windows| self.encoder.encode_spanned_window_batch(windows),
        )
    }

    /// Measures a text for windows of `window_tokens` model inputs without
    /// planning or inference: its content tokens and one window's content room,
    /// as [`Self::encode_batch_windows`] would plan them.
    ///
    /// # Errors
    /// Returns an error wrapping the [`EmbeddingRefusal`] for empty, byte-oversized
    /// or content-free text, or a tokenizer error.
    pub fn window_extent(
        &self,
        text: &str,
        role: Option<Role>,
        window_tokens: usize,
    ) -> Result<WindowExtent> {
        self.encoder
            .measure_spanned_windows(text, role, window_tokens)
            .map_err(|source| window_error("Failed to measure dense windows", source))
    }

    /// Create a new dense embedder with default configuration.
    ///
    /// This is the simplest way to create a dense embedder - it automatically:
    /// - Looks up the model in the registry
    /// - Selects the best available device (Metal > CUDA > CPU)
    /// - Downloads the model from `HuggingFace` if needed
    /// - Initializes the encoder with appropriate pooling strategy
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry (e.g., "bge-base-en-v1.5", "nomic-embed-v1.5")
    ///
    /// # Returns
    ///
    /// Initialized embedder ready for use.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - Model is not a dense model type
    /// - Model cannot be downloaded or loaded
    /// - Device initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraDense;
    ///
    /// let embedder = TesseraDense::new("bge-base-en-v1.5")?;
    /// let embedding = embedder.encode("What is machine learning?")?;
    /// assert_eq!(embedding.dim(), 768);
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        // Use builder with just model ID
        TesseraDenseBuilder::new().model(model_id).build()
    }

    /// Create a builder for advanced configuration.
    ///
    /// Use this for advanced use cases like:
    /// - Specifying a custom device
    /// - Setting Matryoshka dimensions
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::TesseraDense;
    /// use candle_core::Device;
    ///
    /// let embedder = TesseraDense::builder()
    ///     .model("bge-base-en-v1.5")
    ///     .device(Device::Cpu)
    ///     .build()?;
    /// ```
    #[must_use]
    pub const fn builder() -> TesseraDenseBuilder {
        TesseraDenseBuilder::new()
    }

    /// Internal constructor with batch options.
    pub(crate) const fn from_encoder_with_options(
        encoder: CandleDenseEncoder,
        model_id: String,
        batch_size: Option<NonZeroUsize>,
        yield_ms: Option<u64>,
        resource_policy: ResourcePolicy,
        identity: ModelIdentity,
    ) -> Self {
        Self {
            encoder,
            model_id,
            batch_size,
            yield_ms,
            resource_policy,
            identity,
        }
    }

    /// Returns the exact model this embedder loaded, as captured when it was built.
    #[must_use]
    pub const fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    /// Returns the digest of the installed manifest used to load this embedder.
    ///
    /// Returns `None` when built without an installed model directory.
    #[must_use]
    pub fn installed_manifest_sha256(&self) -> Option<&str> {
        self.identity.manifest_sha256.as_deref()
    }

    /// Encode a single text into a dense embedding.
    ///
    /// Returns a single pooled vector representing the entire input text.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode
    ///
    /// # Returns
    ///
    /// `DenseEmbedding` containing the pooled embedding vector.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tokenization fails
    /// - Model inference fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedding = embedder.encode("What is machine learning?")?;
    /// println!("Encoded to {} dimensions", embedding.dim());
    /// ```
    pub fn encode(&self, text: &str) -> Result<DenseEmbedding> {
        let mut tracker = JobTracker::new(self.resource_policy);
        tracker
            .admit_input(text.len())
            .map_err(|error| resource_error("Dense input exceeds job limits", error))?;
        let embedding =
            <CandleDenseEncoder as Encoder>::encode(&self.encoder, text).map_err(|e| {
                TesseraError::EncodingError {
                    context: format!("Failed to encode text ({} UTF-8 bytes)", text.len()),
                    source: e,
                }
            })?;
        tracker
            .retain_output(f32_output_bytes(embedding.dim()))
            .map_err(|error| resource_error("Dense output exceeds collection limit", error))?;
        Ok(embedding)
    }

    /// Returns the text's vector and token counts, or a named input refusal.
    ///
    /// With a role, the model's text for it is joined directly before `text` and
    /// is never cut. It and the special tokens count toward the sequence limit and
    /// the token count; refusals and the byte limit look at `text` alone.
    /// Without a role, `text` is embedded as it stands.
    ///
    /// # Errors
    /// Returns a configuration error if the limit cannot hold special tokens,
    /// the longer of the model's two texts (with a role) and one content token, or an error
    /// for job limits or inference. Empty or whitespace-only input is refused;
    /// oversized nonempty input is refused.
    pub fn encode_outcome(&self, text: &str, role: Option<Role>) -> Result<EmbeddingOutcome> {
        encode_outcome_batch_with(
            &[text],
            (self.encoder.validate_cut_configuration(role), |text| {
                self.encoder.input_refusal(text, role)
            }),
            self.resource_policy,
            self.identity.dimensions,
            NonZeroUsize::MIN,
            None,
            |_| {
                self.encoder
                    .encode_outcome(text, role)
                    .map(|value| vec![value])
            },
        )?
        .into_iter()
        .next()
        .ok_or_else(|| TesseraError::EncodingError {
            context: "Missing dense outcome outcome".to_string(),
            source: anyhow::anyhow!("Single input produced no outcome"),
        })
    }

    /// Returns one vector or named input refusal per text, in input order.
    ///
    /// Each chunk uses sequential model forwards on unpadded, complete inputs. One
    /// role applies to every text, joined as in [`Self::encode_outcome`].
    ///
    /// # Errors
    /// Returns a configuration error before any item is embedded if the sequence
    /// limit cannot hold special tokens, the longer model text and content. Job, batch, output and
    /// inference limits apply to accepted items; refused items use none of them.
    pub fn encode_batch_outcomes(
        &self,
        texts: &[&str],
        role: Option<Role>,
    ) -> Result<Vec<EmbeddingOutcome>> {
        let batch_size = self
            .batch_size
            .or_else(|| NonZeroUsize::new(self.resource_policy.max_batch_items()))
            .unwrap_or(NonZeroUsize::MIN);
        encode_outcome_batch_with(
            texts,
            (self.encoder.validate_cut_configuration(role), |text| {
                self.encoder.input_refusal(text, role)
            }),
            self.resource_policy,
            self.identity.dimensions,
            batch_size,
            self.yield_ms.map(std::time::Duration::from_millis),
            |accepted| self.encoder.encode_batch_outcomes(accepted, role),
        )
    }

    /// Encodes a long input through bounded overlapping windows.
    ///
    /// This is a weighted aggregation mode, not native full-context attention.
    pub fn encode_windowed(
        &self,
        text: &str,
        config: ContextWindowConfig,
    ) -> Result<DenseEmbedding> {
        let mut tracker = JobTracker::new(self.resource_policy);
        tracker
            .admit_input(text.len())
            .map_err(|error| resource_error("Dense windowed input exceeds job limits", error))?;
        let embedding = self
            .encoder
            .encode_windowed(text, config)
            .map_err(|source| TesseraError::EncodingError {
                context: format!(
                    "Failed to encode windowed text ({} UTF-8 bytes)",
                    text.len()
                ),
                source,
            })?;
        tracker
            .retain_output(f32_output_bytes(embedding.dim()))
            .map_err(|error| resource_error("Dense output exceeds collection limit", error))?;
        Ok(embedding)
    }

    /// Encode multiple texts in a batch.
    ///
    /// Uses backend batching rather than one facade call per item. Throughput
    /// depends on input lengths, device, chunk size, and resource policy; no
    /// fixed speedup is guaranteed.
    ///
    /// If `batch_size` was configured via the builder, texts are processed in
    /// chunks with optional yielding between batches to prevent GPU saturation.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to encode
    ///
    /// # Returns
    ///
    /// Vector of `DenseEmbedding`, one per input text.
    ///
    /// # Errors
    ///
    /// Returns error if encoding any text fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embeddings = embedder.encode_batch(&[
    ///     "First document",
    ///     "Second document",
    /// ])?;
    /// ```
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>> {
        let mut tracker = JobTracker::new(self.resource_policy);
        for text in texts {
            tracker
                .admit_input(text.len())
                .map_err(|error| resource_error("Dense batch input exceeds job limits", error))?;
        }

        // No explicit size means the policy's batch ceiling: one inference admission per
        // chunk, never one per text.
        let batch_size = self
            .batch_size
            .or_else(|| NonZeroUsize::new(self.resource_policy.max_batch_items()))
            .unwrap_or(NonZeroUsize::MIN);

        // Process in chunks with optional yielding
        let mut all_embeddings = Vec::with_capacity(texts.len());
        let yield_duration = self.yield_ms.map(std::time::Duration::from_millis);

        for (chunk_idx, chunk) in texts.chunks(batch_size.get()).enumerate() {
            // Yield between batches (not before the first one)
            if chunk_idx > 0 {
                if let Some(duration) = yield_duration {
                    std::thread::sleep(duration);
                }
            }

            // Process this chunk
            let chunk_embeddings =
                <CandleDenseEncoder as Encoder>::encode_batch(&self.encoder, chunk).map_err(
                    |e| TesseraError::EncodingError {
                        context: format!(
                            "Failed to encode batch chunk {} ({} texts)",
                            chunk_idx,
                            chunk.len()
                        ),
                        source: e,
                    },
                )?;

            for embedding in &chunk_embeddings {
                tracker
                    .retain_output(f32_output_bytes(embedding.dim()))
                    .map_err(|error| {
                        resource_error("Dense batch output exceeds collection limit", error)
                    })?;
            }

            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }

    /// Encodes a logical job in bounded chunks and yields each result to a callback.
    ///
    /// Streamed outputs are not accumulated by Tessera, so the output limit is
    /// applied to each individual result. Item and aggregate input-byte limits
    /// still apply to the complete job.
    pub fn encode_stream<'a, I, F>(&self, texts: I, mut consume: F) -> Result<()>
    where
        I: IntoIterator<Item = &'a str>,
        F: FnMut(DenseEmbedding) -> Result<()>,
    {
        let mut tracker = JobTracker::new(self.resource_policy);
        let chunk_size = self.batch_size.unwrap_or(NonZeroUsize::MIN).get();
        let mut chunk = Vec::with_capacity(chunk_size);
        let mut chunk_index = 0_usize;

        for text in texts {
            tracker
                .admit_input(text.len())
                .map_err(|error| resource_error("Dense stream input exceeds job limits", error))?;
            chunk.push(text);
            if chunk.len() == chunk_size {
                self.consume_dense_chunk(&chunk, chunk_index, &tracker, &mut consume)?;
                chunk.clear();
                chunk_index = chunk_index.saturating_add(1);
            }
        }
        if !chunk.is_empty() {
            self.consume_dense_chunk(&chunk, chunk_index, &tracker, &mut consume)?;
        }
        Ok(())
    }

    fn consume_dense_chunk<F>(
        &self,
        chunk: &[&str],
        chunk_index: usize,
        tracker: &JobTracker,
        consume: &mut F,
    ) -> Result<()>
    where
        F: FnMut(DenseEmbedding) -> Result<()>,
    {
        if chunk_index > 0 {
            if let Some(duration) = self.yield_ms.map(std::time::Duration::from_millis) {
                std::thread::sleep(duration);
            }
        }
        let embeddings = <CandleDenseEncoder as Encoder>::encode_batch(&self.encoder, chunk)
            .map_err(|source| TesseraError::EncodingError {
                context: format!("Failed to encode dense stream chunk {chunk_index}"),
                source,
            })?;
        for embedding in embeddings {
            tracker
                .validate_streamed_output(f32_output_bytes(embedding.dim()))
                .map_err(|error| {
                    resource_error("Dense streamed output exceeds per-item limit", error)
                })?;
            consume(embedding)?;
        }
        Ok(())
    }

    /// Compute cosine similarity between two texts.
    ///
    /// Convenience method that encodes both texts and computes cosine similarity.
    /// For normalized embeddings, this is equivalent to dot product.
    ///
    /// # Arguments
    ///
    /// * `text_a` - First text
    /// * `text_b` - Second text
    ///
    /// # Returns
    ///
    /// Similarity score (higher = more similar). Typically in range [-1, 1],
    /// or [0, 1] for normalized embeddings.
    ///
    /// # Errors
    ///
    /// Returns error if encoding or similarity computation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let score = embedder.similarity(
    ///     "What is machine learning?",
    ///     "Machine learning is a subset of AI"
    /// )?;
    /// println!("Similarity: {:.4}", score);
    /// ```
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
        let emb_a = self.encode(text_a)?;
        let emb_b = self.encode(text_b)?;

        // Compute cosine similarity (dot product for normalized embeddings)
        let dot_product: f32 = emb_a
            .values()
            .iter()
            .zip(emb_b.values().iter())
            .map(|(a, b)| a * b)
            .sum();

        Ok(dot_product)
    }

    /// Get the embedding dimension.
    ///
    /// Returns the dimensionality of the output embedding vector.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Embedding dimension: {}", embedder.dimension());
    /// ```
    pub fn dimension(&self) -> usize {
        self.encoder.embedding_dim()
    }

    /// Get the model identifier.
    ///
    /// Returns the model ID from the registry (e.g., "bge-base-en-v1.5").
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Using model: {}", embedder.model());
    /// ```
    pub fn model(&self) -> &str {
        &self.model_id
    }

    /// Parameter dtype selected for this model instance.
    #[must_use]
    pub const fn model_dtype(&self) -> ModelDType {
        self.encoder.model_dtype()
    }
}

fn encode_outcome_batch_with<F, C>(
    texts: &[&str],
    preflight: (anyhow::Result<()>, C),
    policy: ResourcePolicy,
    dimensions: usize,
    batch_size: NonZeroUsize,
    yield_duration: Option<std::time::Duration>,
    mut encode: F,
) -> Result<Vec<EmbeddingOutcome>>
where
    F: FnMut(&[&str]) -> anyhow::Result<Vec<CountedDenseEmbedding>>,
    C: Fn(&str) -> anyhow::Result<Option<EmbeddingRefusal>>,
{
    let (configuration, classify) = preflight;
    configuration.map_err(EmbedFailure::from_cut_configuration)?;
    let mut tracker = JobTracker::new(policy);
    let mut refusals = Vec::with_capacity(texts.len());
    let mut accepted = Vec::new();
    let mut positions = Vec::new();
    for (position, &text) in texts.iter().enumerate() {
        let refusal = match EmbeddingRefusal::for_text(text, policy.max_input_bytes_per_sequence())
        {
            Some(refusal) => Some(refusal),
            None => classify(text).map_err(|source| TesseraError::EncodingError {
                context: format!("Failed to classify input {position}"),
                source,
            })?,
        };
        if refusal.is_none() {
            tracker.admit_input(text.len()).map_err(|error| {
                resource_error("Dense outcome batch input exceeds job limits", error)
            })?;
            accepted.push(text);
            positions.push(position);
        }
        refusals.push(refusal);
    }
    let mut embeddings = Vec::with_capacity(accepted.len());
    for (index, (chunk, chunk_positions)) in accepted
        .chunks(batch_size.get())
        .zip(positions.chunks(batch_size.get()))
        .enumerate()
    {
        if index > 0 {
            if let Some(duration) = yield_duration {
                std::thread::sleep(duration);
            }
        }
        let results = encode(chunk).map_err(|source| TesseraError::EncodingError {
            context: format!(
                "Failed to encode complete batch chunk {index} ({} texts)",
                chunk.len()
            ),
            source: input_positions(source, chunk_positions),
        })?;
        if results.len() != chunk.len() {
            return Err(TesseraError::EncodingError {
                context: "Dense outcome batch outcome count mismatch".to_string(),
                source: anyhow::anyhow!(
                    "Expected {} vectors, received {}",
                    chunk.len(),
                    results.len()
                ),
            });
        }
        for (embedding, &position) in results.iter().zip(chunk_positions) {
            if embedding.dim() != dimensions {
                return Err(TesseraError::EncodingError {
                    context: "Dense outcome output is invalid".to_string(),
                    source: anyhow::Error::new(EmbedFailure::OutputInvalid {
                        index: position,
                        reason: format!(
                            "vector has {} values, expected {dimensions}",
                            embedding.dim()
                        ),
                    }),
                });
            }
            tracker
                .retain_output(f32_output_bytes(embedding.dim()))
                .map_err(|error| {
                    resource_error("Dense outcome batch output exceeds collection limit", error)
                })?;
        }
        embeddings.extend(results);
    }
    let mut embeddings = embeddings.into_iter();
    refusals
        .into_iter()
        .map(|refusal| {
            refusal.map_or_else(
                || {
                    embeddings
                        .next()
                        .map(EmbeddingOutcome::Embedded)
                        .ok_or_else(|| TesseraError::EncodingError {
                            context: "Missing dense outcome outcome".to_string(),
                            source: anyhow::anyhow!("Accepted input produced no vector"),
                        })
                },
                |refusal| Ok(EmbeddingOutcome::Refused(refusal)),
            )
        })
        .collect()
}

/// Renumbers an output failure from its position in a chunk to the caller's input.
fn input_positions(source: anyhow::Error, positions: &[usize]) -> anyhow::Error {
    match source.downcast_ref::<EmbedFailure>() {
        Some(EmbedFailure::OutputInvalid { index, reason }) => match positions.get(*index) {
            Some(&position) => anyhow::Error::new(EmbedFailure::OutputInvalid {
                index: position,
                reason: reason.clone(),
            }),
            None => source,
        },
        _ => source,
    }
}

fn resource_error(context: &str, error: crate::runtime::ResourcePolicyError) -> TesseraError {
    TesseraError::EncodingError {
        context: context.to_string(),
        source: anyhow::Error::new(error),
    }
}

enum PreparedWindowInput {
    Refused(EmbeddingRefusal),
    Accepted(usize, Vec<SpannedTokenWindow>),
}

fn encode_windows_batch_with<P, F>(
    texts: &[&str],
    policy: ResourcePolicy,
    dimensions: usize,
    batch_size: usize,
    plan: P,
    mut encode: F,
) -> Result<Vec<WindowEmbeddingOutcome>>
where
    P: Fn(&str) -> anyhow::Result<(usize, Vec<SpannedTokenWindow>)>,
    F: FnMut(&[&SpannedTokenWindow]) -> anyhow::Result<Vec<CountedDenseEmbedding>>,
{
    let batch_size = NonZeroUsize::new(batch_size).ok_or_else(|| {
        window_error(
            "Window batch limit must be positive",
            anyhow::anyhow!("Zero batch size"),
        )
    })?;
    let mut tracker = JobTracker::new(policy);
    let mut inputs = Vec::with_capacity(texts.len());
    let mut input_bytes = 0_usize;
    let mut window_count = 0_usize;
    for &text in texts {
        if let Some(refusal) =
            EmbeddingRefusal::for_text(text, policy.max_input_bytes_per_sequence())
        {
            inputs.push(PreparedWindowInput::Refused(refusal));
            continue;
        }
        let (total, windows) = match plan(text) {
            Ok(plan) => plan,
            Err(source) => {
                if let Some(refusal) = source.downcast_ref::<EmbeddingRefusal>().copied() {
                    inputs.push(PreparedWindowInput::Refused(refusal));
                    continue;
                }
                return Err(window_error("Failed to plan dense windows", source));
            }
        };
        if total == 0 || windows.is_empty() {
            return Err(window_error(
                "Window planner produced no content",
                anyhow::anyhow!("Empty content plan"),
            ));
        }
        tracker
            .admit_input(text.len())
            .map_err(|error| resource_error("Window input exceeds job limits", error))?;
        input_bytes = input_bytes.saturating_add(text.len());
        window_count = window_count.saturating_add(windows.len());
        policy
            .validate_job(window_count, input_bytes)
            .map_err(|error| resource_error("Windows exceed job limits", error))?;
        for _ in &windows {
            tracker
                .retain_output(f32_output_bytes(dimensions))
                .map_err(|error| resource_error("Window outputs exceed collection limit", error))?;
        }
        inputs.push(PreparedWindowInput::Accepted(total, windows));
    }
    let windows = inputs
        .iter()
        .flat_map(|input| match input {
            PreparedWindowInput::Accepted(_, windows) => windows.as_slice(),
            PreparedWindowInput::Refused(_) => &[],
        })
        .collect::<Vec<_>>();
    let mut embeddings = Vec::with_capacity(windows.len());
    for chunk in windows.chunks(batch_size.get()) {
        let values = encode(chunk)
            .map_err(|source| window_error("Dense window inference failed", source))?;
        if values.len() != chunk.len()
            || values.iter().zip(chunk).any(|(value, input)| {
                value.dim() != dimensions || value.tokens_total() != input.window.token_ids.len()
            })
        {
            return Err(window_error(
                "Dense window output shape or count mismatch",
                anyhow::anyhow!("Invalid per-window output"),
            ));
        }
        embeddings.extend(values);
    }
    let mut embeddings = embeddings.into_iter();
    inputs
        .into_iter()
        .map(|input| match input {
            PreparedWindowInput::Refused(refusal) => Ok(WindowEmbeddingOutcome::Refused(refusal)),
            PreparedWindowInput::Accepted(tokens_total, windows) => {
                let windows = windows
                    .into_iter()
                    .map(|input| {
                        embeddings
                            .next()
                            .map(|embedding| DenseWindowEmbedding {
                                embedding,
                                byte_start: input.byte_start,
                                byte_end: input.byte_end,
                                content_tokens: input.window.content_end
                                    - input.window.content_start,
                            })
                            .ok_or_else(|| {
                                window_error(
                                    "Window vector is missing",
                                    anyhow::anyhow!("Missing per-window output"),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(WindowEmbeddingOutcome::Embedded(WindowedDenseEmbedding {
                    tokens_total,
                    windows,
                }))
            }
        })
        .collect()
}

fn window_error(context: &str, source: anyhow::Error) -> TesseraError {
    TesseraError::EncodingError {
        context: context.to_string(),
        source,
    }
}

#[cfg(test)]
mod window_tests;
