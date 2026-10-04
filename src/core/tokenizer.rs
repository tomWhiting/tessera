//! Tokenization abstraction layer.
//!
//! This module provides a wrapper around the `HuggingFace` tokenizers library
//! for loading and using BERT-compatible tokenizers.
//!
//! Tokenizer artifacts are loaded from the registered model repository. Tessera
//! does not currently redirect missing tokenizers to an assumed base model;
//! models without a complete, audited artifact path remain catalog-only.

use anyhow::{Context, Result};
use tokenizers::{PostProcessor, Tokenizer as HfTokenizer, TruncationParams};

use crate::models::loader::ModelFileResolver;
use crate::runtime::{plan_token_windows, ContextWindowConfig, ResourcePolicy, TokenWindow};

#[cfg(test)]
pub(crate) mod tests;

type TokenizedInput = (Vec<u32>, Vec<u32>);
type UnpaddedBatch = (Vec<TokenizedInput>, usize);

/// A sequence limit that cannot hold the required framing and any content.
#[derive(Debug, thiserror::Error)]
#[error("InvalidCutConfiguration: sequence limit {limit} must exceed special-token count {special_tokens}")]
pub struct CutConfigurationError {
    /// Configured maximum sequence length.
    pub limit: usize,
    /// Special tokens added to a single sequence.
    pub special_tokens: usize,
}

#[derive(Debug)]
pub(crate) struct CutTokenizedInput {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) attention_mask: Vec<u32>,
    pub(crate) tokens_total: usize,
    pub(crate) cut: bool,
}

impl CutTokenizedInput {
    pub(crate) fn tokens_read(&self) -> usize {
        self.token_ids.len()
    }
}

/// Wrapper around `HuggingFace` tokenizer for BERT models.
pub struct Tokenizer {
    inner: HfTokenizer,
    truncating: Option<HfTokenizer>,
    resource_policy: ResourcePolicy,
    pad_token_id: Option<u32>,
}

impl Tokenizer {
    /// Loads a tokenizer from the `HuggingFace` Hub.
    ///
    /// The model must be registered with an immutable Hub revision. Set
    /// `TESSERA_OFFLINE=1` to permit pinned cache lookup only.
    ///
    /// # Arguments
    /// * `model_name` - Name of the model on `HuggingFace` Hub (e.g., "bert-base-uncased")
    ///
    /// # Returns
    /// A new Tokenizer instance
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        Self::from_pretrained_with_policy(model_name, ResourcePolicy::default())
    }

    /// Loads a tokenizer with explicit resource limits.
    ///
    /// Any truncation configured in the tokenizer artifact is disabled so
    /// over-limit inputs are reported rather than silently shortened.
    pub fn from_pretrained_with_policy(
        model_name: &str,
        resource_policy: ResourcePolicy,
    ) -> Result<Self> {
        let model = crate::models::registry::get_model_by_hf_id(model_name).ok_or_else(|| {
            anyhow::anyhow!("Model '{model_name}' is not registered for tokenizer loading")
        })?;
        let files = ModelFileResolver::new(model)?;
        Self::from_model_files_with_policy(&files, resource_policy)
    }

    /// Loads `tokenizer.json` through the shared pinned artifact resolver.
    pub(crate) fn from_model_files_with_policy(
        files: &ModelFileResolver,
        resource_policy: ResourcePolicy,
    ) -> Result<Self> {
        let tokenizer_path = files.get(files.model().tokenizer_file)?;
        let mut inner = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|error| anyhow::anyhow!("Failed to load tokenizer: {error}"))
            .with_context(|| format!("Loading tokenizer from {}", tokenizer_path.display()))?;

        inner
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;
        let pad_token_id = inner.get_padding().map(|parameters| parameters.pad_id);
        inner.with_padding(None);

        Ok(Self {
            inner,
            truncating: None,
            resource_policy,
            pad_token_id,
        })
    }

    /// Encodes text into token IDs.
    ///
    /// # Arguments
    /// * `text` - The text to tokenize
    /// * `add_special_tokens` - Whether to add special tokens like `[CLS]` and `[SEP]`
    ///
    /// # Returns
    /// A tuple of (`token_ids`, `attention_mask`)
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<(Vec<u32>, Vec<u32>)> {
        self.resource_policy
            .validate_input_bytes(text.len())
            .map_err(anyhow::Error::new)?;
        let (token_ids, attention_mask) = self.encode_unchecked(text, add_special_tokens)?;

        self.resource_policy
            .validate_sequence(token_ids.len())
            .map_err(anyhow::Error::new)?;
        self.resource_policy
            .validate_batch(1, token_ids.len())
            .map_err(anyhow::Error::new)?;

        Ok((token_ids, attention_mask))
    }

    pub(crate) fn validate_cut_configuration(&self) -> Result<()> {
        let special_tokens = self.cut_special_tokens();
        let limit = self.resource_policy.max_sequence_tokens();
        if limit <= special_tokens {
            return Err(CutConfigurationError {
                limit,
                special_tokens,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn prepare_cut(&mut self) -> Result<()> {
        if self.resource_policy.max_sequence_tokens() <= self.cut_special_tokens() {
            // Cut calls report invalid limits; ordinary encoding retains its behavior.
            return Ok(());
        }
        let mut truncating = self.inner.clone();
        truncating
            .with_truncation(Some(TruncationParams {
                max_length: self.resource_policy.max_sequence_tokens(),
                ..TruncationParams::default()
            }))
            .map_err(|error| anyhow::anyhow!("Failed to configure cut tokenization: {error}"))?;
        self.truncating = Some(truncating);
        Ok(())
    }

    fn cut_special_tokens(&self) -> usize {
        self.inner
            .get_post_processor()
            .map_or(0, |processor| processor.added_tokens(false))
    }

    pub(crate) fn encode_cut(&self, text: &str) -> Result<CutTokenizedInput> {
        self.validate_cut_configuration()?;
        self.resource_policy.validate_input_bytes(text.len())?;
        let mut encoding = self
            .inner
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("Failed to encode text: {error}"))?;
        let tokens_total = encoding.len();
        let limit = self.resource_policy.max_sequence_tokens();
        let cut = tokens_total > limit;
        if cut {
            drop(encoding);
            encoding = self
                .truncating
                .as_ref()
                .context("Cut tokenizer was not prepared during model loading")?
                .encode(text, true)
                .map_err(|error| anyhow::anyhow!("Failed to encode cut text: {error}"))?;
        }
        self.resource_policy.validate_sequence(encoding.len())?;
        self.resource_policy.validate_batch(1, encoding.len())?;
        Ok(CutTokenizedInput {
            token_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            tokens_total,
            cut,
        })
    }

    pub(crate) fn encode_batch_cut(&self, texts: &[&str]) -> Result<Vec<CutTokenizedInput>> {
        self.validate_cut_configuration()?;
        self.resource_policy.validate_batch(texts.len(), 0)?;
        for text in texts {
            self.resource_policy.validate_input_bytes(text.len())?;
        }
        let inputs = texts
            .iter()
            .map(|text| self.encode_cut(text))
            .collect::<Result<Vec<_>>>()?;
        let max_len = inputs
            .iter()
            .map(CutTokenizedInput::tokens_read)
            .max()
            .unwrap_or(0);
        self.resource_policy.validate_batch(texts.len(), max_len)?;
        Ok(inputs)
    }

    fn encode_unchecked(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {e}"))
            .context("Encoding text with tokenizer")?;

        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();

        Ok((token_ids, attention_mask))
    }

    /// Encodes text for a caller that will apply and validate a bounded
    /// transformation before allocating model tensors.
    ///
    /// Raw input bytes are still bounded here. This deliberately skips the
    /// generic sequence check so callers can form validated context windows or
    /// preserve required role-framing tokens while truncating content.
    pub(crate) fn encode_for_bounded_truncation(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        self.resource_policy
            .validate_input_bytes(text.len())
            .map_err(anyhow::Error::new)?;
        self.encode_unchecked(text, add_special_tokens)
    }

    /// Tokenizes a long input into validated overlapping model inputs.
    ///
    /// The caller must aggregate the window outputs according to its
    /// representation semantics. This method bounds raw input bytes and every
    /// planned forward before returning any tensor-ready token IDs.
    pub(crate) fn encode_windows(
        &self,
        text: &str,
        config: ContextWindowConfig,
    ) -> Result<Vec<TokenWindow>> {
        let (content_ids, _) = self.encode_for_bounded_truncation(text, false)?;
        let (special_prefix, special_suffix) = self.special_token_envelope()?;
        let windows = plan_token_windows(
            &content_ids,
            &special_prefix,
            &special_suffix,
            config,
            self.resource_policy,
        )
        .map_err(anyhow::Error::new)?;

        for window in &windows {
            self.resource_policy
                .validate_sequence(window.token_ids.len())
                .map_err(anyhow::Error::new)?;
            self.resource_policy
                .validate_batch(1, window.token_ids.len())
                .map_err(anyhow::Error::new)?;
        }
        Ok(windows)
    }

    fn special_token_envelope(&self) -> Result<(Vec<u32>, Vec<u32>)> {
        const PROBE: &str = "tessera";
        let (content, _) = self.encode_unchecked(PROBE, false)?;
        let (wrapped, _) = self.encode_unchecked(PROBE, true)?;
        anyhow::ensure!(
            !content.is_empty(),
            "Tokenizer produced no content tokens for special-token probe"
        );
        let start = wrapped
            .windows(content.len())
            .position(|candidate| candidate == content)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Tokenizer special-token processor does not preserve a contiguous single-sequence payload"
                )
            })?;
        let end = start + content.len();
        Ok((wrapped[..start].to_vec(), wrapped[end..].to_vec()))
    }

    /// Decodes token IDs back into text.
    ///
    /// # Arguments
    /// * `token_ids` - The token IDs to decode
    /// * `skip_special_tokens` - Whether to skip special tokens like `[CLS]`, `[SEP]`, and `[PAD]`
    ///
    /// # Returns
    /// The decoded text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {e}"))
            .context("Decoding token IDs")
    }

    /// Returns the vocabulary size of the tokenizer.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    /// Resolves a token ID from the loaded tokenizer artifact.
    ///
    /// ColBERT preprocessing uses this instead of assuming conventional BERT
    /// IDs for role markers, masks, padding, and punctuation.
    pub(crate) fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Returns the hard limits enforced by this tokenizer.
    #[must_use]
    pub const fn resource_policy(&self) -> ResourcePolicy {
        self.resource_policy
    }

    /// Encodes multiple texts into token IDs with padding.
    ///
    /// All sequences are padded to the length of the longest sequence in the batch.
    /// This enables efficient batch processing in neural networks.
    ///
    /// # Arguments
    /// * `texts` - Slice of texts to tokenize
    /// * `add_special_tokens` - Whether to add special tokens like `[CLS]` and `[SEP]`
    ///
    /// # Returns
    /// A vector of tuples (`token_ids`, `attention_mask`), one per input text.
    /// All sequences have the same length (padded to max).
    ///
    /// # Example
    /// ```ignore
    /// let tokenizer = Tokenizer::from_pretrained("bert-base-uncased")?;
    /// let batch = tokenizer.encode_batch(&["Hello", "Hello world"], true)?;
    ///
    /// // Second sequence is longer, so first is padded
    /// assert_eq!(batch[0].0.len(), batch[1].0.len());
    /// ```
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
        let (all_tokenized, max_len) = self.tokenize_batch_unpadded(texts, add_special_tokens)?;

        let pad_token_id = self.padding_token_id()?;

        // Pad all sequences to max length
        let mut padded_batch = Vec::with_capacity(texts.len());
        for (mut token_ids, mut attention_mask) in all_tokenized {
            let current_len = token_ids.len();
            if current_len < max_len {
                let padding_len = max_len - current_len;
                token_ids.extend(vec![pad_token_id; padding_len]);
                attention_mask.extend(vec![0; padding_len]);
            }
            padded_batch.push((token_ids, attention_mask));
        }

        Ok(padded_batch)
    }

    fn padding_token_id(&self) -> Result<u32> {
        if let Some(pad_token_id) = self.pad_token_id {
            return Ok(pad_token_id);
        }
        ["[PAD]", "<pad>"]
            .into_iter()
            .find_map(|token| self.inner.token_to_id(token))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Tokenizer artifact does not define padding metadata or a recognized pad token"
                )
            })
    }

    fn tokenize_batch_unpadded(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<UnpaddedBatch> {
        if texts.is_empty() {
            return Ok((Vec::new(), 0));
        }

        // Reject an oversized item count before doing tokenization work.
        self.resource_policy
            .validate_batch(texts.len(), 0)
            .map_err(anyhow::Error::new)?;
        for text in texts {
            self.resource_policy
                .validate_input_bytes(text.len())
                .map_err(anyhow::Error::new)?;
        }

        let mut all_tokenized = Vec::with_capacity(texts.len());
        let mut max_len = 0;
        for text in texts {
            let (token_ids, attention_mask) = self.encode_unchecked(text, add_special_tokens)?;
            self.resource_policy
                .validate_sequence(token_ids.len())
                .map_err(anyhow::Error::new)?;
            max_len = max_len.max(token_ids.len());
            all_tokenized.push((token_ids, attention_mask));
        }

        // Validate the padded tensor shape before allocating any padding or tensors.
        self.resource_policy
            .validate_batch(texts.len(), max_len)
            .map_err(anyhow::Error::new)?;

        Ok((all_tokenized, max_len))
    }
}
