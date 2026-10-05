//! Tokenization abstraction layer.
//!
//! This module provides a wrapper around the `HuggingFace` tokenizers library
//! for loading and using BERT-compatible tokenizers.
//!
//! Tokenizer artifacts are loaded from the registered model repository. Tessera
//! does not currently redirect missing tokenizers to an assumed base model;
//! models without a complete, audited artifact path remain catalog-only.

use anyhow::{Context, Result};
use tokenizers::models::ModelWrapper;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{PostProcessor, Tokenizer as HfTokenizer};

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

/// A sequence limit that cannot hold the framing, the longer prompt and any content.
#[derive(Debug, thiserror::Error)]
#[error("InvalidPromptConfiguration: sequence limit {limit} must exceed special-token count {special_tokens} plus prompt tokens {prompt_tokens}")]
pub struct PromptConfigurationError {
    /// Configured maximum sequence length.
    pub limit: usize,
    /// Special tokens added to a single sequence.
    pub special_tokens: usize,
    /// Tokens in the longer of the model's prompts.
    pub prompt_tokens: usize,
}

/// Splits on white space before a lone `Metaspace` on a Unigram model.
///
/// SentencePiece references strip edge spaces and collapse repeats before
/// adding the `▁` marker; a `tokenizer.json` declaring `Metaspace` alone keeps a
/// lone `▁` for them instead. Every other tokenizer is left as loaded.
pub(crate) fn split_whitespace_before_metaspace(tokenizer: &mut HfTokenizer) {
    if !matches!(tokenizer.get_model(), ModelWrapper::Unigram(_)) {
        return;
    }
    let Some(PreTokenizerWrapper::Metaspace(metaspace)) = tokenizer.get_pre_tokenizer() else {
        return;
    };
    let sequence = Sequence::new(vec![
        PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
        PreTokenizerWrapper::Metaspace(metaspace.clone()),
    ]);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(sequence)));
}

#[derive(Debug)]
pub(crate) struct WholeTokenizedInput {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) attention_mask: Vec<u32>,
    pub(crate) tokens_total: usize,
}

pub(crate) struct SpannedTokenWindow {
    pub(crate) window: TokenWindow,
    pub(crate) byte_start: usize,
    pub(crate) byte_end: usize,
}

/// Wrapper around `HuggingFace` tokenizer for BERT models.
pub struct Tokenizer {
    inner: HfTokenizer,
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
        split_whitespace_before_metaspace(&mut inner);

        inner
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;
        let pad_token_id = inner.get_padding().map(|parameters| parameters.pad_id);
        inner.with_padding(None);

        Ok(Self {
            inner,
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

    /// Checks the limit holds the special tokens, the longest prompt and one token.
    pub(crate) fn validate_cut_configuration_with(&self, prompts: &[&str]) -> Result<()> {
        self.validate_cut_configuration()?;
        let prompt_tokens = self.longest_prefix_tokens(prompts)?;
        let special_tokens = self.cut_special_tokens();
        let limit = self.resource_policy.max_sequence_tokens();
        if limit <= special_tokens + prompt_tokens {
            return Err(PromptConfigurationError {
                limit,
                special_tokens,
                prompt_tokens,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn longest_prefix_tokens(&self, prefixes: &[&str]) -> Result<usize> {
        let mut longest = 0;
        for prefix in prefixes {
            let encoding = self
                .inner
                .encode(*prefix, false)
                .map_err(|error| anyhow::anyhow!("Failed to encode prompt: {error}"))?;
            longest = longest.max(encoding.len());
        }
        Ok(longest)
    }

    pub(crate) fn cut_special_tokens(&self) -> usize {
        self.inner
            .get_post_processor()
            .map_or(0, |processor| processor.added_tokens(false))
    }

    /// Tokenises `prompt` joined directly before `text`, preserving every token.
    pub(crate) fn encode_with_prompt(
        &self,
        prompt: &str,
        text: &str,
    ) -> Result<WholeTokenizedInput> {
        self.validate_cut_configuration_with(&[prompt])?;
        self.resource_policy.validate_input_bytes(text.len())?;
        let joined = [prompt, text].concat();
        let encoding = self
            .inner
            .encode(joined.as_str(), true)
            .map_err(|error| anyhow::anyhow!("Failed to encode text: {error}"))?;
        if !text.chars().all(char::is_whitespace)
            && !encoding
                .get_special_tokens_mask()
                .iter()
                .zip(encoding.get_offsets())
                .any(|(&special, &(_, end))| special == 0 && end > prompt.len())
        {
            return Err(crate::EmbeddingRefusal::NoContentTokens.into());
        }
        let tokens_total = encoding.len();
        let limit = self.resource_policy.max_sequence_tokens();
        if tokens_total > limit {
            return Err(crate::EmbeddingRefusal::TextLongerThanModel {
                tokens_total,
                tokens_limit: limit,
            }
            .into());
        }
        self.resource_policy.validate_sequence(encoding.len())?;
        self.resource_policy.validate_batch(1, encoding.len())?;
        Ok(WholeTokenizedInput {
            token_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            tokens_total,
        })
    }

    pub(crate) fn encode_batch_with_prompt(
        &self,
        prompt: &str,
        texts: &[&str],
    ) -> Result<Vec<WholeTokenizedInput>> {
        self.validate_cut_configuration()?;
        self.resource_policy.validate_batch(texts.len(), 0)?;
        for text in texts {
            self.resource_policy.validate_input_bytes(text.len())?;
        }
        let inputs = texts
            .iter()
            .map(|text| self.encode_with_prompt(prompt, text))
            .collect::<Result<Vec<_>>>()?;
        let max_len = inputs
            .iter()
            .map(|input| input.token_ids.len())
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

        if !text.chars().all(char::is_whitespace)
            && !encoding.get_special_tokens_mask().contains(&0)
        {
            return Err(crate::EmbeddingRefusal::NoContentTokens.into());
        }

        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();

        Ok((token_ids, attention_mask))
    }

    /// Encodes text for a caller that will apply and validate a bounded
    /// transformation before allocating model tensors.
    ///
    /// Raw input bytes are still bounded here. This deliberately skips the
    /// generic sequence check so callers can form validated context windows or
    /// preserve required role-framing tokens before checking the final sequence.
    pub(crate) fn encode_for_bounded_transform(
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
        let (content_ids, _) = self.encode_for_bounded_transform(text, false)?;
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

    pub(crate) fn encode_spanned_windows(
        &self,
        prompt: &str,
        text: &str,
        config: ContextWindowConfig,
    ) -> Result<(usize, Vec<SpannedTokenWindow>)> {
        if let Some(refusal) = crate::EmbeddingRefusal::for_text(
            text,
            self.resource_policy.max_input_bytes_per_sequence(),
        ) {
            return Err(refusal.into());
        }
        let joined = [prompt, text].concat();
        let encoding = self
            .inner
            .encode(joined.as_str(), false)
            .map_err(|error| anyhow::anyhow!("Failed to tokenize window content: {error}"))?;
        let prefix_bytes = prompt.len();
        let split = encoding
            .get_offsets()
            .partition_point(|offset| offset.1 <= prefix_bytes);
        anyhow::ensure!(
            encoding.get_offsets()[..split]
                .iter()
                .all(|offset| offset.1 <= prefix_bytes)
                && encoding.get_offsets()[split..]
                    .iter()
                    .all(|offset| offset.1 > prefix_bytes),
            "Tokenizer prefix/content offsets are not contiguous"
        );
        let content_ids = &encoding.get_ids()[split..];
        if content_ids.is_empty() {
            return Err(crate::EmbeddingRefusal::NoContentTokens.into());
        }
        let offsets = encoding.get_offsets()[split..]
            .iter()
            .map(|&(start, end)| {
                (
                    start.saturating_sub(prefix_bytes),
                    end.saturating_sub(prefix_bytes),
                )
            })
            .collect::<Vec<_>>();
        let (mut prefix, suffix) = self.special_token_envelope()?;
        prefix.extend_from_slice(&encoding.get_ids()[..split]);
        let planned =
            plan_token_windows(content_ids, &prefix, &suffix, config, self.resource_policy)?;
        let capacity = config.window_tokens() - prefix.len() - suffix.len();
        let expected = if content_ids.len() <= capacity {
            1
        } else {
            1 + (content_ids.len() - capacity).div_ceil(capacity - config.overlap_tokens())
        };
        anyhow::ensure!(
            planned.len() == expected,
            "Window count does not match the content-token formula"
        );
        let last = planned.len() - 1;
        let mut previous = (0, 0);
        let mut windows = Vec::with_capacity(planned.len());
        for (index, window) in planned.iter().enumerate() {
            let byte_start = if index == 0 {
                0
            } else {
                offsets[window.content_start].0
            };
            let byte_end = if index == last {
                text.len()
            } else {
                offsets[window.content_end - 1]
                    .1
                    .max(offsets[planned[index + 1].content_start].0)
            };
            anyhow::ensure!(
                byte_start < byte_end
                    && text.is_char_boundary(byte_start)
                    && text.is_char_boundary(byte_end),
                "Window {index} has invalid UTF-8 span {byte_start}..{byte_end}"
            );
            anyhow::ensure!(
                index == 0
                    || (byte_start >= previous.0
                        && byte_end >= previous.1
                        && byte_start <= previous.1),
                "Window {index} span order or coverage is invalid"
            );
            self.resource_policy
                .validate_sequence(window.token_ids.len())?;
            self.resource_policy
                .validate_batch(1, window.token_ids.len())?;
            anyhow::ensure!(
                window.token_ids.len() <= config.window_tokens(),
                "Window token limit exceeded"
            );
            previous = (byte_start, byte_end);
            windows.push((byte_start, byte_end));
        }
        Ok((
            content_ids.len(),
            planned
                .into_iter()
                .zip(windows)
                .map(|(window, (byte_start, byte_end))| SpannedTokenWindow {
                    window,
                    byte_start,
                    byte_end,
                })
                .collect(),
        ))
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

    pub(crate) fn token_string(&self, id: u32) -> Result<String> {
        self.inner
            .id_to_token(id)
            .with_context(|| format!("Tokenizer has no token string for id {id}"))
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
