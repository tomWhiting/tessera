//! Tokenization abstraction layer.
//!
//! This module provides a wrapper around the `HuggingFace` tokenizers library
//! for loading and using BERT-compatible tokenizers.
//!
//! Tokenizer artifacts are loaded from the registered model repository. Tessera
//! does not currently redirect missing tokenizers to an assumed base model;
//! models without a complete, audited artifact path remain catalog-only.

use anyhow::{Context, Result};
use tokenizers::Tokenizer as HfTokenizer;

use crate::runtime::ResourcePolicy;

#[cfg(test)]
mod tests;

type TokenizedInput = (Vec<u32>, Vec<u32>);
type UnpaddedBatch = (Vec<TokenizedInput>, usize);

/// Wrapper around `HuggingFace` tokenizer for BERT models.
pub struct Tokenizer {
    inner: HfTokenizer,
    resource_policy: ResourcePolicy,
}

impl Tokenizer {
    /// Loads a tokenizer from the `HuggingFace` Hub.
    ///
    /// Uses the tokenizers crate's built-in `from_pretrained` which handles
    /// different tokenizer formats automatically (tokenizer.json, vocab.json + merges.txt, etc.)
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
        // Use tokenizers crate's built-in from_pretrained (requires "http" feature)
        // This handles different tokenizer formats: tokenizer.json, vocab.json + merges.txt, etc.
        let mut inner = match HfTokenizer::from_pretrained(model_name, None) {
            Ok(inner) => inner,
            Err(e) => {
                // A redirect is used only when an audited model explicitly declares one.
                Self::get_base_model_tokenizer(model_name).map_or_else(
                    || {
                        Err(anyhow::anyhow!("Failed to load tokenizer: {e}"))
                            .with_context(|| format!("Loading tokenizer for model: {model_name}"))
                    },
                    |base_model| {
                        HfTokenizer::from_pretrained(base_model, None)
                        .map_err(|e2| anyhow::anyhow!(
                            "Failed to load tokenizer from {model_name} or base model {base_model}: {e2}"
                        ))
                        .with_context(|| format!("Loading tokenizer for model: {model_name}"))
                    },
                )?
            }
        };

        inner
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;
        inner.with_padding(None);

        Ok(Self {
            inner,
            resource_policy,
        })
    }

    /// Returns an explicitly audited base-tokenizer redirect, when one exists.
    ///
    /// # Returns
    /// Tessera currently has no active redirects. Unknown or incomplete model
    /// repositories fail instead of silently borrowing a tokenizer.
    fn get_base_model_tokenizer(_model_name: &str) -> Option<&'static str> {
        None
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

        // Get padding token ID (typically 0 for BERT)
        let pad_token_id = self.inner.token_to_id("[PAD]").unwrap_or(0);

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
