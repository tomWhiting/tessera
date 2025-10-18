//! Tokenization abstraction layer.
//!
//! This module provides a wrapper around the HuggingFace tokenizers library
//! for loading and using BERT-compatible tokenizers.

use anyhow::{Context, Result};
use tokenizers::Tokenizer as HfTokenizer;

/// Wrapper around HuggingFace tokenizer for BERT models.
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    /// Loads a tokenizer from the HuggingFace Hub.
    ///
    /// # Arguments
    /// * `model_name` - Name of the model on HuggingFace Hub (e.g., "bert-base-uncased")
    ///
    /// # Returns
    /// A new Tokenizer instance
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        // Download tokenizer file from HuggingFace Hub
        let api =
            hf_hub::api::sync::Api::new().context("Failed to initialize HuggingFace Hub API")?;
        let repo = api.model(model_name.to_string());

        let tokenizer_path = repo
            .get("tokenizer.json")
            .with_context(|| format!("Downloading tokenizer for {}", model_name))?;

        // Load tokenizer from file
        let inner = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
            .with_context(|| format!("Loading tokenizer for model: {}", model_name))?;

        Ok(Self { inner })
    }

    /// Encodes text into token IDs.
    ///
    /// # Arguments
    /// * `text` - The text to tokenize
    /// * `add_special_tokens` - Whether to add [CLS], [SEP] tokens
    ///
    /// # Returns
    /// A tuple of (token_ids, attention_mask)
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<(Vec<u32>, Vec<u32>)> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))
            .context("Encoding text with tokenizer")?;

        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();

        Ok((token_ids, attention_mask))
    }

    /// Decodes token IDs back into text.
    ///
    /// # Arguments
    /// * `token_ids` - The token IDs to decode
    /// * `skip_special_tokens` - Whether to skip [CLS], [SEP], [PAD] tokens
    ///
    /// # Returns
    /// The decoded text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))
            .context("Decoding token IDs")
    }

    /// Returns the vocabulary size of the tokenizer.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    /// Encodes multiple texts into token IDs with padding.
    ///
    /// All sequences are padded to the length of the longest sequence in the batch.
    /// This enables efficient batch processing in neural networks.
    ///
    /// # Arguments
    /// * `texts` - Slice of texts to tokenize
    /// * `add_special_tokens` - Whether to add [CLS], [SEP] tokens
    ///
    /// # Returns
    /// A vector of tuples (token_ids, attention_mask), one per input text.
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
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts
        let mut all_tokenized = Vec::with_capacity(texts.len());
        for text in texts {
            let (token_ids, attention_mask) = self.encode(text, add_special_tokens)?;
            all_tokenized.push((token_ids, attention_mask));
        }

        // Find max length in batch
        let max_len = all_tokenized
            .iter()
            .map(|(tokens, _)| tokens.len())
            .max()
            .unwrap_or(0);

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
}
