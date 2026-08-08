//! ColPali prompt and token preprocessing.

use super::{ColPaliPreprocessorConfig, ImageProcessor};
use crate::core::Tokenizer;
use anyhow::{Context, Result};

const BOS_TOKEN: &str = "<bos>";
const IMAGE_TOKEN: &str = "<image>";
const PAD_TOKEN: &str = "<pad>";
const IMAGE_PROMPT: &str = "Describe the image.";
const QUERY_PREFIX: &str = "Question: ";
const QUERY_AUGMENTATION_TOKENS: usize = 10;

/// Model-specific preprocessing shared by image and query inference.
pub struct ColPaliProcessor {
    image_processor: ImageProcessor,
    image_seq_length: usize,
    image_prompt_token_ids: Vec<u32>,
}

impl ColPaliProcessor {
    pub(crate) fn new(config: &ColPaliPreprocessorConfig, tokenizer: &Tokenizer) -> Result<Self> {
        validate_special_tokens(tokenizer)?;
        let (image_prompt_token_ids, attention_mask) = tokenizer
            .encode(&render_image_suffix(), false)
            .context("Failed to tokenize the ColPali image prompt")?;
        validate_token_layout("image prompt", &image_prompt_token_ids, &attention_mask)?;
        let bos_id = required_token_id(tokenizer, BOS_TOKEN)?;
        anyhow::ensure!(
            image_prompt_token_ids.first() == Some(&bos_id),
            "ColPali image prompt must begin with {BOS_TOKEN} (token ID {bos_id})"
        );

        Ok(Self {
            image_processor: ImageProcessor::from_preprocessor_config(config),
            image_seq_length: config.image_seq_length(),
            image_prompt_token_ids,
        })
    }

    pub(crate) const fn image_processor(&self) -> &ImageProcessor {
        &self.image_processor
    }

    pub(crate) const fn image_seq_length(&self) -> usize {
        self.image_seq_length
    }

    /// IDs for the text suffix only; Candle prepends the computed image features.
    pub(crate) fn image_prompt_token_ids(&self) -> &[u32] {
        &self.image_prompt_token_ids
    }

    pub(crate) fn tokenize_query(&self, query: &str, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
        let rendered = render_query(query);
        let (token_ids, attention_mask) =
            tokenizer.encode(&rendered, false).with_context(|| {
                format!(
                    "Failed to tokenize ColPali query ({} UTF-8 bytes)",
                    query.len()
                )
            })?;
        validate_token_layout("query", &token_ids, &attention_mask)?;

        let bos_id = required_token_id(tokenizer, BOS_TOKEN)?;
        let pad_id = required_token_id(tokenizer, PAD_TOKEN)?;
        anyhow::ensure!(
            token_ids.first() == Some(&bos_id),
            "ColPali query must begin with {BOS_TOKEN} (token ID {bos_id})"
        );
        anyhow::ensure!(
            token_ids.iter().filter(|&&id| id == pad_id).count() >= QUERY_AUGMENTATION_TOKENS,
            "ColPali query tokenization lost the {QUERY_AUGMENTATION_TOKENS} required {PAD_TOKEN} augmentation tokens"
        );
        Ok(token_ids)
    }
}

fn validate_special_tokens(tokenizer: &Tokenizer) -> Result<()> {
    for token in [BOS_TOKEN, IMAGE_TOKEN, PAD_TOKEN] {
        required_token_id(tokenizer, token)?;
    }
    Ok(())
}

fn required_token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| {
        anyhow::anyhow!("Pinned ColPali tokenizer is missing required token {token:?}")
    })
}

fn validate_token_layout(kind: &str, token_ids: &[u32], attention_mask: &[u32]) -> Result<()> {
    anyhow::ensure!(!token_ids.is_empty(), "ColPali {kind} produced no tokens");
    anyhow::ensure!(
        attention_mask.len() == token_ids.len(),
        "ColPali {kind} attention mask has {} entries for {} token IDs",
        attention_mask.len(),
        token_ids.len()
    );
    anyhow::ensure!(
        attention_mask.iter().all(|&value| value == 1),
        "ColPali {kind} requires every prompt and augmentation token to be active"
    );
    Ok(())
}

fn render_image_suffix() -> String {
    format!("{BOS_TOKEN}{IMAGE_PROMPT}\n")
}

#[cfg(test)]
fn render_full_image_prompt(image_seq_length: usize) -> String {
    let mut prompt = IMAGE_TOKEN.repeat(image_seq_length);
    prompt.push_str(&render_image_suffix());
    prompt
}

fn render_query(query: &str) -> String {
    let augmentation = PAD_TOKEN.repeat(QUERY_AUGMENTATION_TOKENS);
    format!("{BOS_TOKEN}{QUERY_PREFIX}{query}{augmentation}\n")
}

#[cfg(test)]
mod tests;
