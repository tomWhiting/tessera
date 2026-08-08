use serde_json::Value;
use thiserror::Error;

use super::ModelDType;

#[cfg(test)]
mod tests;

/// Transformer dimensions used for conservative inference scratch estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformerProfile {
    hidden_size: usize,
    intermediate_size: usize,
    attention_heads: usize,
}

impl TransformerProfile {
    pub(crate) fn new(
        hidden_size: usize,
        intermediate_size: usize,
        attention_heads: usize,
    ) -> Result<Self, EstimateError> {
        if hidden_size == 0 || intermediate_size == 0 || attention_heads == 0 {
            return Err(EstimateError::ZeroDimension);
        }
        Ok(Self {
            hidden_size,
            intermediate_size,
            attention_heads,
        })
    }

    /// Reads common BERT, XLM-R, ModernBERT, NomicBERT, and JinaBERT keys.
    pub(crate) fn from_config_json(config_json: &str) -> Result<Self, EstimateError> {
        let value: Value = serde_json::from_str(config_json)
            .map_err(|error| EstimateError::InvalidConfig(error.to_string()))?;
        let hidden_size = first_usize(&value, &["hidden_size", "n_embd", "dim"])
            .ok_or(EstimateError::MissingDimension("hidden size"))?;
        let intermediate_size =
            first_usize(&value, &["intermediate_size", "n_inner", "hidden_dim"])
                .unwrap_or_else(|| hidden_size.saturating_mul(4));
        let attention_heads = first_usize(
            &value,
            &["num_attention_heads", "n_head", "n_heads", "num_heads"],
        )
        .ok_or(EstimateError::MissingDimension("attention heads"))?;
        Self::new(hidden_size, intermediate_size, attention_heads)
    }

    /// Estimates peak live transformer scratch bytes with 1.5x headroom.
    ///
    /// Bidirectional encoders do not retain every layer's activations during
    /// inference, so layer count is intentionally not multiplied into peak
    /// residency. The estimate includes Q/K/V and residual buffers, attention
    /// scores plus probabilities, and feed-forward intermediates.
    pub(crate) fn peak_bytes(
        self,
        batch_size: usize,
        sequence_tokens: usize,
        dtype: ModelDType,
    ) -> u128 {
        let batch = batch_size as u128;
        let sequence = sequence_tokens as u128;
        let hidden = self.hidden_size as u128;
        let intermediate = self.intermediate_size as u128;
        let heads = self.attention_heads as u128;
        let element_bytes = dtype.bytes_per_parameter() as u128;

        let hidden_buffers = batch
            .saturating_mul(sequence)
            .saturating_mul(hidden)
            .saturating_mul(6);
        let attention_buffers = batch
            .saturating_mul(heads)
            .saturating_mul(sequence)
            .saturating_mul(sequence)
            .saturating_mul(2);
        let ffn_buffers = batch
            .saturating_mul(sequence)
            .saturating_mul(intermediate)
            .saturating_mul(2);
        hidden_buffers
            .saturating_add(attention_buffers.max(ffn_buffers))
            .saturating_mul(element_bytes)
            .saturating_mul(3)
            / 2
    }
}

fn first_usize(value: &Value, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
    })
}

/// Invalid or incomplete transformer configuration used for estimation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EstimateError {
    #[error("invalid transformer config JSON: {0}")]
    InvalidConfig(String),
    #[error("transformer config is missing {0}")]
    MissingDimension(&'static str),
    #[error("transformer dimensions must be greater than zero")]
    ZeroDimension,
}
