use std::num::NonZeroUsize;

use thiserror::Error;

#[cfg(test)]
mod tests;

const DEFAULT_MAX_INPUT_BYTES_PER_SEQUENCE: usize = 1024 * 1024;
const DEFAULT_MAX_ATTENTION_CELLS: usize = 1_048_576;

/// Hard limits applied before Tessera loads model parameters or allocates text token tensors.
///
/// The default policy is intentionally conservative. Callers may provide a
/// different policy through an embedder builder, but its sequence limit may
/// never exceed the selected model's registered context length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourcePolicy {
    max_sequence_tokens: usize,
    max_batch_items: usize,
    max_batch_tokens: usize,
    max_model_bytes: usize,
    max_input_bytes_per_sequence: usize,
    max_attention_cells: usize,
}

impl ResourcePolicy {
    /// Creates a resource policy with explicit hard limits.
    ///
    /// The pre-tokenization input-byte and attention-cell limits retain their
    /// conservative defaults; use the corresponding `with_*` methods to
    /// override them.
    ///
    /// Zero is a valid limit: it rejects non-empty work while still allowing
    /// empty batches.
    #[must_use]
    pub const fn new(
        max_sequence_tokens: usize,
        max_batch_items: usize,
        max_batch_tokens: usize,
        max_model_bytes: usize,
    ) -> Self {
        Self {
            max_sequence_tokens,
            max_batch_items,
            max_batch_tokens,
            max_model_bytes,
            max_input_bytes_per_sequence: DEFAULT_MAX_INPUT_BYTES_PER_SEQUENCE,
            max_attention_cells: DEFAULT_MAX_ATTENTION_CELLS,
        }
    }

    /// Returns a conservative policy capped to a model's context length.
    #[must_use]
    pub fn for_model_context(model_context_tokens: usize) -> Self {
        let defaults = Self::default();
        Self {
            max_sequence_tokens: defaults.max_sequence_tokens.min(model_context_tokens),
            ..defaults
        }
    }

    /// Maximum tokens permitted in one sequence, including special tokens.
    #[must_use]
    pub const fn max_sequence_tokens(&self) -> usize {
        self.max_sequence_tokens
    }

    /// Maximum number of items permitted in one text tensor forward pass.
    #[must_use]
    pub const fn max_batch_items(&self) -> usize {
        self.max_batch_items
    }

    /// Maximum padded token cells permitted in one batch.
    #[must_use]
    pub const fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }

    /// Maximum estimated bytes permitted for model parameters.
    #[must_use]
    pub const fn max_model_bytes(&self) -> usize {
        self.max_model_bytes
    }

    /// Maximum UTF-8 input bytes permitted in one text sequence.
    #[must_use]
    pub const fn max_input_bytes_per_sequence(&self) -> usize {
        self.max_input_bytes_per_sequence
    }

    /// Maximum attention-matrix cells permitted in one forward pass.
    #[must_use]
    pub const fn max_attention_cells(&self) -> usize {
        self.max_attention_cells
    }

    /// Replaces the per-sequence token limit.
    #[must_use]
    pub const fn with_max_sequence_tokens(mut self, max_sequence_tokens: usize) -> Self {
        self.max_sequence_tokens = max_sequence_tokens;
        self
    }

    /// Replaces the batch item limit.
    #[must_use]
    pub const fn with_max_batch_items(mut self, max_batch_items: usize) -> Self {
        self.max_batch_items = max_batch_items;
        self
    }

    /// Replaces the padded batch token limit.
    #[must_use]
    pub const fn with_max_batch_tokens(mut self, max_batch_tokens: usize) -> Self {
        self.max_batch_tokens = max_batch_tokens;
        self
    }

    /// Replaces the estimated model parameter byte limit.
    #[must_use]
    pub const fn with_max_model_bytes(mut self, max_model_bytes: usize) -> Self {
        self.max_model_bytes = max_model_bytes;
        self
    }

    /// Replaces the per-sequence UTF-8 input byte limit.
    #[must_use]
    pub const fn with_max_input_bytes_per_sequence(
        mut self,
        max_input_bytes_per_sequence: usize,
    ) -> Self {
        self.max_input_bytes_per_sequence = max_input_bytes_per_sequence;
        self
    }

    /// Replaces the attention-matrix cell limit.
    #[must_use]
    pub const fn with_max_attention_cells(mut self, max_attention_cells: usize) -> Self {
        self.max_attention_cells = max_attention_cells;
        self
    }

    /// Returns a conservative item count for a forward pass of uninspected text.
    ///
    /// This assumes every item reaches the configured per-sequence limit. Once
    /// tokenized, the exact padded tensor shape is still validated separately.
    #[must_use]
    pub(crate) const fn conservative_batch_size(&self) -> Option<NonZeroUsize> {
        if self.max_sequence_tokens == 0 {
            return None;
        }
        let items_by_token_budget = self.max_batch_tokens / self.max_sequence_tokens;
        let Some(attention_cells_per_item) = self
            .max_sequence_tokens
            .checked_mul(self.max_sequence_tokens)
        else {
            return None;
        };
        let items_by_attention_budget = self.max_attention_cells / attention_cells_per_item;
        let item_limit = if self.max_batch_items < items_by_token_budget {
            self.max_batch_items
        } else {
            items_by_token_budget
        };
        NonZeroUsize::new(if item_limit < items_by_attention_budget {
            item_limit
        } else {
            items_by_attention_budget
        })
    }

    /// Validates this policy against a model's registered context length.
    pub fn validate_model_context(
        &self,
        model_context_tokens: usize,
    ) -> Result<(), ResourcePolicyError> {
        if self.max_sequence_tokens > model_context_tokens {
            return Err(ResourcePolicyError::ModelContext {
                measured: self.max_sequence_tokens,
                allowed: model_context_tokens,
            });
        }
        Ok(())
    }

    /// Estimates and validates model parameter storage.
    ///
    /// Registry parameter counts use decimal suffixes such as `109M` and
    /// `3B`. `bytes_per_parameter` should match the load dtype (4 for F32).
    pub fn validate_model_parameters(
        &self,
        registry_parameters: &str,
        bytes_per_parameter: usize,
    ) -> Result<u128, ResourcePolicyError> {
        let parameter_count = parse_parameter_count(registry_parameters)?;
        let estimated_bytes = parameter_count.saturating_mul(bytes_per_parameter as u128);
        if estimated_bytes > self.max_model_bytes as u128 {
            return Err(ResourcePolicyError::ModelBytes {
                measured: estimated_bytes,
                allowed: self.max_model_bytes,
            });
        }
        Ok(estimated_bytes)
    }

    /// Validates a tokenized sequence before tensor allocation.
    pub fn validate_sequence(&self, sequence_tokens: usize) -> Result<(), ResourcePolicyError> {
        if sequence_tokens > self.max_sequence_tokens {
            return Err(ResourcePolicyError::SequenceTokens {
                measured: sequence_tokens,
                allowed: self.max_sequence_tokens,
            });
        }
        Ok(())
    }

    /// Validates raw UTF-8 input size before tokenization work.
    pub fn validate_input_bytes(&self, input_bytes: usize) -> Result<(), ResourcePolicyError> {
        if input_bytes > self.max_input_bytes_per_sequence {
            return Err(ResourcePolicyError::InputBytes {
                measured: input_bytes,
                allowed: self.max_input_bytes_per_sequence,
            });
        }
        Ok(())
    }

    /// Validates a padded batch shape before tensor allocation.
    ///
    /// `padded_sequence_tokens` is the longest sequence length in the batch.
    /// Empty batches are always accepted.
    pub fn validate_batch(
        &self,
        batch_items: usize,
        padded_sequence_tokens: usize,
    ) -> Result<(), ResourcePolicyError> {
        if batch_items == 0 {
            return Ok(());
        }
        if batch_items > self.max_batch_items {
            return Err(ResourcePolicyError::BatchItems {
                measured: batch_items,
                allowed: self.max_batch_items,
            });
        }

        let batch_items = batch_items as u128;
        let sequence_tokens = padded_sequence_tokens as u128;
        let padded_batch_tokens = batch_items.saturating_mul(sequence_tokens);
        if padded_batch_tokens > self.max_batch_tokens as u128 {
            return Err(ResourcePolicyError::BatchTokens {
                measured: padded_batch_tokens,
                allowed: self.max_batch_tokens,
            });
        }
        let attention_cells = batch_items
            .saturating_mul(sequence_tokens)
            .saturating_mul(sequence_tokens);
        if attention_cells > self.max_attention_cells as u128 {
            return Err(ResourcePolicyError::AttentionCells {
                measured: attention_cells,
                allowed: self.max_attention_cells,
            });
        }
        Ok(())
    }
}

impl Default for ResourcePolicy {
    fn default() -> Self {
        Self::new(512, 16, 2048, 2 * 1024 * 1024 * 1024)
    }
}

/// Resolves an override against registered model context and parameter metadata.
pub fn resolve_registry_policy(
    override_policy: Option<ResourcePolicy>,
    model_context_tokens: usize,
    registry_parameters: &str,
) -> Result<ResourcePolicy, ResourcePolicyError> {
    let policy =
        override_policy.unwrap_or_else(|| ResourcePolicy::for_model_context(model_context_tokens));
    policy.validate_model_context(model_context_tokens)?;
    policy.validate_model_parameters(registry_parameters, 4)?;
    Ok(policy)
}

fn parse_parameter_count(registry_parameters: &str) -> Result<u128, ResourcePolicyError> {
    let value = registry_parameters.trim();
    let suffix = value
        .chars()
        .last()
        .ok_or_else(|| invalid_parameter_count(registry_parameters))?;
    let multiplier = match suffix.to_ascii_uppercase() {
        'K' => 1_000_u128,
        'M' => 1_000_000_u128,
        'B' => 1_000_000_000_u128,
        _ if suffix.is_ascii_digit() => 1_u128,
        _ => return Err(invalid_parameter_count(registry_parameters)),
    };
    let magnitude = if multiplier == 1 {
        value
    } else {
        &value[..value.len() - suffix.len_utf8()]
    };

    let mut parts = magnitude.split('.');
    let whole = parts
        .next()
        .filter(|part| !part.is_empty())
        .ok_or_else(|| invalid_parameter_count(registry_parameters))?;
    let fraction = parts.next();
    if parts.next().is_some()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || fraction
            .is_some_and(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(invalid_parameter_count(registry_parameters));
    }

    let whole = whole
        .parse::<u128>()
        .map_err(|_| invalid_parameter_count(registry_parameters))?;
    let (fraction, scale) = if let Some(fraction_text) = fraction {
        let fractional_digits = u32::try_from(fraction_text.len())
            .map_err(|_| invalid_parameter_count(registry_parameters))?;
        let fraction = fraction_text
            .parse::<u128>()
            .map_err(|_| invalid_parameter_count(registry_parameters))?;
        let scale = 10_u128
            .checked_pow(fractional_digits)
            .ok_or_else(|| invalid_parameter_count(registry_parameters))?;
        (fraction, scale)
    } else {
        (0, 1)
    };
    let scaled_magnitude = whole
        .checked_mul(scale)
        .and_then(|scaled| scaled.checked_add(fraction))
        .ok_or_else(|| invalid_parameter_count(registry_parameters))?;
    let scaled_parameters = scaled_magnitude
        .checked_mul(multiplier)
        .ok_or_else(|| invalid_parameter_count(registry_parameters))?;
    let rounded_up = u128::from(scaled_parameters % scale != 0);

    Ok(scaled_parameters / scale + rounded_up)
}

fn invalid_parameter_count(value: &str) -> ResourcePolicyError {
    ResourcePolicyError::InvalidParameterCount {
        value: value.to_string(),
    }
}

/// A hard resource limit rejected work before model or text tensor allocation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResourcePolicyError {
    /// A configured sequence limit exceeded the selected model's context.
    #[error("Configured sequence token limit {measured} exceeds model context limit {allowed}")]
    ModelContext {
        /// Configured policy value.
        measured: usize,
        /// Registered model context length.
        allowed: usize,
    },
    /// One tokenized sequence exceeded the policy.
    #[error("Sequence token count {measured} exceeds resource policy limit {allowed}")]
    SequenceTokens {
        /// Token count after adding requested special tokens.
        measured: usize,
        /// Maximum permitted sequence tokens.
        allowed: usize,
    },
    /// A batch contained too many items.
    #[error("Batch item count {measured} exceeds resource policy limit {allowed}")]
    BatchItems {
        /// Number of submitted items.
        measured: usize,
        /// Maximum permitted batch items.
        allowed: usize,
    },
    /// Padding would create too many token cells.
    #[error("Padded batch token count {measured} exceeds resource policy limit {allowed}")]
    BatchTokens {
        /// Product of batch size and longest sequence length.
        measured: u128,
        /// Maximum permitted padded token cells.
        allowed: usize,
    },
    /// One raw text input exceeded the pre-tokenization byte limit.
    #[error("Input byte count {measured} exceeds resource policy limit {allowed}")]
    InputBytes {
        /// UTF-8 bytes in the submitted text.
        measured: usize,
        /// Maximum permitted UTF-8 input bytes.
        allowed: usize,
    },
    /// A forward pass would allocate too many attention-matrix cells.
    #[error("Attention cell count {measured} exceeds resource policy limit {allowed}")]
    AttentionCells {
        /// Product of batch size and squared padded sequence length.
        measured: u128,
        /// Maximum permitted attention cells.
        allowed: usize,
    },
    /// Estimated model parameters exceeded the configured byte budget.
    #[error("Estimated model parameter bytes {measured} exceeds resource policy limit {allowed}")]
    ModelBytes {
        /// Estimated bytes at the selected load dtype.
        measured: u128,
        /// Maximum permitted model parameter bytes.
        allowed: usize,
    },
    /// Registry parameter metadata could not be converted into a count.
    #[error("Invalid registry parameter count '{value}'")]
    InvalidParameterCount {
        /// Unparseable registry value.
        value: String,
    },
}
