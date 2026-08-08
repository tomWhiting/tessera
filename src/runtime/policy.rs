use std::num::NonZeroUsize;

mod error;
mod parse;
#[cfg(test)]
mod tests;

pub use error::ResourcePolicyError;
use parse::parse_parameter_count;

const DEFAULT_MAX_INPUT_BYTES_PER_SEQUENCE: usize = 1024 * 1024;
const DEFAULT_MAX_ATTENTION_CELLS: usize = 1_048_576;
const DEFAULT_MAX_JOB_ITEMS: usize = 1024;
const DEFAULT_MAX_JOB_INPUT_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MAX_ACTIVATION_BYTES: usize = 512 * 1024 * 1024;

/// Resource limits for model loading, inference requests, and collected outputs.
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
    max_job_items: usize,
    max_job_input_bytes: usize,
    max_output_bytes: usize,
    max_activation_bytes: usize,
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
            max_job_items: DEFAULT_MAX_JOB_ITEMS,
            max_job_input_bytes: DEFAULT_MAX_JOB_INPUT_BYTES,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_activation_bytes: DEFAULT_MAX_ACTIVATION_BYTES,
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

    /// Maximum inputs accepted by one logical encode job across all chunks.
    #[must_use]
    pub const fn max_job_items(&self) -> usize {
        self.max_job_items
    }

    /// Maximum aggregate UTF-8 input bytes accepted by one logical job.
    #[must_use]
    pub const fn max_job_input_bytes(&self) -> usize {
        self.max_job_input_bytes
    }

    /// Maximum bytes a collecting API may retain for embedding values.
    #[must_use]
    pub const fn max_output_bytes(&self) -> usize {
        self.max_output_bytes
    }

    /// Maximum estimated live inference scratch bytes for one forward pass.
    #[must_use]
    pub const fn max_activation_bytes(&self) -> usize {
        self.max_activation_bytes
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

    /// Replaces the logical-job item limit.
    #[must_use]
    pub const fn with_max_job_items(mut self, max_job_items: usize) -> Self {
        self.max_job_items = max_job_items;
        self
    }

    /// Replaces the aggregate logical-job input-byte limit.
    #[must_use]
    pub const fn with_max_job_input_bytes(mut self, max_job_input_bytes: usize) -> Self {
        self.max_job_input_bytes = max_job_input_bytes;
        self
    }

    /// Replaces the retained embedding-output byte limit.
    #[must_use]
    pub const fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    /// Replaces the estimated live inference scratch-byte limit.
    #[must_use]
    pub const fn with_max_activation_bytes(mut self, max_activation_bytes: usize) -> Self {
        self.max_activation_bytes = max_activation_bytes;
        self
    }

    /// Validates a model-specific transformer scratch estimate.
    pub(crate) fn validate_transformer_activations(
        &self,
        profile: crate::runtime::TransformerProfile,
        batch_size: usize,
        sequence_tokens: usize,
        dtype: crate::runtime::ModelDType,
    ) -> Result<u128, ResourcePolicyError> {
        let estimated = profile.peak_bytes(batch_size, sequence_tokens, dtype);
        if estimated > self.max_activation_bytes as u128 {
            return Err(ResourcePolicyError::ActivationBytes {
                measured: estimated,
                allowed: self.max_activation_bytes,
            });
        }
        Ok(estimated)
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

    /// Validates cumulative work for one logical job before another forward.
    pub fn validate_job(
        &self,
        job_items: usize,
        total_input_bytes: usize,
    ) -> Result<(), ResourcePolicyError> {
        if job_items > self.max_job_items {
            return Err(ResourcePolicyError::JobItems {
                measured: job_items,
                allowed: self.max_job_items,
            });
        }
        if total_input_bytes > self.max_job_input_bytes {
            return Err(ResourcePolicyError::JobInputBytes {
                measured: total_input_bytes,
                allowed: self.max_job_input_bytes,
            });
        }
        Ok(())
    }

    /// Validates bytes retained by a collecting embedding API.
    pub fn validate_output_bytes(&self, output_bytes: usize) -> Result<(), ResourcePolicyError> {
        if output_bytes > self.max_output_bytes {
            return Err(ResourcePolicyError::OutputBytes {
                measured: output_bytes,
                allowed: self.max_output_bytes,
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

/// Resolves an override using the selected model parameter dtype.
pub fn resolve_registry_policy_with_dtype(
    override_policy: Option<ResourcePolicy>,
    model_context_tokens: usize,
    registry_parameters: &str,
    dtype: crate::runtime::ModelDType,
) -> Result<ResourcePolicy, ResourcePolicyError> {
    let policy =
        override_policy.unwrap_or_else(|| ResourcePolicy::for_model_context(model_context_tokens));
    policy.validate_model_context(model_context_tokens)?;
    policy.validate_model_parameters(registry_parameters, dtype.bytes_per_parameter())?;
    Ok(policy)
}
