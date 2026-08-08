use thiserror::Error;

/// A hard resource limit rejected work before model or text tensor allocation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ResourcePolicyError {
    /// A configured sequence limit exceeded the selected model's context.
    #[error("Configured sequence token limit {measured} exceeds model context limit {allowed}")]
    ModelContext {
        /// Configured policy value.
        measured: usize,
        /// Registered context ceiling.
        allowed: usize,
    },
    /// One tokenized sequence exceeded the policy.
    #[error("Sequence token count {measured} exceeds resource policy limit {allowed}")]
    SequenceTokens {
        /// Tokenized sequence length.
        measured: usize,
        /// Maximum permitted sequence length.
        allowed: usize,
    },
    /// A batch contained too many items.
    #[error("Batch item count {measured} exceeds resource policy limit {allowed}")]
    BatchItems {
        /// Submitted item count.
        measured: usize,
        /// Maximum items in one forward.
        allowed: usize,
    },
    /// Padding would create too many token cells.
    #[error("Padded batch token count {measured} exceeds resource policy limit {allowed}")]
    BatchTokens {
        /// Padded token cells.
        measured: u128,
        /// Maximum padded token cells.
        allowed: usize,
    },
    /// One raw text input exceeded the pre-tokenization byte limit.
    #[error("Input byte count {measured} exceeds resource policy limit {allowed}")]
    InputBytes {
        /// Submitted UTF-8 bytes.
        measured: usize,
        /// Maximum UTF-8 bytes per input.
        allowed: usize,
    },
    /// A forward pass would allocate too many attention-matrix cells.
    #[error("Attention cell count {measured} exceeds resource policy limit {allowed}")]
    AttentionCells {
        /// Batch size multiplied by squared sequence length.
        measured: u128,
        /// Maximum permitted attention cells.
        allowed: usize,
    },
    /// A logical job submitted too many inputs across its forward chunks.
    #[error("Job item count {measured} exceeds resource policy limit {allowed}")]
    JobItems {
        /// Cumulative logical-job items.
        measured: usize,
        /// Maximum logical-job items.
        allowed: usize,
    },
    /// Aggregate raw text input exceeded the logical-job byte limit.
    #[error("Job input byte count {measured} exceeds resource policy limit {allowed}")]
    JobInputBytes {
        /// Cumulative logical-job input bytes.
        measured: usize,
        /// Maximum logical-job input bytes.
        allowed: usize,
    },
    /// A collecting API would retain too many embedding bytes.
    #[error("Collected output byte count {measured} exceeds resource policy limit {allowed}")]
    OutputBytes {
        /// Retained or per-item output bytes.
        measured: usize,
        /// Maximum permitted output bytes.
        allowed: usize,
    },
    /// Estimated live transformer scratch exceeded the configured ceiling.
    #[error("Estimated activation bytes {measured} exceed resource policy limit {allowed}")]
    ActivationBytes {
        /// Architecture-aware scratch estimate.
        measured: u128,
        /// Maximum permitted estimated scratch bytes.
        allowed: usize,
    },
    /// Estimated model parameters exceeded the configured byte budget.
    #[error("Estimated model parameter bytes {measured} exceeds resource policy limit {allowed}")]
    ModelBytes {
        /// Estimated parameter bytes at the selected dtype.
        measured: u128,
        /// Maximum permitted parameter bytes.
        allowed: usize,
    },
    /// Registry parameter metadata could not be converted into a count.
    #[error("Invalid registry parameter count '{value}'")]
    InvalidParameterCount {
        /// Unparseable registry value.
        value: String,
    },
}
