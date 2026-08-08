use thiserror::Error;

use super::ResourcePolicy;

#[cfg(test)]
mod tests;

/// Explicit sliding-window configuration for long text.
///
/// Windowed embeddings are an aggregation mode and are not equivalent to a
/// model performing native full-attention inference over the complete input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextWindowConfig {
    window_tokens: usize,
    overlap_tokens: usize,
}

impl ContextWindowConfig {
    /// Creates a window configuration including model special tokens.
    #[must_use]
    pub const fn new(window_tokens: usize, overlap_tokens: usize) -> Self {
        Self {
            window_tokens,
            overlap_tokens,
        }
    }

    /// Tokens in each model input, including its special-token envelope.
    #[must_use]
    pub const fn window_tokens(self) -> usize {
        self.window_tokens
    }

    /// Content tokens shared with the next window.
    #[must_use]
    pub const fn overlap_tokens(self) -> usize {
        self.overlap_tokens
    }
}

/// One validated model input and its ownership range in the original tokens.
pub struct TokenWindow {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) attention_mask: Vec<u32>,
    pub(crate) owned_start: usize,
    pub(crate) owned_end: usize,
    pub(crate) owned_local_start: usize,
    pub(crate) owned_local_end: usize,
}

impl TokenWindow {
    pub(crate) const fn owned_len(&self) -> usize {
        self.owned_end - self.owned_start
    }

    pub(crate) const fn owned_local_range(&self) -> std::ops::Range<usize> {
        self.owned_local_start..self.owned_local_end
    }
}

/// Builds overlapping, center-owned windows without allocating attention tensors.
pub fn plan_token_windows(
    content_ids: &[u32],
    special_prefix: &[u32],
    special_suffix: &[u32],
    config: ContextWindowConfig,
    policy: ResourcePolicy,
) -> Result<Vec<TokenWindow>, ContextWindowError> {
    if config.window_tokens > policy.max_sequence_tokens() {
        return Err(ContextWindowError::PolicyLimit {
            measured: config.window_tokens,
            allowed: policy.max_sequence_tokens(),
        });
    }
    let special_tokens = special_prefix.len().saturating_add(special_suffix.len());
    let content_capacity = config
        .window_tokens
        .checked_sub(special_tokens)
        .filter(|capacity| *capacity > 0)
        .ok_or(ContextWindowError::SpecialTokenEnvelope {
            window_tokens: config.window_tokens,
            special_tokens,
        })?;
    if config.overlap_tokens >= content_capacity {
        return Err(ContextWindowError::Overlap {
            overlap: config.overlap_tokens,
            content_capacity,
        });
    }
    if content_ids.is_empty() {
        return Ok(vec![make_window(
            content_ids,
            special_prefix,
            special_suffix,
            0,
            0,
            0,
            0,
        )]);
    }

    let stride = content_capacity - config.overlap_tokens;
    let mut windows = Vec::new();
    let mut start = 0_usize;
    while start < content_ids.len() {
        let end = start
            .saturating_add(content_capacity)
            .min(content_ids.len());
        let left_overlap = if start == 0 { 0 } else { config.overlap_tokens };
        let right_overlap = if end == content_ids.len() {
            0
        } else {
            config.overlap_tokens
        };
        let owned_start = start.saturating_add(left_overlap / 2);
        let owned_end = end.saturating_sub(right_overlap - right_overlap / 2);
        windows.push(make_window(
            content_ids,
            special_prefix,
            special_suffix,
            start,
            end,
            owned_start,
            owned_end,
        ));
        policy
            .validate_job(windows.len(), 0)
            .map_err(|_| ContextWindowError::TooManyWindows {
                measured: windows.len(),
                allowed: policy.max_job_items(),
            })?;
        if end == content_ids.len() {
            break;
        }
        start = start.saturating_add(stride);
    }
    Ok(windows)
}

fn make_window(
    content_ids: &[u32],
    prefix: &[u32],
    suffix: &[u32],
    start: usize,
    end: usize,
    owned_start: usize,
    owned_end: usize,
) -> TokenWindow {
    let mut token_ids = Vec::with_capacity(prefix.len() + end.saturating_sub(start) + suffix.len());
    token_ids.extend_from_slice(prefix);
    token_ids.extend_from_slice(&content_ids[start..end]);
    token_ids.extend_from_slice(suffix);
    let attention_mask = vec![1; token_ids.len()];
    TokenWindow {
        token_ids,
        attention_mask,
        owned_start,
        owned_end,
        owned_local_start: owned_start.saturating_sub(start),
        owned_local_end: owned_end.saturating_sub(start),
    }
}

/// Invalid sliding-window configuration or plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ContextWindowError {
    /// Requested window exceeds the per-forward sequence limit.
    #[error("window token count {measured} exceeds resource policy limit {allowed}")]
    PolicyLimit {
        /// Requested total window tokens.
        measured: usize,
        /// Per-forward token ceiling.
        allowed: usize,
    },
    /// Special tokens leave no content capacity.
    #[error(
        "window token count {window_tokens} cannot contain {special_tokens} special tokens and content"
    )]
    SpecialTokenEnvelope {
        /// Requested total window tokens.
        window_tokens: usize,
        /// Required prefix and suffix tokens.
        special_tokens: usize,
    },
    /// Overlap must be smaller than the window content capacity.
    #[error("window overlap {overlap} must be smaller than content capacity {content_capacity}")]
    Overlap {
        /// Requested shared content tokens.
        overlap: usize,
        /// Available content tokens in one window.
        content_capacity: usize,
    },
    /// Windowing would exceed the logical-job item ceiling.
    #[error("window count {measured} exceeds resource policy job limit {allowed}")]
    TooManyWindows {
        /// Planned window count.
        measured: usize,
        /// Logical-job item ceiling.
        allowed: usize,
    },
}
