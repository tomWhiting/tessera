use std::mem::size_of;

use super::{ResourcePolicy, ResourcePolicyError};

#[cfg(test)]
mod tests;

/// Cumulative resource accounting for one logical encode operation.
pub struct JobTracker {
    policy: ResourcePolicy,
    items: usize,
    input_bytes: usize,
    retained_output_bytes: usize,
}

impl JobTracker {
    pub(crate) const fn new(policy: ResourcePolicy) -> Self {
        Self {
            policy,
            items: 0,
            input_bytes: 0,
            retained_output_bytes: 0,
        }
    }

    /// Admits one input before tokenization or model execution.
    pub(crate) fn admit_input(&mut self, input_bytes: usize) -> Result<(), ResourcePolicyError> {
        let items = self.items.saturating_add(1);
        let total_input_bytes = self.input_bytes.saturating_add(input_bytes);
        self.policy.validate_job(items, total_input_bytes)?;
        self.items = items;
        self.input_bytes = total_input_bytes;
        Ok(())
    }

    /// Records bytes retained by a collecting API.
    pub(crate) fn retain_output(&mut self, output_bytes: usize) -> Result<(), ResourcePolicyError> {
        let retained_output_bytes = self.retained_output_bytes.saturating_add(output_bytes);
        self.policy.validate_output_bytes(retained_output_bytes)?;
        self.retained_output_bytes = retained_output_bytes;
        Ok(())
    }

    /// Validates that one streamed item itself fits the output ceiling.
    pub(crate) fn validate_streamed_output(
        &self,
        output_bytes: usize,
    ) -> Result<(), ResourcePolicyError> {
        self.policy.validate_output_bytes(output_bytes)
    }
}

pub const fn f32_output_bytes(elements: usize) -> usize {
    elements.saturating_mul(size_of::<f32>())
}
