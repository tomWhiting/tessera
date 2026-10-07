use haem_frames::embedding::{Limits, Windows};
use tessera::ResourcePolicy;

use crate::failure::Failure;

pub struct Budget {
    pub memory: usize,
    pub threads: usize,
    batch: usize,
    job_items: usize,
    input: usize,
    tokens: usize,
    job_input: usize,
    batch_tokens: usize,
    attention: usize,
}

fn number(value: u64, name: &str) -> Result<usize, Failure> {
    usize::try_from(value).map_err(|_| Failure::limits(format!("{name} exceeds usize")))
}

fn product(left: usize, right: usize, name: &str) -> Result<usize, Failure> {
    left.checked_mul(right)
        .ok_or_else(|| Failure::limits(format!("{name} overflows")))
}

impl Budget {
    /// With windows (protocol 2) a job holds up to `max_windows` window inputs and
    /// vectors per item, whatever the Start names.
    pub fn new(limits: &Limits, windows: Option<Windows>) -> Result<Self, Failure> {
        let batch = number(limits.batch_items, "batch_items")?;
        let job_items = match windows {
            Some(windows) => product(
                batch,
                number(windows.max_windows, "max_windows")?,
                "window job items",
            )?,
            None => batch,
        };
        let input = number(limits.input_bytes, "input_bytes")?;
        let tokens = number(limits.tokens, "tokens")?;
        let batch_tokens = product(batch, tokens, "padded tokens")?;
        Ok(Self {
            memory: number(limits.memory_bytes, "memory_bytes")?,
            threads: number(limits.threads, "threads")?,
            batch,
            job_items,
            input,
            tokens,
            job_input: product(batch, input, "job input bytes")?,
            batch_tokens,
            attention: product(batch_tokens, tokens, "attention cells")?,
        })
    }

    pub fn policy(&self, dimensions: usize, parameters: &str) -> Result<ResourcePolicy, Failure> {
        let output = product(
            product(self.job_items, dimensions, "output dimensions")?,
            4,
            "output bytes",
        )?;
        let policy = ResourcePolicy::new(self.tokens, self.batch, self.batch_tokens, self.memory);
        let model_bytes = policy
            .validate_model_parameters(parameters, 4)
            .map_err(|error| Failure::limits(format!("model bytes and memory_bytes: {error}")))?;
        let model_bytes = usize::try_from(model_bytes)
            .map_err(|_| Failure::limits("model bytes exceed usize"))?;
        let activation = self
            .memory
            .checked_sub(model_bytes)
            .ok_or_else(|| Failure::limits("model bytes exceed memory_bytes"))?;
        Ok(policy
            .with_max_input_bytes_per_sequence(self.input)
            .with_max_job_items(self.job_items)
            .with_max_job_input_bytes(self.job_input)
            .with_max_output_bytes(output)
            .with_max_attention_cells(self.attention)
            .with_max_activation_bytes(activation))
    }
}
