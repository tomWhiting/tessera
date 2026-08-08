use anyhow::{Context, Result};
use ndarray::Array2;

use crate::core::TokenEmbeddings;
use crate::runtime::{f32_output_bytes, ContextWindowConfig, JobTracker};

use super::CandleBertEncoder;

impl CandleBertEncoder {
    /// Encodes a long retrieval document through bounded, overlapping ColBERT
    /// windows. Overlap supplies context, while center ownership retains each
    /// original content-token row once. Every window retains its own
    /// `[CLS]`, `[D]`, and `[SEP]` rows.
    pub fn encode_document_windowed(
        &self,
        text: &str,
        config: ContextWindowConfig,
    ) -> Result<TokenEmbeddings> {
        let policy = self.resource_policy;
        let mut tracker = JobTracker::new(policy);
        tracker
            .admit_input(text.len())
            .context("Admitting ColBERT windowed document input")?;
        let plan_config = self
            .preprocessor
            .document_window_plan_config(config)
            .context("Validating ColBERT document window configuration")?;
        let windows = self
            .tokenizer
            .encode_windows(text, plan_config)
            .with_context(|| {
                format!(
                    "Planning ColBERT document windows for {} UTF-8 bytes",
                    text.len()
                )
            })?;

        let expected_dim = self.config.embedding_dim;
        let mut values = Vec::new();
        let mut total_rows = 0_usize;
        let mut output_dim = None;
        for (index, window) in windows.into_iter().enumerate() {
            let prepared = self
                .preprocessor
                .prepare_document_window(
                    &window.token_ids,
                    &window.attention_mask,
                    window.owned_local_range(),
                )
                .with_context(|| format!("Preparing ColBERT document window {index}"))?;
            policy
                .validate_sequence(prepared.token_ids.len())
                .with_context(|| format!("Validating ColBERT document window {index} length"))?;
            policy
                .validate_batch(1, prepared.token_ids.len())
                .with_context(|| format!("Validating ColBERT document window {index} shape"))?;
            let selected_rows = prepared
                .output_mask
                .iter()
                .filter(|selected| **selected == 1)
                .count();
            tracker
                .retain_output(f32_output_bytes(selected_rows.saturating_mul(expected_dim)))
                .with_context(|| format!("Preflighting ColBERT document window {index} output"))?;

            let embedding = self
                .infer_one(prepared, "")
                .with_context(|| format!("Encoding ColBERT document window {index}"))?;
            let (rows, dimension) = embedding.shape();
            anyhow::ensure!(
                dimension == expected_dim,
                "ColBERT document window produced {dimension} dimensions; expected {expected_dim}"
            );
            if let Some(previous) = output_dim {
                anyhow::ensure!(
                    previous == dimension,
                    "ColBERT output dimension changed from {previous} to {dimension} across document windows"
                );
            } else {
                output_dim = Some(dimension);
            }
            total_rows = total_rows
                .checked_add(rows)
                .context("ColBERT document window row count overflowed")?;
            let matrix = embedding.into_matrix();
            values.extend(matrix.iter().copied());
        }

        let output_dim = output_dim.context("Window planner returned no ColBERT inputs")?;
        let matrix = Array2::from_shape_vec((total_rows, output_dim), values)
            .context("Aggregating ColBERT document-window embeddings")?;
        TokenEmbeddings::new(matrix, text.to_string())
            .context("Creating aggregated ColBERT document embeddings")
    }
}
