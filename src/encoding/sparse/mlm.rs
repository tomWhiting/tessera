use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

/// MLM (Masked Language Modeling) head for SPLADE.
///
/// Projects BERT hidden states to vocabulary logits via:
/// 1. Dense transformation with GELU activation
/// 2. Layer normalization
/// 3. Final linear projection to vocabulary space
pub(super) struct MlmHead {
    /// Transform layer: hidden → hidden with GELU
    transform_dense: Linear,
    /// Layer normalization
    transform_layer_norm: LayerNorm,
    /// Final projection: hidden → `vocab_size`
    decoder: Linear,
}

impl MlmHead {
    /// Load MLM head from weights.
    ///
    /// # Arguments
    /// * `vb` - Variable builder for loading weights
    /// * `hidden_size` - Hidden dimension size
    /// * `vocab_size` - Vocabulary size
    ///
    /// # Returns
    /// Initialized MLM head ready for inference
    pub(super) fn load(vb: VarBuilder, hidden_size: usize, vocab_size: usize) -> Result<Self> {
        // Load transform layer (dense + activation)
        let transform_vb = vb.pp("cls").pp("predictions").pp("transform");
        let transform_dense = linear(hidden_size, hidden_size, transform_vb.pp("dense"))
            .context("Loading MLM transform dense layer")?;

        // Load layer norm
        let transform_layer_norm = layer_norm(
            hidden_size,
            1e-12, // epsilon - standard BERT value
            transform_vb.pp("LayerNorm"),
        )
        .context("Loading MLM layer normalization")?;

        // Load decoder (final projection to vocab)
        let decoder_vb = vb.pp("cls").pp("predictions");
        let decoder = linear(hidden_size, vocab_size, decoder_vb.pp("decoder"))
            .context("Loading MLM decoder layer")?;

        Ok(Self {
            transform_dense,
            transform_layer_norm,
            decoder,
        })
    }

    /// Forward pass: `hidden_states` → `vocab_logits`
    ///
    /// # Arguments
    /// * `hidden_states` - Token representations from BERT [`seq_len`, `hidden_size`]
    ///
    /// # Returns
    /// Vocabulary logits [`seq_len`, `vocab_size`]
    pub(super) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Transform: linear + GELU
        let transformed = self
            .transform_dense
            .forward(hidden_states)
            .context("MLM transform dense forward")?;
        let activated = transformed.gelu().context("MLM GELU activation")?;

        // Layer norm
        let normalized = self
            .transform_layer_norm
            .forward(&activated)
            .context("MLM layer norm forward")?;

        // Project to vocabulary
        let logits = self
            .decoder
            .forward(&normalized)
            .context("MLM decoder forward")?;

        Ok(logits)
    }
}
