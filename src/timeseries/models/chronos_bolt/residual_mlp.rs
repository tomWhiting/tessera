use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// Residual MLP for patch embedding.
///
/// This is a 3-layer MLP with a residual connection:
/// - `hidden_layer`: `input_dim` -> `hidden_dim` (+ `ReLU`)
/// - `output_layer`: `hidden_dim` -> `output_dim`
/// - `residual_layer`: `input_dim` -> `output_dim`
/// - output = `output_layer(ReLU(hidden_layer(x)))` + `residual_layer(x)`
///
/// Used by Chronos Bolt for both input and output patch embeddings.
pub struct ResidualMLP {
    hidden_layer: Linear,
    output_layer: Linear,
    residual_layer: Linear,
}

impl ResidualMLP {
    /// Create a new residual MLP.
    ///
    /// # Arguments
    /// * `input_dim` - Input dimension
    /// * `hidden_dim` - Hidden dimension (typically 4x `d_model`)
    /// * `output_dim` - Output dimension
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Returns
    /// Initialized `ResidualMLP`
    ///
    /// # Errors
    /// Returns error if weight loading fails
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_layer = linear(input_dim, hidden_dim, vb.pp("hidden_layer"))
            .context("Failed to create hidden layer")?;
        let output_layer = linear(hidden_dim, output_dim, vb.pp("output_layer"))
            .context("Failed to create output layer")?;
        let residual_layer = linear(input_dim, output_dim, vb.pp("residual_layer"))
            .context("Failed to create residual layer")?;

        Ok(Self {
            hidden_layer,
            output_layer,
            residual_layer,
        })
    }

    /// Forward pass through residual MLP.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// Output tensor after residual MLP transformation
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Main path: input -> hidden -> relu -> output
        let hidden = self.hidden_layer.forward(x)?;
        let hidden = hidden.relu()?;
        let main_output = self.output_layer.forward(&hidden)?;

        // Residual path: input -> residual
        let residual = self.residual_layer.forward(x)?;

        // Combine with residual connection
        let output = (main_output + residual)?;

        Ok(output)
    }
}
