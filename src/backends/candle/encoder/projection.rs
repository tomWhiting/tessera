use anyhow::Result;

#[cfg(test)]
mod tests;

/// Validates registry and checkpoint dimensions before loading ColBERT weights.
pub(super) fn validate_projection_contract(
    has_projection: bool,
    registered_projection_dim: Option<usize>,
    configured_embedding_dim: usize,
    registered_hidden_dim: usize,
    checkpoint_hidden_dim: usize,
) -> Result<()> {
    anyhow::ensure!(
        has_projection,
        "Runnable ColBERT models must declare a projection layer"
    );
    anyhow::ensure!(
        registered_projection_dim == Some(configured_embedding_dim),
        "ColBERT projection dimension {:?} does not match configured embedding dimension {}",
        registered_projection_dim,
        configured_embedding_dim
    );
    anyhow::ensure!(
        registered_hidden_dim == checkpoint_hidden_dim,
        "ColBERT checkpoint hidden dimension {checkpoint_hidden_dim} does not match registered hidden dimension {registered_hidden_dim}"
    );
    Ok(())
}
