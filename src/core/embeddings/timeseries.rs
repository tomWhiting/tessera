use super::Encoder;
use anyhow::Result;

/// Time series embedding representation for time series foundation models.
///
/// Represents time series data as fixed-size embedding vectors suitable for
/// similarity search, clustering, and retrieval. Unlike forecasting outputs,
/// embeddings are designed to capture the temporal patterns in a compressed
/// representation for downstream tasks.
///
/// # Example Models
/// - amazon/chronos-bolt-small
/// - google/timesfm-1.0-200m
#[derive(Debug, Clone)]
pub struct TimeSeriesEmbedding {
    /// Embedding vectors: [`num_series`, `embedding_dim`]
    ///
    /// For batch processing, this contains embeddings for multiple time series.
    /// Each row represents the embedding for one time series.
    embeddings: Vec<Vec<f32>>,

    /// Number of time series in the batch.
    num_series: usize,

    /// Embedding dimension (e.g., 512 for Chronos Bolt).
    embedding_dim: usize,

    /// Optional: Original time series lengths before padding.
    ///
    /// Useful for tracking which series were padded/truncated during preprocessing.
    original_lengths: Option<Vec<usize>>,

    /// Optional: Source identifier for tracking data origin.
    source: Option<String>,
}

impl TimeSeriesEmbedding {
    /// Create a new time series embedding.
    ///
    /// # Arguments
    /// * `embeddings` - The embedding vectors [`num_series`, `embedding_dim`]
    /// * `num_series` - Number of time series in the batch
    /// * `embedding_dim` - Dimension of each embedding vector
    /// * `original_lengths` - Optional original lengths before preprocessing
    /// * `source` - Optional source identifier
    ///
    /// # Returns
    /// A new validated `TimeSeriesEmbedding` instance
    ///
    /// # Errors
    ///
    /// Returns an error if the declared shape or optional length metadata does
    /// not match the data, or if an embedding contains a non-finite value.
    pub fn new(
        embeddings: Vec<Vec<f32>>,
        num_series: usize,
        embedding_dim: usize,
        original_lengths: Option<Vec<usize>>,
        source: Option<String>,
    ) -> Result<Self> {
        anyhow::ensure!(
            num_series > 0,
            "Time-series embedding must contain a series"
        );
        anyhow::ensure!(
            embedding_dim > 0,
            "Time-series embedding dimension must be greater than zero"
        );
        anyhow::ensure!(
            embeddings.len() == num_series,
            "Time-series embedding contains {} vectors, expected {num_series}",
            embeddings.len()
        );
        for (series_index, embedding) in embeddings.iter().enumerate() {
            anyhow::ensure!(
                embedding.len() == embedding_dim,
                "Time-series vector {series_index} has dimension {}, expected {embedding_dim}",
                embedding.len()
            );
            anyhow::ensure!(
                embedding.iter().all(|value| value.is_finite()),
                "Time-series vector {series_index} contains NaN or Inf values"
            );
        }
        if let Some(lengths) = &original_lengths {
            anyhow::ensure!(
                lengths.len() == num_series,
                "Original-length metadata contains {} entries, expected {num_series}",
                lengths.len()
            );
            anyhow::ensure!(
                lengths.iter().all(|length| *length > 0),
                "Original time-series lengths must be greater than zero"
            );
        }

        Ok(Self {
            embeddings,
            num_series,
            embedding_dim,
            original_lengths,
            source,
        })
    }

    /// Get the number of time series in this embedding.
    #[must_use]
    pub const fn num_series(&self) -> usize {
        self.num_series
    }

    /// Borrow the series embedding vectors.
    #[must_use]
    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// Get the embedding dimension.
    #[must_use]
    pub const fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the shape of the embedding matrix as (`num_series`, `embedding_dim`).
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.num_series, self.embedding_dim)
    }

    /// Borrow original input lengths, when recorded.
    #[must_use]
    pub fn original_lengths(&self) -> Option<&[usize]> {
        self.original_lengths.as_deref()
    }

    /// Get the source identifier if available.
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }
}

#[cfg(test)]
#[path = "timeseries/tests.rs"]
mod tests;

/// Time series encoder producing fixed-size embeddings from temporal data.
///
/// Encodes time series into fixed-size vector representations suitable for
/// similarity search, clustering, and retrieval. These models can typically
/// also perform forecasting, but the primary use case is embedding extraction.
///
/// # Characteristics
/// - Fixed-length output (one vector per time series)
/// - Temporal pattern compression
/// - Designed for similarity-based retrieval
/// - Typically 192-1280 dimensions
/// - Context lengths from 512 to 2048+ timesteps
///
/// # Example Models
/// - amazon/chronos-bolt-small (512-dim)
/// - google/timesfm-1.0-200m (1280-dim)
pub trait TimeSeriesEncoder: Encoder<Output = TimeSeriesEmbedding> {
    /// Get the embedding dimension.
    ///
    /// # Returns
    /// Dimensionality of the output embedding vector
    fn embedding_dim(&self) -> usize;

    /// Get the context length (maximum input timesteps).
    ///
    /// # Returns
    /// Maximum number of timesteps the encoder can process
    fn context_length(&self) -> usize;

    /// Get the prediction length (forecast horizon).
    ///
    /// # Returns
    /// Number of future timesteps the model can predict (if forecasting is supported)
    fn prediction_length(&self) -> usize;

    /// Forecast future values from historical data.
    ///
    /// # Arguments
    /// * `input` - Historical time series data [batch, channels, timesteps]
    ///
    /// # Returns
    /// Predicted future values [batch, channels, `prediction_length`]
    ///
    /// # Errors
    /// Returns error if forecasting fails or is not supported
    fn forecast(&self, input: &candle_core::Tensor) -> Result<candle_core::Tensor>;

    /// Extract embeddings for similarity search.
    ///
    /// # Arguments
    /// * `input` - Time series data [batch, channels, timesteps]
    ///
    /// # Returns
    /// Fixed-size embeddings [batch, `embedding_dim`]
    ///
    /// # Errors
    /// Returns error if embedding extraction fails
    fn extract_embeddings(&self, input: &candle_core::Tensor) -> Result<candle_core::Tensor>;
}
