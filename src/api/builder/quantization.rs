/// Quantization configuration for embeddings.
///
/// Enables an alternative compressed embedding representation. Quantization
/// changes the scoring metric; evaluate retrieval quality and performance on
/// the target model and corpus before relying on it.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum QuantizationConfig {
    /// No quantization (full precision float32)
    None,
    /// Binary quantization (one sign bit per floating-point dimension).
    ///
    /// The packed payload is up to 32 times smaller than an equivalent F32
    /// payload, excluding byte padding and container overhead. No fixed
    /// ranking-retention or speedup claim is made without a checked benchmark.
    Binary,
}
