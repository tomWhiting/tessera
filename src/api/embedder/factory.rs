use super::{TesseraDense, TesseraMultiVector, TesseraSparse, TesseraVision};
use crate::error::{Result, TesseraError};
use crate::models::registry::{get_model, ModelType};

/// Unified embedder that auto-detects model type.
///
/// This enum provides a smart factory pattern that automatically creates
/// the appropriate dense, multi-vector, sparse, or vision variant based on the
/// model type in the registry. Catalog-only entries are rejected before
/// dispatch.
///
/// # Example
///
/// ```ignore
/// use tessera::Tessera;
///
/// // Auto-detects ColBERT model -> creates MultiVector variant
/// let colbert = Tessera::new("colbert-v2")?;
///
/// // Auto-detects dense model -> creates Dense variant
/// let bge = Tessera::new("bge-base-en-v1.5")?;
///
/// // Auto-detects sparse model -> creates Sparse variant
/// let splade = Tessera::new("splade-pp-en-v1")?;
///
/// // Vision is also a factory variant. The current 3B F32 adapter requires an
/// // explicit model-memory policy, so construct it with TesseraVisionBuilder.
///
/// // Pattern match to use specific API
/// match colbert {
///     Tessera::MultiVector(mv) => {
///         let embeddings = mv.encode("query")?;
///         println!("Got {} tokens", embeddings.num_tokens);
///     }
///     Tessera::Dense(d) => {
///         let embedding = d.encode("query")?;
///         println!("Got {} dimensions", embedding.dim());
///     }
///     Tessera::Sparse(s) => {
///         let embedding = s.encode("query")?;
///         println!("Got {} non-zero dimensions", embedding.nnz());
///     }
///     Tessera::Vision(v) => {
///         let doc_emb = v.encode_document("invoice.jpg")?;
///         println!("Got {} patches", doc_emb.num_patches);
///     }
/// }
/// ```
pub enum Tessera {
    /// Dense single-vector embedder
    Dense(TesseraDense),
    /// Multi-vector ColBERT-style embedder
    MultiVector(TesseraMultiVector),
    /// Sparse SPLADE-style embedder
    Sparse(TesseraSparse),
    /// Vision-language ColPali-style embedder
    Vision(TesseraVision),
}

impl Tessera {
    /// Create a new embedder with automatic model type detection.
    ///
    /// Looks up the model in the registry and creates the appropriate
    /// embedder variant based on the model type:
    /// - Dense models -> `Tessera::Dense(TesseraDense)`
    /// - MultiVector/Colbert models -> `Tessera::MultiVector(TesseraMultiVector)`
    /// - Sparse models -> `Tessera::Sparse(TesseraSparse)`
    /// - `VisionLanguage` models -> `Tessera::Vision(TesseraVision)`
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier from the registry
    ///
    /// # Returns
    ///
    /// Tessera enum variant containing the appropriate embedder.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model is not found in the registry
    /// - The entry is catalog-only or its model type has no active runtime
    /// - Model cannot be loaded
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tessera::Tessera;
    ///
    /// let embedder = Tessera::new("colbert-v2")?;
    /// let embedder = Tessera::new("bge-base-en-v1.5")?;
    /// let embedder = Tessera::new("splade-pp-en-v1")?;
    /// ```
    pub fn new(model_id: &str) -> Result<Self> {
        let model_info = get_model(model_id).ok_or_else(|| TesseraError::ModelNotFound {
            model_id: model_id.to_string(),
        })?;
        crate::api::builder::ensure_runnable_model(model_info)?;

        match model_info.model_type {
            ModelType::Dense => {
                let dense = TesseraDense::new(model_id)?;
                Ok(Self::Dense(dense))
            }
            ModelType::Colbert => {
                let mv = TesseraMultiVector::new(model_id)?;
                Ok(Self::MultiVector(mv))
            }
            ModelType::Sparse => {
                let sparse = TesseraSparse::new(model_id)?;
                Ok(Self::Sparse(sparse))
            }
            ModelType::VisionLanguage => {
                let vision = TesseraVision::new(model_id)?;
                Ok(Self::Vision(vision))
            }
            ModelType::Timeseries => Err(TesseraError::ConfigError(
                "Chronos and TimesFM remain catalog entries, but their runtimes are quarantined: \
                 stock Candle 0.11 does not expose the continuous-embedding T5 API required by \
                 Chronos, and TimesFM never had an implementation"
                    .to_string(),
            )),
            ModelType::Unified => Err(TesseraError::ConfigError(
                "Model type 'Unified' is not yet supported. Currently supported: Dense, Colbert (MultiVector), Sparse, VisionLanguage".to_string()
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tessera;
    use crate::error::TesseraError;

    #[test]
    fn unified_factory_rejects_catalog_only_models_before_dispatch() {
        let result = Tessera::new("chronos-bolt-small");

        let Err(error) = result else {
            panic!("a catalog-only model must be rejected before factory dispatch");
        };
        assert!(matches!(
            error,
            TesseraError::ConfigError(message)
                if message.contains("chronos-bolt-small")
                    && message.contains("catalog-only")
                    && message.contains("continuous-embedding")
        ));
    }
}
