use super::TesseraMultiVector;
use crate::api::QuantizedEmbeddings;
use crate::core::TokenEmbeddings;
use crate::error::{Result, TesseraError};
use crate::quantization::{multi_vector_distance, quantize_multi};

impl TesseraMultiVector {
    /// Quantizes a token matrix to one-bit sign vectors.
    pub fn quantize(&self, embeddings: &TokenEmbeddings) -> Result<QuantizedEmbeddings> {
        match &self.quantizer {
            Some(quantizer) => {
                let vectors = embeddings
                    .matrix()
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect::<Vec<_>>();
                QuantizedEmbeddings::new(quantize_multi(quantizer, &vectors)?)
            }
            None => Err(quantizer_not_configured()),
        }
    }

    /// Encodes generic, untyped text and quantizes it.
    pub fn encode_quantized(&self, text: &str) -> Result<QuantizedEmbeddings> {
        let embeddings = self.encode(text)?;
        self.quantize(&embeddings)
    }

    /// Encodes a retrieval query and quantizes it.
    pub fn encode_query_quantized(&self, text: &str) -> Result<QuantizedEmbeddings> {
        let embeddings = self.encode_query(text)?;
        self.quantize(&embeddings)
    }

    /// Encodes a retrieval document and quantizes it.
    pub fn encode_document_quantized(&self, text: &str) -> Result<QuantizedEmbeddings> {
        let embeddings = self.encode_document(text)?;
        self.quantize(&embeddings)
    }

    /// Computes quantized late-interaction similarity.
    pub fn similarity_quantized(
        &self,
        query: &QuantizedEmbeddings,
        document: &QuantizedEmbeddings,
    ) -> Result<f32> {
        self.quantizer.as_ref().map_or_else(
            || Err(quantizer_not_configured()),
            |quantizer| multi_vector_distance(quantizer, query.vectors(), document.vectors()),
        )
    }
}

fn quantizer_not_configured() -> TesseraError {
    TesseraError::QuantizationError(
        "No quantizer configured. Use .quantization(QuantizationConfig::Binary) in builder"
            .to_string(),
    )
}
