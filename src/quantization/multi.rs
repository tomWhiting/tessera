use super::Quantization;
use crate::error::{Result, TesseraError};

/// Quantize every vector in a non-empty, rectangular multi-vector embedding.
///
/// # Errors
///
/// Returns an error when the collection is empty, vector dimensions differ,
/// or a vector cannot be quantized safely.
pub fn quantize_multi<Q: Quantization>(
    quantizer: &Q,
    vectors: &[Vec<f32>],
) -> Result<Vec<Q::Output>> {
    let Some(first) = vectors.first() else {
        return Err(quantization_error(
            "Multi-vector embedding must contain at least one vector",
        ));
    };
    let expected_dim = first.len();
    if expected_dim == 0 {
        return Err(quantization_error(
            "Quantized vectors must have a non-zero dimension",
        ));
    }

    vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            if vector.len() != expected_dim {
                return Err(quantization_error(format!(
                    "Vector {index} has dimension {}, expected {expected_dim}",
                    vector.len()
                )));
            }
            quantizer.quantize_vector(vector)
        })
        .collect()
}

/// Compute quantized `MaxSim` for non-empty query and document collections.
///
/// # Errors
///
/// Returns an error when either collection is empty, vectors are
/// incompatible, or scoring produces a non-finite result.
pub fn multi_vector_distance<Q: Quantization>(
    quantizer: &Q,
    query: &[Q::Output],
    document: &[Q::Output],
) -> Result<f32> {
    if query.is_empty() {
        return Err(quantization_error(
            "MaxSim query must contain at least one vector",
        ));
    }
    if document.is_empty() {
        return Err(quantization_error(
            "MaxSim document must contain at least one vector",
        ));
    }

    let mut total = 0.0_f32;
    for query_vector in query {
        let mut best = f32::NEG_INFINITY;
        for document_vector in document {
            let score = quantizer.distance(query_vector, document_vector)?;
            if !score.is_finite() {
                return Err(quantization_error(
                    "Quantized distance produced a non-finite score",
                ));
            }
            best = best.max(score);
        }

        total += best;
        if !total.is_finite() {
            return Err(quantization_error(
                "Quantized MaxSim score exceeded the finite f32 range",
            ));
        }
    }

    Ok(total)
}

fn quantization_error(message: impl Into<String>) -> TesseraError {
    TesseraError::QuantizationError(message.into())
}

#[cfg(test)]
#[path = "multi/tests.rs"]
mod tests;
