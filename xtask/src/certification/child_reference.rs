use tessera::{CutEmbeddingOutcome, TesseraDense, TesseraMultiVector};

use super::{
    reference_text, validate_probe_token_count, CertResult, LoadedReference, ReferenceOutput,
    SemanticMode,
};

pub(super) fn dense(
    embedder: &TesseraDense,
    official_reference: Option<&LoadedReference>,
    installed: bool,
) -> CertResult<Option<ReferenceOutput>> {
    official_reference
        .map(|reference| {
            let text = reference_text(reference)?;
            let expected_used = reference.document.probe.cut_at_tokens();
            let values = if installed || expected_used.is_some() {
                let CutEmbeddingOutcome::Embedded(output) = embedder.encode_cut(text, None)? else {
                    return Err(if expected_used.is_some() {
                        "cut_reference_refused: reference probe was refused before inference"
                    } else {
                        "installed reference probe was refused before inference"
                    }
                    .into());
                };
                validate_dense_probe_counts(
                    reference.document.probe.token_count(),
                    expected_used,
                    output.tokens_total(),
                    output.tokens_read(),
                    output.cut(),
                )?;
                output
                    .values()
                    .as_slice()
                    .ok_or("installed reference dense output is not contiguous")?
                    .to_vec()
            } else {
                let output = embedder.encode(text)?;
                output
                    .values()
                    .as_slice()
                    .ok_or("official-reference dense output is not contiguous")?
                    .to_vec()
            };
            Ok::<_, Box<dyn std::error::Error>>(ReferenceOutput::Dense { values })
        })
        .transpose()
}

pub(super) fn validate_dense_probe_counts(
    expected_total: usize,
    expected_used: Option<usize>,
    observed_total: usize,
    observed_used: usize,
    cut: bool,
) -> CertResult<()> {
    if let Some(expected_used) = expected_used {
        if !cut || observed_total != expected_total || observed_used != expected_used {
            return Err(format!(
                "cut_reference_token_mismatch: expected total={expected_total}, used={expected_used}; observed total={observed_total}, used={observed_used}, cut={cut}"
            ).into());
        }
        return Ok(());
    }
    validate_probe_token_count(expected_total, observed_total)?;
    if cut {
        return Err("installed reference probe exceeds the admitted token limit".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/cut_reference.rs"]
mod tests;

pub(super) fn multi_vector(
    embedder: &TesseraMultiVector,
    official_reference: Option<&LoadedReference>,
) -> CertResult<Option<ReferenceOutput>> {
    official_reference
        .map(|reference| {
            let text = reference_text(reference)?;
            let output = match reference.document.capability.semantic_mode {
                SemanticMode::LateInteractionQuery => embedder.encode_query(text)?,
                SemanticMode::LateInteractionDocument => embedder.encode_document(text)?,
                _ => return Err("multi-vector reference has an incompatible semantic mode".into()),
            };
            Ok::<_, Box<dyn std::error::Error>>(ReferenceOutput::MultiVector {
                rows: output.num_tokens(),
                columns: output.embedding_dim(),
                values: output.matrix().iter().copied().collect(),
            })
        })
        .transpose()
}
