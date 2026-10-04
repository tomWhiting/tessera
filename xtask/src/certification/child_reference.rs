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
            let values = if installed {
                let CutEmbeddingOutcome::Embedded(output) = embedder.encode_cut(text)? else {
                    return Err("installed reference probe was refused before inference".into());
                };
                validate_probe_token_count(
                    reference.document.probe.token_count(),
                    output.tokens_total(),
                )?;
                if output.cut() {
                    return Err("installed reference probe exceeds the admitted token limit".into());
                }
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
