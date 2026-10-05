use std::path::Path;

use tessera::{EmbeddingOutcome, TesseraDense, TesseraMultiVector};

use super::{reference_text, CertResult, LoadedReference, ReferenceOutput, SemanticMode};

pub(super) fn dense(
    embedder: &TesseraDense,
    official_reference: Option<&LoadedReference>,
    installed: bool,
    tokenizer_path: Option<&Path>,
) -> CertResult<Option<ReferenceOutput>> {
    official_reference
        .map(|reference| {
            let source_text = reference_text(reference)?;
            let expected_used = reference.document.probe.cut_at_tokens();
            let text = match expected_used {
                Some(used) => constructed_probe(
                    source_text,
                    reference.document.probe.token_count(),
                    used,
                    tokenizer_path.ok_or("constructed_probe_tokenizer_missing")?,
                )?,
                None => source_text.to_owned(),
            };
            let values = if installed || expected_used.is_some() {
                let EmbeddingOutcome::Embedded(output) = embedder.encode_outcome(&text, None)?
                else {
                    return Err(if expected_used.is_some() {
                        "cut_reference_refused: constructed probe was refused before inference"
                    } else {
                        "installed constructed probe was refused before inference"
                    }
                    .into());
                };
                validate_dense_probe_counts(
                    reference.document.probe.token_count(),
                    expected_used,
                    output.tokens_total(),
                )?;
                output
                    .values()
                    .as_slice()
                    .ok_or("installed reference dense output is not contiguous")?
                    .to_vec()
            } else {
                let output = embedder.encode(&text)?;
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

pub(in crate::certification) fn validate_dense_probe_counts(
    expected_total: usize,
    expected_used: Option<usize>,
    observed_total: usize,
) -> CertResult<()> {
    let expected = expected_used.unwrap_or(expected_total);
    if observed_total != expected {
        return Err(format!(
            "constructed_probe_token_mismatch: expected={expected}; observed={observed_total}"
        )
        .into());
    }
    Ok(())
}

pub(in crate::certification) fn constructed_probe(
    text: &str,
    expected_total: usize,
    used: usize,
    tokenizer_path: &Path,
) -> CertResult<String> {
    let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|error| format!("constructed_probe_tokenizer_invalid: {error}"))?;
    tokenizer
        .with_truncation(None)
        .map_err(|error| format!("constructed_probe_truncation_invalid: {error}"))?;
    tokenizer.with_padding(None);
    probe_span(text, expected_total, used, &tokenizer).map(str::to_owned)
}

fn probe_span<'a>(
    text: &'a str,
    expected_total: usize,
    used: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> CertResult<&'a str> {
    let encoded = tokenizer
        .encode(text, true)
        .map_err(|error| format!("constructed_probe_tokenization_failed: {error}"))?;
    if encoded.len() != expected_total {
        return Err(format!(
            "constructed_probe_source_token_mismatch: expected={expected_total}; observed={}",
            encoded.len()
        )
        .into());
    }
    let special_tokens = encoded
        .get_special_tokens_mask()
        .iter()
        .filter(|&&mask| mask != 0)
        .count();
    let content_tokens = used
        .checked_sub(special_tokens)
        .filter(|&count| count > 0)
        .ok_or("constructed_probe_limit_invalid: special tokens leave no content")?;
    let (_, end) = encoded
        .get_offsets()
        .iter()
        .zip(encoded.get_special_tokens_mask())
        .filter(|(_, mask)| **mask == 0)
        .nth(content_tokens - 1)
        .map(|(offset, _)| *offset)
        .ok_or("constructed_probe_content_missing")?;
    text.get(..end)
        .filter(|probe| !probe.is_empty())
        .ok_or_else(|| "constructed_probe_offset_invalid".into())
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
