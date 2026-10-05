use std::fs;
use std::path::Path;

use candle_core::Device;
use tessera::{configure_cpu_threads, EmbeddingOutcome, TesseraDense};

use super::{compare_vectors, BatchComparison, ChildMeasurement};
use crate::certification::{artifacts, child, install, reference, spec};
use spec::{CertResult, Representation};

const FILLER_TEXTS: [&str; 15] = [
    "a",
    "red",
    "small",
    "a stone",
    "blue sky",
    "green leaf",
    "a quiet day",
    "one long path",
    "a small garden",
    "the sun is warm",
    "a book on a desk",
    "a calm blue river",
    "a slow walk past trees",
    "fresh grass by water",
    "a short note about rain",
];

pub(crate) fn run(
    repository: &Path,
    model: &str,
    profile_name: &str,
    threads: usize,
    batch: bool,
    outcome: &Path,
    model_dir: Option<&Path>,
) -> CertResult<()> {
    let loaded = spec::load_model(repository, model)?;
    install::require_dense(loaded.spec.model.representation)?;
    if threads == 0 {
        return Err("measurement_threads_invalid: threads must be positive".into());
    }
    let profile = loaded.spec.profile(profile_name)?;
    configure_source(repository, model_dir)?;
    configure_cpu_threads(threads)?;
    let reference = reference::load_optional(repository, &loaded.spec, profile_name)?
        .ok_or("measurement_reference_missing: profile requires a checked reference text")?;
    let reference::ReferenceProbe::Text { text, .. } = &reference.document.probe else {
        return Err("measurement_reference_invalid: dense text reference required".into());
    };
    if reference.document.expected.representation() != Representation::Dense {
        return Err("measurement_reference_invalid: dense output reference required".into());
    }
    if model_dir.is_none() {
        artifacts::verify_cached(repository, &loaded)?;
    }
    let policy = child::resource_policy(profile)
        .with_max_batch_items(16)
        .with_max_job_items(16);
    let mut builder = TesseraDense::builder()
        .model(model)
        .device(Device::Cpu)
        .batch_size(16)
        .resource_policy(policy);
    if let Some(directory) = model_dir {
        builder = builder.model_dir(directory);
    }
    let embedder = builder.build()?;
    if let Some(directory) = model_dir {
        artifacts::verify_directory(directory, &loaded)?;
    }
    let probe_text = if let Some(used) = reference.document.probe.cut_at_tokens() {
        let entry =
            tessera::models::registry::get_model(model).ok_or("constructed_probe_model_missing")?;
        let tokenizer_path = match model_dir {
            Some(directory) => directory.join(entry.tokenizer_file),
            None => artifacts::cached_artifact_path(repository, &loaded, entry.tokenizer_file)?,
        };
        crate::certification::child::child_reference::constructed_probe(
            text,
            reference.document.probe.token_count(),
            used,
            &tokenizer_path,
        )?
    } else {
        text.to_owned()
    };
    let text = probe_text.as_str();
    let vector = encode(&embedder, text, &reference.document.probe)?;
    let comparisons = if batch {
        batch_comparisons(&embedder, text, &vector)?
    } else {
        Vec::new()
    };
    let result = ChildMeasurement {
        vector,
        batch: comparisons,
        installed_manifest_sha256: embedder.installed_manifest_sha256().map(str::to_owned),
    };
    fs::write(outcome, serde_json::to_vec(&result)?)?;
    Ok(())
}

fn configure_source(repository: &Path, model_dir: Option<&Path>) -> CertResult<()> {
    if model_dir.is_some() {
        std::env::remove_var("HF_HOME");
        std::env::remove_var("TESSERA_OFFLINE");
    } else {
        artifacts::configure_cache(repository)?;
        std::env::set_var("TESSERA_OFFLINE", "1");
    }
    Ok(())
}

fn encode(
    embedder: &TesseraDense,
    text: &str,
    probe: &reference::ReferenceProbe,
) -> CertResult<Vec<f32>> {
    if let Some(used) = probe.cut_at_tokens() {
        let EmbeddingOutcome::Embedded(output) = embedder.encode_outcome(text, None)? else {
            return Err("measurement_input_refused: cut reference text was refused".into());
        };
        crate::certification::child::child_reference::validate_dense_probe_counts(
            probe.token_count(),
            Some(used),
            output.tokens_total(),
        )?;
        Ok(output.values().iter().copied().collect())
    } else {
        Ok(embedder.encode(text)?.values().iter().copied().collect())
    }
}

fn batch_comparisons(
    embedder: &TesseraDense,
    text: &str,
    alone: &[f32],
) -> CertResult<Vec<BatchComparison>> {
    [0, 7, 15]
        .into_iter()
        .map(|position| {
            let mut texts = FILLER_TEXTS.to_vec();
            texts.insert(position, text);
            let outputs = embedder.encode_batch(texts.as_slice())?;
            if outputs.len() != 16 {
                return Err("measurement_batch_shape: expected sixteen vectors".into());
            }
            let observed = outputs
                .get(position)
                .ok_or("measurement_batch_shape: reference position absent")?
                .values()
                .iter()
                .copied()
                .collect::<Vec<_>>();
            Ok(BatchComparison {
                position,
                comparison: compare_vectors(alone, &observed)?,
            })
        })
        .collect()
}
