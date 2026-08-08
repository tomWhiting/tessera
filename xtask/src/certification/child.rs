use std::fs;
use std::path::Path;

use candle_core::Device;
use tessera::{
    configure_cpu_threads, max_sim, ResourcePolicy, TesseraDense, TesseraMultiVector,
    TesseraSparse, Tokenizer,
};

use super::artifacts;
use super::evidence::{ChildOutcome, SmokeObservation};
use super::reference::{
    self, ComparisonStatus, LoadedReference, ReferenceComparison, ReferenceOutput, ReferenceProbe,
};
use super::smoke_math::{cosine, cosine_iter, min_max, norm, sparse_cosine, sparse_dot};
use super::smoke_observation::{base_checks, check, observation};
use super::spec::{
    CertResult, CertificationSpec, ProfileKind, ProfileSpec, Representation, SemanticMode,
};

pub(crate) fn run(
    repository: &Path,
    model_id: &str,
    profile_name: &str,
    outcome_path: &Path,
) -> CertResult<()> {
    let loaded = super::spec::load_model(repository, model_id)?;
    let profile = loaded.spec.profile(profile_name)?;
    artifacts::configure_cache(repository)?;
    std::env::set_var("TESSERA_OFFLINE", "1");
    configure_cpu_threads(profile.process.cpu_threads)?;

    let official_reference = reference::load_optional(repository, &loaded.spec, profile_name)?;
    let result = execute(
        repository,
        &loaded.spec,
        profile,
        official_reference.as_ref(),
    );
    let outcome = match result {
        Ok((verified_artifacts, observation, reference_comparison)) => {
            let passed = observation.checks.iter().all(|check| check.passed)
                && reference_comparison.status != ComparisonStatus::Failed;
            ChildOutcome {
                status: if passed { "passed" } else { "failed" }.to_string(),
                error: (!passed).then(|| {
                    "one or more smoke or official-reference contracts failed".to_string()
                }),
                verified_artifacts,
                observation: Some(observation),
                reference_comparison,
            }
        }
        Err(error) => ChildOutcome {
            status: "failed".to_string(),
            error: Some(error.to_string()),
            verified_artifacts: Vec::new(),
            observation: None,
            reference_comparison: official_reference.as_ref().map_or_else(
                ReferenceComparison::not_configured,
                |loaded| {
                    ReferenceComparison::not_run(loaded, "inference did not produce an output")
                },
            ),
        },
    };
    let passed = outcome.status == "passed";
    fs::write(outcome_path, serde_json::to_vec_pretty(&outcome)?)?;
    if passed {
        Ok(())
    } else {
        Err(outcome
            .error
            .unwrap_or_else(|| "smoke failed".to_string())
            .into())
    }
}

fn execute(
    repository: &Path,
    spec: &CertificationSpec,
    profile: &ProfileSpec,
    official_reference: Option<&LoadedReference>,
) -> CertResult<(
    Vec<artifacts::VerifiedArtifact>,
    SmokeObservation,
    ReferenceComparison,
)> {
    if profile.kind == ProfileKind::LongContext && official_reference.is_none() {
        return Err(
            "long-context execution requires a checked near-limit official reference probe".into(),
        );
    }
    let verified = artifacts::verify_cached(
        repository,
        &super::spec::load_model(repository, &spec.model.id)?,
    )?;
    let policy = resource_policy(profile);
    if let Some(reference) = official_reference {
        verify_probe_tokens(spec, policy, reference)?;
    }
    let (observation, observed_reference) = match spec.model.representation {
        Representation::Dense => dense_smoke(spec, policy, official_reference)?,
        Representation::MultiVector => multi_vector_smoke(spec, policy, official_reference)?,
        Representation::Sparse => sparse_smoke(spec, policy, official_reference)?,
        Representation::Vision => {
            super::vision_smoke::run(repository, spec, policy, official_reference)?
        }
    };
    let comparison = match (official_reference, observed_reference) {
        (Some(reference), Some(observed)) => reference::compare(reference, &observed)?,
        (Some(reference), None) => {
            ReferenceComparison::not_run(reference, "reference probe was not executed")
        }
        (None, _) => ReferenceComparison::not_configured(),
    };
    Ok((verified, observation, comparison))
}

fn verify_probe_tokens(
    spec: &CertificationSpec,
    policy: ResourcePolicy,
    reference: &LoadedReference,
) -> CertResult<()> {
    let tokenizer = Tokenizer::from_pretrained_with_policy(&spec.model.repository, policy)?;
    let (text, expected_tokens) = match &reference.document.probe {
        ReferenceProbe::Text { text, token_count } => (text, *token_count),
        ReferenceProbe::Image {
            query,
            query_token_count,
            ..
        } => (query, *query_token_count),
    };
    let observed_tokens = tokenizer.encode(text, true)?.0.len();
    validate_probe_token_count(expected_tokens, observed_tokens)
}

fn validate_probe_token_count(expected_tokens: usize, observed_tokens: usize) -> CertResult<()> {
    if observed_tokens != expected_tokens {
        return Err(format!(
            "official reference probe declares {expected_tokens} tokens but the pinned tokenizer produced {observed_tokens}"
        )
        .into());
    }
    Ok(())
}

fn resource_policy(profile: &ProfileSpec) -> ResourcePolicy {
    let limits = &profile.resource_policy;
    ResourcePolicy::new(
        limits.max_sequence_tokens,
        limits.max_batch_items,
        limits.max_batch_tokens,
        limits.max_model_bytes,
    )
    .with_max_input_bytes_per_sequence(limits.max_input_bytes_per_sequence)
    .with_max_attention_cells(limits.max_attention_cells)
    .with_max_job_items(limits.max_job_items)
    .with_max_job_input_bytes(limits.max_job_input_bytes)
    .with_max_output_bytes(limits.max_output_bytes)
    .with_max_activation_bytes(limits.max_activation_bytes)
}

fn dense_smoke(
    spec: &CertificationSpec,
    policy: ResourcePolicy,
    official_reference: Option<&LoadedReference>,
) -> CertResult<(SmokeObservation, Option<ReferenceOutput>)> {
    let fixture = &spec.smoke.fixture;
    let embedder = TesseraDense::builder()
        .model(&spec.model.id)
        .device(Device::Cpu)
        .batch_size(2)
        .resource_policy(policy)
        .build()?;
    let query = embedder.encode(&fixture.query)?;
    let repeated = embedder.encode(&fixture.query)?;
    let positive = embedder.encode(&fixture.positive)?;
    let negative = embedder.encode(&fixture.negative)?;
    let batch = embedder.encode_batch(&[&fixture.query, &fixture.positive])?;
    let vectors = [
        query
            .values()
            .as_slice()
            .ok_or("query output is not contiguous")?,
        repeated
            .values()
            .as_slice()
            .ok_or("repeated output is not contiguous")?,
        positive
            .values()
            .as_slice()
            .ok_or("positive output is not contiguous")?,
        negative
            .values()
            .as_slice()
            .ok_or("negative output is not contiguous")?,
    ];
    let norms = vectors
        .iter()
        .map(|vector| norm(vector))
        .collect::<Vec<_>>();
    let repeat_similarity = cosine(vectors[0], vectors[1]);
    let relevant_score = cosine(vectors[0], vectors[2]);
    let unrelated_score = cosine(vectors[0], vectors[3]);
    let finite = vectors
        .iter()
        .flat_map(|vector| vector.iter())
        .all(|value| value.is_finite());
    let batch_shapes = batch
        .iter()
        .map(|value| vec![value.dim()])
        .collect::<Vec<_>>();
    let mut checks = base_checks(
        spec,
        query.dim(),
        finite,
        repeat_similarity,
        relevant_score - unrelated_score,
    );
    checks.push(check(
        "batch-shape",
        batch_shapes == vec![vec![spec.smoke.expected_dimension]; 2],
        format!("observed {batch_shapes:?}"),
    ));
    let [batch_query_embedding, batch_positive_embedding] = batch.as_slice() else {
        return Err(format!("dense batch returned {} outputs; expected 2", batch.len()).into());
    };
    let batch_query = batch_query_embedding
        .values()
        .as_slice()
        .ok_or("batch query output is not contiguous")?;
    let batch_positive = batch_positive_embedding
        .values()
        .as_slice()
        .ok_or("batch positive output is not contiguous")?;
    let batch_parity = cosine(vectors[0], batch_query).min(cosine(vectors[2], batch_positive));
    checks.push(check(
        "batch-sequential-parity",
        batch_parity >= spec.smoke.repeat_similarity_minimum,
        format!("minimum cosine {batch_parity}"),
    ));
    if spec.smoke.normalized {
        checks.push(check(
            "l2-normalized",
            norms.iter().all(|value| (value - 1.0).abs() <= 0.01),
            format!("norm range {:?}", min_max(&norms)),
        ));
    }
    let observed_reference = official_reference
        .map(|reference| {
            let text = reference_text(reference)?;
            let output = embedder.encode(text)?;
            let values = output
                .values()
                .as_slice()
                .ok_or("official-reference dense output is not contiguous")?
                .to_vec();
            Ok::<_, Box<dyn std::error::Error>>(ReferenceOutput::Dense { values })
        })
        .transpose()?;
    Ok((
        observation(
            "dense",
            vec![query.dim()],
            batch_shapes,
            finite,
            Some(&norms),
            None,
            repeat_similarity,
            relevant_score,
            unrelated_score,
            checks,
        ),
        observed_reference,
    ))
}

fn multi_vector_smoke(
    spec: &CertificationSpec,
    policy: ResourcePolicy,
    official_reference: Option<&LoadedReference>,
) -> CertResult<(SmokeObservation, Option<ReferenceOutput>)> {
    let fixture = &spec.smoke.fixture;
    let embedder = TesseraMultiVector::builder()
        .model(&spec.model.id)
        .device(Device::Cpu)
        .resource_policy(policy)
        .build()?;
    let query = embedder.encode_query(&fixture.query)?;
    let repeated = embedder.encode_query(&fixture.query)?;
    let positive = embedder.encode_document(&fixture.positive)?;
    let negative = embedder.encode_document(&fixture.negative)?;
    let mut batch = embedder.encode_query_batch(&[&fixture.query])?;
    batch.extend(embedder.encode_document_batch(&[&fixture.positive])?);
    let repeated_shape_matches = query.shape() == repeated.shape();
    let repeat_similarity = if repeated_shape_matches {
        cosine_iter(query.matrix().iter(), repeated.matrix().iter())
    } else {
        f32::NEG_INFINITY
    };
    let relevant_score = max_sim(&query, &positive)?;
    let unrelated_score = max_sim(&query, &negative)?;
    let finite = [&query, &repeated, &positive, &negative]
        .iter()
        .flat_map(|value| value.matrix().iter())
        .all(|value| value.is_finite());
    let norms = [&query, &repeated, &positive, &negative]
        .iter()
        .flat_map(|value| value.matrix().rows())
        .map(|row| row.iter().map(|value| value * value).sum::<f32>().sqrt())
        .collect::<Vec<_>>();
    let batch_shapes = batch
        .iter()
        .map(|value| vec![value.num_tokens(), value.embedding_dim()])
        .collect::<Vec<_>>();
    let mut checks = base_checks(
        spec,
        query.embedding_dim(),
        finite,
        repeat_similarity,
        relevant_score - unrelated_score,
    );
    checks.push(check(
        "batch-shape",
        batch_shapes.len() == 2
            && batch_shapes
                .iter()
                .all(|shape| shape[0] > 0 && shape[1] == spec.smoke.expected_dimension),
        format!("observed {batch_shapes:?}"),
    ));
    let [batch_query, batch_positive] = batch.as_slice() else {
        return Err(format!(
            "multi-vector batch returned {} outputs; expected 2",
            batch.len()
        )
        .into());
    };
    let batch_shape_matches =
        query.shape() == batch_query.shape() && positive.shape() == batch_positive.shape();
    let batch_parity = cosine_iter(query.matrix().iter(), batch_query.matrix().iter()).min(
        cosine_iter(positive.matrix().iter(), batch_positive.matrix().iter()),
    );
    checks.push(check(
        "batch-sequential-parity",
        batch_shape_matches && batch_parity >= spec.smoke.repeat_similarity_minimum,
        format!("shape-match={batch_shape_matches}, minimum cosine {batch_parity}"),
    ));
    if spec.smoke.normalized {
        checks.push(check(
            "row-normalized",
            norms.iter().all(|value| (value - 1.0).abs() <= 0.01),
            format!("norm range {:?}", min_max(&norms)),
        ));
    }
    let observed_reference = official_reference
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
        .transpose()?;
    Ok((
        observation(
            "multi_vector",
            vec![query.num_tokens(), query.embedding_dim()],
            batch_shapes,
            finite,
            Some(&norms),
            None,
            repeat_similarity,
            relevant_score,
            unrelated_score,
            checks,
        ),
        observed_reference,
    ))
}

fn sparse_smoke(
    spec: &CertificationSpec,
    policy: ResourcePolicy,
    official_reference: Option<&LoadedReference>,
) -> CertResult<(SmokeObservation, Option<ReferenceOutput>)> {
    let fixture = &spec.smoke.fixture;
    let embedder = TesseraSparse::builder()
        .model(&spec.model.id)
        .device(Device::Cpu)
        .resource_policy(policy)
        .build()?;
    let query = embedder.encode(&fixture.query)?;
    let repeated = embedder.encode(&fixture.query)?;
    let positive = embedder.encode(&fixture.positive)?;
    let negative = embedder.encode(&fixture.negative)?;
    let batch = embedder.encode_batch(&[&fixture.query, &fixture.positive])?;
    let repeat_similarity = sparse_cosine(query.entries(), repeated.entries());
    let relevant_score = sparse_dot(query.entries(), positive.entries());
    let unrelated_score = sparse_dot(query.entries(), negative.entries());
    let finite = [&query, &repeated, &positive, &negative]
        .iter()
        .flat_map(|value| value.entries().iter())
        .all(|(_, value)| value.is_finite() && *value > 0.0);
    let ordered = [&query, &repeated, &positive, &negative]
        .iter()
        .all(|value| value.entries().windows(2).all(|pair| pair[0].0 < pair[1].0));
    let batch_shapes = batch
        .iter()
        .map(|value| vec![value.nnz()])
        .collect::<Vec<_>>();
    let expected_vocab = spec.smoke.expected_vocabulary_size.unwrap_or(0);
    let mut checks = base_checks(
        spec,
        query.vocab_size(),
        finite,
        repeat_similarity,
        relevant_score - unrelated_score,
    );
    checks.push(check(
        "vocabulary-size",
        query.vocab_size() == expected_vocab && embedder.vocab_size() == expected_vocab,
        format!("observed {}", query.vocab_size()),
    ));
    checks.push(check(
        "sorted-unique-positive-weights",
        ordered && finite,
        format!("ordered={ordered}, finite-positive={finite}"),
    ));
    let [batch_query, batch_positive] = batch.as_slice() else {
        return Err(format!("sparse batch returned {} outputs; expected 2", batch.len()).into());
    };
    let batch_shape_matches = query.vocab_size() == batch_query.vocab_size()
        && positive.vocab_size() == batch_positive.vocab_size();
    let batch_parity = sparse_cosine(query.entries(), batch_query.entries())
        .min(sparse_cosine(positive.entries(), batch_positive.entries()));
    checks.push(check(
        "batch-sequential-parity",
        batch_shape_matches && batch_parity >= spec.smoke.repeat_similarity_minimum,
        format!("shape-match={batch_shape_matches}, minimum cosine {batch_parity}"),
    ));
    let observed_reference = official_reference
        .map(|reference| {
            let text = reference_text(reference)?;
            let output = embedder.encode(text)?;
            let (indices, values) = output.entries().iter().copied().unzip();
            Ok::<_, Box<dyn std::error::Error>>(ReferenceOutput::Sparse {
                vocabulary_size: output.vocab_size(),
                indices,
                values,
            })
        })
        .transpose()?;
    Ok((
        observation(
            "sparse",
            vec![query.vocab_size()],
            batch_shapes,
            finite,
            None,
            Some(query.nnz()),
            repeat_similarity,
            relevant_score,
            unrelated_score,
            checks,
        ),
        observed_reference,
    ))
}

fn reference_text(reference: &LoadedReference) -> CertResult<&str> {
    match &reference.document.probe {
        ReferenceProbe::Text { text, .. } => Ok(text),
        ReferenceProbe::Image { .. } => Err("text model received an image reference probe".into()),
    }
}

#[cfg(test)]
#[path = "tests/child.rs"]
mod tests;
