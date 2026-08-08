use std::fs;
use std::path::Path;

use candle_core::Device;
use tessera::{
    configure_cpu_threads, max_sim, ResourcePolicy, TesseraDense, TesseraMultiVector, TesseraSparse,
};

use super::artifacts;
use super::evidence::{CheckEvidence, ChildOutcome, SmokeObservation};
use super::spec::{CertResult, CertificationSpec, ProfileSpec, Representation};

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

    let result = execute(repository, &loaded.spec, profile);
    let outcome = match result {
        Ok((verified_artifacts, observation)) => {
            let passed = observation.checks.iter().all(|check| check.passed);
            ChildOutcome {
                status: if passed { "passed" } else { "failed" }.to_string(),
                error: (!passed).then(|| "one or more smoke contracts failed".to_string()),
                verified_artifacts,
                observation: Some(observation),
            }
        }
        Err(error) => ChildOutcome {
            status: "failed".to_string(),
            error: Some(error.to_string()),
            verified_artifacts: Vec::new(),
            observation: None,
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
) -> CertResult<(Vec<artifacts::VerifiedArtifact>, SmokeObservation)> {
    let verified = artifacts::verify_cached(
        repository,
        &super::spec::load_model(repository, &spec.model.id)?,
    )?;
    let policy = resource_policy(profile);
    let observation = match spec.model.representation {
        Representation::Dense => dense_smoke(spec, policy)?,
        Representation::MultiVector => multi_vector_smoke(spec, policy)?,
        Representation::Sparse => sparse_smoke(spec, policy)?,
        Representation::Vision => {
            return Err(
                "vision certification requires a checked image fixture in a later profile".into(),
            );
        }
    };
    Ok((verified, observation))
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
}

fn dense_smoke(spec: &CertificationSpec, policy: ResourcePolicy) -> CertResult<SmokeObservation> {
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
            .embedding
            .as_slice()
            .ok_or("query output is not contiguous")?,
        repeated
            .embedding
            .as_slice()
            .ok_or("repeated output is not contiguous")?,
        positive
            .embedding
            .as_slice()
            .ok_or("positive output is not contiguous")?,
        negative
            .embedding
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
        .embedding
        .as_slice()
        .ok_or("batch query output is not contiguous")?;
    let batch_positive = batch_positive_embedding
        .embedding
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
    Ok(observation(
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
    ))
}

fn multi_vector_smoke(
    spec: &CertificationSpec,
    policy: ResourcePolicy,
) -> CertResult<SmokeObservation> {
    let fixture = &spec.smoke.fixture;
    let embedder = TesseraMultiVector::builder()
        .model(&spec.model.id)
        .device(Device::Cpu)
        .resource_policy(policy)
        .build()?;
    let query = embedder.encode(&fixture.query)?;
    let repeated = embedder.encode(&fixture.query)?;
    let positive = embedder.encode(&fixture.positive)?;
    let negative = embedder.encode(&fixture.negative)?;
    let batch = embedder.encode_batch(&[&fixture.query, &fixture.positive])?;
    let repeated_shape_matches =
        query.num_tokens == repeated.num_tokens && query.embedding_dim == repeated.embedding_dim;
    let repeat_similarity = if repeated_shape_matches {
        cosine_iter(query.embeddings.iter(), repeated.embeddings.iter())
    } else {
        f32::NEG_INFINITY
    };
    let relevant_score = max_sim(&query, &positive)?;
    let unrelated_score = max_sim(&query, &negative)?;
    let finite = [&query, &repeated, &positive, &negative]
        .iter()
        .flat_map(|value| value.embeddings.iter())
        .all(|value| value.is_finite());
    let norms = [&query, &repeated, &positive, &negative]
        .iter()
        .flat_map(|value| value.embeddings.rows())
        .map(|row| row.iter().map(|value| value * value).sum::<f32>().sqrt())
        .collect::<Vec<_>>();
    let batch_shapes = batch
        .iter()
        .map(|value| vec![value.num_tokens, value.embedding_dim])
        .collect::<Vec<_>>();
    let mut checks = base_checks(
        spec,
        query.embedding_dim,
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
    let batch_shape_matches = query.num_tokens == batch_query.num_tokens
        && query.embedding_dim == batch_query.embedding_dim
        && positive.num_tokens == batch_positive.num_tokens
        && positive.embedding_dim == batch_positive.embedding_dim;
    let batch_parity = cosine_iter(query.embeddings.iter(), batch_query.embeddings.iter()).min(
        cosine_iter(positive.embeddings.iter(), batch_positive.embeddings.iter()),
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
    Ok(observation(
        "multi_vector",
        vec![query.num_tokens, query.embedding_dim],
        batch_shapes,
        finite,
        Some(&norms),
        None,
        repeat_similarity,
        relevant_score,
        unrelated_score,
        checks,
    ))
}

fn sparse_smoke(spec: &CertificationSpec, policy: ResourcePolicy) -> CertResult<SmokeObservation> {
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
    let repeat_similarity = sparse_cosine(&query.weights, &repeated.weights);
    let relevant_score = sparse_dot(&query.weights, &positive.weights);
    let unrelated_score = sparse_dot(&query.weights, &negative.weights);
    let finite = [&query, &repeated, &positive, &negative]
        .iter()
        .flat_map(|value| value.weights.iter())
        .all(|(_, value)| value.is_finite() && *value > 0.0);
    let ordered = [&query, &repeated, &positive, &negative]
        .iter()
        .all(|value| value.weights.windows(2).all(|pair| pair[0].0 < pair[1].0));
    let batch_shapes = batch
        .iter()
        .map(|value| vec![value.nnz()])
        .collect::<Vec<_>>();
    let expected_vocab = spec.smoke.expected_vocabulary_size.unwrap_or(0);
    let mut checks = base_checks(
        spec,
        query.vocab_size,
        finite,
        repeat_similarity,
        relevant_score - unrelated_score,
    );
    checks.push(check(
        "vocabulary-size",
        query.vocab_size == expected_vocab && embedder.vocab_size() == expected_vocab,
        format!("observed {}", query.vocab_size),
    ));
    checks.push(check(
        "sorted-unique-positive-weights",
        ordered && finite,
        format!("ordered={ordered}, finite-positive={finite}"),
    ));
    let [batch_query, batch_positive] = batch.as_slice() else {
        return Err(format!("sparse batch returned {} outputs; expected 2", batch.len()).into());
    };
    let batch_shape_matches = query.vocab_size == batch_query.vocab_size
        && positive.vocab_size == batch_positive.vocab_size;
    let batch_parity = sparse_cosine(&query.weights, &batch_query.weights)
        .min(sparse_cosine(&positive.weights, &batch_positive.weights));
    checks.push(check(
        "batch-sequential-parity",
        batch_shape_matches && batch_parity >= spec.smoke.repeat_similarity_minimum,
        format!("shape-match={batch_shape_matches}, minimum cosine {batch_parity}"),
    ));
    Ok(observation(
        "sparse",
        vec![query.vocab_size],
        batch_shapes,
        finite,
        None,
        Some(query.nnz()),
        repeat_similarity,
        relevant_score,
        unrelated_score,
        checks,
    ))
}

fn base_checks(
    spec: &CertificationSpec,
    dimension: usize,
    finite: bool,
    repeat_similarity: f32,
    margin: f32,
) -> Vec<CheckEvidence> {
    vec![
        check(
            "dimension",
            dimension == spec.smoke.expected_dimension,
            format!("observed {dimension}"),
        ),
        check("finite", finite, format!("finite={finite}")),
        check(
            "repeat-similarity",
            repeat_similarity >= spec.smoke.repeat_similarity_minimum,
            format!("observed {repeat_similarity}"),
        ),
        check(
            "retrieval-margin",
            margin > spec.smoke.minimum_score_margin,
            format!("observed {margin}"),
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn observation(
    representation: &str,
    primary_shape: Vec<usize>,
    batch_shapes: Vec<Vec<usize>>,
    finite: bool,
    norms: Option<&[f32]>,
    non_zero: Option<usize>,
    repeat_similarity: f32,
    relevant_score: f32,
    unrelated_score: f32,
    checks: Vec<CheckEvidence>,
) -> SmokeObservation {
    let (norm_min, norm_max) = norms.map_or((None, None), |values| {
        let (minimum, maximum) = min_max(values);
        (Some(minimum), Some(maximum))
    });
    SmokeObservation {
        representation: representation.to_string(),
        primary_shape,
        batch_shapes,
        finite,
        norm_min,
        norm_max,
        non_zero,
        repeat_similarity,
        relevant_score,
        unrelated_score,
        score_margin: relevant_score - unrelated_score,
        checks,
    }
}

fn check(name: &str, passed: bool, detail: String) -> CheckEvidence {
    CheckEvidence {
        name: name.to_string(),
        passed,
        detail,
    }
}

fn norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return f32::NEG_INFINITY;
    }
    cosine_iter(left.iter(), right.iter())
}

fn cosine_iter<'a>(
    left: impl Iterator<Item = &'a f32>,
    right: impl Iterator<Item = &'a f32>,
) -> f32 {
    let (mut dot, mut left_norm, mut right_norm) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (left, right) in left.zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt()).max(f32::MIN_POSITIVE)
}

fn sparse_dot(left: &[(usize, f32)], right: &[(usize, f32)]) -> f32 {
    let (mut left_index, mut right_index, mut score) = (0, 0, 0.0_f32);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].0.cmp(&right[right_index].0) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                score += left[left_index].1 * right[right_index].1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    score
}

fn sparse_cosine(left: &[(usize, f32)], right: &[(usize, f32)]) -> f32 {
    let dot = sparse_dot(left, right);
    let left_norm = left
        .iter()
        .map(|(_, value)| value * value)
        .sum::<f32>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|(_, value)| value * value)
        .sum::<f32>()
        .sqrt();
    dot / (left_norm * right_norm).max(f32::MIN_POSITIVE)
}

fn min_max(values: &[f32]) -> (f32, f32) {
    values.iter().copied().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
    )
}

#[cfg(test)]
#[path = "tests/child.rs"]
mod tests;
