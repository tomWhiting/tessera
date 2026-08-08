use sha2::{Digest, Sha256};

use super::reference::{
    ComparisonStatus, LoadedReference, NumericTolerance, ReferenceComparison, ReferenceOutput,
};
use super::spec::CertResult;

pub(super) fn compare(
    reference: &LoadedReference,
    observed: &ReferenceOutput,
) -> CertResult<ReferenceComparison> {
    validate_output(observed)?;
    let expected = &reference.document.expected;
    let expected_shape = expected.shape()?;
    let observed_shape = observed.shape()?;
    let shape_matches = expected_shape == observed_shape;
    let sparse_indices_match = expected.sparse_indices() == observed.sparse_indices();
    let representation_matches = expected.representation() == observed.representation();
    let expected_values = expected.values();
    let observed_values = observed.values();
    let (numeric_passed, max_absolute_error, max_relative_error) = compare_values(
        expected_values,
        observed_values,
        reference.document.tolerance,
    );
    let minimum_cosine = if shape_matches && sparse_indices_match && representation_matches {
        output_minimum_cosine(expected, observed)
    } else {
        None
    };
    let cosine_passed =
        minimum_cosine.is_some_and(|value| value >= reference.document.tolerance.minimum_cosine);
    let passed = shape_matches
        && sparse_indices_match
        && representation_matches
        && numeric_passed
        && cosine_passed;
    Ok(ReferenceComparison {
        status: if passed {
            ComparisonStatus::Passed
        } else {
            ComparisonStatus::Failed
        },
        reference_path: Some(reference.path.clone()),
        reference_sha256: Some(reference.sha256.clone()),
        expected_output_sha256: Some(digest(&serde_json::to_vec(expected)?)),
        observed_output_sha256: Some(digest(&serde_json::to_vec(observed)?)),
        probe_tokens: Some(reference.document.probe.token_count()),
        observed_shape,
        compared_values: expected_values.len().min(observed_values.len()),
        max_absolute_error,
        max_relative_error,
        minimum_cosine,
        detail: format!(
            "representation-match={representation_matches}, shape-match={shape_matches}, sparse-indices-match={sparse_indices_match}, numeric-pass={numeric_passed}, cosine-pass={cosine_passed}"
        ),
    })
}

pub(super) fn validate_output(output: &ReferenceOutput) -> CertResult<()> {
    let shape = output.shape()?;
    if shape.contains(&0) || output.values().is_empty() {
        return Err("official reference output is empty".into());
    }
    if !output.values().iter().all(|value| value.is_finite()) {
        return Err("official reference output contains a non-finite value".into());
    }
    if let ReferenceOutput::Sparse {
        vocabulary_size,
        indices,
        values,
    } = output
    {
        if indices.len() != values.len()
            || indices.windows(2).any(|pair| pair[0] >= pair[1])
            || indices.iter().any(|index| *index >= *vocabulary_size)
            || values.iter().any(|value| *value <= 0.0)
        {
            return Err(
                "sparse reference entries must be sorted, unique, in range, and positive".into(),
            );
        }
    }
    Ok(())
}

fn compare_values(
    expected: &[f32],
    observed: &[f32],
    tolerance: NumericTolerance,
) -> (bool, Option<f32>, Option<f32>) {
    if expected.len() != observed.len() || expected.is_empty() {
        return (false, None, None);
    }
    let mut passed = true;
    let mut max_absolute = 0.0_f32;
    let mut max_relative = 0.0_f32;
    for (&expected, &observed) in expected.iter().zip(observed) {
        let absolute = (observed - expected).abs();
        let relative = absolute / expected.abs().max(tolerance.absolute).max(f32::EPSILON);
        max_absolute = max_absolute.max(absolute);
        max_relative = max_relative.max(relative);
        passed &= absolute <= tolerance.absolute + tolerance.relative * expected.abs();
    }
    (passed, Some(max_absolute), Some(max_relative))
}

fn output_minimum_cosine(expected: &ReferenceOutput, observed: &ReferenceOutput) -> Option<f32> {
    let columns = expected.row_width()?;
    expected
        .values()
        .chunks_exact(columns)
        .zip(observed.values().chunks_exact(columns))
        .map(|(left, right)| cosine(left, right))
        .reduce(f32::min)
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let (dot, left_norm, right_norm) = left.iter().zip(right).fold(
        (0.0_f32, 0.0_f32, 0.0_f32),
        |(dot, left_norm, right_norm), (&left, &right)| {
            (
                dot + left * right,
                left_norm + left * left,
                right_norm + right * right,
            )
        },
    );
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    if denominator <= f32::MIN_POSITIVE {
        if left == right {
            1.0
        } else {
            0.0
        }
    } else {
        dot / denominator
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
