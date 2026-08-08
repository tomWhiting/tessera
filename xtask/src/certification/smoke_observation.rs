use super::evidence::{CheckEvidence, SmokeObservation};
use super::smoke_math::min_max;
use super::spec::CertificationSpec;

pub(super) fn base_checks(
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
pub(super) fn observation(
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

pub(super) fn check(name: &str, passed: bool, detail: String) -> CheckEvidence {
    CheckEvidence {
        name: name.to_string(),
        passed,
        detail,
    }
}
