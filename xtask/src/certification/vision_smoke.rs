use std::path::Path;

use candle_core::Device;
use tessera::{ResourcePolicy, TesseraVision};

use super::evidence::{CheckEvidence, SmokeObservation};
use super::reference::{self, LoadedReference, ReferenceOutput, ReferenceProbe};
use super::smoke_math::{min_max, norm};
use super::spec::{CertResult, CertificationSpec};

pub(super) fn run(
    repository: &Path,
    spec: &CertificationSpec,
    policy: ResourcePolicy,
    official_reference: Option<&LoadedReference>,
) -> CertResult<(SmokeObservation, Option<ReferenceOutput>)> {
    let reference = official_reference
        .ok_or("vision certification requires a checked official reference and image fixture")?;
    let ReferenceProbe::Image { query, .. } = &reference.document.probe else {
        return Err("vision certification requires an image reference probe".into());
    };
    let image_path = reference::resolve_image(repository, &reference.document.probe)?;
    let image_path = image_path
        .to_str()
        .ok_or("reference image path is not valid UTF-8")?;
    let embedder = TesseraVision::builder()
        .model(&spec.model.id)
        .device(Device::Cpu)
        .resource_policy(policy)
        .build()?;
    let document = embedder.encode_document(image_path)?;
    let query = embedder.encode_query(query)?;
    let relevant_score = embedder.search(&query, &document)?;
    let values = document
        .vectors()
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let finite = values.iter().all(|value| value.is_finite());
    let norms = document
        .vectors()
        .iter()
        .map(|row| norm(row))
        .collect::<Vec<_>>();
    let norm_range = min_max(&norms);
    let mut checks = vec![
        check(
            "dimension",
            document.embedding_dim() == spec.smoke.expected_dimension,
            format!("observed {}", document.embedding_dim()),
        ),
        check("finite", finite, format!("finite={finite}")),
        check(
            "non-empty-patches",
            document.num_patches() > 0,
            format!("observed {} patches", document.num_patches()),
        ),
    ];
    if spec.smoke.normalized {
        checks.push(check(
            "row-normalized",
            norms.iter().all(|value| (value - 1.0).abs() <= 0.01),
            format!("norm range {norm_range:?}"),
        ));
    }
    let observed = ReferenceOutput::Vision {
        rows: document.num_vectors(),
        columns: document.embedding_dim(),
        values,
    };
    Ok((
        SmokeObservation {
            representation: "vision".to_string(),
            primary_shape: vec![document.num_vectors(), document.embedding_dim()],
            batch_shapes: Vec::new(),
            finite,
            norm_min: Some(norm_range.0),
            norm_max: Some(norm_range.1),
            non_zero: None,
            repeat_similarity: 1.0,
            relevant_score,
            unrelated_score: 0.0,
            score_margin: relevant_score,
            checks,
        },
        Some(observed),
    ))
}

fn check(name: &str, passed: bool, detail: String) -> CheckEvidence {
    CheckEvidence {
        name: name.to_string(),
        passed,
        detail,
    }
}
