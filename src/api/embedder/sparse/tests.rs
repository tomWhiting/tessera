use std::cell::Cell;

use crate::core::SparseEmbedding;
use crate::runtime::ResourcePolicy;

use super::{collect_sparse_batch, sorted_sparse_dot, sparse_output_bytes};

fn one_entry_embedding(text: &str) -> SparseEmbedding {
    SparseEmbedding::new(vec![(1, 1.0)], 4, text.to_string()).unwrap()
}

#[test]
fn sparse_dot_merges_sorted_indices() {
    let left = [(1, 2.0), (4, 3.0), (9, -1.0)];
    let right = [(0, 8.0), (4, 5.0), (7, 2.0), (9, 4.0)];

    assert!((sorted_sparse_dot(&left, &right) - 11.0).abs() < f32::EPSILON);
    assert!(sorted_sparse_dot(&[], &right).abs() < f32::EPSILON);
}

#[test]
fn collecting_batch_stops_before_encoding_past_output_budget() {
    let calls = Cell::new(0_usize);
    let policy = ResourcePolicy::default().with_max_output_bytes(sparse_output_bytes(2));
    let texts = ["first", "second", "third", "must-not-run"];

    let error = collect_sparse_batch(&texts, policy, |_, text| {
        calls.set(calls.get().saturating_add(1));
        Ok(one_entry_embedding(text))
    })
    .unwrap_err();

    assert_eq!(calls.get(), 3);
    assert!(error.to_string().contains("Collected output byte count"));
}

#[test]
fn collecting_batch_preflights_the_complete_input_job() {
    let calls = Cell::new(0_usize);
    let policy = ResourcePolicy::default().with_max_job_items(1);

    let error = collect_sparse_batch(&["first", "second"], policy, |_, text| {
        calls.set(calls.get().saturating_add(1));
        Ok(one_entry_embedding(text))
    })
    .unwrap_err();

    assert_eq!(calls.get(), 0);
    assert!(error.to_string().contains("Job item count"));
}

#[test]
fn collecting_batch_preserves_input_order() {
    let texts = ["first", "second", "third"];

    let embeddings = collect_sparse_batch(&texts, ResourcePolicy::default(), |_, text| {
        Ok(one_entry_embedding(text))
    })
    .unwrap();

    let encoded_texts = embeddings
        .iter()
        .map(SparseEmbedding::text)
        .collect::<Vec<_>>();
    assert_eq!(encoded_texts, texts);
}
