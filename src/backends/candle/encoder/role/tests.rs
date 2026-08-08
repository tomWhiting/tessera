use std::collections::HashSet;

use crate::runtime::{plan_token_windows, ContextWindowConfig, ResourcePolicy};

use super::{
    document_window_plan_config, prepare_document_window_tokens, prepare_role_tokens, ArtifactIds,
    ColbertConfig, InputRole, PreparedInput, DEFAULT_DOCUMENT_MAX_LENGTH, DEFAULT_QUERY_MAX_LENGTH,
};

fn ids() -> ArtifactIds {
    ArtifactIds {
        cls: 101,
        sep: 102,
        mask: 103,
        pad: 0,
        query_marker: 1,
        document_marker: 2,
        punctuation: HashSet::from([12, 14]),
    }
}

fn config(query: usize, document: usize) -> ColbertConfig {
    ColbertConfig {
        query_max_length: query,
        document_max_length: document,
    }
}

#[test]
fn query_inserts_artifact_marker_and_augments_with_masks() {
    let prepared = prepare_role_tokens(&[101, 11, 12, 102], InputRole::Query, config(7, 8), &ids())
        .expect("query preprocessing should succeed");

    assert_eq!(
        prepared,
        PreparedInput {
            token_ids: vec![101, 1, 11, 12, 102, 103, 103],
            attention_mask: vec![1; 7],
            output_mask: vec![1; 7],
        }
    );
}

#[test]
fn query_truncation_preserves_role_framing_and_separator() {
    let prepared = prepare_role_tokens(
        &[101, 10, 11, 12, 13, 102],
        InputRole::Query,
        config(5, 8),
        &ids(),
    )
    .expect("query preprocessing should succeed");

    assert_eq!(prepared.token_ids, [101, 1, 10, 11, 102]);
}

#[test]
fn document_masks_punctuation_but_keeps_special_and_role_tokens() {
    let prepared = prepare_role_tokens(
        &[101, 11, 12, 13, 14, 102],
        InputRole::Document,
        config(7, 8),
        &ids(),
    )
    .expect("document preprocessing should succeed");

    assert_eq!(prepared.token_ids, [101, 2, 11, 12, 13, 14, 102]);
    assert_eq!(prepared.attention_mask, [1; 7]);
    assert_eq!(prepared.output_mask, [1, 1, 1, 0, 1, 0, 1]);
}

#[test]
fn document_truncation_preserves_separator_after_inserted_marker() {
    let prepared = prepare_role_tokens(
        &[101, 10, 11, 12, 13, 102],
        InputRole::Document,
        config(7, 5),
        &ids(),
    )
    .expect("document preprocessing should succeed");

    assert_eq!(prepared.token_ids, [101, 2, 10, 11, 102]);
    assert_eq!(prepared.attention_mask, [1; 5]);
    assert_eq!(prepared.output_mask, [1; 5]);
}

#[test]
fn batch_padding_is_excluded_from_attention_and_output() {
    let mut prepared = PreparedInput {
        token_ids: vec![101, 2, 102],
        attention_mask: vec![1; 3],
        output_mask: vec![1; 3],
    };

    prepared.pad_to(5, 0);

    assert_eq!(prepared.token_ids, [101, 2, 102, 0, 0]);
    assert_eq!(prepared.attention_mask, [1, 1, 1, 0, 0]);
    assert_eq!(prepared.output_mask, [1, 1, 1, 0, 0]);
}

#[test]
fn default_lengths_are_clamped_to_resource_policy() {
    let policy = ResourcePolicy::default().with_max_sequence_tokens(128);
    let resolved = ColbertConfig::resolve(None, None, &policy).expect("defaults should fit");

    assert_eq!(resolved.query_max_length, DEFAULT_QUERY_MAX_LENGTH);
    assert_eq!(resolved.document_max_length, 128);
    assert!(DEFAULT_DOCUMENT_MAX_LENGTH > resolved.document_max_length);
}

#[test]
fn explicit_lengths_must_fit_role_framing_and_policy() {
    let policy = ResourcePolicy::default().with_max_sequence_tokens(64);

    let too_short = ColbertConfig::resolve(Some(2), None, &policy).unwrap_err();
    assert!(too_short.to_string().contains("at least 3"));

    let too_long = ColbertConfig::resolve(None, Some(65), &policy).unwrap_err();
    assert!(too_long.to_string().contains("exceeds resource policy"));
}

#[test]
fn framing_tokens_must_come_from_the_tokenizer_artifact() {
    let error = prepare_role_tokens(&[999, 11, 102], InputRole::Document, config(7, 8), &ids())
        .unwrap_err();

    assert!(error.to_string().contains("framed by artifact"));
}

#[test]
fn document_window_budget_includes_the_inserted_role_marker() {
    let plan = document_window_plan_config(ContextWindowConfig::new(8, 2), config(7, 8))
        .expect("final ColBERT window should fit");

    assert_eq!(plan.window_tokens(), 7);
    assert_eq!(plan.overlap_tokens(), 2);
    assert!(document_window_plan_config(ContextWindowConfig::new(3, 0), config(7, 8)).is_err());
    assert!(document_window_plan_config(ContextWindowConfig::new(9, 0), config(7, 8)).is_err());
}

#[test]
fn document_window_offsets_ownership_past_cls_and_document_marker() {
    let prepared = prepare_document_window_tokens(
        &[101, 11, 12, 13, 14, 102],
        &[1; 6],
        1..3,
        config(7, 8),
        &ids(),
    )
    .expect("document window should be framed");

    assert_eq!(prepared.token_ids, [101, 2, 11, 12, 13, 14, 102]);
    assert_eq!(prepared.attention_mask, [1; 7]);
    assert_eq!(prepared.output_mask, [1, 1, 0, 0, 1, 0, 1]);
}

#[test]
fn center_owned_document_windows_select_each_content_token_once() {
    let content = (20_u32..28).collect::<Vec<_>>();
    let requested = ContextWindowConfig::new(7, 2);
    let planned = document_window_plan_config(requested, config(7, 7))
        .expect("ColBERT window budget should adjust for [D]");
    let windows = plan_token_windows(&content, &[101], &[102], planned, ResourcePolicy::default())
        .expect("overlapping windows should plan");

    let mut selected_content = Vec::new();
    for window in windows {
        let prepared = prepare_document_window_tokens(
            &window.token_ids,
            &window.attention_mask,
            window.owned_local_range(),
            config(7, 7),
            &ids(),
        )
        .expect("window should receive document framing");
        assert!(prepared.token_ids.len() <= requested.window_tokens());
        assert_eq!(prepared.output_mask[0..2], [1, 1]);
        assert_eq!(prepared.output_mask.last(), Some(&1));
        selected_content.extend(
            prepared.token_ids[2..prepared.token_ids.len() - 1]
                .iter()
                .zip(&prepared.output_mask[2..prepared.output_mask.len() - 1])
                .filter_map(|(id, selected)| (*selected == 1).then_some(*id)),
        );
    }

    assert_eq!(selected_content, content);
}
