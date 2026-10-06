use super::encode_windows_batch_with;
use crate::core::tokenizer::tests::cut_tokenizer_with_policy;
use crate::runtime::ResourcePolicy;
use crate::{ContextWindowConfig, TesseraDense};
use crate::{CountedDenseEmbedding, EmbeddingRefusal, WindowEmbeddingOutcome};

#[test]
fn dense_window_path_keeps_every_content_token_and_source_span() {
    let tokenizer = cut_tokenizer_with_policy(ResourcePolicy::new(5, 16, 2048, usize::MAX));
    let text = "one two three one";
    let (total, windows) = tokenizer
        .encode_spanned_windows("", text, ContextWindowConfig::new(5, 1))
        .unwrap();
    assert_eq!(total, 4);
    assert_eq!(windows.len(), 2);
    assert_eq!(windows[0].byte_start, 0);
    assert_eq!(windows[0].byte_end, 13);
    assert_eq!(windows[1].byte_start, 8);
    assert_eq!(windows[1].byte_end, text.len());
    assert_eq!(windows[0].window.token_ids, [10, 2, 3, 4, 11]);
    assert_eq!(windows[1].window.token_ids, [10, 4, 2, 11]);
    std::hint::black_box(TesseraDense::encode_windows);
    std::hint::black_box(TesseraDense::encode_batch_windows);
}

#[test]
fn window_count_matches_the_closed_formula() {
    let tokenizer = cut_tokenizer_with_policy(ResourcePolicy::new(512, 16, 10_000, usize::MAX));
    for (total, expected) in [(1, 1), (510, 1), (511, 2), (956, 2), (957, 3)] {
        let text = vec!["one"; total].join(" ");
        let (observed, windows) = tokenizer
            .encode_spanned_windows("", &text, ContextWindowConfig::new(512, 64))
            .unwrap();
        assert_eq!(observed, total);
        assert_eq!(windows.len(), expected);
        assert_eq!(windows[0].byte_start, 0);
        assert_eq!(windows.last().unwrap().byte_end, text.len());
        for pair in windows.windows(2) {
            assert!(pair[1].byte_start >= pair[0].byte_start);
            assert!(pair[1].byte_end >= pair[0].byte_end);
            assert!(pair[1].byte_start <= pair[0].byte_end);
        }
        assert!(windows
            .iter()
            .all(|window| window.byte_start < window.byte_end
                && text.is_char_boundary(window.byte_start)
                && text.is_char_boundary(window.byte_end)
                && window.window.token_ids.len() <= 512));
    }
}

#[test]
fn prefix_tokens_reduce_capacity_without_changing_content_ids() {
    let tokenizer = cut_tokenizer_with_policy(ResourcePolicy::new(5, 16, 2048, usize::MAX));
    let (total, windows) = tokenizer
        .encode_spanned_windows("two ", "one two three", ContextWindowConfig::new(5, 1))
        .unwrap();
    assert_eq!(total, 3);
    assert_eq!(windows.len(), 2);
    assert_eq!(windows[0].window.token_ids, [10, 3, 2, 3, 11]);
    assert_eq!(windows[1].window.token_ids, [10, 3, 3, 4, 11]);
}

#[test]
fn multibyte_boundaries_and_edge_whitespace_are_covered() {
    let tokenizer = cut_tokenizer_with_policy(ResourcePolicy::new(4, 16, 2048, usize::MAX));
    let text = "  one 😀 two  ";
    let (total, windows) = tokenizer
        .encode_spanned_windows("", text, ContextWindowConfig::new(4, 1))
        .unwrap();
    assert_eq!(total, 3);
    assert_eq!(windows.len(), 2);
    assert_eq!((windows[0].byte_start, windows[0].byte_end), (0, 10));
    assert_eq!(
        (windows[1].byte_start, windows[1].byte_end),
        (6, text.len())
    );
    assert!(windows
        .iter()
        .all(|window| text.is_char_boundary(window.byte_start)
            && text.is_char_boundary(window.byte_end)));
}

#[test]
fn no_overlap_windows_include_the_whitespace_between_tokens() {
    let tokenizer = cut_tokenizer_with_policy(ResourcePolicy::new(3, 16, 2048, usize::MAX));
    let text = "one  two";
    let (_, windows) = tokenizer
        .encode_spanned_windows("", text, ContextWindowConfig::new(3, 0))
        .unwrap();
    assert_eq!((windows[0].byte_start, windows[0].byte_end), (0, 5));
    assert_eq!((windows[1].byte_start, windows[1].byte_end), (5, 8));
}

#[test]
fn batch_windows_preserve_vectors_spans_and_refusals_in_order() {
    let policy = ResourcePolicy::new(4, 2, 64, usize::MAX).with_max_input_bytes_per_sequence(64);
    let tokenizer = cut_tokenizer_with_policy(policy);
    let mut groups = Vec::new();
    let outcomes = encode_windows_batch_with(
        &["one two three one", "  ", "two", &"x".repeat(65)],
        policy,
        1,
        2,
        |text| tokenizer.encode_spanned_windows("", text, ContextWindowConfig::new(4, 1)),
        |windows| {
            groups.push(windows.len());
            windows
                .iter()
                .map(|input| {
                    CountedDenseEmbedding::new(
                        ndarray::array![f32::from(
                            u16::try_from(input.window.token_ids[1]).unwrap()
                        )],
                        input.window.token_ids.len(),
                    )
                })
                .collect()
        },
    )
    .unwrap();
    assert_eq!(groups, [2, 2]);
    let WindowEmbeddingOutcome::Embedded(first) = &outcomes[0] else {
        panic!("Missing first input");
    };
    assert_eq!(first.tokens_total(), 4);
    assert_eq!(first.windows().len(), 3);
    assert_eq!(
        first
            .windows()
            .iter()
            .map(|window| window.values()[0])
            .collect::<Vec<_>>(),
        [2.0, 3.0, 4.0]
    );
    assert_eq!(first.windows()[0].tokens(), 4);
    assert_eq!(first.windows()[0].byte_start(), 0);
    assert_eq!(first.windows()[2].byte_end(), 17);
    assert!(matches!(
        outcomes[1],
        WindowEmbeddingOutcome::Refused(EmbeddingRefusal::Empty)
    ));
    assert!(matches!(outcomes[2], WindowEmbeddingOutcome::Embedded(_)));
    assert!(matches!(
        outcomes[3],
        WindowEmbeddingOutcome::Refused(EmbeddingRefusal::TooLarge {
            input_bytes: 65,
            limit: 64
        })
    ));
}

#[test]
fn all_plans_and_output_budgets_are_checked_before_forwarding() {
    let policy = ResourcePolicy::new(4, 2, 64, usize::MAX).with_max_output_bytes(4);
    let tokenizer = cut_tokenizer_with_policy(policy);
    let result = encode_windows_batch_with(
        &["one two three"],
        policy,
        1,
        2,
        |text| tokenizer.encode_spanned_windows("", text, ContextWindowConfig::new(4, 1)),
        |_| panic!("An over-budget job reached inference"),
    );
    assert!(result.is_err());
}

#[test]
fn zero_token_middle_item_is_refused_without_losing_its_neighbors() {
    let policy = ResourcePolicy::new(4, 2, 64, usize::MAX)
        .with_max_job_items(2)
        .with_max_job_input_bytes(6)
        .with_max_output_bytes(8);
    let tokenizer = crate::core::tokenizer::tests::drop_controls_tokenizer(policy);
    let mut forwarded = Vec::new();
    let outcomes = encode_windows_batch_with(
        &["one", "\u{0}", "two"],
        policy,
        1,
        2,
        |text| tokenizer.encode_spanned_windows("", text, ContextWindowConfig::new(4, 1)),
        |windows| {
            windows
                .iter()
                .map(|input| {
                    forwarded.push(input.window.token_ids[1]);
                    CountedDenseEmbedding::new(ndarray::array![1.0], input.window.token_ids.len())
                })
                .collect()
        },
    )
    .unwrap();
    assert_eq!(forwarded, [2, 3]);
    assert_eq!(outcomes.len(), 3);
    assert!(matches!(outcomes[0], WindowEmbeddingOutcome::Embedded(_)));
    let WindowEmbeddingOutcome::Refused(refusal) = outcomes[1] else {
        panic!("Zero-token text was not refused");
    };
    assert_eq!(refusal.code(), "embed_input_no_content_tokens");
    assert!(matches!(outcomes[2], WindowEmbeddingOutcome::Embedded(_)));
}

#[test]
fn a_measured_extent_counts_the_windows_the_planner_forms() {
    let tokenizer = cut_tokenizer_with_policy(ResourcePolicy::new(64, 16, 2048, usize::MAX));
    let text = "one two three one two three one two three one  ";
    for (prompt, window_tokens, overlap) in [("", 5, 1), ("", 6, 2), ("two ", 6, 1), ("", 64, 0)] {
        let extent = tokenizer
            .spanned_window_extent(prompt, text, window_tokens)
            .unwrap();
        let (total, windows) = tokenizer
            .encode_spanned_windows(
                prompt,
                text,
                ContextWindowConfig::new(window_tokens, overlap),
            )
            .unwrap();
        assert_eq!(extent.tokens_total, total);
        assert_eq!(extent.windows(overlap), Some(windows.len()));
        let last = windows.last().unwrap();
        assert_eq!(last.byte_end, text.len());
    }
}
