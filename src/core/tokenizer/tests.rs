use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;

use super::{HfTokenizer, Tokenizer};
use crate::runtime::{ContextWindowConfig, ResourcePolicy};

pub fn tokenizer(resource_policy: ResourcePolicy) -> Tokenizer {
    let vocabulary = [
        ("[UNK]".to_string(), 0),
        ("[PAD]".to_string(), 1),
        ("one".to_string(), 2),
        ("two".to_string(), 3),
        ("three".to_string(), 4),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".to_string())
        .build()
        .expect("in-memory tokenizer model should be valid");
    let mut inner = HfTokenizer::new(model);
    inner.with_pre_tokenizer(Some(Whitespace {}));

    Tokenizer {
        inner,
        truncating: None,
        resource_policy,
        pad_token_id: Some(1),
    }
}

fn cut_tokenizer(limit: usize) -> Tokenizer {
    cut_tokenizer_with_policy(ResourcePolicy::new(limit, 16, 2048, usize::MAX))
}

#[test]
fn prefix_tokens_take_the_longer_role_without_special_tokens() {
    let tokenizer = cut_tokenizer(8);
    assert_eq!(
        tokenizer
            .longest_prefix_tokens(&["one", "one two three"])
            .unwrap(),
        3
    );
    assert_eq!(
        tokenizer
            .longest_prefix_tokens(&["one two three", "one"])
            .unwrap(),
        3
    );
}

#[test]
fn prefix_tokens_are_zero_when_no_prefix_is_present() {
    let tokenizer = cut_tokenizer(8);
    assert_eq!(tokenizer.longest_prefix_tokens(&[]).unwrap(), 0);
    assert_eq!(tokenizer.longest_prefix_tokens(&["", ""]).unwrap(), 0);
}

#[test]
fn prefix_tokens_count_the_whole_prefix_above_the_sequence_limit() {
    let tokenizer = cut_tokenizer(4);
    assert_eq!(
        tokenizer
            .longest_prefix_tokens(&["one two three one two three one"])
            .unwrap(),
        7
    );
}

pub fn cut_tokenizer_with_policy(policy: ResourcePolicy) -> Tokenizer {
    let mut tokenizer = tokenizer(policy);
    tokenizer.inner.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[START] $A [END]")
            .unwrap()
            .special_tokens(vec![("[START]", 10), ("[END]", 11)])
            .build()
            .unwrap(),
    ));
    tokenizer.prepare_cut().unwrap();
    tokenizer
}

#[test]
fn cut_under_limit_preserves_existing_tokens_and_counts() {
    let tokenizer = cut_tokenizer(5);
    let input = tokenizer.encode_cut("", "one").unwrap();
    assert_eq!(input.token_ids, [10, 2, 11]);
    assert_eq!(input.token_ids, tokenizer.encode("one", true).unwrap().0);
    assert_eq!(input.tokens_read(), 3);
    assert_eq!(input.tokens_total, 3);
    assert!(!input.cut);
}

#[test]
fn cut_exact_limit_preserves_existing_tokens_and_counts() {
    let tokenizer = cut_tokenizer(5);
    let input = tokenizer.encode_cut("", "one two three").unwrap();
    assert_eq!(input.token_ids, [10, 2, 3, 4, 11]);
    assert_eq!(
        input.token_ids,
        tokenizer.encode("one two three", true).unwrap().0
    );
    assert_eq!(input.tokens_read(), 5);
    assert_eq!(input.tokens_total, 5);
    assert!(!input.cut);
}

#[test]
fn cut_over_limit_keeps_special_tokens_and_reports_whole_count() {
    let input = cut_tokenizer(5)
        .encode_cut("", "one two three one")
        .unwrap();
    assert_eq!(input.token_ids, [10, 2, 3, 4, 11]);
    assert_eq!(input.tokens_read(), 5);
    assert_eq!(input.tokens_total, 6);
    assert!(input.cut);
}

#[test]
fn cut_far_over_limit_keeps_start_and_reports_whole_count() {
    let text = vec!["one"; 50].join(" ");
    let input = cut_tokenizer(5).encode_cut("", &text).unwrap();
    assert_eq!(input.token_ids, [10, 2, 2, 2, 11]);
    assert_eq!(input.tokens_read(), 5);
    assert_eq!(input.tokens_total, 52);
    assert!(input.cut);
}

#[test]
fn cut_configuration_requires_special_tokens_plus_content() {
    for limit in [0, 1, 2] {
        let tokenizer = cut_tokenizer(limit);
        let error = tokenizer.encode_cut("", "one").unwrap_err();
        let error = error
            .downcast_ref::<super::CutConfigurationError>()
            .unwrap();
        assert_eq!(error.limit, limit);
        assert_eq!(error.special_tokens, 2);
        assert!(error.to_string().starts_with("InvalidCutConfiguration:"));
        let error = tokenizer.encode_batch_cut("", &["one", "two"]).unwrap_err();
        assert!(error
            .downcast_ref::<super::CutConfigurationError>()
            .is_some());
        assert!(tokenizer.encode_batch_cut("", &[]).is_err());
    }
}

#[test]
fn cut_batch_preserves_order_and_accepts_long_items() {
    let inputs = cut_tokenizer(5)
        .encode_batch_cut("", &["two", "one two three one", "three"])
        .unwrap();
    assert_eq!(inputs.len(), 3);
    assert_eq!(inputs[0].token_ids, [10, 3, 11]);
    assert_eq!(inputs[1].token_ids, [10, 2, 3, 4, 11]);
    assert_eq!(inputs[1].tokens_total, 6);
    assert!(inputs[1].cut);
    assert_eq!(inputs[2].token_ids, [10, 4, 11]);
}

#[test]
fn cut_methods_still_refuse_the_byte_limit() {
    let mut tokenizer = cut_tokenizer(5);
    tokenizer.resource_policy = tokenizer
        .resource_policy
        .with_max_input_bytes_per_sequence(3);
    for error in [
        tokenizer.encode_cut("", "three").unwrap_err(),
        tokenizer
            .encode_batch_cut("", &["one", "three"])
            .unwrap_err(),
    ] {
        assert_eq!(
            error.to_string(),
            "Input byte count 5 exceeds resource policy limit 3"
        );
    }
}

#[test]
fn cut_uses_the_tokenizer_prepared_once_before_calls() {
    let tokenizer = cut_tokenizer(5);
    let prepared = std::ptr::from_ref(tokenizer.truncating.as_ref().unwrap());
    tokenizer.encode_cut("", "one two three one").unwrap();
    tokenizer.encode_cut("", "three two one three").unwrap();
    assert_eq!(
        prepared,
        std::ptr::from_ref(tokenizer.truncating.as_ref().unwrap())
    );
}

#[test]
fn sequence_limit_rejects_instead_of_truncating() {
    let tokenizer = tokenizer(ResourcePolicy::new(2, 16, 2048, usize::MAX));

    let error = tokenizer
        .encode("one two three", false)
        .expect_err("three tokens must exceed a two-token limit");

    assert_eq!(
        error.to_string(),
        "Sequence token count 3 exceeds resource policy limit 2"
    );
}

#[test]
fn padded_batch_limit_is_checked_before_padding() {
    let tokenizer = tokenizer(ResourcePolicy::new(3, 2, 5, usize::MAX));

    let error = tokenizer
        .encode_batch(&["one", "one two three"], false)
        .expect_err("two sequences padded to three tokens require six token cells");

    assert_eq!(
        error.to_string(),
        "Padded batch token count 6 exceeds resource policy limit 5"
    );
}

#[test]
fn single_input_obeys_batch_token_limit() {
    let tokenizer = tokenizer(ResourcePolicy::new(3, 1, 2, usize::MAX));

    let error = tokenizer
        .encode("one two three", false)
        .expect_err("a single input still occupies one padded batch");

    assert_eq!(
        error.to_string(),
        "Padded batch token count 3 exceeds resource policy limit 2"
    );
}

#[test]
fn item_limit_is_checked_before_batch_tokenization() {
    let tokenizer = tokenizer(ResourcePolicy::new(3, 2, 6, usize::MAX));

    let error = tokenizer
        .encode_batch(&["one", "two", "three"], false)
        .expect_err("three items must exceed a two-item limit");

    assert_eq!(
        error.to_string(),
        "Batch item count 3 exceeds resource policy limit 2"
    );
}

#[test]
fn exact_batch_boundary_and_empty_batch_are_valid() {
    let bounded_tokenizer = tokenizer(ResourcePolicy::new(3, 2, 6, usize::MAX));
    let batch = bounded_tokenizer
        .encode_batch(&["one", "one two three"], false)
        .expect("exact policy boundary should be accepted");

    assert_eq!(batch.len(), 2);
    assert!(batch.iter().all(|(tokens, _)| tokens.len() == 3));

    let zero_policy_tokenizer = tokenizer(ResourcePolicy::new(0, 0, 0, 0));
    assert!(zero_policy_tokenizer
        .encode_batch(&[], false)
        .expect("empty batches should always be accepted")
        .is_empty());
}

#[test]
fn raw_input_bytes_are_rejected_before_tokenization() {
    let bounded_tokenizer = tokenizer(
        ResourcePolicy::default()
            .with_max_sequence_tokens(3)
            .with_max_input_bytes_per_sequence(3),
    );

    let error = bounded_tokenizer
        .encode("four", false)
        .expect_err("four UTF-8 bytes must exceed a three-byte input limit");

    assert_eq!(
        error.to_string(),
        "Input byte count 4 exceeds resource policy limit 3"
    );
}

#[test]
fn unregistered_tokenizers_are_rejected_before_network_access() {
    let error = Tokenizer::from_pretrained("example/unregistered-tokenizer")
        .err()
        .expect("unregistered tokenizer must fail");

    assert!(error.to_string().contains("not registered"));
}

#[test]
fn tokenizers_without_a_registry_pin_are_rejected_before_network_access() {
    let error = Tokenizer::from_pretrained("jinaai/jina-colbert-v2-96")
        .err()
        .expect("an unpinned tokenizer must fail");

    assert!(error
        .to_string()
        .contains("has no pinned HuggingFace revision"));
}

#[test]
fn artifact_token_ids_are_resolved_without_numeric_assumptions() {
    let tokenizer = tokenizer(ResourcePolicy::default());

    assert_eq!(tokenizer.token_to_id("[PAD]"), Some(1));
    assert_eq!(tokenizer.token_to_id("three"), Some(4));
    assert_eq!(tokenizer.token_to_id("[MASK]"), None);
}

#[test]
fn bounded_truncation_encoding_still_enforces_raw_bytes() {
    let tokenizer =
        tokenizer(ResourcePolicy::new(1, 1, 1, usize::MAX).with_max_input_bytes_per_sequence(32));

    let (ids, _) = tokenizer
        .encode_for_bounded_truncation("one two three", false)
        .expect("the role-specific caller is responsible for truncating tokens");
    assert_eq!(ids, [2, 3, 4]);

    let error = tokenizer
        .encode_for_bounded_truncation("one two three one two three one two", false)
        .expect_err("raw bytes remain bounded before tokenization");
    assert!(error.to_string().contains("Input byte count"));
}

#[test]
fn window_encoding_covers_content_once_by_center_ownership() {
    let tokenizer = tokenizer(
        ResourcePolicy::default()
            .with_max_sequence_tokens(3)
            .with_max_job_items(4),
    );
    let windows = tokenizer
        .encode_windows("one two three one", ContextWindowConfig::new(3, 1))
        .expect("valid overlapping windows");

    assert_eq!(windows.len(), 2);
    assert_eq!(
        windows
            .iter()
            .map(crate::runtime::TokenWindow::owned_len)
            .sum::<usize>(),
        4
    );
    assert!(windows.iter().all(|window| window.token_ids.len() <= 3));
}

#[test]
fn prompt_is_joined_before_the_text_and_counted() {
    let tokenizer = cut_tokenizer(8);
    let query = tokenizer.encode_cut("three two ", "one").unwrap();
    assert_eq!(query.token_ids, [10, 4, 3, 2, 11]);
    assert_eq!(query.tokens_read(), 5);
    assert_eq!(query.tokens_total, 5);
    assert!(!query.cut);
    let document = tokenizer.encode_cut("", "one").unwrap();
    assert_eq!(document.token_ids, [10, 2, 11]);
    assert_eq!(document.tokens_total, 3);
}

#[test]
fn cutting_keeps_the_prompt_and_removes_from_the_end() {
    let input = cut_tokenizer(5)
        .encode_cut("three ", "one two three one")
        .unwrap();
    assert_eq!(input.token_ids, [10, 4, 2, 3, 11]);
    assert_eq!(input.tokens_read(), 5);
    assert_eq!(input.tokens_total, 7);
    assert!(input.cut);
}

#[test]
fn batch_joins_the_one_prompt_before_every_text() {
    let inputs = cut_tokenizer(8)
        .encode_batch_cut("three ", &["one", "two"])
        .unwrap();
    let ids: Vec<_> = inputs.iter().map(|input| input.token_ids.clone()).collect();
    assert_eq!(ids, [vec![10, 4, 2, 11], vec![10, 4, 3, 11]]);
}

#[test]
fn byte_limit_counts_the_callers_text_alone() {
    let policy = ResourcePolicy::new(8, 16, 2048, usize::MAX).with_max_input_bytes_per_sequence(3);
    let input = cut_tokenizer_with_policy(policy)
        .encode_cut("three two ", "one")
        .unwrap();
    assert_eq!(input.tokens_total, 5);
}

#[test]
fn limit_must_hold_special_tokens_longer_prompt_and_one_token() {
    let prompts = ["three two ", "one "];
    let error = cut_tokenizer(4)
        .validate_cut_configuration_with(&prompts)
        .unwrap_err();
    let error = error
        .downcast_ref::<super::PromptConfigurationError>()
        .expect("prompt configuration error");
    assert_eq!(
        (error.limit, error.special_tokens, error.prompt_tokens),
        (4, 2, 2)
    );
    assert!(error
        .to_string()
        .starts_with("InvalidPromptConfiguration: "));
    cut_tokenizer(5)
        .validate_cut_configuration_with(&prompts)
        .unwrap();
}

mod metaspace_whitespace {
    use tokenizers::models::unigram::Unigram;
    use tokenizers::models::wordpiece::WordPiece;
    use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
    use tokenizers::pre_tokenizers::PreTokenizerWrapper;
    use tokenizers::AddedToken;

    use super::super::{split_whitespace_before_metaspace, HfTokenizer, Tokenizer};
    use crate::runtime::ResourcePolicy;

    const LONE: u32 = 1;
    const ONE: u32 = 2;
    const TWO: u32 = 3;

    fn metaspace() -> PreTokenizerWrapper {
        PreTokenizerWrapper::Metaspace(Metaspace::new('▁', PrependScheme::Always, true))
    }

    fn unigram(pre_tokenizer: PreTokenizerWrapper) -> HfTokenizer {
        let vocabulary = ["<unk>", "▁", "▁one", "▁two", "one", "two"]
            .into_iter()
            .map(|piece| (piece.to_string(), -1.0))
            .collect();
        let mut tokenizer = HfTokenizer::new(Unigram::from(vocabulary, Some(0), false).unwrap());
        tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
        tokenizer
    }

    fn ids(tokenizer: &HfTokenizer, text: &str) -> Vec<u32> {
        tokenizer.encode(text, false).unwrap().get_ids().to_vec()
    }

    #[test]
    fn lone_metaspace_without_the_rule_keeps_a_lone_marker() {
        let tokenizer = unigram(metaspace());
        assert_eq!(ids(&tokenizer, "one "), [ONE, LONE]);
    }

    #[test]
    fn trailing_space_gives_no_lone_marker() {
        let mut tokenizer = unigram(metaspace());
        split_whitespace_before_metaspace(&mut tokenizer);
        assert_eq!(ids(&tokenizer, "one "), [ONE]);
        assert_eq!(ids(&tokenizer, "one   "), [ONE]);
    }

    #[test]
    fn leading_spaces_give_no_lone_marker() {
        let mut tokenizer = unigram(metaspace());
        split_whitespace_before_metaspace(&mut tokenizer);
        assert_eq!(ids(&tokenizer, " one"), [ONE]);
        assert_eq!(ids(&tokenizer, "  one"), [ONE]);
    }

    #[test]
    fn repeated_inner_spaces_give_no_lone_marker() {
        let mut tokenizer = unigram(metaspace());
        split_whitespace_before_metaspace(&mut tokenizer);
        assert_eq!(ids(&tokenizer, "one  two"), [ONE, TWO]);
        assert_eq!(ids(&tokenizer, "one \t\n two"), [ONE, TWO]);
    }

    #[test]
    fn rule_reaches_plain_and_cut_encoding() {
        let mut inner = unigram(metaspace());
        split_whitespace_before_metaspace(&mut inner);
        let mut tokenizer = Tokenizer {
            inner,
            truncating: None,
            resource_policy: ResourcePolicy::new(2, 16, 2048, usize::MAX),
            pad_token_id: None,
        };
        tokenizer.prepare_cut().unwrap();
        assert_eq!(tokenizer.encode("one  two ", false).unwrap().0, [ONE, TWO]);
        let cut = tokenizer.encode_cut("", "one  two  one ").unwrap();
        assert_eq!(cut.token_ids, [ONE, TWO]);
        assert_eq!(cut.tokens_total, 3);
    }

    #[test]
    fn word_piece_is_left_as_loaded() {
        let mut tokenizer = HfTokenizer::new(WordPiece::default());
        tokenizer.with_pre_tokenizer(Some(metaspace()));
        split_whitespace_before_metaspace(&mut tokenizer);
        assert_eq!(tokenizer.get_pre_tokenizer(), Some(&metaspace()));
    }

    #[test]
    fn unigram_with_a_sequence_is_left_as_loaded() {
        let sequence = PreTokenizerWrapper::Sequence(Sequence::new(vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            metaspace(),
        ]));
        let mut tokenizer = unigram(sequence.clone());
        split_whitespace_before_metaspace(&mut tokenizer);
        assert_eq!(tokenizer.get_pre_tokenizer(), Some(&sequence));
    }

    /// R5: what the rule does for these inputs, unchecked against the reference.
    #[test]
    fn recorded_behaviour_for_other_white_space_and_markers() {
        let mut tokenizer = unigram(metaspace());
        split_whitespace_before_metaspace(&mut tokenizer);
        // U+00A0 and U+3000 are white space to char::is_whitespace, so they split.
        assert_eq!(ids(&tokenizer, "one\u{a0}two\u{a0}"), [ONE, TWO]);
        assert_eq!(ids(&tokenizer, "one\u{3000}two"), [ONE, TWO]);
        // A literal marker is not white space; Metaspace splits before it as before.
        assert_eq!(ids(&tokenizer, "one▁two"), [ONE, TWO]);
        assert_eq!(ids(&tokenizer, "one▁"), [ONE, LONE]);
        // An added token is matched before pre-tokenisation and keeps its id.
        let added = tokenizer.add_special_tokens(&[AddedToken::from("<x>", true)]);
        assert_eq!(added, 1);
        let x = tokenizer.token_to_id("<x>").unwrap();
        assert_eq!(ids(&tokenizer, "one <x> two"), [ONE, x, TWO]);
    }
}
