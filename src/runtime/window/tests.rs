use super::{plan_token_windows, ContextWindowConfig, ContextWindowError};
use crate::runtime::ResourcePolicy;

#[test]
fn windows_cover_content_once_through_center_ownership() {
    let content = (10_u32..30).collect::<Vec<_>>();
    let windows = plan_token_windows(
        &content,
        &[1],
        &[2],
        ContextWindowConfig::new(10, 2),
        ResourcePolicy::default(),
    )
    .expect("valid windows should plan");

    assert_eq!(windows.len(), 3);
    assert!(windows.iter().all(|window| window.token_ids.len() <= 10));
    assert!(windows
        .iter()
        .all(|window| window.attention_mask.iter().all(|value| *value == 1)));
    let owned = windows
        .iter()
        .flat_map(|window| window.owned_start..window.owned_end)
        .collect::<Vec<_>>();
    assert_eq!(owned, (0..content.len()).collect::<Vec<_>>());
}

#[test]
fn invalid_overlap_and_policy_are_rejected() {
    let policy = ResourcePolicy::default().with_max_sequence_tokens(8);
    assert!(matches!(
        plan_token_windows(
            &[1, 2],
            &[10],
            &[11],
            ContextWindowConfig::new(9, 0),
            policy
        ),
        Err(ContextWindowError::PolicyLimit { .. })
    ));
    assert!(matches!(
        plan_token_windows(
            &[1, 2],
            &[10],
            &[11],
            ContextWindowConfig::new(8, 6),
            policy
        ),
        Err(ContextWindowError::Overlap { .. })
    ));
}

#[test]
fn empty_content_still_produces_the_special_token_input() {
    let windows = plan_token_windows(
        &[],
        &[1],
        &[2],
        ContextWindowConfig::new(8, 1),
        ResourcePolicy::default(),
    )
    .expect("empty content should be representable");
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].token_ids, [1, 2]);
    assert_eq!(windows[0].owned_len(), 0);
}
