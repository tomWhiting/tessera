use haem_frames::embedding::{Embed, FailedCode, Input, Kind, Message};
use serde_json::{json, Value};

use super::{execute, fixture, frames, limits, messages, mixed_documents, start};

#[test]
fn an_overlength_item_refuses_the_whole_embed_and_exits() {
    let model = fixture::installed();
    let output = execute(&frames(&[
        Message::Start(start(model.path(), limits())),
        Message::Embed(mixed_documents()),
        Message::Embed(Embed {
            kind: Kind::Document,
            items: vec![Input {
                id: "later".into(),
                text: "one".into(),
            }],
        }),
    ]));
    assert_eq!(output.status.code(), Some(1), "{:?}", output.stderr);
    let decoded = messages(&output);
    let [Message::Ready(_), Message::Failed(failed)] = decoded.as_slice() else {
        panic!("expected Ready then Failed only, got {decoded:?}");
    };
    assert_eq!(failed.code, FailedCode::EmbedLimits);
    assert_eq!(
        failed.message,
        "text_longer_than_model {\"items\":[{\"id\":\"cut\",\"tokens_total\":26,\"tokens_limit\":16}],\"omitted\":0}"
    );
}

#[test]
fn long_ids_leave_whole_ordered_items_and_an_omitted_count() {
    let model = fixture::installed();
    let ids: Vec<String> = (0..8)
        .map(|index| format!("{index}{}\"\\\n", "é".repeat(180)))
        .collect();
    let mut policy = limits();
    policy.batch_items = 8;
    let batch = Embed {
        kind: Kind::Document,
        items: ids
            .iter()
            .map(|id| Input {
                id: id.clone(),
                text: vec!["one"; 24].join(" "),
            })
            .collect(),
    };
    let output = execute(&frames(&[
        Message::Start(start(model.path(), policy)),
        Message::Embed(batch),
    ]));
    assert_eq!(output.status.code(), Some(1), "{:?}", output.stderr);
    let decoded = messages(&output);
    let [Message::Ready(_), Message::Failed(failed)] = decoded.as_slice() else {
        panic!("expected Ready then Failed only, got {decoded:?}");
    };
    assert_eq!(failed.code, FailedCode::EmbedLimits);
    assert!(failed.message.chars().count() <= haem_frames::embedding::FAILED_MESSAGE_CHARS);
    let object: Value = serde_json::from_str(
        failed
            .message
            .strip_prefix("text_longer_than_model ")
            .unwrap(),
    )
    .unwrap();
    let items = object["items"].as_array().unwrap();
    let omitted = usize::try_from(object["omitted"].as_u64().unwrap()).unwrap();
    assert!(omitted > 0 && !items.is_empty());
    assert_eq!(items.len() + omitted, ids.len());
    for (item, id) in items.iter().zip(&ids) {
        assert_eq!(
            item,
            &json!({"id":id, "tokens_total":26, "tokens_limit":16})
        );
    }
}

#[test]
fn a_nonempty_text_with_no_content_tokens_refuses_the_embed() {
    let model = fixture::installed_without_content_tokens();
    let output = execute(&frames(&[
        Message::Start(start(model.path(), limits())),
        Message::Embed(Embed {
            kind: Kind::Document,
            items: vec![
                Input {
                    id: "short".into(),
                    text: "one".into(),
                },
                Input {
                    id: "no-content".into(),
                    text: "\u{200b}".into(),
                },
            ],
        }),
    ]));
    assert_eq!(output.status.code(), Some(1), "{:?}", output.stderr);
    let decoded = messages(&output);
    let [Message::Ready(_), Message::Failed(failed)] = decoded.as_slice() else {
        panic!("expected Ready then Failed only, got {decoded:?}");
    };
    assert_eq!(failed.code, FailedCode::EmbedLimits);
    assert!(failed.message.contains("embed_input_no_content_tokens"));
    assert!(failed.message.contains("no-content"));
}
