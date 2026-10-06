//! Protocol 2 (EMB-E2): long documents answer whole windows.
//!
//! The fixture model reads 32 tokens with 2 specials and no document prompt, so
//! a document window holds 30 content tokens. Its 7-token query prompt makes
//! Ready's content tokens 32 - 2 - 7 = 23.

use haem_frames::embedding::{
    check_vectors_windowed, decode_vector, Embed, FailedCode, Input, ItemCode, Kind, Limits,
    Message, Outcome, Ready, Start, Vectors, Windows, PROTOCOL_WINDOWS,
};

use super::{execute, fixture, frames, limits, messages, mixed_documents, start};

const WINDOW_CONTENT: u64 = 30;
const READY_CONTENT: u64 = 23;

const fn windowed_limits() -> Limits {
    Limits {
        memory_bytes: 1 << 30,
        threads: 1,
        batch_items: 4,
        input_bytes: 4_096,
        tokens: 32,
        frame_bytes: 1 << 20,
    }
}

fn windowed(directory: &std::path::Path, limits: Limits, overlap: u64, cap: u64) -> Start {
    Start {
        protocol: PROTOCOL_WINDOWS,
        windows: Some(Windows {
            overlap_tokens: overlap,
            max_windows: cap,
        }),
        ..start(directory, limits)
    }
}

/// `count` whitespace-separated words cycling through `vocabulary`, one token each.
fn words(count: usize, vocabulary: &[&str]) -> String {
    (0..count)
        .map(|index| vocabulary[index % vocabulary.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

fn documents(items: &[(&str, &str)]) -> Embed {
    Embed {
        kind: Kind::Document,
        items: items
            .iter()
            .map(|(id, text)| Input {
                id: (*id).into(),
                text: (*text).into(),
            })
            .collect(),
    }
}

/// Runs one Start and one Embed; the worker must answer Ready then Vectors and end cleanly.
fn answer(start: Start, embed: Embed) -> (Ready, Vectors) {
    let output = execute(&frames(&[Message::Start(start), Message::Embed(embed)]));
    assert!(output.status.success(), "{:?}", output.stderr);
    let mut decoded = messages(&output).into_iter();
    let (Some(Message::Ready(ready)), Some(Message::Vectors(vectors)), None) =
        (decoded.next(), decoded.next(), decoded.next())
    else {
        panic!("expected Ready then Vectors only");
    };
    (ready, vectors)
}

fn expected_windows(tokens: u64, overlap: u64) -> usize {
    usize::try_from(1 + (tokens - WINDOW_CONTENT).div_ceil(WINDOW_CONTENT - overlap)).unwrap()
}

/// Checks one Windows answer covers `text` whole, in the planner's count, and returns its bounds.
fn assert_whole_cover(outcome: &Outcome, text: &str, tokens: u64, overlap: u64) -> Vec<(u64, u64)> {
    let Outcome::Windows {
        windows,
        tokens_total,
        ..
    } = outcome
    else {
        panic!("expected Windows, got {outcome:?}");
    };
    assert_eq!(*tokens_total, tokens);
    assert_eq!(windows.len(), expected_windows(tokens, overlap));
    assert_eq!(windows[0].start, 0);
    assert_eq!(
        windows.last().unwrap().end,
        u64::try_from(text.len()).unwrap()
    );
    for pair in windows.windows(2) {
        let (before, after) = (&pair[0], &pair[1]);
        assert!(after.start > before.start, "no progress in {pair:?}");
        assert!((..=before.end).contains(&after.start), "gap in {pair:?}");
        assert!(after.end > before.end);
    }
    for window in windows {
        assert!((1..=WINDOW_CONTENT).contains(&window.tokens_read));
        assert_eq!(decode_vector(&window.vector).unwrap().len(), 768);
    }
    windows
        .iter()
        .map(|window| (window.start, window.end))
        .collect()
}

#[test]
fn protocol_two_ready_names_protocol_two() {
    let model = fixture::installed();
    for (start, protocol) in [
        (windowed(model.path(), windowed_limits(), 5, 7), 2),
        (start(model.path(), limits()), 1),
    ] {
        let output = execute(&frames(&[Message::Start(start)]));
        assert!(output.status.success(), "{:?}", output.stderr);
        let decoded = messages(&output);
        let [Message::Ready(ready)] = decoded.as_slice() else {
            panic!("expected Ready, got {decoded:?}")
        };
        assert_eq!(ready.protocol, protocol);
    }
}

#[test]
fn a_long_document_answers_whole_windows() {
    let model = fixture::installed();
    let long = words(90, &["one", "two", "three"]);
    let items = [("short", "one two three"), ("long", long.as_str())];
    let (ready, vectors) = answer(
        windowed(model.path(), windowed_limits(), 5, 7),
        documents(&items),
    );
    assert!(matches!(
        vectors.items[0],
        Outcome::Vector { tokens_read: 5, .. }
    ));
    let bounds = assert_whole_cover(&vectors.items[1], &long, 90, 5);
    assert_eq!(bounds.len(), 4);
    let Outcome::Windows { windows, .. } = &vectors.items[1] else {
        unreachable!()
    };
    assert_eq!(windows[0].tokens_read, WINDOW_CONTENT);
    check_vectors_windowed(
        &vectors,
        &documents(&items),
        &ready,
        &windowed_limits(),
        Some(Windows {
            overlap_tokens: 5,
            max_windows: 7,
        }),
    )
    .unwrap();
}

#[test]
fn trailing_whitespace_is_inside_the_last_window() {
    let model = fixture::installed();
    let text = format!("{}   \n ", words(70, &["one", "two"]));
    let (_, vectors) = answer(
        windowed(model.path(), windowed_limits(), 5, 7),
        documents(&[("trailing", &text)]),
    );
    let bounds = assert_whole_cover(&vectors.items[0], &text, 70, 5);
    assert_eq!(bounds.last().unwrap().1, u64::try_from(text.len()).unwrap());
}

#[test]
fn windows_never_cross_a_char_boundary() {
    let model = fixture::installed();
    let text = words(75, &["é", "naïve", "🦀", "日本", "one"]);
    let items = [("multibyte", text.as_str())];
    let (ready, vectors) = answer(
        windowed(model.path(), windowed_limits(), 3, 6),
        documents(&items),
    );
    for (start, end) in assert_whole_cover(&vectors.items[0], &text, 75, 3) {
        let (start, end) = (
            usize::try_from(start).unwrap(),
            usize::try_from(end).unwrap(),
        );
        assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
    }
    let asked = Windows {
        overlap_tokens: 3,
        max_windows: 6,
    };
    check_vectors_windowed(
        &vectors,
        &documents(&items),
        &ready,
        &windowed_limits(),
        Some(asked),
    )
    .unwrap();
}

/// The model's weights are NaN, so any inference would end the worker with
/// `embed_output_invalid`; a clean Vectors proves none ran.
#[test]
fn over_the_cap_is_refused_and_not_embedded() {
    let model = fixture::installed_with_bias(f32::NAN);
    let text = words(60, &["one", "two", "three"]);
    assert_eq!(expected_windows(60, 5), 3);
    let (_, vectors) = answer(
        windowed(model.path(), windowed_limits(), 5, 2),
        documents(&[("capped", &text)]),
    );
    let [Outcome::Refused {
        code,
        tokens_total,
        tokens_limit,
        ..
    }] = vectors.items.as_slice()
    else {
        panic!("expected one refusal, got {:?}", vectors.items);
    };
    assert_eq!(*code, ItemCode::WindowsOverCap);
    assert_eq!((*tokens_total, *tokens_limit), (Some(60), Some(32)));
}

/// Start's tokens (16) are below the model's 32, so a window the model reads
/// whole cannot be fed: the long item is refused by name and gets no vector,
/// while the page still answers its short item.
#[test]
fn a_window_the_model_would_cut_is_refused_by_name() {
    let model = fixture::installed();
    let mut policy = windowed_limits();
    policy.tokens = 16;
    let long = words(40, &["one", "two", "three"]);
    let items = [("short", "one two"), ("long", long.as_str())];
    let asked = Windows {
        overlap_tokens: 5,
        max_windows: 7,
    };
    let (ready, vectors) = answer(windowed(model.path(), policy, 5, 7), documents(&items));
    assert!(matches!(vectors.items[0], Outcome::Vector { .. }));
    let Outcome::Refused {
        code,
        tokens_total,
        tokens_limit,
        ..
    } = &vectors.items[1]
    else {
        panic!(
            "expected a refusal with no vector, got {:?}",
            vectors.items[1]
        );
    };
    assert_eq!(*code, ItemCode::TextLongerThanModel);
    assert_eq!((*tokens_total, *tokens_limit), (Some(40), Some(32)));
    assert!(*tokens_total > Some(READY_CONTENT));
    check_vectors_windowed(&vectors, &documents(&items), &ready, &policy, Some(asked)).unwrap();
}

#[test]
fn a_query_is_never_windowed() {
    let model = fixture::installed();
    let output = execute(&frames(&[
        Message::Start(windowed(model.path(), windowed_limits(), 5, 7)),
        Message::Embed(Embed {
            kind: Kind::Query,
            items: vec![Input {
                id: "question".into(),
                text: words(40, &["one", "two"]),
            }],
        }),
    ]));
    assert_eq!(output.status.code(), Some(1), "{:?}", output.stderr);
    let decoded = messages(&output);
    let [Message::Ready(_), Message::Failed(failed)] = decoded.as_slice() else {
        panic!("expected Ready then Failed only, got {decoded:?}");
    };
    assert_eq!(failed.code, FailedCode::EmbedLimits);
    assert!(failed
        .message
        .starts_with("text_longer_than_model {\"items\":[{\"id\":\"question\""));
}

#[test]
fn protocol_one_is_unchanged() {
    let model = fixture::installed();
    let output = execute(&frames(&[
        Message::Start(start(model.path(), limits())),
        Message::Embed(mixed_documents()),
    ]));
    assert_eq!(output.status.code(), Some(1), "{:?}", output.stderr);
    let decoded = messages(&output);
    let [Message::Ready(ready), Message::Failed(failed)] = decoded.as_slice() else {
        panic!("expected Ready then Failed only, got {decoded:?}");
    };
    assert_eq!(ready.protocol, 1);
    assert_eq!(failed.code, FailedCode::EmbedLimits);
    assert_eq!(
        failed.message,
        "text_longer_than_model {\"items\":[{\"id\":\"cut\",\"tokens_total\":26,\"tokens_limit\":16}],\"omitted\":0}"
    );
}

#[test]
fn a_start_with_overlap_at_content_tokens_fails_limits() {
    let model = fixture::installed();
    let output = execute(&frames(&[Message::Start(windowed(
        model.path(),
        windowed_limits(),
        READY_CONTENT,
        7,
    ))]));
    assert_eq!(output.status.code(), Some(1), "{:?}", output.stderr);
    let decoded = messages(&output);
    let [Message::Failed(failed)] = decoded.as_slice() else {
        panic!("expected Failed only, got {decoded:?}");
    };
    assert_eq!(failed.code, FailedCode::EmbedLimits);
    assert!(failed.message.contains("overlap_tokens"));
    let below = execute(&frames(&[Message::Start(windowed(
        model.path(),
        windowed_limits(),
        READY_CONTENT - 1,
        7,
    ))]));
    assert!(below.status.success(), "{:?}", below.stderr);
}
