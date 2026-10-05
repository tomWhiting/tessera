//! Runs the built worker as a child process and checks its frames.

use std::io::{Cursor, Write};
use std::num::NonZeroU32;
use std::process::{Command, Output, Stdio};

use haem_frames::embedding::{
    decode_vector, read_message, write_message, Embed, FailedCode, Input, ItemCode, Kind, Limits,
    Message, Outcome, Start,
};

#[path = "support/fixture.rs"]
mod fixture;

#[path = "protocol/never_cut.rs"]
mod never_cut;

const fn limits() -> Limits {
    Limits {
        memory_bytes: 1 << 30,
        threads: 1,
        batch_items: 4,
        input_bytes: 512,
        tokens: 16,
        frame_bytes: 65_536,
    }
}

fn start(directory: &std::path::Path, limits: Limits) -> Start {
    Start {
        protocol: 1,
        model_dir: directory.to_str().unwrap().to_string(),
        limits,
    }
}

fn frames(messages: &[Message]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for message in messages {
        write_message(&mut bytes, message, NonZeroU32::new(1 << 30).unwrap()).unwrap();
    }
    bytes
}

fn execute(bytes: &[u8]) -> Output {
    execute_with_environment(bytes, None)
}

fn execute_with_environment(bytes: &[u8], environment: Option<(&str, &str)>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tessera-worker"));
    command.env_clear();
    if let Some((name, value)) = environment {
        command.env(name, value);
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    input.write_all(bytes).unwrap();
    drop(input);
    child.wait_with_output().unwrap()
}

#[test]
fn other_environment_variables_are_still_reported() {
    let model = fixture::installed();
    let output = execute_with_environment(
        &frames(&[Message::Start(start(model.path(), limits()))]),
        Some(("TESSERA_ENVIRONMENT_TEST", "present")),
    );
    assert!(output.status.success(), "{:?}", output.stderr);
    let decoded = messages(&output);
    let [Message::Ready(ready)] = decoded.as_slice() else {
        panic!("expected Ready, got {decoded:?}")
    };
    assert_eq!(ready.environment, 1);
}

fn messages(output: &Output) -> Vec<Message> {
    let mut input = Cursor::new(&output.stdout);
    let mut decoded = Vec::new();
    while let Some(message) = read_message(&mut input, NonZeroU32::new(1 << 30).unwrap()).unwrap() {
        decoded.push(message);
    }
    decoded
}

fn assert_failed(output: &Output, expected: FailedCode) {
    assert!(!output.status.success());
    let decoded = messages(output);
    let Some(Message::Failed(failed)) = decoded.last() else {
        panic!(
            "expected Failed, got {decoded:?}; stderr={:?}",
            output.stderr
        );
    };
    assert_eq!(failed.code, expected);
    assert!(failed.message.chars().count() <= 1024);
}

#[test]
fn empty_standard_input_is_a_clean_end() {
    let output = execute(&[]);
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn a_truncated_header_is_protocol_failure() {
    assert_failed(&execute(&[0, 0]), FailedCode::EmbedProtocol);
}

#[test]
fn an_oversized_start_is_refused_before_its_payload() {
    assert_failed(&execute(&8193_u32.to_be_bytes()), FailedCode::EmbedProtocol);
}

#[test]
fn a_wrong_first_tag_is_protocol_failure() {
    assert_failed(
        &execute(&[0, 0, 0, 3, 18, b'{', b'}']),
        FailedCode::EmbedProtocol,
    );
}

#[test]
fn unknown_start_members_are_protocol_failure() {
    let payload = br#"{"protocol":1,"model_dir":"/none","limits":{"memory_bytes":1,"threads":1,"batch_items":1,"input_bytes":1,"tokens":1,"frame_bytes":8192},"extra":true}"#;
    let mut bytes = u32::try_from(payload.len() + 1)
        .unwrap()
        .to_be_bytes()
        .to_vec();
    bytes.push(16);
    bytes.extend_from_slice(payload);
    assert_failed(&execute(&bytes), FailedCode::EmbedProtocol);
}

#[test]
fn a_missing_manifest_is_named_with_its_os_error() {
    let dir = tempfile::tempdir().unwrap();
    let output = execute(&frames(&[Message::Start(start(dir.path(), limits()))]));
    assert_failed(&output, FailedCode::EmbedModelMissing);
    let decoded = messages(&output);
    let Message::Failed(failed) = &decoded[0] else {
        unreachable!()
    };
    assert!(failed.message.contains("manifest.json"));
    assert!(failed.message.contains("No such file"));
}

#[test]
fn malformed_manifest_contents_are_model_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("manifest.json"), b"{}").unwrap();
    assert_failed(
        &execute(&frames(&[Message::Start(start(dir.path(), limits()))])),
        FailedCode::EmbedModelMismatch,
    );
}

#[test]
fn derived_limit_overflow_is_a_named_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let mut policy = limits();
    policy.batch_items = u64::MAX;
    policy.input_bytes = u64::MAX;
    assert_failed(
        &execute(&frames(&[Message::Start(start(dir.path(), policy))])),
        FailedCode::EmbedLimits,
    );
}

/// A short text, a text over the token limit, an empty text and a text over the byte limit.
fn mixed_documents() -> Embed {
    let long = vec!["one"; 24].join(" ");
    Embed {
        kind: Kind::Document,
        items: vec![
            Input {
                id: "first".into(),
                text: "one two three".into(),
            },
            Input {
                id: "cut".into(),
                text: long,
            },
            Input {
                id: "empty".into(),
                text: "\u{2003}\t\n".into(),
            },
            Input {
                id: "bytes".into(),
                text: "é".repeat(257),
            },
        ],
    }
}

#[test]
fn installed_worker_reports_identity_roles_and_ordered_refusals() {
    let model = fixture::installed();
    let mut batch = mixed_documents();
    batch.items.remove(1);
    let query = Embed {
        kind: Kind::Query,
        items: vec![Input {
            id: "query".into(),
            text: "one".into(),
        }],
    };
    let output = execute(&frames(&[
        Message::Start(start(model.path(), limits())),
        Message::Embed(batch),
        Message::Embed(query),
    ]));
    assert!(output.status.success(), "{:?}", output.stderr);
    let decoded = messages(&output);
    assert_eq!(decoded.len(), 3);
    let Message::Ready(ready) = &decoded[0] else {
        panic!("missing Ready")
    };
    assert_eq!(ready.environment, 0);
    assert_eq!(ready.core_limit, 0);
    assert_eq!(ready.model.name, "bge-base-en-v1.5");
    assert_eq!(ready.model.dimensions, 768);
    assert_eq!(ready.model.max_tokens, 32);
    assert_eq!(ready.model.special_tokens, 2);
    assert!(ready.model.normalised);
    assert!(ready.model.prefix_tokens > 0);
    let Message::Vectors(vectors) = &decoded[1] else {
        panic!("missing Vectors")
    };
    assert_eq!(vectors.items.len(), 3);
    assert_eq!(
        vectors.items.iter().map(Outcome::id).collect::<Vec<_>>(),
        ["first", "empty", "bytes"]
    );
    let Outcome::Vector {
        vector,
        tokens_read,
        tokens_total,
        ..
    } = &vectors.items[0]
    else {
        panic!("missing first vector")
    };
    assert_eq!((*tokens_read, *tokens_total), (5, 5));
    assert_eq!(decode_vector(vector).unwrap().len(), 768);
    assert!(matches!(
        vectors.items[1],
        Outcome::Refused {
            code: ItemCode::EmbedInputEmpty,
            ..
        }
    ));
    assert!(matches!(
        vectors.items[2],
        Outcome::Refused {
            code: ItemCode::EmbedInputTooLarge,
            ..
        }
    ));
    let Message::Vectors(query) = &decoded[2] else {
        panic!("missing query Vectors")
    };
    let Outcome::Vector {
        tokens_read,
        tokens_total,
        ..
    } = &query.items[0]
    else {
        panic!("missing query vector")
    };
    assert!(tokens_read > &3);
    assert_eq!(tokens_read, tokens_total);
}

#[test]
fn crossing_the_rust_heap_bound_exits_86_without_a_failure_frame() {
    let model = fixture::installed();
    let mut policy = limits();
    policy.memory_bytes = 512 << 20;
    policy.frame_bytes = 1 << 30;
    let mut bytes = frames(&[Message::Start(start(model.path(), policy))]);
    bytes.extend_from_slice(&(600_u32 << 20).to_be_bytes());
    let output = execute(&bytes);
    assert_eq!(output.status.code(), Some(haem_worker::MEMORY_LIMIT_EXIT));
    let decoded = messages(&output);
    assert!(matches!(decoded.as_slice(), [Message::Ready(_)]));
}

#[test]
fn a_start_memory_estimate_that_cannot_hold_the_model_is_limits() {
    let model = fixture::installed();
    let mut policy = limits();
    policy.memory_bytes = 1 << 20;
    assert_failed(
        &execute(&frames(&[Message::Start(start(model.path(), policy))])),
        FailedCode::EmbedLimits,
    );
}

#[test]
fn activation_and_model_estimates_share_the_start_memory_budget() {
    let model = fixture::installed();
    let mut policy = limits();
    policy.memory_bytes = 436_000_001;
    assert_failed(
        &execute(&frames(&[Message::Start(start(model.path(), policy))])),
        FailedCode::EmbedLimits,
    );
}

#[test]
fn prefix_and_special_tokens_must_leave_content_room_before_ready() {
    let model = fixture::installed();
    let mut policy = limits();
    policy.tokens = 3;
    let output = execute(&frames(&[Message::Start(start(model.path(), policy))]));
    assert_failed(&output, FailedCode::EmbedLimits);
    assert!(matches!(messages(&output).as_slice(), [Message::Failed(_)]));
}

fn invalid_batch(items: Vec<Input>, expected: FailedCode) {
    let model = fixture::installed();
    assert_failed(
        &execute(&frames(&[
            Message::Start(start(model.path(), limits())),
            Message::Embed(Embed {
                kind: Kind::Document,
                items,
            }),
        ])),
        expected,
    );
}

#[test]
fn a_batch_over_its_item_count_ends_with_its_named_failure() {
    invalid_batch(
        (0..5)
            .map(|index| Input {
                id: index.to_string(),
                text: "one".into(),
            })
            .collect(),
        FailedCode::EmbedBatchTooLarge,
    );
}

#[test]
fn repeated_ids_are_protocol_failure() {
    invalid_batch(
        vec![
            Input {
                id: "same".into(),
                text: "one".into(),
            },
            Input {
                id: "same".into(),
                text: "two".into(),
            },
        ],
        FailedCode::EmbedProtocol,
    );
}

#[test]
fn an_empty_batch_is_protocol_failure() {
    invalid_batch(Vec::new(), FailedCode::EmbedProtocol);
}

#[test]
fn nonfinite_model_output_is_named_and_ends_without_vectors() {
    let model = fixture::installed_with_bias(f32::NAN);
    let output = execute(&frames(&[
        Message::Start(start(model.path(), limits())),
        Message::Embed(Embed {
            kind: Kind::Document,
            items: vec![Input {
                id: "bad".into(),
                text: "one".into(),
            }],
        }),
    ]));
    assert_failed(&output, FailedCode::EmbedOutputInvalid);
    assert!(matches!(
        messages(&output).as_slice(),
        [Message::Ready(_), Message::Failed(_)]
    ));
}

#[test]
fn ready_names_the_source_commit_and_compute_build() {
    let model = fixture::installed();
    let output = execute(&frames(&[Message::Start(start(model.path(), limits()))]));
    assert!(output.status.success(), "{:?}", output.stderr);
    let decoded = messages(&output);
    let [Message::Ready(ready)] = decoded.as_slice() else {
        panic!("expected Ready, got {decoded:?}")
    };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let head = Command::new("git")
        .current_dir(&root)
        .args(["rev-parse", "--verify", "HEAD"])
        .output()
        .unwrap();
    assert!(head.status.success(), "{:?}", head.stderr);
    let head = String::from_utf8(head.stdout).unwrap();
    let head = head.trim();
    let status = Command::new("git")
        .current_dir(&root)
        .args(["status", "--porcelain=v1", "--untracked-files=normal"])
        .output()
        .unwrap();
    assert!(status.status.success(), "{:?}", status.stderr);
    let changes = if status.stdout.is_empty() {
        ""
    } else {
        "+changes"
    };
    let commit = format!("{}{changes}", &head[..12]);
    assert_eq!(env!("TESSERA_SOURCE_COMMIT"), commit);
    let compute = if cfg!(feature = "accelerate") {
        "accelerate"
    } else {
        "plain"
    };
    assert_eq!(
        ready.worker,
        format!(
            "tessera-worker {} commit {commit} {compute}",
            env!("CARGO_PKG_VERSION")
        )
    );
    haem_frames::embedding::check_ready(ready).unwrap();
}
