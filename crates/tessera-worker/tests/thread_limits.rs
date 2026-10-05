//! Observe persistent compute threads after a completed fixture embedding.
#![cfg(target_os = "macos")]

use std::io::{BufReader, Write};
use std::num::NonZeroU32;
use std::process::{Command, Stdio};

use haem_frames::embedding::{
    read_message, write_message, Embed, Input, Kind, Limits, Message, Start,
};

#[path = "support/fixture.rs"]
mod fixture;

fn observe_threads(limit: u64) -> usize {
    let model = fixture::installed();
    let frame_limit = NonZeroU32::new(65_536).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_tessera-worker"))
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    write_message(
        &mut input,
        &Message::Start(Start {
            protocol: 1,
            model_dir: model.path().to_str().unwrap().to_string(),
            limits: Limits {
                memory_bytes: 1 << 30,
                threads: limit,
                batch_items: 1,
                input_bytes: 512,
                tokens: 32,
                frame_bytes: 65_536,
            },
        }),
        frame_limit,
    )
    .unwrap();
    input.flush().unwrap();
    let ready = read_message(&mut output, frame_limit).unwrap();
    write_message(
        &mut input,
        &Message::Embed(Embed {
            kind: Kind::Document,
            items: vec![Input {
                id: "document".into(),
                text: vec!["one"; 30].join(" "),
            }],
        }),
        frame_limit,
    )
    .unwrap();
    input.flush().unwrap();
    let vectors = read_message(&mut output, frame_limit).unwrap();
    let threads = Command::new("/bin/ps")
        .args(["-M", "-p", &child.id().to_string()])
        .output()
        .unwrap();
    drop(input);
    drop(output);
    let exit = child.wait_with_output().unwrap();
    assert!(exit.status.success(), "{:?}", exit.stderr);
    assert!(matches!(ready, Some(Message::Ready(_))), "{ready:?}");
    assert!(matches!(vectors, Some(Message::Vectors(_))), "{vectors:?}");
    assert!(threads.status.success(), "{:?}", threads.stderr);
    let listing = String::from_utf8(threads.stdout).unwrap();
    let count = listing.lines().skip(1).count();
    assert!(count > 0, "no worker threads: {listing}");
    count
}

#[test]
fn one_compute_thread_from_empty_environment() {
    let observed = observe_threads(1);
    assert_eq!(observed, 2, "limit 1 permits main plus one worker");
}

#[test]
fn two_compute_threads_from_empty_environment() {
    let observed = observe_threads(2);
    assert!(
        observed <= 3,
        "limit 2 permits main plus two workers, observed {observed}"
    );
}
