# Embedding codec costs

This S0b addendum records source costs, not measured performance. Tessera's
baseline is `09dbefdd1d59d79680b57fa5949468282d3566fc`, tree
`a50628b5205db68e4b8591c254d3d7078e65b8c2`. Only this document changes.
There was no compilation, test, inference, model fetch or timing run.

The worker pins haem-frames to haematite commit
`1605372c1cd176cf9b1d54f3290e1a94a03c39cd`
(`crates/tessera-worker/Cargo.lock:971–973`). References beginning with
`crates/haem-frames/` below refer to that exact dependency commit, read from
`/Users/tom/.cargo/git/checkouts/haematite-eef79b256661262b/1605372/`.
The dependency checkout's tree is
`5f970dde00349b90f5a211901748595971ecc351`.

## Vector representation and byte arithmetic

`encode_vector` reserves eight characters per member, rejects non-finite
members, and writes each f32's four little-endian bytes as two lowercase hex
digits per byte (`crates/haem-frames/src/embedding/vector.rs:3–23`). The returned
string owns this representation; it does not borrow the original vector
(`crates/haem-frames/src/embedding/vector.rs:5–10,18–23`).

For one 768-dimensional f32 vector, the derived counts are:

| Quantity | Arithmetic | Bytes or characters |
| --- | --- | --- |
| Raw f32 member bytes | 768 × 4 | 3,072 bytes |
| Hex representation | 768 × 4 × 2 | 6,144 ASCII characters, hence 6,144 bytes |
| Representation expansion | 6,144 / 3,072 | 2 times the raw member bytes |
| Decoded members | 768 × 4 | 3,072 logical member bytes per decode result |

The four-byte f32 word and two hex digits per byte follow the encode/decode
loops (`crates/haem-frames/src/embedding/vector.rs:18–20,38,48–52`). These
counts exclude JSON field names, IDs, counts, punctuation, frame headers and
allocator overhead. The twofold representation expansion is not a latency
estimate. Vec capacity and actual allocator traffic were not measured.

## Decode and validation passes

`decode_vector` requires an exact multiple of eight characters, checks the
lowercase hex alphabet, reconstructs little-endian f32 members, rejects
non-finite values, and collects an owned `Vec<f32>`
(`crates/haem-frames/src/embedding/vector.rs:26–61`).

`check_vectors` checks response count and ID order against the request, then
calls `decode_vector` for every vector outcome. It checks dimensions and token
counts; when the model declares normalization, it also scans the decoded
members to compute an f64 squared norm and requires norm error at most 0.0001
(`crates/haem-frames/src/embedding/checks.rs:163–214`). Its decoded vector is
local to the check and is not returned to the caller
(`crates/haem-frames/src/embedding/checks.rs:195–220`). Refusal outcomes do not
take that decode path (`crates/haem-frames/src/embedding/checks.rs:183–195`).

The concrete worker builds the hex string with `encode_vector` and calls
`check_vectors` before returning its response
(`crates/tessera-worker/src/engine.rs:150–160,197–205`). Thus a successful
worker vector has one encode pass and one validation decode pass before the
response leaves the engine. The validation temporarily materializes 768
decoded members for this example; normalization adds another member scan,
not another hex decode.

Receiving a `Vectors` frame does not itself run `check_vectors`:
`read_message` reads the frame and dispatches its tag; the `Vectors` branch
only parses JSON (`crates/haem-frames/src/embedding/codec.rs:85–89,118–119,128–135`).
Consequently the following are distinct counts, derived from those call sites:

| Caller behavior | Decode passes per successful vector, including the worker's check |
| --- | --- |
| Worker check, then receiver only parses the message | 1 |
| Receiver also calls `check_vectors` | 2 |
| Receiver calls `check_vectors`, then separately calls `decode_vector` to use the members | 3 |

The latter two rows are conditional caller costs, not claims about an unread
consumer. Each additional `check_vectors` invocation adds a decode and, for a
normalized model, a norm scan. A separate consuming decode allocates a separate
result; the validation API does not supply its temporary vector to reuse.

## JSON and framing handoffs

| Boundary | Source-confirmed work |
| --- | --- |
| Response to JSON payload | `write_message` selects `Vectors` and calls `payload`; `serde_json::to_vec` creates an owned payload `Vec<u8>` (`crates/haem-frames/src/embedding/codec.rs:149–168`). Exact serializer allocations and total JSON bytes remain unmeasured. |
| Payload to stream | `write_message` passes the tag and borrowed payload bytes to the frame writer (`crates/haem-frames/src/embedding/codec.rs:174–182`). |
| Frame size admission | The writer checks payload length plus one tag byte against the limit before any output (`crates/haem-frames/src/frame.rs:102–112`). |
| Successful frame output | Three `write_all` calls write the four-byte big-endian length, one-byte tag and payload; then one `flush` call (`crates/haem-frames/src/frame.rs:113–116`). The framing overhead is therefore five bytes per frame, outside the JSON payload. |
| Frame input | The reader handles the four-byte length, rejects zero/over-limit lengths, reserves and fills an owned body Vec, then reads the body (`crates/haem-frames/src/frame.rs:65–92`). The body includes the tag. |
| Body to response | `read_message` splits off the tag and parses the remaining byte slice (`crates/haem-frames/src/embedding/codec.rs:128–135`). `parse` uses `DeserializeOwned` and rejects trailing JSON data (`crates/haem-frames/src/embedding/codec.rs:85–89`). Exact parser allocations remain unmeasured. |

One flush per frame means one explicit stream `Write::flush` invocation after
the three successful `write_all` calls. An earlier error returns before that
flush (`crates/haem-frames/src/frame.rs:113–116`). This is not a disk-sync call.
These generic `Write` calls do not determine syscall counts, buffering behavior,
backpressure duration or concrete stream-lock behavior; all remain unmeasured.

## Cost boundaries and measurement implications

The inspected routines traverse the current vector, current response items or
current frame. They contain no explicit lock acquisition, whole-model/state
clone, history traversal, disk-sync call, model load or network fetch
(`crates/haem-frames/src/embedding/vector.rs:5–61`,
`crates/haem-frames/src/embedding/checks.rs:163–220`,
`crates/haem-frames/src/embedding/codec.rs:128–182`,
`crates/haem-frames/src/frame.rs:65–116`). Concrete I/O and dependency internals
can add costs; they were not instrumented or exhaustively read. JSON payloads,
hex strings, frame bodies and decoded vectors are separate owned representations
at the boundaries identified above.

An end-to-end worker measurement should state whether its completion boundary
includes message parsing, `check_vectors` and a consuming decode. Otherwise two
measurements can include different amounts of work. This is a measurement
requirement inferred from the source trace; it proposes no codec or protocol
change. No speedup, syscall count, allocation-call count, peak memory or wall-time
result is established by this addendum.
