# Tessera worker from a queue: scouting note, 8 October 2026

Read-only scouting. Nothing was built, compiled, run or tested.

- Tessera: main at `49b0ad5`. Tessera paths below are relative to the repository root.
- haematite: the worker's `crates/tessera-worker/Cargo.lock:971-983` pins `haem-frames` and `haem-worker` to
  haematite main `6f049352e3a7ff67102d1d4f98fe80e54de4ade9`. haematite paths are written
  `haematite@6f04935:path:line` and were read with `git show`, without a checkout.
- Anything that is not in the code is written ABSENT. Anything I concluded rather than read is labelled "Inferred:".

The question being scouted: could a Tessera worker take embedding jobs from a Liminal work queue instead of
from its standard input, so embedding can be pooled across machines? This note records findings and gaps only. It
does not propose a design.

## 1. Protocol 2 on the wire

### 1.1 Framing

- Every frame starts with a 4-byte length, big-endian `u32` (`haematite@6f04935:crates/haem-frames/src/frame.rs:66-76`).
- The length counts one tag byte plus the payload (`frame.rs:102-106`, `frame.rs:113-115`).
- A length of zero is refused as `Fault::Empty` (`frame.rs:77-79`).
- A length over the reader's limit is refused as `Fault::TooLarge`. This happens on the header, before the body is
  allocated (`frame.rs:14-16`, `frame.rs:80-86`).
- End of input is clean only before a header. End of input inside a header or body is `Fault::Cut`
  (`frame.rs:54-62`, `frame.rs:64-75`).
- The writer checks the limit before writing anything, then writes length, tag, payload, and flushes
  (`frame.rs:96-116`).
- The frame layer has no deadlines and no process ownership: "Length framing leaves deadlines and process ownership
  to the caller" (`frame.rs:1`).

### 1.2 Encoding

- The payload is one JSON object, parsed with `serde_json`. Trailing bytes after the object are refused
  (`haematite@6f04935:crates/haem-frames/src/embedding/codec.rs:85-90`). Writing uses `serde_json::to_vec`
  (`codec.rs:149-151`).
- Every message struct uses `deny_unknown_fields`, so an unknown member is a protocol failure
  (`haematite@6f04935:crates/haem-frames/src/embedding.rs:114`, `:125`, `:142`, `:149`, `:161`, `:190`, `:198`, `:205`,
  `:212`, `:253`, `:262`).
- A vector is a string of lowercase hexadecimal: each `f32` member as its 4 little-endian bytes, 8 characters per
  member (`haematite@6f04935:crates/haem-frames/src/embedding/vector.rs:5-24`). A non-finite member is refused on
  encode and decode (`vector.rs:12-17`, `vector.rs:53-58`).

### 1.3 Frame size limits

- The Start frame has a fixed limit of 8,192 bytes, `MINIMUM_FRAME_BYTES` (`embedding.rs:21`). It is read with that
  limit before anything else is known (`codec.rs:138-147`) and checked again on decode (`codec.rs:101-108`).
- Every later frame uses `Start.limits.frame_bytes` as its limit (`crates/tessera-worker/src/session.rs:31`, `:48`).
- `frame_bytes` is a `u32` (`embedding.rs:121`) and must be at least 8,192 (`haematite@6f04935:crates/haem-frames/src/embedding/checks.rs:34-38`).
  So the largest possible frame is 4 GiB less one byte.
- Before Ready, both sides check that `frame_bytes` can carry the largest legal answer for `batch_items`, the model's
  dimensions and, under protocol 2, `max_windows` windows per item (`checks.rs:142-171`; worker side
  `crates/tessera-worker/src/engine.rs:120`; haem side `haematite@6f04935:crates/haem/src/serve/embed/worker.rs:517-520`).
  The size bounds are in `checks.rs:76-130`.
- A Failed message is at most 1,024 characters, `FAILED_MESSAGE_CHARS` (`embedding.rs:22`, `codec.rs:92-97`).

### 1.4 Message types and direction

Tags (`embedding.rs:25-33`):

| Tag | Message | Direction |
|---|---|---|
| 16 | Start | parent to worker, exactly once, first |
| 17 | Ready | worker to parent, once, after Start |
| 18 | Embed | parent to worker, repeated |
| 19 | Vectors | worker to parent, one per Embed |
| 20 | Failed | worker to parent, then the worker exits |

- An unknown tag is `embed_protocol` (`embedding.rs:35-49`).
- Direction is read from use: the worker reads Start then Embed and writes Ready, Vectors, Failed
  (`crates/tessera-worker/src/session.rs:26-71`); haem writes Start and Embed and reads Ready, Vectors, Failed
  (`haematite@6f04935:crates/haem/src/serve/embed/worker.rs:476-501`, `:280`, `:383-413`).
- The worker accepts only Embed after Ready. Anything else is Failed `embed_protocol` and the session ends
  (`session.rs:51-60`).

### 1.5 Start

`embedding.rs:124-134`, `embedding.rs:113-122`, `embedding.rs:141-146`:

- `protocol: u32`: 1, or 2 for windows (`embedding.rs:17-20`).
- `model_dir: String`: must be an absolute path (`checks.rs:40-42`). It is a path on the worker's own filesystem.
- `limits`:
  - `memory_bytes: u64`
  - `threads: u64`
  - `batch_items: u64`
  - `input_bytes: u64`
  - `tokens: u64`
  - `frame_bytes: u32`
  - All must be positive; `frame_bytes` at least 8,192 (`checks.rs:25-39`).
- `windows: Option<{ overlap_tokens: u64, max_windows: u64 }>`: present exactly when `protocol` is 2
  (`embedding.rs:131-133`, `checks.rs:14-24`). `max_windows` must be at least 2 and `overlap_tokens` below `tokens`
  (`checks.rs:43-49`). The overlap must also be below the model's content tokens (`checks.rs:172-180`).
- Worker-side extra rule: with windows, `tokens` must equal the model's `max_tokens`, so windows are fed whole
  (`crates/tessera-worker/src/engine.rs:121-127`).
- haem's defaults: `max_windows` 32, `overlap_tokens` 64 (`haematite@6f04935:crates/haem/src/serve/embed/settings.rs:69-73`).

### 1.6 Ready

`embedding.rs:148-172`:

- `protocol: u32`: the worker answers the protocol its Start named (`embedding.rs:18-19`;
  `crates/tessera-worker/src/engine.rs:102`).
- `worker: String`: the worker's build string, from `TESSERA_WORKER_BUILD`
  (`engine.rs:103`; set in `crates/tessera-worker/build.rs:80` as name, version, commit and compute).
- `core_limit`, `descriptors_closed`, `environment: u64`: numbers from process setup (`engine.rs:104-106`).
- `model`:
  - `name`, `revision`, `manifest_sha256` (64 lowercase hexadecimal characters)
  - `dimensions`, `max_tokens`, `special_tokens`, `prefix_tokens`
  - `normalised: bool`
  - `distance`: `cosine`, `dot` or `euclidean` (`embedding.rs:174-180`)
- Checked by `check_ready` (`checks.rs:55-74`) on both sides (`codec.rs:113-117`, `engine.rs:119`).

### 1.7 Embed (the request)

`embedding.rs:189-202`:

- `kind`: `document` or `query` (`embedding.rs:182-187`).
- `items`: a list of `{ id: String, text: String }`.
- `check_embed` (`checks.rs:188-217`): at least one item (else `embed_protocol`); at most `batch_items`
  (else `embed_batch_too_large`); every id non-empty, at most 512 bytes, no NUL, and unique within the Embed
  (else `embed_protocol`). The worker runs it on every Embed (`session.rs:63-65`).

### 1.8 Vectors (the answer)

`embedding.rs:204-259`. `items` is a list of outcomes, tagged by an `outcome` member:

- `vector`: `{ id, vector, tokens_read, tokens_total }`.
- `windows` (protocol 2, documents only): `{ id, windows: [{ start, end, vector, tokens_read }], tokens_total }`.
  `start` and `end` are byte offsets `[start, end)` on char and token boundaries (`embedding.rs:250-259`).
- `refused`: `{ id, code, tokens_total?, tokens_limit? }`. The two counts are present only with
  `text_longer_than_model` or `windows_over_cap` (`embedding.rs:229-239`, `checks.rs:250-268`).

The worker checks every Vectors with the shared validator before writing it (`engine.rs:152`), and haem checks it
again on receipt (`haematite@6f04935:crates/haem/src/serve/embed/worker.rs:311-320`).

### 1.9 Failed and its codes

`Failed { code, message }` (`embedding.rs:261-266`). Codes (`embedding.rs:52-76`):

- `embed_model_mismatch`
- `embed_model_missing`
- `embed_limits`
- `embed_batch_too_large`
- `embed_output_invalid`
- `embed_protocol`
- `embed_inference_failed`

A Failed always ends the session (section 2).

### 1.10 Per-item refusals

Item codes (`embedding.rs:78-88`):

- `embed_input_empty`: the text is all whitespace (`checks.rs:219-221`).
- `embed_input_too_large`: the text is over `input_bytes` UTF-8 bytes (`checks.rs:223-226`).
- `text_longer_than_model`: the text has more tokens than the model reads.
- `windows_over_cap`: protocol 2 only; the text needs more than `max_windows` windows (`embedding.rs:84-87`).

Where each one comes from:

- Empty and too-large are decided by a shared byte classifier, `input_refusal`, on both sides
  (`checks.rs:219-227`; worker `engine.rs:192`, `engine.rs:229-234`).
- Text longer than the model, protocol 1 or any `query` Embed: the whole Embed fails. The worker sends Failed
  `embed_limits` with a message `text_longer_than_model {"items":[...],"omitted":K}` and no Vectors
  (`engine.rs:176-178`, `engine.rs:218-227`, `crates/tessera-worker/src/failure.rs:23-55`,
  `crates/tessera-worker/README.md:32-39`). Windows apply only to `document` (`engine.rs:141`).
- Text longer than the model, protocol 2 `document`: each item is planned by tokenizing alone before any
  inference (`crates/tessera-worker/src/engine/windows.rs:15-23`, `:119-144`). It is answered as `windows`, or
  refused per item as `windows_over_cap` (`windows.rs:137-139`) or `text_longer_than_model` when no window can move
  forward (`windows.rs:141-142`) or windows tie or overrun (`windows.rs:159-177`).
- A text with no content tokens fails the whole Embed with `embed_limits` naming `embed_input_no_content_tokens`
  (`engine.rs:212-216`, `windows.rs:33-49`). There is no item code for it (`README.md:41-44`).

Inferred: haem pre-filters empty and too-large texts before sending (`haematite@6f04935:crates/haem/src/serve/embed/embedder.rs:206-214`),
so haem treats any per-item refusal other than `text_longer_than_model` and `windows_over_cap` as the worker's
fault, `embed_worker_refused_late`, and kills the worker (`worker.rs:294-310`, `worker.rs:373-376`).

### 1.11 Correlation and ordering

- Per-request id or batch id: ABSENT. Embed has only `kind` and `items` (`embedding.rs:189-195`). Vectors has only
  `items` (`embedding.rs:204-209`).
- Per-item id: present. The caller chooses it (`embedding.rs:197-202`). haem uses each text's position in the whole
  request as a decimal string (`haematite@6f04935:crates/haem/src/serve/embed/worker.rs:57-79`).
- Ordering: the answer must have one outcome per item, in the same order, with the same ids. A count mismatch or
  an id out of order is `embed_protocol` (`checks.rs:361-374`).
- One answer per request: the worker writes exactly one Vectors or one Failed for each Embed, and handles them one
  at a time in a loop (`session.rs:47-71`). haem sends one Embed and reads one answer
  (`worker.rs:250-262`, `worker.rs:382-413`).
- Pipelining (several Embeds in flight): ABSENT. Inferred: with no request id, the only correlation is "the next
  answer belongs to the last request", which works only on one ordered stream.

### 1.12 Model identity

- Start names the model only by `model_dir`, a local path (`embedding.rs:128`). Start carries no model name or
  digest: ABSENT.
- The worker reads the installed manifest, maps it to a registry id, and refuses an unregistered, non-runnable or
  non-dense model as `embed_model_mismatch` (`engine.rs:44-63`). A missing artifact is `embed_model_missing`
  (`failure.rs:57-68`). It requires a manifest digest (`engine.rs:84-89`).
- The worker declares identity in Ready (`engine.rs:101-118`).
- haem checks identity after Ready: the protocol must equal the one it sent, else `embed_protocol_old` or
  `embed_worker_not_ready` (`worker.rs:502-516`); limits must fit the model (`worker.rs:517-520`); `core_limit` and
  `environment` must be 0 (`worker.rs:521-526`); `manifest_sha256` must equal the configured `embed_model_sha256`,
  else `embed_model_pinned_mismatch` (`worker.rs:527-535`).
- With several workers, every share must name the same model, else `embed_model_mixed`
  (`haematite@6f04935:crates/haem/src/serve/embed/workers.rs:86-91`). The comparison is on name, revision, digest,
  dimensions, distance and normalisation (`embedder.rs:12-21`).

## 2. Session lifecycle

- One Start per process. `read_start` is called once (`crates/tessera-worker/src/session.rs:26`). After Ready only
  Embed is accepted; a second Start is Failed `embed_protocol` and the worker exits (`session.rs:51-60`).
- One model per process. The engine is loaded once from the one Start (`session.rs:42-45`). A second model or a
  second Start in the same process: ABSENT.
- Inferred: two process-global settings also tie a process to one session. `configure_cpu_threads` keeps its first
  result ("The first call wins", `src/runtime/threading.rs:72`, `:78-85`). The heap limit is one static
  (`haematite@6f04935:crates/haem-worker/src/allocator.rs:6-16`).
- EOF before Start: clean end, exit 0 (`session.rs:28`, `crates/tessera-worker/src/main.rs:33`).
- EOF between Embeds: clean end, exit 0 (`session.rs:50`). This is how the parent stops a kept worker
  (`haematite@6f04935:crates/haem/src/serve/embed/worker.rs:30`: input is dropped first).
- EOF inside a frame: `Fault::Cut`, mapped to `embed_protocol` (`codec.rs:128-131`), Failed, exit 1
  (`session.rs:61`).
- Malformed frame (bad length, unknown tag, bad JSON, unknown member, wrong first message): Failed with the codec's
  code, exit 1 (`session.rs:29`, `session.rs:61`, `failure.rs:177-181`).
- Error mid-batch: any failure in `check_embed`, encoding or output checks sends Failed and ends the session
  (`session.rs:63-69`). No partial Vectors is sent. The session is not continued. Per-item refusals inside a Vectors
  are not errors and the session goes on.
- Exit codes (`main.rs:20-39`, `README.md:46-47`):
  - 0: clean EOF before a frame.
  - 1: process setup failed (`main.rs:22-24`), or a Failed frame was sent.
  - 2: writing a frame to standard output failed (`main.rs:35-37`).
  - 86: the Rust heap crossed `memory_bytes`. The process ends through `_exit`, with no frame
    (`allocator.rs:18-21`; constant `haematite@6f04935:crates/haem-worker/src/lib.rs:9` and `embedding.rs:23`;
    the two are asserted equal in `crates/tessera-worker/src/process.rs:1`).

## 3. Process and resource policy

### 3.1 Process setup

- Before any stream or thread, `prepare` disables core files, closes inherited descriptors above 2, and counts
  the environment (`haematite@6f04935:crates/haem-worker/src/setup.rs:11-20`, `:22-49`, `:61-82`;
  called at `crates/tessera-worker/src/main.rs:20`; described in `crates/tessera-worker/README.md:8-14`).
- haem starts the worker with an empty environment, working directory `/`, and three pipes
  (`haematite@6f04935:crates/haem/src/serve/embed/worker.rs:200-212`), and refuses a Ready that reports a core
  limit or an environment (`worker.rs:521-526`).

### 3.2 Memory

- Set by `Start.limits.memory_bytes` (`crates/tessera-worker/src/policy.rs:44`).
- Armed with `process::arm` after Start is checked (`session.rs:34-38`, `process.rs:7-9`), which stores the limit
  in the counting allocator (`haematite@6f04935:crates/haem-worker/src/allocator.rs:11-16`; later bare `allocator.rs` citations are this file). The allocator is the worker's global allocator
  (`main.rs:11-12`).
- Enforced on every allocation: crossing the limit calls `_exit(86)` (`allocator.rs:23-42`). Allocations made
  before arming count too (`allocator.rs:1`).
- Also checked up front: the model's parameter bytes plus the activation estimate must fit `memory_bytes`, else
  `embed_limits` (`policy.rs:56-78`).
- It is not a resident-memory limit. It does not cover the weights mapping during load, the tokenizer's C
  allocations, or the stack (`README.md:22-24`).
- haem notes that a separate question worker holds its own copy of the model, so the service may use twice
  `embed_memory_bytes` (`haematite@6f04935:crates/haem/src/serve/embedding.rs:383-384`).

### 3.3 Threads

- Set by `Start.limits.threads` (`policy.rs:45`) and applied with `tessera::configure_cpu_threads`
  (`session.rs:39-41`).
- It caps `RAYON_NUM_THREADS` and `CANDLE_NUM_THREADS`, and on macOS with `accelerate` forces
  `VECLIB_MAXIMUM_THREADS=1` (`src/runtime/threading.rs:60-71`, `:105-112`).

### 3.4 Batch limits

- `batch_items` per Embed (`checks.rs:201-206`). Under protocol 2 a job may hold `batch_items * max_windows`
  window inputs (`policy.rs:28-39`).
- Job input bytes `batch_items * input_bytes`, padded tokens `batch_items * tokens`, attention cells, and output
  bytes are all bounded with checked products (`policy.rs:40-78`, `README.md:26-30`).
- haem side: a request is cut into batches of `batch_items` (`embedder.rs:116-120`); at most
  `embed_request_items` texts per request (`embedder.rs:196-205`); a walk page holds 1 to 1,024 records
  (`settings.rs:29-41`, `:131-140`).

### 3.5 stderr

- A Failed message longer than 1,024 characters is first written whole to stderr, then the frame keeps whole
  characters and says how many it left out and where the full text is (`crates/tessera-worker/src/failure.rs:129-174`).
  Nothing is dropped silently (`failure.rs:134-136`).
- Setup failure and frame-write failure are written to stderr (`main.rs:23`, `main.rs:36`).
- haem reads stderr through its process owner and keeps the last 512 bytes for the operator only, never in a
  caller's answer (`worker.rs:42-43`, `worker.rs:98-108`, `worker.rs:173-179`).
- Inferred: under a queue there is no parent reading stderr, so the "full text on the worker's stderr" promise in
  `failure.rs:154` would point at a stream nobody collects.

### 3.6 Timeouts

- In the worker: ABSENT. The worker reads and blocks without a deadline (`session.rs:47-49`), and the frame layer
  leaves deadlines to the caller (`frame.rs:1`).
- In haem: a deadline from Start to Ready, `embed_ready_ms` (`worker.rs:197-199`); a deadline per call,
  `embed_call_ms`, multiplied by `max_windows` for documents with windows on (`settings.rs:102-113`,
  `worker.rs:264-272`); no deadline while idle (`haematite@6f04935:crates/haem-query/src/worker_owner.rs:77-83`).
  Past a deadline the owner kills the worker and the call is `embed_worker_timeout` (`worker.rs:415-427`).

### 3.7 How haem runs and restarts workers

- How many: the walk uses one `Embedder`, which keeps one worker behind one lock (`embedder.rs:62-70`,
  `haematite@6f04935:crates/haem/src/serve/embedding.rs:563-567`). An optional second worker answers questions only,
  so searches do not wait behind a walk batch (`embedding.rs:375-390`, `:599-612`).
- Several workers: `Workers` splits a page's texts into contiguous shares of near-equal size, one per worker, each
  on its own scoped thread, and joins the answers in text order (`workers.rs:1-14`, `:56-78`, `:105-119`). The
  answer is whole or nothing: one failed share fails the call, naming the worker, and the other shares are thrown
  away (`workers.rs:9-12`, `:79-85`). The count comes from `embed_workers`, 1 to 16 (`settings.rs:361-383`).
- Inferred: at `6f04935`, `Workers::new` and `settings::workers` have no caller outside tests (a `git grep` for
  `Workers::new|settings::workers(` in `crates/haem/src` finds only the definitions). The local `origin/main`
  (`01ce4ad`) shows the same. So pooling across several local workers is written but not yet wired into serve.
- Restart inside a call: a kept worker found dead before any byte of the batch reached it is replaced once, and the
  batch goes to the new one (`embedder.rs:94-99`, `:218-251`). haem tells "unsent" from "sent" by counting the bytes
  the pipe accepted (`worker.rs:116-124`, `:155-171`, `:280-289`). A batch that reached a worker is never sent again
  (`worker.rs:122-123`).
- Restart over time: the first start after a death is at once; later ones wait `embed_restart_ms` (default 1 s),
  doubling up to `embed_restart_most_ms` (default 5 minutes), for as long as the service runs
  (`settings.rs:281-351`; driver `embedding.rs:16-25`, `:330-345`). The question worker follows the same rule
  (`embedding.rs:440-470`).
- A failed page commits nothing; its field goes to the back of the queue with its own doubling delay, and the walk
  takes the same page next time (`haematite@6f04935:crates/haem/src/serve/embed/job.rs:9-15`; `embedding.rs:255-287`).
- Back-pressure: `embed_waiting` callers may wait for the one worker; past that a call is `embed_busy`
  (`embedder.rs:178-188`).

## 4. What would change to take jobs from a queue

### 4.1 Pieces that do not depend on the transport

- `session::run` takes any `impl Read` and `impl Write` (`crates/tessera-worker/src/session.rs:19-23`). Only `main`
  binds it to stdin and stdout (`main.rs:27-31`).
- `Engine::load` and `Engine::encode` take a Start and an Embed and return a Ready and Vectors or a `Failure`
  (`engine.rs:39-133`, `engine.rs:140-154`, `engine/windows.rs:56-117`). No I/O.
- `Budget` (`policy.rs:27-80`) and `Failure` (`failure.rs:6-175`) do no transport I/O, except that
  `Failure::into_message` writes overlong text to stderr (`failure.rs:130-132`).
- The codec and checks work over any `Read`/`Write` or byte slice: `read_message`, `write_message`,
  `decode_message` (`codec.rs:99-183`), and the pure checks `check_start`, `check_ready`, `check_embed`,
  `check_model_limits_windowed`, `check_vectors_windowed` (`checks.rs`).
- Inferred: `decode_message` takes a tag and a payload slice (`codec.rs:99`), so a message could be carried in a
  queue payload without the length prefix.

### 4.2 Pieces that assume a pipe and one parent

- Standard input and output as the only channel (`main.rs:27-31`).
- Process setup closes every inherited descriptor above 2 (`setup.rs:22-49`). Inferred: a queue client that opens
  sockets after setup is not affected, but anything inherited would be closed.
- haem requires an empty environment (`worker.rs:521-526`). Inferred: a queue worker that needs credentials or
  a queue address from its environment would fail haem's Ready check as it stands, if haem still checked it.
- One Start per process, and the process exits on the first Failed (section 2). Inferred: one bad job ends the
  worker; a queue consumer would need a supervisor to start it again, which haem is today
  (`embedding.rs:16-25`).
- Memory overrun ends the process with exit 86 and no frame (`allocator.rs:18-21`). Only a parent that reads the
  exit status can tell what happened.
- Timeouts live only in the parent (section 3.6). A worker that hangs is ended by a parent kill.
- "Unsent" against "sent" is measured by bytes the pipe accepted (`worker.rs:155-171`). That is the only
  redelivery-safety rule today, and it depends on owning the pipe.
- Failure detail beyond 1,024 characters goes to stderr, read by the parent (section 3.5).
- `model_dir` is a path on the worker's machine (`embedding.rs:128`, `checks.rs:40-42`). Inferred: a remote worker
  would need its own installed copy at its own path; haem's pin on `manifest_sha256` (`worker.rs:527-535`) is what
  would keep the model the same across machines.

### 4.3 What a queue consumer would need that is ABSENT today

- Job id or correlation id per Embed: ABSENT (section 1.11). Item ids exist and are unique only within one Embed
  (`checks.rs:207-215`).
- Acknowledgement after the answer is stored: ABSENT. There is no message after Vectors (`embedding.rs:25-33`).
- Idempotence or redelivery safety: ABSENT in the protocol. haem's rule is "never resend a batch that reached a
  worker" (`worker.rs:122-123`). Inferred: embedding is a pure function of text and model, but whether a repeat
  gives identical bytes was not determined here.
- Model selection per job: ABSENT. The model is fixed by the one Start (section 2). A job cannot name a model or
  digest.
- Limits per job: ABSENT. `batch_items`, `input_bytes`, `tokens`, `frame_bytes` and windows are fixed for the
  session by Start (`embedding.rs:113-134`).
- Payload size bound against the queue's own message limit: the frame bound is negotiated per session
  (`frame_bytes`, up to `u32::MAX`). Any bound on a queue message: ABSENT here. A worst-case protocol 2 answer can
  be far larger than its request (`checks.rs:102-130`).
- Back-pressure: today it is the synchronous pipe plus `embed_waiting` in haem (`embedder.rs:178-188`). In the
  worker: ABSENT.
- Health or heartbeat: ABSENT. There are five tags and no ping (`embedding.rs:25-33`). haem knows a worker is
  alive only by its Ready and its answers.
- Deadline or cancellation carried in a message: ABSENT.
- Identity of the worker: Ready carries a build string (`engine.rs:103`) and setup numbers. A host, instance or
  session name: ABSENT. Authentication of either side: ABSENT.
- Partial progress inside a job: ABSENT. A job is whole or nothing at both the Embed level (`session.rs:63-69`) and
  the multi-worker level (`workers.rs:9-12`).
- Where full failure text goes when there is no parent reading stderr: ABSENT (section 3.5).
- Liminal itself: nothing in the Tessera repository mentions Liminal (a search of the repository, excluding local
  state and build output, found no match). No Liminal code was read for this note.

## 5. Open questions

For Apollo (haematite):

1. Is `Workers` (box 43) meant to be wired into serve, and is a queue a replacement for it or a second kind of
   `Embed` behind the same trait (`job.rs:42-50`)?
2. Should the queue carry protocol 2 frames as they are, or a new message family with a job id? Should Embed and
   Vectors gain a request id?
3. Who keeps the "never resend a batch that reached a worker" rule (`worker.rs:122-123`) when delivery is by queue?
   Is resending a job after a lost acknowledgement acceptable if the model digest is the same?
4. How should haem pin the model for remote workers: still `embed_model_sha256` against Ready, per job, or per
   worker registration?
5. Do the deadlines (`embed_ready_ms`, `embed_call_ms`) and restart backoff stay in haem, or move to the queue?
6. Is the empty-environment and zero-core-limit Ready check (`worker.rs:521-526`) still required of a remote worker
   that haem did not start?
7. How should a remote memory overrun (exit 86, no frame) be reported back to the caller?
8. The whole-or-nothing page rule (`job.rs:9-15`, `workers.rs:9-12`): does it hold across machines, where one slow
   or lost share holds the whole page?

For Hermes (Liminal):

1. What does a Liminal work queue offer for delivery: at most once, at least once, acknowledgement, visibility
   timeout or lease, dead letters?
2. What is the largest message, and can a job's answer be larger than its request?
3. Does a consumer register an identity, and can a job be routed to consumers that hold a given model digest?
4. Does Liminal give health or heartbeat for consumers, or must the worker send its own?
5. How does back-pressure work: does a consumer pull one job at a time?
6. Where does a consumer's diagnostic text (stderr today) go?
