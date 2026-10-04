# Local model certification

Tessera's certification lane is deliberately separate from `cargo test`. It is
local-only, opt-in, networked only during an explicit fetch, and designed to
release all model memory between runs.

> **Full repository checkout required:** the commands below use the unpublished
> `tessera-xtask` workspace runner. That repository tooling is not included in
> Tessera's crates.io or Python source-distribution archives.

The checked specifications in `certification/specs/` pin every experimental
model to an immutable Hugging Face revision and declare every artifact's exact
byte count and SHA-256 digest. Schema version 2 also scopes every profile by
device, dtype, semantic mode, maximum admitted sequence length, and registry
context window. Per-sequence and whole-job input, collected output, estimated
activation, model, artifact, disk, timeout, thread, attention, and peak-RSS
limits remain independently enforced. Each profile's model-plus-activation
budget must fit below its parent-process RSS watchdog.

## Commands

Build the optional runner and list its checked specifications without loading a
model:

```bash
cargo run --locked --offline -p tessera-xtask --features certification -- \
  cert list
```

Fetching is the only command allowed to use the network. It downloads one
pinned model into `.tessera/cert-cache/`, first reserves enough space for both
the expected download and the specification's retained free-space allowance,
then streams every artifact through size and SHA-256 verification:

```bash
cargo run --locked -p tessera-xtask --features certification -- \
  cert fetch --model bge-base-en-v1.5
```

Run one CPU smoke in a fresh process. The parent sets `HF_HOME` to the dedicated
cache and `TESSERA_OFFLINE=1`, so a missing pinned artifact fails instead of
falling back to the network. `--repeat 2` means two serial child processes, not
two models retained in one process:

```bash
cargo run --locked --offline --release -p tessera-xtask \
  --features certification -- cert run \
  --model bge-base-en-v1.5 --device cpu --profile smoke --repeat 2
```

Make an installed dense model folder from the already verified certification
cache. The destination must be absent or empty, and its parent must exist.
This command copies each specified artifact, verifies the copied bytes, writes
a schema-1 manifest, and prints the manifest's SHA-256. It performs no fetch:

```bash
cargo run --locked --offline -p tessera-xtask --features certification -- \
  cert install --model bge-base-en-v1.5 --dir /tmp/tessera-bge-base
```

Run the installed folder without using a Hugging Face cache. The parent removes
`HF_HOME` and `TESSERA_OFFLINE` from the child's environment. The dense loader
validates the manifest and every listed artifact, and certification separately
checks the specification's size and hash for each artifact before inference.
Successful evidence includes `installed_manifest_sha256` from that embedder;
older records without this optional member remain readable. Other model
representations refuse `--model-dir`:

```bash
cargo run --locked --offline -p tessera-xtask --features certification -- \
  cert run --model bge-base-en-v1.5 --model-dir /tmp/tessera-bge-base \
  --device cpu --profile smoke --repeat 2
```

Run every specification serially with the same one-model-per-process boundary:

```bash
cargo run --locked --offline --release -p tessera-xtask \
  --features certification -- cert run-all \
  --device cpu --profile smoke --repeat 2
```

Dense, sparse, multi-vector, and vision outputs have distinct reference
comparators. Vision execution additionally requires a content-hashed image
fixture. `colpali-v1.2` remains an integrity-only specification until such an
official reference is checked in, so `run-all` must not be treated as a green
vision certification.

The 8K-capable dense models have a separate `long-context-8k` profile. A long
profile never falls back to the short smoke fixture: it refuses to execute
until its checked reference probe exercises at least 87.5% of the profile's
token limit. Its timeout and RSS watchdog are still applied by the parent.

Inspect promotion readiness or remove one model's re-downloadable cache:

```bash
cargo run --locked --offline -p tessera-xtask --features certification -- \
  cert readiness --model bge-base-en-v1.5

cargo run --locked --offline -p tessera-xtask --features certification -- \
  cert purge --model bge-base-en-v1.5
```

## Evidence and safety boundary

Each child constructs exactly one model instance on CPU. The parent enforces
the specification timeout and, on Unix systems with `ps`, samples process RSS
every 50 ms and kills the child when the declared ceiling is exceeded. A
platform without live RSS samples may still produce diagnostic evidence, but
`readiness` refuses promotion when enforceable RSS evidence is required.
The recorded value is the largest sample observed, not an operating-system
lifetime high-water mark.

Working evidence is compact JSON under `.tessera/cert-evidence/<model>/`. It
contains artifact hashes, shapes, finite/norm checks, batch-versus-sequential
parity, repeatability and retrieval scores, the exact capability scope,
official-reference comparison metrics and output fingerprints, source state,
resource limits, duration, and sampled peak RSS. It never contains model
weights or full embedding vectors. Both cache and working evidence are ignored
by Git.

Readiness remains conservative. It requires matching successful runs from the
current clean `HEAD`, the current specification digest, verified artifacts,
enforced RSS evidence, and a passed official-reference comparison for every
required profile. The initial model specifications intentionally leave their
references unset, so a structural smoke cannot accidentally promote a model to
`Supported`.

## Official-reference contract

`official_reference` is configured inside a profile, not in the global
promotion block. It contains a path relative to `certification/references/` and
the SHA-256 of that exact JSON file. A syntactically valid but invented 64-digit
hash fails while loading the specification because the runner reads and hashes
the referenced bytes.

The reference document pins all of the following:

- model ID, upstream repository, immutable revision, profile, and capability;
- the official producer, framework, framework version, and upstream source;
- a canonical text probe and upstream token count, or a content-hashed image
  fixture plus query (the child verifies the declared count against the pinned
  tokenizer, including its normal single-sequence special tokens);
- a typed dense vector, sorted sparse coordinates and values, multi-vector
  matrix, or vision patch matrix; and
- absolute and relative numeric tolerances plus a minimum cosine threshold.

Dense comparison checks every coordinate and whole-vector cosine. Sparse
comparison additionally requires identical sorted vocabulary coordinates.
Multi-vector and vision comparison require identical shapes and check every
coordinate plus the minimum cosine over all token or patch rows. Failed and
not-configured comparisons are different evidence states; readiness accepts
only `passed`. Reference JSON and image fixtures are individually capped at
16 MiB before reading, and checked tolerances cannot exceed 0.001 absolute or
0.01 relative error or fall below 0.999 cosine. Readiness recomputes the
expected-output fingerprint and rejects incomplete metrics, shapes, probe
counts, or malformed observed fingerprints even if an evidence file claims a
passed status.

Small model-free examples for all four representations live in
`certification/references/contract/`. They test the contract and tamper gate;
they are not model certifications. A real reference must be generated with the
upstream model's documented inference implementation, reviewed, placed under
`certification/references/`, hashed, and connected to exactly one capability
profile. Fetching Tessera's pinned weights remains a separate explicit command,
and certification children remain offline.

## Full-window measurements owed

`gte-modernbert-base` has an unmeasured 2,048-token certification window. Its
required long profile will use a 3,000-token source cut to 2,048 tokens, recording
both counts through `cut_at_tokens`. No reference or run is recorded yet. The
upstream 8,192-token window still needs two isolated full-window runs; the
current scratch estimator requires 9,890,168,832 activation bytes at 8,192
tokens (one item, f32). An 8k profile must be added when the registry window is
raised; it cannot coexist with the current 2,048-token admission limit.

`snowflake-arctic-l` likewise has no reference or run for its required
2,048-token profile. Its future 3,000-token source must record the original
count and the 2,048-token cut. Two isolated full-window runs remain owed for
the upstream 8,192-token window; the current scratch estimator requires
13,186,891,776 activation bytes there (one item, f32). The 8k profile must
be added together with a registry window increase after measurement.

## Jina v2 window qualification

Both Jina v2 entries currently admit 2,048 tokens. Their required long profile
is `long-context-2k`, with a 3,000-token source cut at 2,048 including special
tokens. Neither 2k reference nor run is recorded yet. The eight short references were produced before the registry window was
lowered. Only their declared `capability.context_window_tokens` changed from
8,192 to 2,048; texts, vectors, tolerances, provenance and the 128-token limits
remain byte for byte. Their specifications bind the new file hashes. All
profiles must be rerun at the new clean head to check that the vectors still
hold; both Jina models remain uncertified at that head until those runs pass.

The two checked probe files were built with each specification's
SHA-256-verified tokenizer. Rebuild them with the following commands. The helper reads the same two repository
files at its fixed source commit, preserving source bytes and word boundaries:

```bash
uv run --offline certification/tools/make_long_probe.py \
  certification/specs/jina-embeddings-v2-small-en.json \
  certification/probes/jina-embeddings-v2-small-en-3k.txt \
  --minimum-tokens 3000 --maximum-tokens 3000
uv run --offline certification/tools/make_long_probe.py \
  certification/specs/jina-embeddings-v2-base-en.json \
  certification/probes/jina-embeddings-v2-base-en-3k.txt \
  --minimum-tokens 3000 --maximum-tokens 3000
```

After the checked tokenizer files are present, both reference entry points
accept `--cut-at-tokens 2048`. Use the legacy entry for these Jina checkpoints,
together with the existing pinned code repository/revision and explicit
0.001 absolute, 0.01 relative and 0.999 cosine tolerances. `token_count` records
the whole source before truncation; `cut_at_tokens` records the inference cut.
Inputs at or below that cut are refused. Python reference RSS must remain
below 4,000,000,000 bytes; generation and certification await the compile turn.

The 2k profiles use activation ceilings rounded upward by 100 MB from the
current f32 scratch estimator: small 500,000,000 bytes (estimate
440,401,920), base 700,000,000 bytes (estimate 660,602,880). Model ceilings remain 200,000,000 and
700,000,000 bytes; artifact ceilings 100,000,000 and 350,000,000 bytes. RSS
ceilings are 1,500,000,000 and 2,200,000,000 bytes respectively. One item,
2,048 batch tokens and 4,194,304 attention cells are admitted. These ceilings
are limits, not measurements. The hashed pinned configs give hidden/intermediate
sizes 512/2048 with 8 heads for small, and 768/3072 with 12 heads for base.

Two isolated 8,000-token runs remain owed for each model before the upstream
8,192-token window can be offered. The current f32 activation estimator gives
6,593,445,888 bytes for small and 9,890,168,832 bytes for base at an
8,192-token limit, one item. No 8k profile is defined while the registry admits
2,048. The small model's old 8,000-token reference is retained unbound at
`references/jina-embeddings-v2-small-en/long-context-8k.json`, originally
committed in `b0320d59df676df122ebb54c9ca38010e56d8c1a`; it is not evidence
for the new cut profile. The base model has no 8k reference.

## Nomic native-vector qualification owed

`nomic-embed-v1.5` is entered for native 768-dimensional vectors and a
2,048-token window. No reference or run is recorded. Its four short profiles
and one-item `long-context-2k` need official references and two clean-head runs
apiece. The long source must exceed the limit: use a 3,000-token probe with
`cut_at_tokens: 2048` and record both counts.

The pinned card requires `search_query: ` for queries and `search_document: `
for documents. Its sentence-transformers modules contain mean pooling and no
Normalize module; its inference example applies L2 explicitly. The card's
Matryoshka example applies pooled layer normalization before truncation and
L2. Tessera does not apply that pooled layer norm, so the registry currently
declares only native 768 dimensions. Lower upstream dimensions are unqualified.
The card does not explicitly prescribe cosine or dot comparison. The agreed
normalized-vector policy selects cosine for the service; that declaration is
to be added when the identity metadata reaches this branch.

Config `max_position_embeddings` is 2,048, while `n_positions` is 8,192. The
card's extension recipe changes rotary scaling; the current pinned recipe
and the current loader have not been measured beyond 2,048. Full-window
qualification remains owed before an 8,192-token window can be offered. The
current one-item f32 scratch estimator requires 9,890,168,832 bytes at 8,192,
and 660,602,880 at 2,048. The 2k activation cap is 700,000,000 bytes. Short
model/artifact/RSS ceilings are 700,000,000 / 700,000,000 / 1,700,000,000
bytes; long RSS is 2,400,000,000. No 8k profile is defined at this admission
window, and these limits are estimates, not measured evidence.
