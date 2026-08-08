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
