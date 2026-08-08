# Local model certification

Tessera's certification lane is deliberately separate from `cargo test`. It is
local-only, opt-in, networked only during an explicit fetch, and designed to
release all model memory between runs.

The checked specifications in `certification/specs/` pin every experimental
model to an immutable Hugging Face revision and declare every artifact's exact
byte count and SHA-256 digest. They also set model, artifact, disk, timeout,
thread, input-shape, attention, and peak-RSS limits.

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

The first tranche implements dense, sparse, and multi-vector smoke execution.
`colpali-v1.2` is intentionally present as a fully pinned integrity
specification, but its run records a clear failure until a checked image fixture
and a safe vision dtype/device profile exist. Consequently `run-all` currently
finishes nonzero after processing the models serially; it must not be treated as
a green vision certification.

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
parity, repeatability and retrieval scores, source state, resource limits,
duration, and sampled peak RSS. It never contains model weights or full
embedding vectors. Both cache and working evidence are ignored by Git.

Readiness remains conservative. It requires matching successful runs from the
current clean `HEAD`, the current specification digest, verified artifacts,
enforced RSS evidence, and a checked official-reference fingerprint. The
initial specifications intentionally leave that final fingerprint unset, so a
basic smoke cannot accidentally promote a model to `Supported`.
