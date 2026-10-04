# Tessera embedding worker

This crate loads one installed dense model on the CPU and serves the embedding
messages from `haem-frames` over standard input and standard output. It uses
Tessera's strict installed-manifest reader, verified artifacts, loaded identity
and role-aware bounded encoding. It has no model-fetch feature dependency.

The parent supplies an empty initial environment. Process setup runs before
streams, threads or model construction: it disables core files, closes inherited
descriptors above two and records the initial environment count. After Start is
read and checked, the Rust heap allowance is armed, CPU threads are configured,
the model is built, and Ready reports the process setup's original numbers.
Thread configuration creates the two CPU thread environment variables; Ready's
environment count describes the initial environment, before that configuration.

`memory_bytes` bounds requested live Rust heap bytes through the shared counting
allocator. It also bounds the sum of the registry model-parameter byte estimate
and the existing activation estimate at the requested token and batch limits.
An estimate that cannot fit is `embed_limits`. An allocation crossing the armed
Rust heap allowance ends the process with `MEMORY_LIMIT_EXIT`, without a frame.

The heap allowance does not cover the transient weights mapping during loading,
the tokenizer's C allocations, or the stack. It is not a resident-memory or
whole-process-memory limit.

Job input bytes are `batch_items * input_bytes`; retained output bytes are
`batch_items * dimensions * 4`; padded tokens are `batch_items * tokens`.
All conversions and products are checked. The shared check requires enough
tokens for specials, the longest role prefix and one content token, and enough
frame bytes for the largest valid vector batch.

Unicode whitespace inputs and inputs above the UTF-8 byte limit receive the
shared item refusal. Other overlength inputs are cut at the token limit with
the role prefix preserved, and both counts are returned. Each output batch is
checked with the shared validator before it is written. All failures end the
worker; no partial batch is sent. Clean EOF before a frame exits zero. Failure
frames exit one; a failed frame write exits two.

The worker is its own workspace, excluded from the library's, and is not
published: its two shared crates come from the private haematite repository,
and the library and `xtask` must resolve and build for someone without that
access. Those crates come from haematite main without a manifest revision;
`crates/tessera-worker/Cargo.lock` records the resolved revision. Every cargo
command for the worker names its manifest, for example
`cargo build --release --locked --manifest-path crates/tessera-worker/Cargo.toml`,
and builds into `crates/tessera-worker/target` unless `CARGO_TARGET_DIR` says
otherwise. Its profiles are kept equal to the library's. Resolve/fetch after
the shared API is on main, commit the updated lock, then use
`cargo fetch --locked` on a gate checkout before `--locked --offline` checks.
Do not fetch model files for builds or tests. Tests generate a small installed
BERT fixture locally and wait on child exit, with no clock-based waits.

For the fetch command only, use `CARGO_NET_GIT_FETCH_WITH_CLI=true cargo fetch --manifest-path crates/tessera-worker/Cargo.toml`; after committing the lock, repeat with `--locked`.
