# Speed baseline — 6 October 2026

Incomplete observation matrix: 286/322 observations. This is a baseline measurement; library and worker implementation behaviour is unchanged.

Harness commit `b65bd0b37f9380d1938e7051ce7c19c152a2cfe1`, tree `2bfddf11f8b549ee25144395176d5f2488a0eed1`, based on `1d600f426d9439e7440116bb11ce69249451e139`. Local branch `lane/speed-baseline-20261006`; no push or landing claim. The report is a later documentation-only commit.

Raw records and full command logs: `.tessera/speed-evidence/B1-b65bd0b37f9380d1938e7051ce7c19c152a2cfe1/` in the main checkout. `results.json` supplies every table value; `command-verdicts.json` records every command argv, exit and unrun leg; inventories give each file size and SHA-256.

## Fixtures and timing boundary

Model: CPU F32 bge-base-en-v1.5, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. Source and copied config, tokenizer and weights matched every pinned size and SHA-256 (438,667,685 bytes). No model was fetched. Frozen token assertions were validated before timing.

Fixture manifest SHA-256: `2cd2cc0cc72f5dd4e9872f5f0285f849317972367368d8911a952f30524c36f2`.

| Fixture | Source content tokens | Complete tokens | UTF-8 bytes |
|---|---:|---:|---:|
| t(30) | 30 | 32 | 119 |
| t(126) | 126 | 128 | 503 |
| t(510) | 510 | 512 | 2,039 |
| t(4096) | 4,096 | 4,098 | 16,383 |
| where is the red book? | 6 | 8; Query prefix gives 16 | 22 |

Text t(n) is n copies of `red`, joined by one ASCII space. Mixed jobs repeat `[t(30), t(30), t(30), t(126), t(510)]` 400 times. Short jobs have 2,000 items; full jobs have 500. Each J1 variant requires three complete jobs in one fresh process; each thread variant starts another process. Four untimed four-item J1 calls precede measurement.

J2 uses one untimed query then 100 sequential observations. J3 uses one untimed call then 20 observations per API. J4 has no warmup: 20 fresh library processes and 20 fresh workers. The operating-system file cache is uncontrolled. No request contention is deliberately added; shared-host load is recorded without waiting for an idle machine.

Library clocks include each public call, count/dimension/finiteness checks and output SHA-256 consumption. J4 library clocks start at the model constructor and end at its first checked query vector; fixture validation, thread configuration and process metadata precede that clock. Worker clocks include parent framing through shared validation, vector decoding and hashing; J4 additionally includes worker spawn and Ready. JSONL writing and RSS sampling occur outside those clocks.

Batch size is 4 for J1/J3 and 1 for J2/J4. The library policy has sequence length 512, maximum batch items 4, batch token rows 2,048, memory 2 GiB, attention cells 1,048,576, per-sequence input 65,536 bytes, job items 1,024, job input/output 64 MiB each and activation 1 GiB. Jobs call each public batch API in four-item chunks. Worker limits are memory 2 GiB, threads 2, batch items 1, input 65,536 bytes, tokens 512 and frame 131,072 bytes.

The measured runs report two threads in both Rayon and Candle. The planned four/eight-thread variants remain unrun; VECLIB_MAXIMUM_THREADS=1 is configured for Accelerate. The internal thread count in Accelerate is not independently measured. Exact build features, binary hashes, limits, source identity and host metadata are in the raw records and binary readbacks. The root inference binary and worker link Accelerate.framework; the framing-only parent probe links libSystem.

## Observed timings

p50 is the median; p95 uses the nearest rank. J1 has three samples, so its p95 is its maximum. Values include the output sink. Incomplete rows retain their actual observation counts.

| Journey | Route | Dataset | Threads | Observed/required | p50 seconds | p95 seconds | Items/second at p50 |
|---|---|---|---:|---:|---:|---:|---:|
| J1 | batch | mixed | 2 | 3/3; complete | 1594.409625 | 2929.350267 | 1.254 |
| J1 | batch | short | 2 | 3/3; complete | 106.529617 | 209.980575 | 18.774 |
| J1 | batch | full | 2 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | outcomes | mixed | 2 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | outcomes | short | 2 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | outcomes | full | 2 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | batch | short | 4 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | batch | full | 4 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | outcomes | short | 4 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | outcomes | full | 4 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | batch | short | 8 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | batch | full | 8 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | outcomes | short | 8 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J1 | outcomes | full | 8 | 0/3; unmeasured | unchecked | unchecked | unchecked |
| J2 | outcomes | query | 2 | 100/100; complete | 0.040925 | 0.070930 | 24.435 |
| J2 | worker | query | 2 | 100/100; complete | 0.056923 | 0.177410 | 17.567 |
| J3 | aggregate | long | 2 | 20/20; complete | 7.535500 | 10.041056 | 0.133 |
| J3 | windows | long | 2 | 20/20; complete | 4.357899 | 5.562374 | 0.229 |
| J4 | outcomes | query | 2 | 20/20; complete | 2.373645 | 4.055522 | 0.421 |
| J4 | worker | query | 2 | 20/20; complete | 2.381744 | 2.867681 | 0.420 |

A completed observation is one whole checked public-call job. A variant is complete only after its required count and a successful process exit. The recorder opens its JSONL before model construction and warmup, so an empty file does not locate the process within those phases; process elapsed time is not a job timing.

Measurement stop: `j1-batch-full-t2`, exit `-15`, reason `deadline`. Incomplete work has no timing row.

## Derived work and memory

Forward calls, physical rows P, admission rows and squared lengths are derived from the fixed route and chunk/window plan, rather than instrumented counters. R includes special/prompt tokens per forward. Source content is counted separately. Windows have nine complete lengths of 512 and one of 84: R/P 4,692, squared lengths 2,366,352, ten forwards. Aggregate returns one vector; the windows API returns ten.

| Journey/route/dataset/threads | Items | Forwards | R | P | Real squared lengths | Physical squared lengths | Raw vector bytes | Maximum sampled RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J1/batch/mixed/2 | 2000 | 500 | 294400 | 870400 | 112640000 | 425984000 | 6144000 | 712.17 |
| J1/batch/short/2 | 2000 | 500 | 64000 | 64000 | 2048000 | 2048000 | 6144000 | 359.06 |
| J2/outcomes/query/2 | 1 | 1 | 16 | 16 | 256 | 256 | 3072 | 447.08 |
| J2/worker/query/2 | 1 | 1 | 16 | 16 | 256 | 256 | 3072 | 444.33 |
| J3/aggregate/long/2 | 1 | 10 | 4692 | 4692 | 2366352 | 2366352 | 3072 | 453.55 |
| J3/windows/long/2 | 1 | 10 | 4692 | 4692 | 2366352 | 2366352 | 30720 | 554.28 |
| J4/outcomes/query/2 | 1 | 1 | 16 | 16 | 256 | 256 | 3072 | 451.48 |
| J4/worker/query/2 | 1 | 1 | 16 | 16 | 256 | 256 | 3072 | 444.84 |

RSS in this table is sampled after observations. Each library leg also has a wait4 process peak in its command JSON; that includes model construction. Worker rows sample the worker child; a worker peak was not measured independently. Every leg records load average at its start and df before/after; each observation carries its leg-start load. In J4 worker mode, one parent leg contains twenty fresh workers.

Every consumed vector passed dimension/finiteness checks. Item counts and window spans/counts are checked where the public result exposes them. Output hashes record repeatability; this harness does not compare vectors against reference values or establish cross-route numerical tolerance.

## Gate evidence

All nine root commands and all four worker states ran on the harness commit. Full stdout/stderr and exact argv/exit codes are retained, including the initial rejected attempts. The required successful matrix is below.

| Leg | Exit | Full summary / FAIL / LEAK | Largest timed individual test |
|---|---:|---|---:|
| root-clippy-default | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| root-fmt | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| root-clippy-no-default | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| root-clippy-certification | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| root-nextest-default | 0 | Summary [   2.080s] 423 tests run: 423 passed, 0 skipped; FAIL 0; LEAK 0 | 0.121s |
| root-nextest-no-default | 0 | Summary [   2.373s] 421 tests run: 421 passed, 0 skipped; FAIL 0; LEAK 0 | 0.177s |
| root-nextest-certification-resumed | 0 | Summary [   2.535s] 491 tests run: 491 passed, 0 skipped; FAIL 0; LEAK 0 | 0.158s |
| root-cert-tool-resumed | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| root-doctests-resumed | 0 | test result: ok. 30 passed; 0 failed; 50 ignored; 0 measured; 0 filtered out; finished in 7.63s; FAIL 0; LEAK 0 | unreported |
| worker-plain-dev-clippy | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-plain-dev-binary-list | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-plain-dev-first-launch | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-plain-dev-nextest | 0 | Summary [   1.418s] 31 tests run: 31 passed, 0 skipped; FAIL 0; LEAK 0 | 0.891s |
| worker-plain-release-clippy | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-plain-release-binary-list | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-plain-release-first-launch | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-plain-release-nextest | 0 | Summary [   1.406s] 31 tests run: 31 passed, 0 skipped; FAIL 0; LEAK 0 | 0.759s |
| worker-accelerate-dev-clippy | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-accelerate-dev-binary-list | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-accelerate-dev-first-launch | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-accelerate-dev-nextest | 0 | Summary [   1.245s] 31 tests run: 31 passed, 0 skipped; FAIL 0; LEAK 0 | 0.833s |
| worker-accelerate-release-clippy | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-accelerate-release-binary-list | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-accelerate-release-first-launch | 0 | no tests in this command; FAIL 0; LEAK 0 | unreported |
| worker-accelerate-release-nextest | 0 | Summary [   1.057s] 31 tests run: 31 passed, 0 skipped; FAIL 0; LEAK 0 | 0.717s |

Default/no-default counts remain 423/421. Certification is 483 plus exactly these eight new tests:

- `absent_pinned_artifact_list_is_refused`
- `mixed_tensor_batch_cost_includes_padding`
- `installed_asset_digest_cannot_drift_from_the_pin`
- `final_tensor_chunk_counts_only_present_items`
- `fixture_hash_tampering_is_refused`
- `token_assertion_mismatch_is_refused`
- `outcome_cost_keeps_physical_rows_unpadded`
- `unknown_fixture_manifest_members_are_refused`

No timed nextest individual exceeded 2 seconds. Doc-test per-test times were not reported; its entire suite took 7.63 seconds and included 50 existing ignored cases. All-features CUDA checks are not claimed. The runtime-feature Clippy and both runtime build commands also exited 0 at this commit.

## Stops, cleanup and verification boundary

The initial offline Clippy stopped on missing aligned 0.4.3. Apollo agreement quoted by Daisy permitted one locked root crate fetch: exit 0, 200 downloaded crate/version entries. Both Cargo.lock hashes remained unchanged; no worker fetch ran. An initial gate start was refused at 29.63 GiB under the original 30 GiB floor. Apollo agreement quoted in post 27cbbdd7 changed the start floor to 25 GiB; the live stop stayed 20 GiB and the deadline stayed 20:45. Both records are retained.

Daisy withdrew the roundwise reorder in post 24c5012b. The owned first-record watcher PID 60824 was terminated and reaped with exit 143, without touching the original mixed variant PID 36175. Candidate 34cdb7f54df96c0fea2b89045eff82d29f6a7d4e was withdrawn and the branch restored to the qualified b65bd0b harness. Its selector red run remains evidence only: exit 100, one test run, zero passed, one failed and 87 skipped. No selector implementation build or green run is claimed.

Before package cleanup after each compile verdict, full evidence was copied to main and its inventory posted. The exact cleanup argv, removed counts and df readbacks are retained. Copied model files and the owned reaper hold remain until final evidence is listed; their final purge/readback is a later cleanup record. Dependency compilation stayed in the single held speed-baseline/target directory. No other worker process, installed model directory or installed binary directory was touched.

## Changed harness hot paths

| Path | Locks and whole-state clones added | Syncs | History loop | Model load / network per timed call |
|---|---|---|---|---|
| Library J1 chunks | No harness lock or whole-state clone; public API runtime is unchanged | 0 per observation; one sync on successful JSONL completion | None; loops only the fixed job | 0 / 0 |
| Library J2 queries | No harness lock or whole-state clone | 0 per observation; one sync on successful file completion | None | 0 / 0 |
| Library J3 windows/aggregate | No harness lock or whole-state clone; consumes each returned vector | 0 per observation; one sync on successful file completion | None; fixed window plan | 0 / 0 |
| Library J4 constructor/query | No harness lock or whole-state clone | One sync after its observation | None | 1 / 0 |
| Worker J2 parent probe | No harness lock or whole-state clone; builds two small query messages and decodes output | 0 per observation; one sync on successful file completion | None | 0 / 0 |
| Worker J4 parent probe | No harness lock or whole-state clone; owns and reaps each child PID | 0 per observation; one sync on successful file completion | None; twenty observations | 1 / 0 |

Interrupted files retain flushed observations without a successful finish sync. The wrappers add file flushes per observation and process/RSS metadata reads outside the clocks. Public calls take the existing CPU inference permit. Its FIFO queue mutex covers admission and release, not the forward; single/outcome/window permits cover CPU materialization and pooling, while batch permits release before per-item pooling. No whole model/state is cloned. Existing pooling still clones the flattened hidden-output matrix. Candle internal kernel/storage locks remain unchecked. These library details were mapped in accepted S0 commit 33b8d4810e1bf17ce1d0eb17f88f459e2601301c and are unchanged here. Source: xtask/src/speed/measure.rs, record.rs, fixtures.rs and crates/tessera-worker/examples/speed_probe.rs at the harness commit.

No optimisation, model certification result, installation, remote verification or landing is claimed. Daisy reviews this local branch.
