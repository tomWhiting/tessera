# B1 speed evidence listing — 6 October 2026

Qualified harness: `b65bd0b37f9380d1938e7051ce7c19c152a2cfe1`, tree `2bfddf11f8b549ee25144395176d5f2488a0eed1`. The baseline report and this listing are documentation added after that harness gate. No push or landing claim.

Archive root: `/Users/tom/Developer/ablative/libs/tessera/.tessera/speed-evidence/B1-b65bd0b37f9380d1938e7051ce7c19c152a2cfe1/`. Each inventory enumerates every copied relative path, size and SHA-256. The final snapshot was read back and verified before cleanup.

| Inventory | Files | Bytes | Inventory SHA-256 |
|---|---:|---:|---|
| `inventory-final-before-cleanup.json` | 297 | 1146014 | `7a985c1269cde2aa4d331a8d3f45273b484218c06090c6c0aac7b48930817cfc` |
| `inventory-gate-before-cleanup.json` | 117 | 407846 | `d5f78301a56402541f8dafcda2898d5a69e524c963fc14dc9ce446464d6962d8` |
| `inventory-runtime-before-cleanup.json` | 139 | 420594 | `1d1afb11b9697939955afe5d2d077ca6a935bb00e303e0db930ce3e803f31ac3` |
| `inventory-selector-red-before-cleanup.json` | 262 | 927237 | `c4c323dfded20b28d965fda90b256855fec1496deb1a097c84b525241e43726c` |

`final-before-cleanup/speed-state/command-verdicts.json` contains 88 command records, including exact argv, exits, refused starts and unrun measurement legs. `results.json` holds 286/322 observations. Full stdout/stderr, all 286 JSONL observations, the empty interrupted full-batch JSONL, eight new test names, pins, lock hashes, binary readbacks and exact harness source are retained. The withdrawn selector patch and its red-only result are in `selector-red-before-cleanup`.

The interrupted full-batch process was reaped at the deadline with exit -15, driver exit 1. The other eleven J1 variant processes were not started. Mixed and short batch/thread-2 variants each finished three jobs with exit 0.

Copied models awaiting the listed cleanup: config.json (777 bytes), tokenizer.json (711396 bytes), model.safetensors (437955512 bytes), and the owned manifest.json. The three pinned assets total 438667685 bytes. Their exact digests are in model-readback.json; the manifest digest is `3d7e18a9b8ec2cb72c6268a322a26c9e4a9610e47c14db793c63a8d7d60092e0`. No source or installed model is part of the purge.

Owned executable copies awaiting cleanup: tessera-xtask-certification, tessera-xtask-accelerate, tessera-worker-accelerate and speed_probe-accelerate. Exact sizes and SHA-256 are retained in the binary readbacks.

The single target directory is `.worktrees/speed-baseline/target`. Final package/default/release cleanup, target cleanup, copied-model and executable removal, the exact owned reaper-hold removal and free-disk readback are recorded in a later `after-cleanup/` snapshot. The final documentation commit/tree and report digest are read back there.
