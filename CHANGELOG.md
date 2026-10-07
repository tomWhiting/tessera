# Changelog

## 0.3.0 (8 October 2026)

### Breaking

- `CpuThreadConfigError::InvalidEnvironmentOverride` now has discriminant 2 (it was 1). Code that
  casts this enum to a number must be updated (`cargo semver-checks`, against 0.2.0).

### Added since 0.2.0

- Long-text windows: texts over a model's token limit are split into overlapping windows, never cut
  short. Each window carries its span and token count.
- Role-aware cut APIs, with the certification checks kept against the reference documents.
- Registry and loading:
  - ModernBERT weights in both bare and prefixed layouts;
  - an e5-small entry;
  - retrieval metadata bound to new dense entries.
- Thread safety: proofs, device re-exports, and documentation for Metal, CUDA and threading.
- `tessera-worker`, a separate workspace in `crates/tessera-worker`: the installed embedding worker for
  haematite's embedding protocol, now at protocol 2 with windows. It is not published to crates.io yet,
  because it takes haematite's frame crates from git.
- A speed timing harness (`xtask speed`) and a baseline report.

## 0.2.0 (8 August 2026)

Published 8 August 2026. No changelog was kept before 0.3.0; see the git history.
