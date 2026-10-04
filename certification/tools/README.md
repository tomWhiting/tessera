Run the reference generator from the repository root:

```sh
uv run certification/tools/make_reference.py \
  certification/specs/bge-base-en-v1.5.json smoke \
  'What is machine learning?' /existing/scratch/directory/reference.json
```

The output must not exist and its parent directory must exist. The generator
reads the model identity and capability from the selected specification profile.
It uses the existing official reference only for its comparison tolerances.
The three framework dependencies are pinned in the script's inline metadata.

Downloads use the specification's immutable revision and stay under
`.tessera/reference-cache`. Each downloaded file is printed as `fetched`; reused
files are printed as `cached`. Specification artifacts are checked for size and
SHA-256 before the local snapshot is loaded. Model loading and inference use
CPU float32 with network access disabled. Remote model code is disabled.

No instruction is prepended to the probe. An explicit empty prompt suppresses
model-configured prompts, and `provenance.probe_prefix: null` records that no text
was prepended. Token counts include the tokenizer's special tokens. Probes over
the profile's token limit are refused rather than truncated.

Deterministic inference and serialization omit times and output paths from the
reference. Exclusive publication never replaces an existing file or symlink.

Boundary and publication checks run without model downloads:

```sh
uv run --no-project python -m unittest discover -s certification/tools -v
```
