Run the reference generator from the repository root:

```sh
uv run certification/tools/make_reference.py \
  certification/specs/bge-base-en-v1.5.json smoke \
  'What is machine learning?' /existing/scratch/directory/reference.json
```

The output must not exist and its parent directory must exist. The generator
reads the model identity and capability from the selected specification profile.
It uses the existing official reference only for its comparison tolerances.
When that reference does not exist, all three of `--absolute-tolerance`,
`--relative-tolerance`, and `--minimum-cosine` must be supplied explicitly.
These arguments have no defaults. For example, append
`--absolute-tolerance 0.001 --relative-tolerance 0.01 --minimum-cosine 0.999`
to generate the first reference for a profile. Once a reference exists, its
tolerances are used and the arguments do not override them.
The three framework dependencies are pinned in the script's inline metadata.

Downloads use the specification's immutable revision and stay under
`.tessera/reference-cache`. Each downloaded file is printed as `fetched`; reused
files are printed as `cached`. Specification artifacts are checked for size and
SHA-256 before the local snapshot is loaded. Model loading and inference use
CPU float32 with network access disabled. Remote model code is disabled unless
both `--code-repository owner/name` and `--code-revision <40-digit-commit>` are
supplied. Every configuration code reference must name that repository. Its
Python files are fetched at the code commit before offline mode is enabled;
the code commit is passed to configuration, model, and processor loading.
The code identity is recorded in the provenance producer text.

Use `--probe-file path/to/probe.txt` instead of the positional probe to read
UTF-8 text without changing whitespace or line endings. For example:

```sh
uv run certification/tools/make_reference.py \
  certification/specs/bge-base-en-v1.5.json smoke \
  /existing/scratch/directory/reference.json --probe-file /existing/probe.txt
```

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

For Jina v2, use `make_reference_legacy.py`, which imports the same generator
and pins sentence-transformers 5.7.0, torch 2.14.0, and transformers 4.57.6.
Transformers 4.57.6 is the newest released 4.x version; 5.x removed the
`transformers.onnx` package imported by the pinned Jina configuration code
([upstream removal](https://github.com/huggingface/transformers/pull/41700)).
Sentence-transformers 5.7.0 accepts Transformers >=4.41,<6 and keeps the same
inference API as the primary entry script. Torch 2.14.0 keeps the primary
entry's CPU float32 runtime and satisfies Transformers' >=2.2 requirement.
These pins leave the primary entry's dependencies unchanged.

The Jina code repository is `jinaai/jina-bert-implementation`, revision
`f3ec4cf7de7e561007f27c9efc7148b0bd713f81`. Its configuration and model import
under the legacy stack. The generator loads the ordered built-in Transformer,
Pooling, and optional Normalize modules directly because sentence-transformers
5.7.0's module-class resolver consumes `model_kwargs.code_revision` before the
underlying model is loaded. Direct loading preserves the pin on all three
configuration, model, and processor paths, with local-files-only loading.

The long probe comes from commit `979f1fb06d31d27beb4fbfb213b6229e673d2562`,
`docs/vision_board/part5_dense.md` followed by
`docs/vision_board/part1_multi_vector.md`, joined with one newline. Markdown
and code blocks are unchanged. The generator takes the longest prefix ending
at a word boundary with at most 8000 tokens and refuses fewer than 6000 tokens.
It uses each specification's SHA-256-checked tokenizer, with truncation and
padding disabled; token counts include special tokens. First generate smoke
to fill the pinned model cache, then rebuild a probe independently for each
model (replace `small` with `base` for the other model):

```sh
uv run --python 3.13.11 certification/tools/make_long_probe.py \
  certification/specs/jina-embeddings-v2-small-en.json /tmp/jina-small-long.txt
uv run --python 3.13.11 certification/tools/make_reference_legacy.py \
  certification/specs/jina-embeddings-v2-small-en.json long-context-8k \
  /existing/scratch/directory/long-reference.json \
  --probe-file /tmp/jina-small-long.txt \
  --code-repository jinaai/jina-bert-implementation \
  --code-revision f3ec4cf7de7e561007f27c9efc7148b0bd713f81 \
  --absolute-tolerance 0.001 --relative-tolerance 0.01 --minimum-cosine 0.999
```

The output files in both commands must not already exist. The three new smoke
profiles reuse their model's smoke limits; all five profiles are required for
promotion. Certification results do not change the registry's support tier.
