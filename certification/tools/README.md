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

## Sparse and multi-vector references

Both comparisons already exist. `xtask/src/certification/reference.rs:24`
requires schema version, model id/repository/revision, profile, exact capability,
provenance, probe, tolerances and expected output. Provenance records the same
model repository and revision as the specification (`reference.rs:292`); the
additional encoder-code pin belongs in the producer/framework-version strings.
A text probe contains its unchanged text and the ordinary tokenizer count,
including special tokens (`reference.rs:51`, `child.rs:174`). ColBERT's added
marker and query augmentation do not enter that ordinary count.

- Sparse output is `representation: sparse`, `vocabulary_size`, ascending unique
  in-range `indices`, and equally many strictly positive finite `values`
  (`reference.rs:78`, `reference_compare.rs:68`). It must be nonempty. The child
  compares the actual sparse encoder's entries (`child.rs:482`). Its smoke
  checks also enforce vocabulary size and sorted, unique, positive values
  (`child.rs:441`).
- Multi-vector output is `representation: multi_vector`, positive `rows` and
  `columns`, and exactly `rows * columns` finite, row-major `values`
  (`reference.rs:83,385`, `reference_compare.rs:60`). The child uses query or
  document encoding according to the semantic mode (`child_reference.rs:52`).
  Its smoke checks include column dimension and row normalization
  (`child.rs:377,397`).

`reference::compare` delegates at `reference.rs:228`. The comparator requires
representation and shape equality, exact sparse index equality, and elementwise
`abs(observed - expected) <= absolute + relative * abs(expected)`. It also
requires the minimum corresponding-row cosine to meet `minimum_cosine`
(`reference_compare.rs:16,87,108`). Sparse cosine uses the compact value vector
once support matches. Matrix comparison preserves row order; it does not search
for row permutations or compare MaxSim scores. Tolerances must be finite,
absolute <= 0.001, relative <= 0.01, their sum positive, and cosine in
[0.999, 1] (`reference.rs:342`). References are bound by path and SHA-256
(`reference.rs:184`). Neither representation needs a comparator change here.

Use the pinned retrieval script once the compile slot is released:

```sh
uv run certification/tools/make_retrieval_reference.py \
  certification/specs/splade-pp-en-v1.json smoke \
  /existing/scratch/directory/reference.json \
  --absolute-tolerance 0.001 --relative-tolerance 0.01 --minimum-cosine 0.999
```

Substitute `colbert-small.json` for ColBERT. The same output publication,
immutable snapshot verification, shared cache, offline inference and CPU float32
rules apply. An existing output is refused. The script requires all three
explicit tolerance arguments and checks the current comparator bounds. Real
reference runs remain owed, with a 4 GB Python RSS ceiling.

Four fixed probes are in `make_retrieval_reference.PROBES`: `smoke` asks about
machine learning; `paragraph` describes learning from data and examples;
`non-ascii` contains accented Latin and Japanese text; `whitespace` contains two
leading/trailing spaces, a tab, a newline and an internal double space. The
literal text is recorded in each reference. An explicit positional probe or
`--probe-file` overrides the fixed text without changing its whitespace.
All four profiles have max sequence 128 and context 512. Paragraph uses the
model's document mode; the other three use its query mode. All eight reference
bindings remain absent until generation; all four profiles are required for
promotion.

### Pinned encoder settings

| Setting | SPLADE++ EN v1 | ColBERT Small |
|---|---|---|
| Model pin | prithivida/Splade_PP_en_v1 @ 762be6a7206e2f299182705972a65e5c46e62be2 | answerdotai/answerai-colbert-small-v1 @ c72aa89bc61afdd85373643f3a1a75b2aad6e0fe |
| Architecture | BERT masked-language-model head; hidden 768, FFN 3072, 12 layers/12 heads | BERT plus bias-free projection; hidden 384, FFN 1536, 12 layers/12 heads |
| Positions / text scope | 512 positions; short profile 128; card describes a document encoder, with a separate query model planned | 512 positions; short document limit 128; training metadata doc limit 300 and card indexing example 512 |
| Output | Vocabulary-sized 30522; log(1 + ReLU(logits)), attention-mask multiplication, max over positions; positive coordinates retained | 96 per kept token after linear projection; row L2 normalization |
| Tokenizer | Uncased WordPiece, BertNormalizer and BertPreTokenizer; lowercase and Chinese-character handling enabled | Same tokenizer family and normalization |
| Tokens kept/masked | Padding positions excluded by attention mask; unmasked CLS/SEP and other specials participate; no punctuation filter | CLS, SEP and role marker retained; query MASK augmentation retained, with augmentation attention off; document padding and ASCII punctuation tokens removed |
| Role markers / padding | No prefix/marker or query padding recipe in the card's example | Q=[unused0], D=[unused1], inserted after CLS; query padded to 32 using MASK; card recommends the nearest higher multiple of 16 |
| Vector normalization / metric | No final normalization; card does not explicitly name a metric | Normalized token vectors; metadata declares cosine similarity, upstream retrieval uses dot-product MaxSim |
| License | Apache-2.0 | Apache-2.0 |

Sources: pinned [SPLADE card](https://huggingface.co/prithivida/Splade_PP_en_v1/blob/762be6a7206e2f299182705972a65e5c46e62be2/README.md)
and [configuration](https://huggingface.co/prithivida/Splade_PP_en_v1/blob/762be6a7206e2f299182705972a65e5c46e62be2/config.json);
pinned [ColBERT card](https://huggingface.co/answerdotai/answerai-colbert-small-v1/blob/c72aa89bc61afdd85373643f3a1a75b2aad6e0fe/README.md),
[configuration](https://huggingface.co/answerdotai/answerai-colbert-small-v1/blob/c72aa89bc61afdd85373643f3a1a75b2aad6e0fe/config.json)
and [artifact metadata](https://huggingface.co/answerdotai/answerai-colbert-small-v1/blob/c72aa89bc61afdd85373643f3a1a75b2aad6e0fe/artifact.metadata).

SPLADE uses Transformers' actual `AutoModelForMaskedLM` and the pinned card's
pooling recipe, with torch 2.14.0 and transformers 4.57.6. ColBERT code is pinned
to [stanford-futuredata/ColBERT @ cc4f3dc91c0b45d2d08c251d9d95178285c65f1c](https://github.com/stanford-futuredata/ColBERT/tree/cc4f3dc91c0b45d2d08c251d9d95178285c65f1c).
Its installed VCS record must match that commit. The script uses the upstream
HF_ColBERT class, query/document tokenizers and the upstream
[query, document and mask methods](https://github.com/stanford-futuredata/ColBERT/blob/cc4f3dc91c0b45d2d08c251d9d95178285c65f1c/colbert/modeling/colbert.py).
The Transformers parent loader supplies explicit local-only loading and loading
information; missing/unexpected/mismatched weights are refused. This encoding
path does not instantiate a scoring constructor or request scoring extension
compilation, and does not use the wrapper's unpinned model-name fallback.
Import behavior in the real dependency environment remains unchecked until the
reference run. Document output uses upstream `keep_dims=False` to
remove masked rows. Inputs that would be cut by marker insertion or the fixed
32-token query length are refused before tensorization. No extra normalization,
projection or masking is applied to the upstream output.

## miniCOIL encoder weights

```sh
uv run certification/tools/minicoil_weights.py
```

Fetches `onnx/model.onnx` from `Qdrant/minicoil-v1` at
`4a7b05822a7a246d25778508593fff58fe574dfe` and `model.safetensors` from
`jinaai/jina-embeddings-v2-small-en` at
`44e7d1d6caec8c883c2d4b207588504d519788d0`, and prints:

- the stored number types on each side;
- for every safetensors tensor, the ONNX float initialiser of the same shape, or
  of the transposed shape for a matrix, whose values are closest after both are
  converted to float32, marked `exact`, `close` (largest absolute difference at
  most 0.001) or `unmatched`;
- the counts of each mark and the largest absolute difference among matches.

It writes no file other than the downloads.

## miniCOIL fixtures

```sh
uv run certification/tools/minicoil_fixture.py certification/fixtures/minicoil
```

The directory must exist and none of the eight fixtures may exist; the script
refuses before loading the model if any does, and publishes each file with an
exclusive link so an existing file is never replaced. It fetches the eight
`Qdrant/minicoil-v1` files FastEmbed needs at the pinned revision and loads them
with FastEmbed 0.8.1's `MiniCOIL` class through its `specific_model_path`
argument, on CPU with one onnxruntime thread.

For each of four texts it writes `<text>-document.json` (FastEmbed `embed`) and
`<text>-question.json` (FastEmbed `query_embed`). The script checks that the
stage-by-stage final vector is identical to the public method's output and that
its recorded word resolution agrees with FastEmbed's own.

FastEmbed internals used: `MiniCOIL.onnx_embed` (token ids, attention mask and
the encoder output), `MiniCOIL.vocab_resolver` (`convert_ids_to_tokens`,
`_reconstruct_bpe`, `resolve_tokens`, `vocab`, `stem_mapping`, `stopwords`,
`stemmer`, `vocab_size`), `MiniCOIL.encoder.encoder_weights` and `output_dim`,
`MiniCOIL._post_process_onnx_output`, `MiniCOIL.k`, `b`, `avg_len`, and `GAP` from
`fastembed.sparse.utils.sparse_vectors_converter`.

### Fixture members

- `schema_version`: 1.
- `model`: the Hugging Face repository and revision.
- `producer`: the library and run settings.
- `text`, `role`: the input and whether it was embedded as a `document` or a
  `question`.
- `token_ids`, `tokens`: the tokenizer's ids and token strings for every
  unmasked position, special tokens included, in order.
- `token_vectors`: the encoder's 512-value output for each of those positions,
  as float32 values. Not pooled and not normalised.
- `words`: each word rebuilt by joining `##` pieces, in order, with
  `token_positions` (indices into `token_ids`), `resolution` and `vocab_id`.
  `resolution` is the first branch that matched, in FastEmbed's order:
  `stop_word` (id 0), `exact` (the word is in the vocabulary), `stem_mapping`
  (the word is a key of the stem mapping), `stemmed` (its Snowball English stem
  is a key of the stem mapping), or `unknown` (id 0).
- `projection_rows`: for each non-zero vocabulary id used, the 512 × 4 matrix
  that projects the word's averaged token vector, as rows of 4 float32 values.
- `constants`: BM25 `k`, `b` and `avg_len`; `gap`; `vocab_size` (vocabulary plus
  one for the unknown id); `embedding_size` (4); `unknown_words_shift`, where
  indices of unknown words start; `token_max_length`, the longest stem kept for
  an unknown word; `tf_applied`, true for documents and false for questions.
- `sparse`: the final sparse vector from FastEmbed, `indices` ascending and
  `values` (float32) in the same order. Values are signed. IDF is not applied;
  Qdrant applies it.
