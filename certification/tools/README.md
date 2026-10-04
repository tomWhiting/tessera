# Certification tools

Both scripts run from the repository root with `uv run`; their dependencies are
pinned to exact versions in each script's inline metadata. Downloads use
immutable revisions and stay under `.tessera/reference-cache` (ignored by Git).
Download progress (`fetched:` or `cached:`) goes to standard error, so standard
output is identical between runs.

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
