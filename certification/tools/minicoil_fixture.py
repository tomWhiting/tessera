# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastembed==0.8.1",
#   "huggingface-hub==1.33.0",
#   "mmh3==5.3.1",
#   "numpy==2.5.3",
#   "onnxruntime==1.30.0",
#   "py-rust-stemmers==0.1.8",
#   "tokenizers==0.23.2",
# ]
# ///

"""Write miniCOIL fixtures from FastEmbed 0.8.1, one per text and role, stage by stage."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

REPOSITORY = "Qdrant/minicoil-v1"
REVISION = "4a7b05822a7a246d25778508593fff58fe574dfe"
FILES = (
    "config.json",
    "minicoil.triplet.model.npy",
    "minicoil.triplet.model.vocab",
    "onnx/model.onnx",
    "special_tokens_map.json",
    "stopwords.txt",
    "tokenizer.json",
    "tokenizer_config.json",
)
TEXTS = (
    ("machine-learning", "What is machine learning?"),
    ("bear", "The bear bears a bearing, and the bearings bore the bears."),
    ("cafe", "Zoë's naïve café costs €12.50, no?"),
    ("tessera", "Tessera embeds haematite prolly-trees quickly."),
)
ROLES = ("document", "question")


def fetch(cache):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    paths = {}
    for name in FILES:
        arguments = {
            "repo_id": REPOSITORY,
            "filename": name,
            "revision": REVISION,
            "cache_dir": str(cache),
            "token": False,
        }
        try:
            path = hf_hub_download(**arguments, local_files_only=True)
            action = "cached"
        except LocalEntryNotFoundError:
            path = hf_hub_download(**arguments)
            action = "fetched"
        print(f"{action}: {name}", file=sys.stderr, flush=True)
        paths[name] = Path(path)
    snapshot = paths["config.json"].parent
    if snapshot.name != REVISION:
        raise ValueError("downloaded snapshot does not match the pinned revision")
    return snapshot


def floats(array):
    """Float32 values as JSON numbers: the shortest decimal that reads back to the same float32."""
    import numpy as np

    return [
        float(np.format_float_positional(np.float32(value), unique=True))
        for value in array
    ]


def resolve_kind(resolver, word):
    """Mirror VocabResolver.resolve_tokens' branch order and name the branch taken."""
    if word in resolver.stopwords:
        return "stop_word", 0
    if word in resolver.vocab:
        return "exact", resolver.vocab[word]
    if word in resolver.stem_mapping:
        return "stem_mapping", resolver.vocab[resolver.stem_mapping[word]]
    stem = resolver.stemmer.stem_word(word)
    if stem in resolver.stem_mapping:
        return "stemmed", resolver.vocab[resolver.stem_mapping[stem]]
    return "unknown", 0


def fixture(model, text, role):
    import numpy as np
    from fastembed.sparse.utils.sparse_vectors_converter import GAP

    resolver = model.vocab_resolver
    is_query = role == "question"

    output = model.onnx_embed([text])
    mask = output.attention_mask[0] == 1
    token_ids = output.input_ids[0, mask].astype(np.int64)
    token_vectors = output.model_output[0, mask].astype(np.float32)
    tokens = resolver.convert_ids_to_tokens(token_ids)

    words = []
    for word, positions in resolver._reconstruct_bpe(enumerate(tokens)):
        kind, vocab_id = resolve_kind(resolver, word)
        words.append(
            {
                "word": word,
                "token_positions": positions,
                "resolution": kind,
                "vocab_id": vocab_id,
            }
        )

    # FastEmbed's own resolution must agree with the branch names recorded above.
    resolved_ids, _, _, _ = resolver.resolve_tokens(token_ids.copy())
    for entry in words:
        for position in entry["token_positions"]:
            if int(resolved_ids[position]) != entry["vocab_id"]:
                raise ValueError(
                    f"resolution disagrees with FastEmbed for {entry['word']!r}"
                )

    used_ids = sorted({entry["vocab_id"] for entry in words if entry["vocab_id"] > 0})
    projection = {
        str(vocab_id): [floats(row) for row in model.encoder.encoder_weights[vocab_id]]
        for vocab_id in used_ids
    }

    expected = list(model._post_process_onnx_output(output, is_query=is_query))[0]
    public = list((model.query_embed if is_query else model.embed)([text]))[0]
    if not (
        np.array_equal(expected.indices, public.indices)
        and np.array_equal(expected.values, public.values)
    ):
        raise ValueError("stage-by-stage output differs from FastEmbed's public method")
    order = np.argsort(expected.indices, kind="stable")

    vocab_size = resolver.vocab_size()
    embedding_size = model.encoder.output_dim
    return {
        "schema_version": 1,
        "model": {"repository": REPOSITORY, "revision": REVISION},
        "producer": "fastembed 0.8.1 MiniCOIL on CPU, onnxruntime, one thread",
        "text": text,
        "role": role,
        "token_ids": [int(value) for value in token_ids],
        "tokens": tokens,
        "token_vectors": [floats(row) for row in token_vectors],
        "words": words,
        "projection_rows": projection,
        "constants": {
            "k": model.k,
            "b": model.b,
            "avg_len": model.avg_len,
            "gap": GAP,
            "vocab_size": vocab_size,
            "embedding_size": embedding_size,
            "unknown_words_shift": ((vocab_size * embedding_size) // GAP + 2) * GAP,
            "token_max_length": 40,
            "tf_applied": not is_query,
        },
        "sparse": {
            "indices": [int(value) for value in expected.indices[order]],
            "values": floats(expected.values[order]),
        },
    }


def publish(path, document):
    payload = json.dumps(document, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".fixture-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        # Exclusive publication never replaces an existing fixture.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    try:
        output_dir = arguments.output_dir
        if not output_dir.is_dir():
            raise ValueError("output directory must already exist")
        targets = [
            (output_dir / f"{slug}-{role}.json", text, role)
            for slug, text in TEXTS
            for role in ROLES
        ]
        existing = [str(path) for path, _, _ in targets if os.path.lexists(path)]
        if existing:
            raise FileExistsError(f"fixtures already exist: {existing}")

        cache = Path(__file__).resolve().parents[2] / ".tessera" / "reference-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        snapshot = fetch(cache)
        os.environ["HF_HUB_OFFLINE"] = "1"

        from fastembed.sparse.minicoil import MiniCOIL

        model = MiniCOIL(
            REPOSITORY,
            cache_dir=str(cache / "fastembed"),
            threads=1,
            providers=["CPUExecutionProvider"],
            specific_model_path=str(snapshot),
        )
        for path, text, role in targets:
            publish(path, fixture(model, text, role))
            print(f"wrote: {path} ({path.stat().st_size} bytes)")
    except Exception as error:
        print(
            f"minicoil_fixture failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
