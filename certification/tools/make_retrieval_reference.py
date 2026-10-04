# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.0",
#   "transformers==4.57.6",
#   "colbert-ai[faiss-cpu] @ git+https://github.com/stanford-futuredata/ColBERT.git@cc4f3dc91c0b45d2d08c251d9d95178285c65f1c",
# ]
# ///

"""Produce sparse or token-matrix references using pinned upstream encoders."""

import argparse
from importlib import metadata
import json
import math
import os
from pathlib import Path
import string
import sys
from types import MethodType, SimpleNamespace

from make_reference import fetch_model, publish, read_inputs, read_probe

COLBERT_REPOSITORY = "https://github.com/stanford-futuredata/ColBERT"
COLBERT_REVISION = "cc4f3dc91c0b45d2d08c251d9d95178285c65f1c"
PROBES = {
    "smoke": "What is machine learning?",
    "paragraph": "Machine learning allows computers to learn patterns from data. The model uses examples to make predictions.",
    "non-ascii": "Café naïve résumé — 東京.",
    "whitespace": "  Machine\tlearning\nuses  data.  ",
}


def validate_tolerance(tolerance):
    absolute, relative, cosine = (
        tolerance[name] for name in ("absolute", "relative", "minimum_cosine")
    )
    if (
        not all(math.isfinite(value) for value in [absolute, relative, cosine])
        or not 0 <= absolute <= 0.001
        or not 0 <= relative <= 0.01
        or absolute + relative <= 0
        or not 0.999 <= cosine <= 1
    ):
        raise ValueError("reference tolerance is outside the current comparator bounds")


def validate_loading_info(info):
    failures = {
        key: info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(key)
    }
    if failures:
        raise ValueError(f"upstream model loading failed: {failures}")


def probe_count(tokenizer, text, limit, extra_tokens=0):
    count = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    if count == 0 or count + extra_tokens > limit:
        raise ValueError(
            f"probe and marker exceed token limit {limit}: {count}+{extra_tokens}"
        )
    return count


def sparse_values(values):
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("sparse reference values must be finite and nonnegative")
    entries = [(index, value) for index, value in enumerate(values) if value > 0]
    if not entries:
        raise ValueError("sparse reference must contain positive values")
    return {
        "representation": "sparse",
        "vocabulary_size": len(values),
        "indices": [index for index, _ in entries],
        "values": [value for _, value in entries],
    }


def sparse_output(model, inputs, torch):
    logits = model(**inputs).logits
    weighted = torch.log(1 + torch.relu(logits)) * inputs["attention_mask"].unsqueeze(
        -1
    )
    vector = torch.max(weighted, dim=1).values[0]
    if vector.dtype != torch.float32 or vector.ndim != 1:
        raise ValueError("sparse output must be a float32 vector")
    return sparse_values(vector.tolist())


def matrix_values(matrix, columns):
    if not matrix or any(len(row) != columns for row in matrix):
        raise ValueError("reference matrix has the wrong dimensions")
    values = [value for row in matrix for value in row]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("reference matrix contains non-finite values")
    return {
        "representation": "multi_vector",
        "rows": len(matrix),
        "columns": columns,
        "values": values,
    }


def colbert_output(session, inputs, role, operations, torch, columns):
    if role == "late_interaction_query":
        matrix = operations.query(session, *inputs)[0]
    elif role == "late_interaction_document":
        matrix = operations.doc(session, *inputs, keep_dims=False)[0]
    else:
        raise ValueError("unsupported ColBERT semantic mode")
    if matrix.dtype != torch.float32 or matrix.ndim != 2:
        raise ValueError("ColBERT output must be a float32 matrix")
    return matrix_values(matrix.tolist(), columns)


def colbert_source_version():
    distribution = metadata.distribution("colbert-ai")
    source = distribution.read_text("direct_url.json")
    if source is None:
        raise ValueError("ColBERT installation has no immutable source record")
    source = json.loads(source)
    if (
        source.get("url", "").removesuffix(".git") != COLBERT_REPOSITORY
        or source.get("vcs_info", {}).get("commit_id") != COLBERT_REVISION
    ):
        raise ValueError("ColBERT installation differs from the pinned upstream commit")
    return distribution.version


def colbert_settings(artifact, columns, defaults, tokenizer):
    if type(columns) is not int or columns <= 0:
        raise ValueError("ColBERT projection dimension must be a positive integer")
    required = {
        "dim": columns,
        "query_maxlen": 32,
        "query_token_id": defaults.query_token_id,
        "doc_token_id": defaults.doc_token_id,
        "mask_punctuation": True,
        "attend_to_mask_tokens": False,
    }
    markers = {"query_token_id", "doc_token_id"}
    settings, recorded = {}, {}
    vocabulary = tokenizer.get_vocab()
    for name, value in required.items():
        selected = artifact.get(name, value) if name in markers else artifact.get(name)
        if type(selected) is not type(value) or selected != value:
            raise ValueError(
                f"pinned ColBERT artifact settings differ from its recipe: {name}"
            )
        settings[name] = selected
        recorded[name] = {
            "value": selected,
            "source": "metadata" if name in artifact else "upstream_default",
        }
        if name in markers:
            token_id = vocabulary.get(selected)
            if (
                type(token_id) is not int
                or token_id < 0
                or token_id == tokenizer.unk_token_id
                or tokenizer.convert_tokens_to_ids(selected) != token_id
            ):
                raise ValueError(
                    f"pinned ColBERT tokenizer is missing marker: {selected}"
                )
            recorded[name]["token_id"] = token_id
    return settings, recorded


def load_colbert(snapshot, limit, columns, torch):
    from colbert.infra import ColBERTConfig
    from colbert.modeling.colbert import ColBERT
    from colbert.modeling.hf_colbert import class_factory
    from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
    from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
    from transformers import PreTrainedModel

    artifact = json.loads((snapshot / "artifact.metadata").read_text(encoding="utf-8"))
    defaults = ColBERTConfig()
    model_class = class_factory(str(snapshot))
    tokenizer = model_class.raw_tokenizer_from_pretrained(str(snapshot))
    settings, recorded = colbert_settings(artifact, columns, defaults, tokenizer)
    config = ColBERTConfig(
        checkpoint=str(snapshot),
        model_name=str(snapshot),
        doc_maxlen=limit,
        gpus=0,
        nranks=1,
        **settings,
    )
    model, loading = PreTrainedModel.from_pretrained.__func__(
        model_class,
        str(snapshot),
        colbert_config=config,
        local_files_only=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    validate_loading_info(loading)
    model.to("cpu").float().eval()
    query = QueryTokenizer(config, verbose=0)
    document_tokenizer = DocTokenizer(config)
    skiplist = {}
    for symbol in string.punctuation:
        ids = query.tok.encode(symbol, add_special_tokens=False)
        if not ids:
            raise ValueError("ColBERT punctuation has no tokenizer id")
        skiplist[symbol] = True
        skiplist[ids[0]] = True
    session = SimpleNamespace(
        bert=model.LM,
        linear=model.linear,
        device=torch.device("cpu"),
        use_gpu=False,
        pad_token=query.tok.pad_token_id,
        skiplist=skiplist,
    )
    session.mask = MethodType(ColBERT.mask, session)
    return session, query, document_tokenizer, ColBERT, recorded


def document(
    identity,
    capability,
    tolerance,
    profile,
    probe,
    count,
    output,
    versions,
    settings=None,
):
    producer = "pinned model card recipe on CPU, float32"
    if identity["representation"] == "multi_vector":
        producer += f"; upstream code {COLBERT_REPOSITORY}@{COLBERT_REVISION}"
        if settings is not None:
            producer += "; ColBERT settings " + json.dumps(settings, sort_keys=True)
    return {
        "schema_version": 1,
        "model_id": identity["id"],
        "repository": identity["repository"],
        "revision": identity["revision"],
        "profile": profile,
        "capability": capability,
        "provenance": {
            "producer": producer,
            "framework": "transformers",
            "framework_version": versions,
            "source_repository": identity["repository"],
            "source_revision": identity["revision"],
            "probe_prefix": None,
        },
        "probe": {"kind": "text", "text": probe, "token_count": count},
        "tolerance": tolerance,
        "expected": output,
    }


def make_reference(spec, capability, tolerance, profile, probe, snapshot):
    import torch
    import transformers

    validate_tolerance(tolerance)

    identity = spec["model"]
    selected = {
        "splade-pp-en-v1": "sparse",
        "splade-pp-en-v2": "sparse",
        "colbert-small": "multi_vector",
        "colbert-v2": "multi_vector",
    }
    if selected.get(identity["id"]) != identity["representation"]:
        raise ValueError("model has no pinned retrieval recipe")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("auto_map") or config.get("model_type") != "bert":
        raise ValueError(
            "retrieval recipe requires the pinned built-in BERT architecture"
        )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    versions = f"{transformers.__version__} (torch {torch.__version__})"
    limit = capability["max_sequence_tokens"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, token=False, trust_remote_code=False
    )
    settings = None
    if identity["representation"] == "sparse":
        count = probe_count(tokenizer, probe, limit)
        model, loading = transformers.AutoModelForMaskedLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            token=False,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            output_loading_info=True,
        )
        validate_loading_info(loading)
        model.to("cpu").float().eval()
        inputs = tokenizer(probe, return_tensors="pt", truncation=False)
        with torch.inference_mode():
            output = sparse_output(model, inputs, torch)
    else:
        versions += f"; colbert-ai {colbert_source_version()}@{COLBERT_REVISION}"
        role = capability["semantic_mode"]
        count = probe_count(
            tokenizer,
            probe,
            min(limit, 32) if role == "late_interaction_query" else limit,
            extra_tokens=1,
        )
        columns = spec["smoke"]["expected_dimension"]
        session, query, doc, operations, settings = load_colbert(
            snapshot, limit, columns, torch
        )
        inputs = (query if role == "late_interaction_query" else doc).tensorize([probe])
        with torch.inference_mode():
            output = colbert_output(session, inputs, role, operations, torch, columns)
    return document(
        identity,
        capability,
        tolerance,
        profile,
        probe,
        count,
        output,
        versions,
        settings,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("profile")
    parser.add_argument("probe", nargs="?")
    parser.add_argument("output", type=Path)
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, required=True)
    parser.add_argument("--relative-tolerance", type=float, required=True)
    parser.add_argument("--minimum-cosine", type=float, required=True)
    args = parser.parse_args()
    try:
        probe = (
            PROBES[args.profile]
            if args.probe is None and args.probe_file is None
            else read_probe(args.probe, args.probe_file)
        )
        if os.path.lexists(args.output):
            raise FileExistsError(f"output already exists: {args.output}")
        if not args.output.parent.is_dir():
            raise ValueError("output parent directory must already exist")
        tolerances = {
            "absolute": args.absolute_tolerance,
            "relative": args.relative_tolerance,
            "minimum_cosine": args.minimum_cosine,
        }
        spec, capability, tolerance = read_inputs(
            args.spec,
            args.profile,
            probe,
            tolerances,
            representations=("sparse", "multi_vector"),
        )
        validate_tolerance(tolerance)
        cache = Path(__file__).resolve().parents[2] / ".tessera" / "reference-cache"
        cache.mkdir(parents=True, exist_ok=True)
        for name in ["HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"]:
            os.environ[name] = str(cache)
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        snapshot = fetch_model(
            spec["model"], spec["artifacts"], cache, require_modules=False
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from huggingface_hub import constants

        constants.HF_HUB_OFFLINE = True
        result = make_reference(
            spec, capability, tolerance, args.profile, probe, snapshot
        )
        publish(args.output, result)
        print(f"token_count: {result['probe']['token_count']}")
        print(f"uv_environment: {sys.prefix}")
        print(f"wrote: {args.output}")
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f"reference refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
