# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sentence-transformers==5.7.0",
#   "torch==2.14.0",
#   "transformers==5.17.0",
# ]
# ///

"""Generate a deterministic dense reference from an immutable model specification."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile


def read_inputs(spec_path, profile_name, probe):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if profile_name not in spec["profiles"]:
        raise ValueError(f"unknown profile: {profile_name}")
    profile = spec["profiles"][profile_name]
    model = spec["model"]
    capability = profile["capability"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model["repository"]):
        raise ValueError("model repository must be an owner/name identifier")
    if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("model revision must be an immutable 40-digit commit")
    if model["representation"] != "dense":
        raise ValueError("only dense references are supported")
    if capability["device"] != "cpu" or capability["dtype"] != "f32":
        raise ValueError("profile must specify cpu and f32")
    if capability["semantic_mode"] not in ("query", "document"):
        raise ValueError("unsupported semantic mode")
    limit = capability["max_sequence_tokens"]
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_sequence_tokens must be a positive integer")
    if not isinstance(probe, str) or not probe.strip():
        raise ValueError("probe text must not be empty")
    relative = PurePosixPath(profile["official_reference"]["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("reference path must stay within the references directory")
    reference = json.loads(
        (spec_path.parent.parent / "references" / relative).read_text(encoding="utf-8")
    )
    tolerance = reference["tolerance"]
    for name in ("absolute", "relative", "minimum_cosine"):
        value = tolerance[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid tolerance: {name}")
    return spec, capability, tolerance


def fetch_model(model, artifacts, cache):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repository = model["repository"]
    revision = model["revision"]
    files = HfApi(token=False).list_repo_files(repository, revision=revision)
    required = {artifact["path"] for artifact in artifacts}
    ancillary = {
        "modules.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    }
    selected = sorted(
        name
        for name in files
        if name in required or name in ancillary or name.endswith("/config.json")
    )
    missing = required.difference(selected) | {"modules.json"}.difference(selected)
    if missing:
        raise ValueError(f"pinned model is missing required files: {sorted(missing)}")
    downloaded = {}
    for name in selected:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe model file path: {name}")
        arguments = {
            "repo_id": repository,
            "filename": name,
            "revision": revision,
            "cache_dir": str(cache),
            "token": False,
        }
        try:
            path = hf_hub_download(**arguments, local_files_only=True)
            action = "cached"
        except LocalEntryNotFoundError:
            path = hf_hub_download(**arguments)
            action = "fetched"
        print(f"{action}: {name}", flush=True)
        downloaded[name] = Path(path)
    for artifact in artifacts:
        path = downloaded[artifact["path"]]
        if path.stat().st_size != artifact["size_bytes"]:
            raise ValueError(f"artifact size mismatch: {artifact['path']}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {artifact['path']}")
    snapshot = downloaded["config.json"].parent
    if snapshot.name != revision:
        raise ValueError("downloaded snapshot does not match the pinned revision")
    return snapshot


def make_reference(spec, capability, tolerance, profile, probe, snapshot, cache):
    import sentence_transformers
    import torch
    import transformers

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = sentence_transformers.SentenceTransformer(
        str(snapshot),
        revision=spec["model"]["revision"],
        device="cpu",
        cache_folder=str(cache),
        local_files_only=True,
        token=False,
        trust_remote_code=False,
        model_kwargs={"dtype": torch.float32},
    )
    model.float()
    model.eval()
    model.max_seq_length = capability["max_sequence_tokens"]
    token_count = len(model.tokenizer.encode(probe, truncation=False))
    if token_count > model.max_seq_length:
        raise ValueError(
            f"probe has {token_count} tokens, above the profile limit {model.max_seq_length}"
        )
    with torch.inference_mode():
        vector = model.encode(
            probe,
            prompt="",
            task=capability["semantic_mode"],
            device="cpu",
            precision="float32",
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
    if vector.dtype != torch.float32 or vector.ndim != 1:
        raise ValueError("reference output must be a float32 vector")
    values = vector.tolist()
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("reference output must contain finite values")
    identity = spec["model"]
    return {
        "schema_version": 1,
        "model_id": identity["id"],
        "repository": identity["repository"],
        "revision": identity["revision"],
        "profile": profile,
        "capability": capability,
        "provenance": {
            "producer": "sentence-transformers reference run on CPU, float32, L2-normalised",
            "framework": "sentence-transformers",
            "framework_version": (
                f"{sentence_transformers.__version__} "
                f"(torch {torch.__version__}, transformers {transformers.__version__})"
            ),
            "source_repository": identity["repository"],
            "source_revision": identity["revision"],
            "probe_prefix": None,
        },
        "probe": {"kind": "text", "text": probe, "token_count": token_count},
        "tolerance": tolerance,
        "expected": {"representation": identity["representation"], "values": values},
    }


def publish(output, reference):
    payload = (
        json.dumps(reference, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix=".reference-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        # Exclusive publication preserves an existing destination even during a race.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("profile")
    parser.add_argument("probe")
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    try:
        if os.path.lexists(arguments.output):
            raise FileExistsError(f"output already exists: {arguments.output}")
        if not arguments.output.parent.is_dir():
            raise ValueError("output parent directory must already exist")
        spec, capability, tolerance = read_inputs(
            arguments.spec, arguments.profile, arguments.probe
        )
        cache = Path(__file__).resolve().parents[2] / ".tessera" / "reference-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
        os.environ["HF_HUB_CACHE"] = str(cache)
        os.environ["XDG_CACHE_HOME"] = str(cache)
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        snapshot = fetch_model(spec["model"], spec["artifacts"], cache)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        reference = make_reference(
            spec,
            capability,
            tolerance,
            arguments.profile,
            arguments.probe,
            snapshot,
            cache,
        )
        publish(arguments.output, reference)
        print(f"token_count: {reference['probe']['token_count']}")
        print(f"uv_environment: {sys.prefix}")
        print(f"wrote: {arguments.output}")
    except Exception as error:
        print(
            f"make_reference failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
