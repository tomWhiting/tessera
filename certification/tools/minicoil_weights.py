# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub==2.1.1",
#   "numpy==2.5.3",
#   "onnx==1.23.1",
# ]
# ///

"""Compare miniCOIL's ONNX encoder weights with jina-embeddings-v2-small-en's safetensors."""

import json
import os
from pathlib import Path
import struct
import sys

MINICOIL = (
    "Qdrant/minicoil-v1",
    "4a7b05822a7a246d25778508593fff58fe574dfe",
    "onnx/model.onnx",
)
JINA = (
    "jinaai/jina-embeddings-v2-small-en",
    "44e7d1d6caec8c883c2d4b207588504d519788d0",
    "model.safetensors",
)
CLOSE = 0.001

SAFETENSORS_TYPES = {"F32": "<f4", "F16": "<f2", "BF16": "bf16", "F64": "<f8"}


def fetch(cache, repository, revision, filename):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    arguments = {
        "repo_id": repository,
        "filename": filename,
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
    print(f"{action}: {repository}@{revision}/{filename}", file=sys.stderr, flush=True)
    return Path(path)


def read_safetensors(path):
    """Read every tensor as float32 without a framework, so BF16 is handled exactly."""
    import numpy as np

    with path.open("rb") as stream:
        (header_len,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(header_len))
        data = stream.read()
    tensors = {}
    stored = set()
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = entry["dtype"]
        if dtype not in SAFETENSORS_TYPES:
            raise ValueError(f"unsupported safetensors dtype {dtype} for {name}")
        stored.add(dtype)
        start, end = entry["data_offsets"]
        raw = data[start:end]
        if dtype == "BF16":
            halves = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            values = halves.view(np.float32)
        else:
            values = np.frombuffer(raw, dtype=SAFETENSORS_TYPES[dtype]).astype(
                np.float32
            )
        tensors[name] = values.reshape(entry["shape"])
    return tensors, sorted(stored)


def read_onnx(path):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    initialisers = {}
    stored = set()
    for tensor in model.graph.initializer:
        stored.add(onnx.TensorProto.DataType.Name(tensor.data_type))
        array = numpy_helper.to_array(tensor)
        if np.issubdtype(array.dtype, np.floating):
            initialisers[tensor.name] = array.astype(np.float32)
    return initialisers, sorted(stored)


def best_match(values, candidates):
    """Return (onnx name, orientation, max abs difference) of the closest same-shape initialiser."""
    import numpy as np

    best = None
    for name, candidate in candidates.items():
        orientations = []
        if candidate.shape == values.shape:
            orientations.append(("same", candidate))
        if values.ndim == 2 and candidate.shape == values.shape[::-1]:
            orientations.append(("transposed", candidate.T))
        for orientation, aligned in orientations:
            difference = float(np.max(np.abs(aligned - values))) if values.size else 0.0
            if (
                best is None
                or difference < best[2]
                or (difference == best[2] and (name, orientation) < (best[0], best[1]))
            ):
                best = (name, orientation, difference)
    return best


def main():
    cache = Path(__file__).resolve().parents[2] / ".tessera" / "reference-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    onnx_path = fetch(cache, *MINICOIL)
    safetensors_path = fetch(cache, *JINA)

    tensors, safetensors_types = read_safetensors(safetensors_path)
    initialisers, onnx_types = read_onnx(onnx_path)

    print(f"onnx: {MINICOIL[0]}@{MINICOIL[1]}/{MINICOIL[2]}")
    print(f"onnx stored types: {', '.join(onnx_types)}")
    print(f"onnx float initialisers: {len(initialisers)}")
    print(f"safetensors: {JINA[0]}@{JINA[1]}/{JINA[2]}")
    print(f"safetensors stored types: {', '.join(safetensors_types)}")
    print(f"safetensors tensors: {len(tensors)}")
    print("comparison: both sides converted to float32; same shape or 2-D transpose")

    exact = close = unmatched = 0
    largest = 0.0
    for name in sorted(tensors):
        values = tensors[name]
        match = best_match(values, initialisers)
        shape = "x".join(str(size) for size in values.shape)
        if match is None:
            unmatched += 1
            print(f"unmatched {name} [{shape}]: no initialiser of this shape")
            continue
        onnx_name, orientation, difference = match
        if difference == 0.0:
            verdict = "exact"
            exact += 1
        elif difference <= CLOSE:
            verdict = "close"
            close += 1
        else:
            unmatched += 1
            print(
                f"unmatched {name} [{shape}]: nearest {onnx_name} ({orientation}) "
                f"max abs difference {difference:.9g}"
            )
            continue
        largest = max(largest, difference)
        print(
            f"{verdict} {name} [{shape}] = {onnx_name} ({orientation}) max abs difference {difference:.9g}"
        )

    print(f"equal exactly: {exact}")
    print(f"equal within {CLOSE}: {close}")
    print(f"unmatched: {unmatched}")
    print(f"largest absolute difference among matched: {largest:.9g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
