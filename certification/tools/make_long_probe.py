# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.22.2"]
# ///

"""Rebuild the fixed repository prose probe with the model's checked tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

SOURCE_COMMIT = "979f1fb06d31d27beb4fbfb213b6229e673d2562"
SOURCE_FILES = (
    "docs/vision_board/part5_dense.md",
    "docs/vision_board/part1_multi_vector.md",
)


def cut_probe(text, count_tokens, minimum=6000, maximum=8000):
    boundaries = [match.end() for match in re.finditer(r"\S+", text)]
    low, high = 0, len(boundaries)
    while low < high:
        middle = (low + high) // 2
        if count_tokens(text[: boundaries[middle]]) <= maximum:
            low = middle + 1
        else:
            high = middle
    if low == 0:
        raise ValueError("source has no word boundary within the token limit")
    probe = text[: boundaries[low - 1]]
    count = count_tokens(probe)
    if not minimum <= count <= maximum:
        raise ValueError(
            f"source probe has {count} tokens, outside {minimum}..{maximum}"
        )
    return probe, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    try:
        from tokenizers import Tokenizer

        root = Path(__file__).resolve().parents[2]
        spec = json.loads(arguments.spec.read_text(encoding="utf-8"))
        model = spec["model"]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model["repository"]):
            raise ValueError("model repository must be an owner/name identifier")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError("model revision must be an immutable 40-digit commit")
        tokenizer_artifact = next(
            artifact
            for artifact in spec["artifacts"]
            if artifact["path"] == "tokenizer.json"
        )
        tokenizer_path = (
            root
            / ".tessera"
            / "reference-cache"
            / ("models--" + model["repository"].replace("/", "--"))
            / "snapshots"
            / model["revision"]
            / "tokenizer.json"
        )
        with tokenizer_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != tokenizer_artifact["sha256"]:
            raise ValueError("cached tokenizer SHA-256 differs from the specification")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        sections = [
            subprocess.run(
                ["git", "show", f"{SOURCE_COMMIT}:{file}"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout.decode("utf-8")
            for file in SOURCE_FILES
        ]
        probe, count = cut_probe(
            "\n".join(sections), lambda text: len(tokenizer.encode(text).ids)
        )
        with arguments.output.open("x", encoding="utf-8", newline="") as stream:
            stream.write(probe)
        print(f"token_count: {count}")
        print(f"wrote: {arguments.output}")
    except Exception as error:
        print(
            f"make_long_probe failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
