"""Fetch pinned files or run fresh local miniCOIL qualification processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / ".tessera/cert-evidence/minicoil-L/assets"
FIXTURES = tuple(f"{text}-{role}" for text in ("machine-learning", "bear", "cafe", "tessera") for role in ("document", "question"))
RSS_LIMIT = 2 * 1024**3


def disk():
    result = subprocess.run(["df", "-k", str(ROOT)], capture_output=True, text=True, check=True)
    return {"command": result.args, "exit_code": result.returncode, "stdout": result.stdout,
            "free_bytes": int(result.stdout.splitlines()[-1].split()[3]) * 1024}


def source():
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain=v1"], text=True)
    if dirty:
        raise ValueError("qualification_source_dirty")
    return {"commit": head, "dirty": False}


def entry():
    registry = json.loads((ROOT / "models.json").read_text())
    return next(model for model in registry["model_categories"]["sparse"]["models"] if model["id"] == "minicoil-v1")


def fetch(bootstrap):
    before = disk()
    if before["free_bytes"] < 62 * 1024**3:
        raise ValueError(f"fetch_start_floor: {before['free_bytes']}")
    if ASSETS.exists():
        raise ValueError("fetch_destination_exists")
    model = entry()
    assets = model["minicoil_assets"]
    required = ("minicoil.triplet.model.npy", "minicoil.triplet.model.vocab", "stopwords.txt")
    declarations = {item["path"]: item for item in assets["files"]}
    if "config.json" not in declarations:
        if not bootstrap:
            raise ValueError("encoder_config_not_declared")
        declarations["config.json"] = {"path": "config.json", "size_bytes": 1175}
    groups = [
        ("encoder", assets["encoder_repository"], assets["encoder_revision"],
         [assets["encoder_weights"], declarations["config.json"], declarations["tokenizer.json"]]),
        ("tables", model["huggingface_id"], model["revision"], [declarations[name] for name in required]),
    ]
    ASSETS.mkdir(parents=True)
    report = {"source": source(), "before": before, "files": [], "inventories": []}
    try:
        for directory, repository, revision, files in groups:
            with urllib.request.urlopen(f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true") as response:
                inventory = json.load(response)
            if inventory["sha"] != revision:
                raise ValueError("inventory_revision_mismatch")
            report["inventories"].append({"repository": repository, "revision": revision, "inventory": inventory})
            siblings = {item["rfilename"]: item for item in inventory["siblings"]}
            destination = ASSETS / directory
            destination.mkdir()
            for artifact in files:
                metadata = siblings[artifact["path"]]
                if metadata["size"] != artifact["size_bytes"]:
                    raise ValueError("inventory_size_mismatch")
                path = destination / artifact["path"]
                temporary = path.with_suffix(path.suffix + ".part")
                digest = hashlib.sha256()
                blob = hashlib.sha1(f"blob {metadata['size']}\0".encode())
                size = 0
                try:
                    with urllib.request.urlopen(f"https://huggingface.co/{repository}/resolve/{revision}/{artifact['path']}") as response, temporary.open("xb") as output:
                        while chunk := response.read(65536):
                            if disk()["free_bytes"] < 60 * 1024**3:
                                raise ValueError("fetch_live_floor")
                            output.write(chunk)
                            digest.update(chunk)
                            blob.update(chunk)
                            size += len(chunk)
                            if size > metadata["size"]:
                                raise ValueError("payload_size_exceeded")
                    sha256 = digest.hexdigest()
                    if size != metadata["size"]:
                        raise ValueError("payload_size_mismatch")
                    if "lfs" in metadata:
                        if sha256 != metadata["lfs"]["sha256"]:
                            raise ValueError("inventory_lfs_digest_mismatch")
                    elif blob.hexdigest() != metadata["blobId"]:
                        raise ValueError("inventory_git_blob_mismatch")
                    if artifact.get("sha256") is not None and sha256 != artifact["sha256"]:
                        raise ValueError("registry_digest_mismatch")
                    temporary.rename(path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                report["files"].append({"directory": directory, "repository": repository, "revision": revision,
                                        "path": artifact["path"], "size_bytes": size, "sha256": sha256})
        encoder = [dict(path=x["path"], size_bytes=x["size_bytes"], sha256=x["sha256"]) for x in report["files"] if x["directory"] == "encoder"]
        manifest = {"schema_version": 1, "model": {"id": assets["encoder_model"], "repository": assets["encoder_repository"],
                                                 "revision": assets["encoder_revision"]}, "artifacts": encoder}
        (ASSETS / "encoder/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report["after"] = disk()
        (ASSETS / "fetch.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"files": report["files"], "before": report["before"], "after": report["after"]}))
    except BaseException:
        purge()
        raise


def observe(command):
    child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env={**os.environ, "TESSERA_OFFLINE": "1", "RAYON_NUM_THREADS": "1", "CANDLE_NUM_THREADS": "1"})
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    output = bytearray()
    refusal = None
    sampled_peak = 0
    try:
        while selector.get_map():
            measurement = subprocess.run(["ps", "-o", "rss=", "-p", str(child.pid)], capture_output=True, text=True)
            if measurement.stdout.strip():
                sampled_peak = max(sampled_peak, int(measurement.stdout.strip()) * 1024)
            if sampled_peak > RSS_LIMIT or disk()["free_bytes"] < 60 * 1024**3:
                refusal = "resident_ceiling" if sampled_peak > RSS_LIMIT else "native_live_disk_floor"
                try:
                    os.kill(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for key, mask in selector.select(timeout=0.05):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    output.extend(chunk)
                else:
                    selector.unregister(key.fileobj)
        pid, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        if pid != child.pid:
            raise ValueError("native_wait_pid_mismatch")
        peak = int(usage.ru_maxrss)
        if peak > RSS_LIMIT:
            refusal = "resident_ceiling"
        return {"pid": pid, "exit_code": child.returncode, "peak_rss_bytes": peak, "sampled_peak_rss_bytes": sampled_peak,
                "rss_ceiling_bytes": RSS_LIMIT, "rss_method": "Darwin wait4 ru_maxrss bytes; live ps RSS guard",
                "refusal": refusal, "output": output.decode()}
    finally:
        selector.close()
        child.stdout.close()
        if child.returncode is None:
            child.kill()
            child.wait()


def run(binary, evidence):
    if sys.platform != "darwin":
        raise ValueError("native_controller_requires_darwin")
    identity = source()
    binary = binary.resolve(strict=True)
    evidence.mkdir(parents=True, exist_ok=False)
    binary_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    for repetition in (1, 2):
        for name in FIXTURES:
            before = disk()
            if before["free_bytes"] < 62 * 1024**3:
                raise ValueError(f"native_start_floor: {before['free_bytes']}")
            if source() != identity:
                raise ValueError("qualification_source_changed")
            command = [str(binary), "--encoder-dir", str(ASSETS / "encoder"), "--tables-dir", str(ASSETS / "tables"),
                       "--fixture", str(ROOT / "certification/fixtures/minicoil" / f"{name}.json")]
            record = {"source": identity, "binary_sha256": binary_digest, "command": command, "fixture": name,
                      "repetition": repetition, "before": before, **observe(command), "after": disk()}
            try:
                record["result"] = json.loads(record["output"].splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as error:
                record["decode_error"] = str(error)
            record["passed"] = record["exit_code"] == 0 and record["refusal"] is None and record.get("result", {}).get("passed") is True
            if source() != identity:
                record["passed"] = False
                record["source_changed"] = True
            (evidence / f"{name}-{repetition}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if not record["passed"]:
                raise ValueError(f"native_fixture_failed: {name} repetition {repetition}")


def purge():
    if not ASSETS.exists():
        return
    before = disk()
    removed = []
    for path in sorted(ASSETS.rglob("*"), reverse=True):
        if path.is_file():
            information = path.lstat()
            if path.is_symlink() or information.st_nlink != 1:
                raise ValueError("purge_ownership_refused")
            path.unlink()
            removed.append({"path": str(path), "size_bytes": information.st_size, "link_count": information.st_nlink, "exit_code": 0})
        elif path.is_dir():
            path.rmdir()
        else:
            raise ValueError("purge_shape_refused")
    ASSETS.rmdir()
    print(json.dumps({"purge": removed, "before": before, "after": disk()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetching = commands.add_parser("fetch")
    fetching.add_argument("--bootstrap-config", action="store_true")
    running = commands.add_parser("run")
    running.add_argument("--binary", type=Path, required=True)
    running.add_argument("--evidence", type=Path, required=True)
    commands.add_parser("purge")
    arguments = parser.parse_args()
    if arguments.command == "fetch":
        fetch(arguments.bootstrap_config)
    elif arguments.command == "run":
        run(arguments.binary, arguments.evidence)
    else:
        purge()


if __name__ == "__main__":
    main()
