"""Metadata-only inventory of alpha's RoboRender trees; never opens payloads.

The output is a migration inventory, not a checksum or a deletion allowlist.
Authentication material is explicitly excluded and must be reprovisioned.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fnmatch
import json
import os
from pathlib import Path
import socket
import stat


SEARCH_ROOTS = [
    "/mnt/data1/ldu", "/mnt/data2/ldu",
    "/mnt/data1/ldu/research", "/mnt/data2/ldu/research",
    "/mnt/data1/ldu/research/artifacts", "/mnt/data2/ldu/research/artifacts",
]
PATTERNS = [
    "*roborender*", "*dagger*", "BC_render", "raven5*", "raven_5step*",
    "fixed_panel*", "rtx_compact_audit*", "h200_multisession_pool*",
    "rdb4_codec*", "v4_exchange_optimization*", "eval-bc-*",
]
PRIVATE_PARTS = {".ssh", "secrets", "credentials", "roborender_round_dagger_credentials"}
PRIVATE_NAMES = {".netrc", ".git-credentials", ".env", "workspace_info.yaml", "workspace_info.json"}


def private(path: Path) -> bool:
    name = path.name.lower()
    return (bool(PRIVATE_PARTS.intersection(path.parts)) or name in PRIVATE_NAMES
            or name.endswith((".pem", ".key")) or "bearer" in name
            or name in {"token", "token.txt", "auth_token", "auth_token.txt"})


def category(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine", ".plan")):
        return "model_checkpoint_engine"
    if name.endswith((".npz", ".npy", ".h5", ".hdf5", ".parquet")):
        return "data"
    if name.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".zst", ".bundle", ".sif")):
        return "archive"
    if ".git" in path.parts:
        return "git_history"
    if any(p in {".venv", "__pycache__", ".pytest_cache", "persistent-cache"} for p in path.parts):
        return "environment_or_cache"
    return "source_metadata_media_other"


def discover() -> list[Path]:
    found = []
    for root in SEARCH_ROOTS:
        if not Path(root).is_dir():
            continue
        found.extend(p for p in Path(root).iterdir()
                     if any(fnmatch.fnmatch(p.name.lower(), pattern.lower()) for pattern in PATTERNS))
    roots = []
    for path in sorted(set(found), key=lambda p: (len(p.parts), str(p))):
        if not any(path.is_relative_to(parent) for parent in roots):
            roots.append(path)
    return roots


def inventory() -> dict:
    if socket.gethostname() != "8f68286-lcedt":
        raise RuntimeError("inventory is restricted to RTX alpha")
    roots = discover()
    rows, links, errors, omitted = [], [], [], []
    seen = set()
    totals, by_root = Counter(), defaultdict(Counter)

    def visit(path: Path, root: Path):
        try:
            info = path.lstat()
            if private(path):
                omitted.append(str(path))
                return
            if stat.S_ISLNK(info.st_mode):
                links.append({"path": str(path), "target": os.readlink(path)})
            elif stat.S_ISREG(info.st_mode):
                identity = (info.st_dev, info.st_ino)
                unique = identity not in seen
                seen.add(identity)
                kind = category(path)
                rows.append({"path": str(path), "root": str(root), "bytes": info.st_size,
                             "mtime_ns": info.st_mtime_ns, "device": info.st_dev,
                             "inode": info.st_ino, "links": info.st_nlink, "category": kind})
                by_root[str(root)]["files"] += 1
                by_root[str(root)]["logical_bytes"] += info.st_size
                if unique:
                    totals[kind] += info.st_size
                    by_root[str(root)]["unique_inode_bytes"] += info.st_size
                    by_root[str(root)]["allocated_unique_inode_bytes"] += info.st_blocks * 512
        except OSError as exc:
            errors.append({"path": str(path), "error": str(exc)})

    for root in roots:
        if private(root):
            omitted.append(str(root))
            continue
        if root.is_symlink() or not root.is_dir():
            visit(root, root)
            continue
        def onerror(exc):
            errors.append({"path": exc.filename, "error": str(exc)})
        for parent, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
            for name in list(dirs):
                path = Path(parent) / name
                if private(path):
                    omitted.append(str(path))
                    dirs.remove(name)
                elif path.is_symlink():
                    visit(path, root)
                    dirs.remove(name)
            for name in files:
                visit(Path(parent) / name, root)
    return {"schema": "roborender.alpha-migration-inventory.v1", "host": socket.gethostname(),
            "metadata_only": True, "payload_content_hashes_verified": False,
            "roots": [str(p) for p in roots], "files": rows, "symlinks": links,
            "errors": errors, "private_paths_not_migrated": omitted,
            "summary": {"files": len(rows), "unique_inodes": len(seen),
                        "unique_inode_bytes": sum(totals.values()), "by_category": totals,
                        "by_root": dict(by_root), "symlinks": len(links),
                        "errors": len(errors), "private_paths_omitted": len(omitted)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result = inventory()
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    args.output.chmod(0o600)
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
