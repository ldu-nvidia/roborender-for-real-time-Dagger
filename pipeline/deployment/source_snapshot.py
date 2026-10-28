"""Source-only release snapshots and cross-host drift verification.

No Git checkout/reset, network access, asset copying, or GPU initialization.
Run before deployment, never in the per-step training path.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile

SCHEMA = "roborender.source-snapshot.v1"
MAX_FILE_BYTES = 8 << 20
SUFFIXES = {".py", ".md", ".sh", ".json", ".toml", ".txt", ".patch", ".lock", ".yaml", ".yml", ".sbatch", ".ini"}
SPECIAL_NAMES = {".gitignore", ".gitattributes", ".dockerignore", "Dockerfile"}
FORBIDDEN_PARTS = {".git", ".ssh", "runtime", "secrets", "credentials", "__pycache__", "wandb", "logs", "venv", ".venv"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe_relative(name: str) -> Path:
    value = PurePosixPath(name)
    if not name or value.is_absolute() or ".." in value.parts or str(value) != name or "\\" in name:
        raise ValueError(f"unsafe source path: {name!r}")
    if any(part in FORBIDDEN_PARTS for part in value.parts):
        raise ValueError(f"host-local artifact in source set: {name}")
    if not (value.suffix in SUFFIXES or value.name in SPECIAL_NAMES or value.name.startswith("Dockerfile.")):
        raise ValueError(f"non-source file in source set: {name}")
    if value.parts[0] in {"data", "checkpoints", "renderer"} and value.suffix not in {".md", ".json"} and value.name != ".gitattributes":
        raise ValueError(f"asset payload is not source: {name}")
    return Path(name)


def source_names(root: Path) -> list[str]:
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
    ).decode().split("\0")
    # Unstaged removals/renames remain in `git ls-files --cached`. Inventory the
    # current working tree, including new paths, without resurrecting removed
    # tests or legacy files. Verification against an older manifest still fails
    # on every removed/renamed path; this is not an omission from drift checks.
    deleted = subprocess.check_output(
        ["git", "ls-files", "-z", "--deleted"], cwd=root
    ).decode().split("\0")
    return sorted(set(filter(None, names)) - set(filter(None, deleted)))


def file_record(root: Path, name: str) -> dict:
    relative = safe_relative(name)
    path = root / relative
    # Refuse symlinks in the file or any parent; assets must never leak through
    # an apparently innocuous source path.
    if any((root / parent).is_symlink() for parent in (relative, *relative.parents)):
        raise ValueError(f"source symlink is forbidden: {name}")
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode) or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"not a bounded regular source file: {name}")
    raw = path.read_bytes()
    raw.decode("utf-8")
    if re.search(rb"(?m)^-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----", raw):
        raise ValueError(f"private key in source file: {name}")
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "executable": bool(mode & 0o111)}


def inventory(root: Path) -> dict:
    root = root.resolve(strict=True)
    rows = [file_record(root, name) for name in source_names(root)]
    return {"schema": SCHEMA, "source_sha256": hashlib.sha256(canonical(rows)).hexdigest(),
            "file_count": len(rows), "files": rows}


def verify(root: Path, manifest: dict) -> dict:
    if set(manifest) != {"schema", "source_sha256", "file_count", "files"} or manifest["schema"] != SCHEMA:
        raise ValueError("invalid source manifest")
    rows = manifest["files"]
    if not isinstance(rows, list) or manifest["file_count"] != len(rows):
        raise ValueError("invalid manifest file count")
    if hashlib.sha256(canonical(rows)).hexdigest() != manifest["source_sha256"]:
        raise ValueError("source manifest digest mismatch")
    names = [row["path"] for row in rows]
    if names != sorted(set(names)):
        raise ValueError("manifest paths must be unique and sorted")
    actual = inventory(root)
    if actual != manifest:
        expected_rows = {r["path"]: r for r in rows}
        actual_rows = {r["path"]: r for r in actual["files"]}
        changed = sorted(k for k in expected_rows.keys() | actual_rows.keys()
                         if expected_rows.get(k) != actual_rows.get(k))
        raise ValueError(f"source drift: {changed}")
    return {"ok": True, "source_sha256": actual["source_sha256"], "file_count": len(rows)}


def create(root: Path, output: Path) -> dict:
    manifest = inventory(root)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source.tar.gz"
    # Pack only the small reviewed source set. Read each file again and compare
    # against its manifest so concurrent edits cannot create a mixed release.
    with tarfile.open(archive, "w:gz") as bundle:
        for row in manifest["files"]:
            raw = (root / row["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"source changed during snapshot: {row['path']}")
            info = tarfile.TarInfo(row["path"])
            info.size = len(raw)
            info.mode = 0o755 if row["executable"] else 0o644
            bundle.addfile(info, io.BytesIO(raw))
    verify(root, manifest)
    (output / "source-manifest.json").write_bytes(canonical(manifest) + b"\n")
    (output / "source-files.txt").write_text("".join(row["path"] + "\n" for row in manifest["files"]))
    return {"source_sha256": manifest["source_sha256"], "files": manifest["file_count"],
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "create", "verify"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.action == "create":
        if args.output is None:
            parser.error("create requires --output (a new directory)")
        result = create(root, args.output)
    elif args.action == "verify":
        if args.manifest is None:
            parser.error("verify requires --manifest")
        result = verify(root, json.loads(args.manifest.read_text()))
    else:
        result = inventory(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
