#!/usr/bin/env python3
"""Fail-closed, content-addressed verification for the production anchor pool.

The inventory algorithm is intentionally identical to
``scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py::inventory_sha256``:
for every NPZ in lexicographic relative-path order, hash the 8-byte big-endian
path length, UTF-8 path bytes, 8-byte big-endian file size, and the 32 binary
bytes of the file SHA-256.  This avoids delimiter ambiguity while binding every
anchor name, size, and byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence


SHA256_RE = re.compile(r"[0-9a-f]{64}")
SCHEMA = "fast_roborender.round_dagger.anchor_pool_verification.v1"
PRODUCTION_ANCHOR_COUNT = 2048
PRODUCTION_INVENTORY_SHA256 = (
    "ee358a97581512a5d03730d06efd5804e53844f517112dc0a845ed86b6dae46f"
)


class AnchorPoolError(RuntimeError):
    """Raised when the anchor pool is not the exact immutable input set."""


@dataclass(frozen=True)
class AnchorRecord:
    relative_path: str
    size: int
    sha256: str


def _file_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _safe_file_record(root: Path, path: Path) -> AnchorRecord:
    relative = path.relative_to(root).as_posix()
    if not relative or "\t" in relative or "\n" in relative or "\r" in relative:
        raise AnchorPoolError(f"unsafe anchor relative path: {relative!r}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AnchorPoolError(f"cannot safely open anchor {relative}: {exc}") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AnchorPoolError(f"anchor is not a regular file: {relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = _file_sha256(stream)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise AnchorPoolError(f"anchor mutated while hashing: {relative}")
    return AnchorRecord(relative, before.st_size, digest)


def _walk(root: Path) -> tuple[list[Path], tuple[tuple[str, int, int, int, int], ...]]:
    anchors: list[Path] = []
    tree: list[tuple[str, int, int, int, int]] = []
    try:
        entries = sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix())
    except OSError as exc:
        raise AnchorPoolError(f"cannot enumerate anchor pool: {exc}") from None
    for path in entries:
        relative = path.relative_to(root).as_posix()
        try:
            info = path.lstat()
        except OSError as exc:
            raise AnchorPoolError(f"cannot inspect anchor-pool entry {relative}: {exc}") from None
        if stat.S_ISLNK(info.st_mode):
            raise AnchorPoolError(f"symbolic links are forbidden in anchor pool: {relative}")
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise AnchorPoolError(f"special files are forbidden in anchor pool: {relative}")
        tree.append((relative, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        if stat.S_ISREG(info.st_mode) and path.suffix == ".npz":
            anchors.append(path)
    return anchors, tuple(tree)


def inventory_sha256(records: Sequence[AnchorRecord]) -> str:
    """Return the established, unambiguous production inventory digest."""
    digest = hashlib.sha256()
    for record in records:
        relative = record.relative_path.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(record.size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(record.sha256))
    return digest.hexdigest()


def verify_anchor_pool(
    root: Path,
    *,
    expected_count: int,
    expected_sha256: str,
) -> dict[str, object]:
    if type(expected_count) is not int or expected_count <= 0:
        raise AnchorPoolError("expected anchor count must be a positive integer")
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise AnchorPoolError("expected inventory SHA-256 must be lowercase hexadecimal")
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise AnchorPoolError(f"cannot inspect anchor-pool root {root}: {exc}") from None
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise AnchorPoolError("anchor-pool root must be a real directory, not a symlink")
    root = root.resolve(strict=True)

    anchors, tree_before = _walk(root)
    if len(anchors) != expected_count:
        raise AnchorPoolError(
            f"anchor pool contains {len(anchors)} NPZ files; expected exactly {expected_count}"
        )
    records = [_safe_file_record(root, path) for path in anchors]
    # Detect additions, removals, replacements, or metadata mutations that race
    # the scan.  The per-file fd checks above cover mutations during each hash.
    anchors_after, tree_after = _walk(root)
    if tree_after != tree_before or [p.relative_to(root) for p in anchors_after] != [
        p.relative_to(root) for p in anchors
    ]:
        raise AnchorPoolError("anchor pool mutated while inventory was calculated")

    actual = inventory_sha256(records)
    if actual != expected_sha256:
        raise AnchorPoolError(
            f"anchor inventory SHA-256 mismatch: {actual} != {expected_sha256}"
        )
    return {
        "schema": SCHEMA,
        "status": "verified",
        "root": str(root),
        "anchor_count": len(records),
        "total_file_bytes": sum(record.size for record in records),
        "inventory_sha256": actual,
        "algorithm": "len64be(path_utf8)+path_utf8+size64be+sha256_binary",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_anchor_pool(
            args.root,
            expected_count=args.expected_count,
            expected_sha256=args.expected_sha256,
        )
    except AnchorPoolError as exc:
        print(f"anchor-pool verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
