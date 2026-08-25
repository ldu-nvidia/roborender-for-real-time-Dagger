#!/usr/bin/env python3
"""Create and verify immutable file inventories for remote learner handoff.

The round coordinator keeps absolute replay paths in its semantic snapshot.
The OCI learner therefore receives an exact copy of the campaign tree and the
job mounts that copy at the original absolute RTX path.  This utility proves
the copy byte-for-byte before it can be mounted, and proves returned learner
artifacts before they can be atomically published on RTX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SCHEMA = "fast_roborender.round_dagger.remote_tree_inventory.v1"
_HEX = frozenset("0123456789abcdef")
_PROJECTION_REQUIRED_FILES = ("campaign.json", "coordinator_state.json")
_PROJECTION_REQUIRED_DIRECTORIES = ("policies", "replay", "rounds")
_PROJECTION_EXCLUDED_TOP_LEVEL = (
    ".coordinator_state.lock",
    "actors",
    "operator",
)


class TransferInventoryError(RuntimeError):
    """A remote-transfer tree or its inventory was unsafe or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise TransferInventoryError(f"inventory entry is not regular: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _safe_root(value: str | os.PathLike[str], *, create: bool = False) -> Path:
    path = Path(os.path.abspath(Path(value).expanduser()))
    if path.is_symlink():
        raise TransferInventoryError("tree root must not be a symlink")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise TransferInventoryError("tree root must be a directory")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise TransferInventoryError("tree root path contains a symlink")
    return path


def _regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(dirnames):
            child = directory_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise TransferInventoryError(
                    f"remote-transfer tree contains an unsafe directory: {child}"
                )
        for name in sorted(filenames):
            child = directory_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise TransferInventoryError(
                    f"remote-transfer tree contains a non-regular file: {child}"
                )
            files.append(child)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def validate_tree_structure(root: str | os.PathLike[str]) -> dict[str, int]:
    """Reject links/special entries without hashing a potentially large tree."""

    canonical = _safe_root(root)
    files = _regular_files(canonical)
    return {
        "file_count": len(files),
        "total_bytes": sum(os.lstat(path).st_size for path in files),
    }


def validate_campaign_projection(root: str | os.PathLike[str]) -> dict[str, int]:
    """Validate the immutable, learner-facing subset of a campaign tree.

    The local actor outputs, operator state, coordinator lock, and per-round
    orchestration telemetry are mutable producer-side state.  They are neither
    learner inputs nor safe remote-transfer material, so a projection that
    contains any of them must fail closed before its inventory is published.
    """

    canonical = _safe_root(root)
    report = validate_tree_structure(canonical)
    for relative in _PROJECTION_REQUIRED_FILES:
        path = canonical / relative
        if not path.is_file() or path.is_symlink():
            raise TransferInventoryError(
                f"campaign projection is missing required file: {relative}"
            )
    for relative in _PROJECTION_REQUIRED_DIRECTORIES:
        path = canonical / relative
        if not path.is_dir() or path.is_symlink():
            raise TransferInventoryError(
                f"campaign projection is missing required directory: {relative}"
            )
    for relative in _PROJECTION_EXCLUDED_TOP_LEVEL:
        if os.path.lexists(canonical / relative):
            raise TransferInventoryError(
                f"campaign projection contains excluded path: {relative}"
            )
    for round_path in (canonical / "rounds").iterdir():
        if round_path.is_dir() and os.path.lexists(round_path / "orchestration_run"):
            relative = (round_path / "orchestration_run").relative_to(canonical)
            raise TransferInventoryError(
                "campaign projection contains excluded path: "
                f"{relative.as_posix()}"
            )
    return report


def build_inventory(root: str | os.PathLike[str]) -> dict[str, Any]:
    canonical = _safe_root(root)
    records: list[dict[str, Any]] = []
    for path in _regular_files(canonical):
        relative = path.relative_to(canonical).as_posix()
        if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise TransferInventoryError("inventory produced an unsafe relative path")
        info = os.lstat(path)
        records.append(
            {
                "path": relative,
                "size": info.st_size,
                "sha256": _sha256_file(path),
            }
        )
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    body["tree_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def _validate_inventory(document: Any) -> dict[str, Any]:
    keys = {"schema", "file_count", "total_bytes", "files", "tree_sha256"}
    if not isinstance(document, Mapping) or set(document) != keys:
        raise TransferInventoryError("remote tree inventory key-set mismatch")
    result = dict(document)
    if result["schema"] != SCHEMA:
        raise TransferInventoryError("remote tree inventory schema mismatch")
    if type(result["file_count"]) is not int or result["file_count"] < 0:
        raise TransferInventoryError("inventory file_count is invalid")
    if type(result["total_bytes"]) is not int or result["total_bytes"] < 0:
        raise TransferInventoryError("inventory total_bytes is invalid")
    if not isinstance(result["files"], list):
        raise TransferInventoryError("inventory files must be a list")
    paths: list[str] = []
    for item in result["files"]:
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise TransferInventoryError("inventory file record key-set mismatch")
        relative = item["path"]
        parsed = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("/")
        if (
            not isinstance(relative, str)
            or not relative
            or parsed.is_absolute()
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise TransferInventoryError("inventory contains an unsafe path")
        if type(item["size"]) is not int or item["size"] < 0:
            raise TransferInventoryError("inventory file size is invalid")
        digest = item["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
        ):
            raise TransferInventoryError("inventory file SHA-256 is invalid")
        paths.append(relative)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise TransferInventoryError("inventory paths are not unique and sorted")
    if result["file_count"] != len(result["files"]):
        raise TransferInventoryError("inventory file_count mismatch")
    if result["total_bytes"] != sum(item["size"] for item in result["files"]):
        raise TransferInventoryError("inventory total_bytes mismatch")
    tree_digest = result.pop("tree_sha256")
    encoded = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    if tree_digest != expected:
        raise TransferInventoryError("inventory semantic tree digest mismatch")
    result["tree_sha256"] = tree_digest
    return result


def load_inventory(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TransferInventoryError("inventory must be a regular non-symlink file")
    try:
        with source.open("r", encoding="utf-8") as handle:
            return _validate_inventory(json.load(handle))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransferInventoryError(
            f"cannot read strict remote tree inventory ({type(exc).__name__})"
        ) from exc


def verify_inventory(
    root: str | os.PathLike[str], inventory: str | os.PathLike[str]
) -> dict[str, Any]:
    expected = load_inventory(inventory)
    observed = build_inventory(root)
    if observed != expected:
        raise TransferInventoryError("remote tree differs from its immutable inventory")
    return observed


def publish_inventory(path: str | os.PathLike[str], document: Mapping[str, Any]) -> None:
    validated = _validate_inventory(document)
    destination = Path(os.path.abspath(Path(path).expanduser()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(validated, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise TransferInventoryError("refusing to replace a different inventory")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != payload:
                raise TransferInventoryError("competing inventory publication differs")
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def publish_tree(
    stage: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    inventory: str | os.PathLike[str],
) -> dict[str, Any]:
    stage_path = _safe_root(stage)
    expected = verify_inventory(stage_path, inventory)
    destination_path = Path(os.path.abspath(Path(destination).expanduser()))
    if (
        stage_path == destination_path
        or stage_path in destination_path.parents
        or destination_path in stage_path.parents
    ):
        raise TransferInventoryError(
            "tree stage and destination must be disjoint paths"
        )
    if destination_path.is_symlink():
        raise TransferInventoryError("tree destination must not be a symlink")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        observed = verify_inventory(destination_path, inventory)
        if observed != expected:
            raise TransferInventoryError("existing destination has different bytes")
        # The destination already won publication.  An exact redundant stage
        # is safe to remove and must not survive to obstruct a resumed rsync.
        shutil.rmtree(stage_path)
        directory = os.open(
            destination_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return observed
    os.rename(stage_path, destination_path)
    directory = os.open(
        destination_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return verify_inventory(destination_path, inventory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--root", required=True)
    inventory.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--inventory", required=True)
    structure = sub.add_parser("validate-structure")
    structure.add_argument("--root", required=True)
    projection = sub.add_parser("validate-projection")
    projection.add_argument("--root", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--stage", required=True)
    publish.add_argument("--destination", required=True)
    publish.add_argument("--inventory", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "inventory":
        report = build_inventory(arguments.root)
        publish_inventory(arguments.output, report)
    elif arguments.command == "verify":
        report = verify_inventory(arguments.root, arguments.inventory)
    elif arguments.command == "validate-structure":
        report = validate_tree_structure(arguments.root)
    elif arguments.command == "validate-projection":
        report = validate_campaign_projection(arguments.root)
    else:
        report = publish_tree(
            arguments.stage, arguments.destination, arguments.inventory
        )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TransferInventoryError as exc:
        print(f"remote campaign transfer failed: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
