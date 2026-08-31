"""Verified, resumable cohort handoff between RTX collection and H100 replay."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .artifacts import sha256_file, verify_cohort


class HandoffError(RuntimeError):
    """A cohort transfer was partial, unsafe, corrupt, or conflicting."""


def _safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise HandoffError("artifact path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise HandoffError(f"unsafe artifact path: {value!r}")
    return Path(*path.parts)


def _expected_files(report: Mapping[str, Any]) -> dict[Path, str]:
    expected = {
        Path("contract.json"): str(report["contract_sha256"]),
        Path("seal.json"): str(report["cohort_sha256"]),
    }
    for shard in report["shards"]:
        expected[_safe_relative(shard["manifest_path"])] = str(
            shard["manifest_sha256"]
        )
        expected[_safe_relative(shard["payload_path"])] = str(shard["payload_sha256"])
    return expected


def _absolute_directory(path: str | Path, *, create: bool) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise HandoffError("handoff paths must be absolute")
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    try:
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HandoffError(f"missing or unsafe directory: {raw}") from exc
    if raw.is_symlink() or not resolved.is_dir():
        raise HandoffError(f"handoff directory is unsafe: {raw}")
    return resolved


def _copy_one(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy one regular file atomically; an exact staged file is resumable."""

    if source.is_symlink() or not source.is_file():
        raise HandoffError(f"source artifact is not a safe regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise HandoffError(f"staged artifact is a symbolic link: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise HandoffError(f"staged artifact is not regular: {destination}")
        if sha256_file(destination) == expected_sha256:
            return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            with os.fdopen(source_fd, "rb", closefd=True) as source_handle, os.fdopen(
                descriptor, "wb", closefd=True
            ) as destination_handle:
                descriptor = -1
                shutil.copyfileobj(source_handle, destination_handle, 8 * 1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        finally:
            # fdopen owns source_fd after it succeeds.
            pass
        if sha256_file(temporary) != expected_sha256:
            raise HandoffError(f"copied artifact digest mismatch: {source.name}")
        os.replace(temporary, destination)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def stage_cohort(source: str | Path, stage: str | Path) -> dict[str, Any]:
    """Copy a sealed cohort into a deterministic resumable staging directory."""

    source_root = _absolute_directory(source, create=False)
    report = verify_cohort(source_root)
    stage_path = Path(stage).expanduser()
    if not stage_path.is_absolute():
        raise HandoffError("stage path must be absolute")
    parent = _absolute_directory(stage_path.parent, create=True)
    stage_path = parent / stage_path.name
    if not stage_path.name.startswith(".incoming-"):
        raise HandoffError("stage directory must use the .incoming- prefix")
    if stage_path.is_symlink():
        raise HandoffError("stage directory must not be a symlink")
    stage_path.mkdir(mode=0o700, exist_ok=True)
    if not stage_path.is_dir():
        raise HandoffError("stage path is not a directory")

    for relative, digest in sorted(_expected_files(report).items()):
        _copy_one(source_root / relative, stage_path / relative, digest)
    staged = verify_exact_stage(stage_path)
    if staged["cohort_sha256"] != report["cohort_sha256"]:
        raise HandoffError("staged cohort identity differs from source")
    return staged


def verify_exact_stage(stage: str | Path) -> dict[str, Any]:
    """Verify every byte and reject unreferenced files in a handoff stage."""

    root = _absolute_directory(stage, create=False)
    report = verify_cohort(root)
    expected = set(_expected_files(report))
    actual: set[Path] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            child = Path(directory) / name
            if child.is_symlink():
                raise HandoffError("symbolic links are forbidden in a handoff stage")
        for name in filenames:
            child = Path(directory) / name
            if child.is_symlink() or not stat.S_ISREG(os.lstat(child).st_mode):
                raise HandoffError("handoff stage contains a non-regular file")
            actual.add(child.relative_to(root))
    if actual != expected:
        raise HandoffError(
            f"handoff stage file set mismatch: missing={sorted(expected-actual)} "
            f"extra={sorted(actual-expected)}"
        )
    return report


def publish_stage(stage: str | Path, destination: str | Path) -> dict[str, Any]:
    """Atomically publish one verified stage, with exact idempotent retry."""

    stage_path = _absolute_directory(stage, create=False)
    report = verify_exact_stage(stage_path)
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        raise HandoffError("destination path must be absolute")
    parent = _absolute_directory(destination_path.parent, create=True)
    destination_path = parent / destination_path.name
    if destination_path.is_symlink():
        raise HandoffError("destination must not be a symlink")
    if destination_path.exists():
        existing = verify_exact_stage(destination_path)
        if existing["cohort_sha256"] != report["cohort_sha256"]:
            raise HandoffError("destination already contains a different cohort")
        shutil.rmtree(stage_path)
        return existing
    try:
        os.rename(stage_path, destination_path)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        if not destination_path.exists():
            raise
        existing = verify_exact_stage(destination_path)
        if existing["cohort_sha256"] != report["cohort_sha256"]:
            raise HandoffError("concurrent handoff published a different cohort") from exc
        shutil.rmtree(stage_path)
        return existing
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return verify_exact_stage(destination_path)


def transfer_local(
    source: str | Path, destination: str | Path, *, stage_name: str | None = None
) -> dict[str, Any]:
    """Resumably stage and atomically publish a cohort on a local/shared FS."""

    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        raise HandoffError("destination path must be absolute")
    source_report = verify_cohort(source)
    suffix = source_report["cohort_sha256"][:16]
    chosen = stage_name or f".incoming-{destination_path.name}-{suffix}"
    if "/" in chosen or chosen in {".", ".."}:
        raise HandoffError("stage_name must be one safe path component")
    stage = destination_path.parent / chosen
    stage_cohort(source, stage)
    return publish_stage(stage, destination_path)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("local")
    local.add_argument("--source", required=True)
    local.add_argument("--destination", required=True)
    verify = subparsers.add_parser("verify-stage")
    verify.add_argument("--stage", required=True)
    publish = subparsers.add_parser("publish-stage")
    publish.add_argument("--stage", required=True)
    publish.add_argument("--destination", required=True)
    args = parser.parse_args()
    if args.command == "local":
        report = transfer_local(args.source, args.destination)
    elif args.command == "verify-stage":
        report = verify_exact_stage(args.stage)
    else:
        report = publish_stage(args.stage, args.destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "HandoffError",
    "publish_stage",
    "stage_cohort",
    "transfer_local",
    "verify_exact_stage",
]
