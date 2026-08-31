"""Cross-UID handoff for RTX actor run telemetry.

Production actors run in a root-owned Isaac container, while the cohort
supervisor and recovery watcher run as the unprivileged host operator.  A
container must never be allowed to create the shared telemetry directories
with ownership that prevents the supervisor from publishing the cohort seal,
quality projection, or terminal evidence.

``prepare_host_layout`` is run by the host before any actor container starts.
It creates every shared mutable directory and proves that the host can publish
into it.  ``normalize_container_outputs`` is the deliberately narrow root
operation used after an older/root-owned container has exited: it changes only
W&B/telemetry ownership and modes below one already host-owned run directory.
It never visits the cohort/shard tree.  The host then calls
``verify_host_access`` again before accepting or recovering a sealed cohort.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any


SCHEMA = "fast_roborender.round_dagger.actor_host_access.v1"

# These directories are created by the host before the root-owned actor/W&B
# sidecar starts.  Root can publish into them, but it cannot strand a directory
# that the host needs later for finalization or recovery.
HOST_MUTABLE_DIRECTORIES = (
    "logs",
    "actor_pair_attempts",
    "quality",
    "ranks",
    "telemetry",
    "telemetry/scalars",
    "telemetry/images",
    "telemetry/videos",
    ".wandb_media",
)

WANDB_RECREATABLE_TOP_LEVEL_SYMLINKS = frozenset(
    {
        "wandb/latest-run",
        "wandb/debug.log",
        "wandb/debug-internal.log",
    }
)

class ActorHostAccessError(RuntimeError):
    """The host/container ownership boundary is unsafe or unusable."""


def _absolute_run_root(value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ActorHostAccessError("actor run directory is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ActorHostAccessError(
            "actor run directory must be a non-symlink directory"
        )
    if path.resolve(strict=True) != path:
        raise ActorHostAccessError("actor run directory must be canonical")
    return path


def _safe_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ActorHostAccessError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ActorHostAccessError(f"{label} must be a non-symlink directory")
    return info


def _mkdir_below_run(root: Path, relative: str) -> Path:
    current = root
    for component in Path(relative).parts:
        current /= component
        try:
            os.mkdir(current, 0o700)
        except FileExistsError:
            pass
        _safe_directory(current, label=f"actor run layout {relative}")
    return current


def _host_publish_probe(path: Path) -> None:
    """Create, fsync, and remove one unpredictable file through a directory fd."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(path, flags)
    except OSError as exc:
        raise ActorHostAccessError(
            f"host cannot open actor run layout directory: {path.name}"
        ) from exc
    name = f".host_access_probe_{os.getpid()}_{secrets.token_hex(8)}"
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            os.write(descriptor, b"host-access\n")
            os.fsync(descriptor)
        except OSError as exc:
            raise ActorHostAccessError(
                f"host cannot publish into actor run layout directory: {path.name}"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _iter_tree(root: Path) -> list[Path]:
    """Return a no-symlink, no-special-file tree in children-first order."""

    if not root.exists():
        return []
    _safe_directory(root, label=f"managed actor directory {root.name}")
    result: list[Path] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        _safe_directory(current_path, label="managed actor directory")
        for name in sorted(directories):
            child = current_path / name
            _safe_directory(child, label="managed actor subdirectory")
        for name in sorted(files):
            child = current_path / name
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ActorHostAccessError(
                    "managed actor output contains a symlink or special file"
                )
            result.append(child)
        result.extend(current_path / name for name in sorted(directories))
    # Files before their containing directories, deepest paths first.
    return sorted(result, key=lambda item: (-len(item.parts), str(item)))


def _is_recreatable_wandb_symlink(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if relative.as_posix() in WANDB_RECREATABLE_TOP_LEVEL_SYMLINKS:
        return True
    return bool(
        len(relative.parts) == 4
        and relative.parts[0] == "wandb"
        and re.fullmatch(
            r"run-[0-9]{8}_[0-9]{6}-[A-Za-z0-9_-]+", relative.parts[1]
        )
        and relative.parts[2:] == ("logs", "debug-core.log")
    )


def _remove_recreatable_wandb_symlinks(root: Path) -> int:
    """Unlink only W&B's four canonical link shapes, never their targets."""

    candidates: list[Path] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        _safe_directory(current_path, label="W&B working directory")
        for name in sorted(list(directories)):
            child = current_path / name
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode):
                if not _is_recreatable_wandb_symlink(root, child):
                    raise ActorHostAccessError(
                        "W&B working state contains an unknown symlink"
                    )
                directories.remove(name)
                candidates.append(child)
            elif not stat.S_ISDIR(info.st_mode):
                raise ActorHostAccessError(
                    "W&B working state contains a special directory entry"
                )
        for name in sorted(files):
            child = current_path / name
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode):
                if not _is_recreatable_wandb_symlink(root, child):
                    raise ActorHostAccessError(
                        "W&B working state contains an unknown symlink"
                    )
                candidates.append(child)
            elif not stat.S_ISREG(info.st_mode):
                raise ActorHostAccessError(
                    "W&B working state contains a special file"
                )
    for path in candidates:
        info = os.lstat(path)
        if not stat.S_ISLNK(info.st_mode):
            raise ActorHostAccessError("W&B symbolic link changed before removal")
        os.unlink(path)
    return len(candidates)


def _require_host_readable(path: Path, *, host_uid: int, host_gid: int) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ActorHostAccessError("actor telemetry contains an unsafe file")
    mode = stat.S_IMODE(info.st_mode)
    readable = (
        bool(mode & stat.S_IRUSR)
        if info.st_uid == host_uid
        else (
            bool(mode & stat.S_IRGRP)
            if info.st_gid == host_gid
            else bool(mode & stat.S_IROTH)
        )
    )
    if not readable:
        raise ActorHostAccessError(
            "actor telemetry is not readable by the host identity"
        )


def verify_host_access(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Prove the current host identity can finalize and recover this run."""

    root = _absolute_run_root(run_dir)
    expected_uid = os.geteuid()
    expected_gid = os.getegid()
    root_info = _safe_directory(root, label="actor run directory")
    if root_info.st_uid != expected_uid:
        raise ActorHostAccessError(
            "actor run directory is not owned by the current host identity"
        )
    directories: list[dict[str, Any]] = []
    for relative in HOST_MUTABLE_DIRECTORIES:
        path = root / relative
        info = _safe_directory(path, label=f"actor run layout {relative}")
        if info.st_uid != expected_uid or not stat.S_IMODE(info.st_mode) & stat.S_IWUSR:
            raise ActorHostAccessError(
                f"actor run layout is not host-writable: {relative}"
            )
        _host_publish_probe(path)
        directories.append(
            {
                "path": relative,
                "uid": info.st_uid,
                "gid": info.st_gid,
                "mode": f"{stat.S_IMODE(info.st_mode):04o}",
            }
        )

    telemetry = root / "telemetry"
    readable_files = 0
    for path in _iter_tree(telemetry):
        if path.is_dir():
            continue
        _require_host_readable(path, host_uid=expected_uid, host_gid=expected_gid)
        readable_files += 1

    # W&B's private working tree is mutable rather than an immutable telemetry
    # surface.  Recovery executes as the host UID and therefore needs ownership
    # of every existing entry, not merely write access to its top-level parent.
    wandb_root = root / ".wandb_media"
    for path in [wandb_root, *_iter_tree(wandb_root)]:
        info = os.lstat(path)
        if info.st_uid != expected_uid:
            raise ActorHostAccessError(
                "W&B working state is not owned by the current host identity"
            )
        if stat.S_ISDIR(info.st_mode):
            if not stat.S_IMODE(info.st_mode) & stat.S_IWUSR:
                raise ActorHostAccessError("W&B working directory is not host-writable")
        elif not stat.S_IMODE(info.st_mode) & stat.S_IWUSR:
            raise ActorHostAccessError("W&B working file is not host-writable")

    return {
        "schema": SCHEMA,
        "operation": "verify",
        "status": "HOST_ACCESSIBLE",
        "run_dir": str(root),
        "host_uid": expected_uid,
        "host_gid": expected_gid,
        "directories": directories,
        "cross_identity_readable_files": readable_files,
    }


def prepare_host_layout(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Create the host-owned layout before any root-owned container starts."""

    root = _absolute_run_root(run_dir)
    root_info = _safe_directory(root, label="actor run directory")
    if root_info.st_uid != os.geteuid():
        raise ActorHostAccessError(
            "actor run directory is not owned by the current host identity"
        )
    for relative in HOST_MUTABLE_DIRECTORIES:
        _mkdir_below_run(root, relative)
    report = verify_host_access(root)
    return {**report, "operation": "prepare"}


def _chown_chmod(path: Path, *, uid: int, gid: int, mode: int) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise ActorHostAccessError("managed actor output contains a symbolic link")
    os.chown(path, uid, gid, follow_symlinks=False)
    os.chmod(path, mode, follow_symlinks=False)


def normalize_container_outputs(
    run_dir: str | os.PathLike[str], *, host_uid: int, host_gid: int
) -> dict[str, Any]:
    """Normalize only container-created telemetry/W&B state for host recovery."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (host_uid, host_gid)
    ):
        raise ActorHostAccessError("host UID and GID must be positive integers")
    root = _absolute_run_root(run_dir)
    root_info = _safe_directory(root, label="actor run directory")
    if root_info.st_uid != host_uid or root_info.st_gid != host_gid:
        raise ActorHostAccessError(
            "requested host identity does not own the actor run directory"
        )

    # Materialize missing layout components without visiting any cohort/shard
    # path, then hand their directory inodes back to the host identity.
    for relative in HOST_MUTABLE_DIRECTORIES:
        path = _mkdir_below_run(root, relative)
        _chown_chmod(path, uid=host_uid, gid=host_gid, mode=0o700)

    telemetry = root / "telemetry"
    telemetry_entries = _iter_tree(telemetry)
    for path in telemetry_entries:
        info = os.lstat(path)
        mode = 0o700 if stat.S_ISDIR(info.st_mode) else 0o444
        _chown_chmod(path, uid=host_uid, gid=host_gid, mode=mode)
    _chown_chmod(telemetry, uid=host_uid, gid=host_gid, mode=0o700)

    wandb_root = root / ".wandb_media"
    removed_wandb_symlinks = _remove_recreatable_wandb_symlinks(wandb_root)
    wandb_entries = _iter_tree(wandb_root)
    for path in wandb_entries:
        info = os.lstat(path)
        mode = 0o700 if stat.S_ISDIR(info.st_mode) else 0o600
        _chown_chmod(path, uid=host_uid, gid=host_gid, mode=mode)
    _chown_chmod(wandb_root, uid=host_uid, gid=host_gid, mode=0o700)

    return {
        "schema": SCHEMA,
        "operation": "normalize",
        "status": "NORMALIZED",
        "run_dir": str(root),
        "host_uid": host_uid,
        "host_gid": host_gid,
        "telemetry_entries": len(telemetry_entries),
        "wandb_entries": len(wandb_entries),
        "removed_wandb_symlinks": removed_wandb_symlinks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("prepare", "verify"):
        command = subparsers.add_parser(operation)
        command.add_argument("--run-dir", required=True, type=Path)
    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--run-dir", required=True, type=Path)
    normalize.add_argument("--host-uid", required=True, type=int)
    normalize.add_argument("--host-gid", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.operation == "prepare":
            report = prepare_host_layout(arguments.run_dir)
        elif arguments.operation == "verify":
            report = verify_host_access(arguments.run_dir)
        else:
            report = normalize_container_outputs(
                arguments.run_dir,
                host_uid=arguments.host_uid,
                host_gid=arguments.host_gid,
            )
    except (ActorHostAccessError, OSError) as exc:
        detail = str(exc) if isinstance(exc, ActorHostAccessError) else type(exc).__name__
        print(f"actor host-access handoff failed: {detail}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ActorHostAccessError",
    "HOST_MUTABLE_DIRECTORIES",
    "SCHEMA",
    "normalize_container_outputs",
    "prepare_host_layout",
    "verify_host_access",
]
