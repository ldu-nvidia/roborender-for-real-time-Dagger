"""Immutable terminal evidence and W&B scalar telemetry for one actor-pair attempt.

The actor-pair shell supervisor owns process reaping, while this module owns the
small durable record that explains the outcome.  Publishing the record, its
numeric W&B projection, and (optionally) the sidecar finish marker in one
ordered operation prevents a failed collection from looking like a successful
or still-running W&B job.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .actor_pair import ActorPairError, inspect_actor_pair, inspect_actor_rank
from .artifacts import load_strict_json, sha256_file


TERMINAL_SCHEMA = "fast_roborender.round_dagger.actor_pair_terminal.v1"
SCALAR_SCHEMA = "fast_roborender.round_dagger.actor_pair_terminal_scalar.v1"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class ActorPairTerminalError(RuntimeError):
    """Terminal evidence could not be published safely."""


def _check_path_chain(path: Path, label: str) -> None:
    """Reject every existing symlink or non-directory ancestor."""

    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(parts):
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ActorPairTerminalError(
                f"{label} contains a symbolic link: {current}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ActorPairTerminalError(
                f"{label} contains a non-directory ancestor: {current}"
            )


def _ensure_directory(path: Path, label: str) -> None:
    _check_path_chain(path, label)
    path.mkdir(parents=True, exist_ok=True)
    _check_path_chain(path, label)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ActorPairTerminalError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ActorPairTerminalError(f"{label} must be a non-symlink directory")


def _absolute_directory(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    _check_path_chain(path, label)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ActorPairTerminalError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ActorPairTerminalError(f"{label} must be a non-symlink directory")
    return path


def _inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        raise ActorPairTerminalError(f"{label} must be inside the actor run") from None


def _exit_code(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ActorPairTerminalError(f"{label} must be an exit code in [0,255]")
    return value


def _immutable_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish once, or accept a byte-identical immutable record.

    Only a deterministic hidden staging inode is written directly.  The final
    name appears via a same-directory create-only hard link after the staging
    bytes and mode are fsynced. A process killed mid-write can therefore never
    expose a truncated terminal/scalar at the final name.
    """

    _ensure_directory(path.parent, "terminal output parent")
    _check_path_chain(path, "terminal output path")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(path.parent, directory_flags)
    staging_name = f".{path.name}.staging"
    staging_created = False

    def existing_bytes() -> bytes:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ActorPairTerminalError(
                "existing terminal output conflicts"
            ) from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
                raise ActorPairTerminalError("existing terminal output conflicts")
            if info.st_size != len(payload):
                raise ActorPairTerminalError("existing terminal output conflicts")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            os.close(descriptor)

    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        try:
            staging_info = os.stat(
                staging_name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            staging_info = None
        if staging_info is not None:
            if not stat.S_ISREG(staging_info.st_mode):
                raise ActorPairTerminalError(
                    "terminal output staging artifact is unsafe"
                )
            os.unlink(staging_name, dir_fd=directory_fd)
            os.fsync(directory_fd)

        try:
            existing = existing_bytes()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != payload:
                raise ActorPairTerminalError("existing terminal output conflicts")
            return

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(staging_name, flags, 0o400, dir_fd=directory_fd)
        staging_created = True
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as output:
                output.write(payload)
                output.flush()
                os.fchmod(output.fileno(), 0o444)
                os.fsync(output.fileno())
            try:
                os.link(
                    staging_name,
                    path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if existing_bytes() != payload:
                    raise ActorPairTerminalError(
                        "existing terminal output conflicts"
                    ) from None
            os.fsync(directory_fd)
        finally:
            if staging_created:
                try:
                    os.unlink(staging_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                except FileNotFoundError:
                    pass
    finally:
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def _atomic_finish_marker(path: Path, exit_code: int) -> None:
    """Publish the finish marker create-only after all terminal evidence."""

    _check_path_chain(path, "sidecar finish marker")
    payload = f"{exit_code}\n".encode("ascii")
    try:
        _immutable_bytes(path, payload)
    except ActorPairTerminalError as exc:
        raise ActorPairTerminalError("existing sidecar finish marker conflicts") from exc


def _failure_code(
    *,
    rank0_exit_code: int,
    rank1_exit_code: int,
    lifecycle_exit_code: int,
    quality_exit_code: int,
    wandb_wrapper_exit_code: int,
    sealed: bool,
) -> str:
    if rank0_exit_code or rank1_exit_code:
        return "actor_rank_nonzero"
    if lifecycle_exit_code:
        return "image3f_lifecycle_failure"
    if quality_exit_code:
        return "quality_finalization_failure"
    if wandb_wrapper_exit_code:
        return "wandb_wrapper_failure"
    if not sealed:
        return "cohort_unsealed"
    return "none"


def publish_actor_pair_terminal(
    *,
    attempt_dir: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    cohort_dir: str | os.PathLike[str],
    campaign_id: str,
    macro_round: int,
    cohort_index: int,
    rank0_exit_code: int,
    rank1_exit_code: int,
    lifecycle_exit_code: int,
    quality_exit_code: int,
    wandb_wrapper_exit_code: int = 0,
    delivery_pending: bool = False,
    finish_marker: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Publish immutable attempt outcome, scalar projection, then finish marker."""

    run_root = _absolute_directory(run_dir, "actor run directory")
    attempt_root = _absolute_directory(attempt_dir, "actor attempt directory")
    cohort_root = _absolute_directory(cohort_dir, "actor cohort directory")
    _inside(attempt_root, run_root, "actor attempt directory")
    if not isinstance(campaign_id, str) or TOKEN_RE.fullmatch(campaign_id) is None:
        raise ActorPairTerminalError("campaign ID is malformed")
    for value, label in ((macro_round, "macro round"), (cohort_index, "cohort index")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ActorPairTerminalError(f"{label} must be non-negative")
    statuses = {
        "0": _exit_code(rank0_exit_code, "rank-zero exit code"),
        "1": _exit_code(rank1_exit_code, "rank-one exit code"),
    }
    lifecycle = _exit_code(lifecycle_exit_code, "lifecycle exit code")
    quality = _exit_code(quality_exit_code, "quality exit code")
    wrapper = _exit_code(wandb_wrapper_exit_code, "W&B wrapper exit code")

    contract_path = cohort_root / "contract.json"
    try:
        contract = load_strict_json(contract_path)
        contract_sha256 = sha256_file(contract_path)
    except Exception as exc:
        raise ActorPairTerminalError("actor cohort contract is invalid") from exc
    if not isinstance(contract, Mapping):
        raise ActorPairTerminalError("actor cohort contract is not an object")
    expected_identity = {
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "cohort_index": cohort_index,
    }
    mismatches = [
        key for key, expected in expected_identity.items() if contract.get(key) != expected
    ]
    if mismatches:
        raise ActorPairTerminalError(
            "terminal identity does not match actor cohort contract: "
            + ", ".join(mismatches)
        )

    rank_reports: dict[str, Mapping[str, Any]] = {}
    for rank in (0, 1):
        try:
            rank_reports[str(rank)] = inspect_actor_rank(cohort_root, rank)
        except ActorPairError:
            # A killed exporter can leave an uncommitted payload.  Terminal
            # telemetry must still say FAILED without trusting/counting it.
            rank_reports[str(rank)] = {
                "state": "invalid",
                "transition_count": 0,
                "expected_transitions": 32 * 512,
            }
    try:
        pair = inspect_actor_pair(cohort_root)
    except ActorPairError:
        pair = {
            "sealed": False,
            "rank_states": {
                rank: str(report["state"]) for rank, report in rank_reports.items()
            },
        }
    committed_rank_transitions = sum(
        int(report["transition_count"])
        for report in rank_reports.values()
        if report["state"] == "complete"
    )
    sealed = bool(pair["sealed"])
    expected_transitions = sum(
        int(report["expected_transitions"]) for report in rank_reports.values()
    )
    # "Accepted" means learner-ready, not merely salvageable.  Completed rank
    # shards survive retry but do not become official replay input until the
    # exact joint cohort seal exists.
    accepted_transitions = expected_transitions if sealed else 0
    failure_code = _failure_code(
        rank0_exit_code=statuses["0"],
        rank1_exit_code=statuses["1"],
        lifecycle_exit_code=lifecycle,
        quality_exit_code=quality,
        wandb_wrapper_exit_code=wrapper,
        sealed=sealed,
    )
    succeeded = failure_code == "none"
    if delivery_pending and not succeeded:
        raise ActorPairTerminalError(
            "W&B delivery can be pending only for an otherwise successful cohort"
        )
    wandb_exit_code = 0 if succeeded else 2
    document: dict[str, Any] = {
        "schema": TERMINAL_SCHEMA,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "cohort_index": cohort_index,
        "cohort_contract_sha256": contract_sha256,
        "attempt_name": attempt_root.name,
        "outcome": (
            "PENDING_WANDB_DELIVERY"
            if delivery_pending
            else ("SUCCEEDED" if succeeded else "FAILED")
        ),
        "delivery_pending": delivery_pending,
        "failure_code": failure_code,
        "wandb_exit_code": wandb_exit_code,
        "rank_exit_codes": statuses,
        "lifecycle_exit_code": lifecycle,
        "quality_exit_code": quality,
        "wandb_wrapper_exit_code": wrapper,
        "cohort_sealed": sealed,
        "rank_states": pair["rank_states"],
        "accepted_transitions": accepted_transitions,
        "committed_rank_transitions": committed_rank_transitions,
        "expected_transitions": expected_transitions,
    }
    terminal_path = attempt_root / "actor_pair_terminal.json"
    terminal_payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _immutable_bytes(terminal_path, terminal_payload)

    scalar = {
        "schema": SCALAR_SCHEMA,
        "step": macro_round * 4 + cohort_index,
        "metrics": {
            "collection/data_ready": int(sealed),
            "collection/failed": int(not succeeded),
            "collection/succeeded": int(succeeded and not delivery_pending),
            "collection/terminal_pending": int(delivery_pending),
            "collection/cohort_sealed": int(sealed),
            "collection/accepted_transitions": accepted_transitions,
            "collection/committed_rank_transitions": committed_rank_transitions,
            "collection/expected_transitions": document["expected_transitions"],
            "collection/completion_fraction": (
                accepted_transitions / document["expected_transitions"]
            ),
            "collection/committed_rank_fraction": (
                committed_rank_transitions / document["expected_transitions"]
            ),
            "collection/rank0_exit_code": statuses["0"],
            "collection/rank1_exit_code": statuses["1"],
            "collection/lifecycle_exit_code": lifecycle,
            "collection/quality_exit_code": quality,
            "collection/wandb_wrapper_exit_code": wrapper,
            "collection/macro_round": macro_round,
            "collection/cohort_index": cohort_index,
        },
    }
    scalar_name = f"step_{scalar['step']:06d}_actor_pair_terminal_{attempt_root.name}.json"
    scalar_path = run_root / "telemetry" / "scalars" / scalar_name
    scalar_payload = (
        json.dumps(scalar, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _immutable_bytes(scalar_path, scalar_payload)

    if finish_marker is not None:
        finish_path = Path(os.path.abspath(os.path.expanduser(os.fspath(finish_marker))))
        _inside(finish_path, run_root, "sidecar finish marker")
        _check_path_chain(finish_path, "sidecar finish marker")
        _absolute_directory(finish_path.parent, "sidecar finish-marker parent")
        _atomic_finish_marker(finish_path, wandb_exit_code)
    return {**document, "terminal_path": str(terminal_path), "scalar_path": str(scalar_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--macro-round", type=int, required=True)
    parser.add_argument("--cohort-index", type=int, required=True)
    parser.add_argument("--rank0-exit-code", type=int, required=True)
    parser.add_argument("--rank1-exit-code", type=int, required=True)
    parser.add_argument("--lifecycle-exit-code", type=int, required=True)
    parser.add_argument("--quality-exit-code", type=int, required=True)
    parser.add_argument("--wandb-wrapper-exit-code", type=int, default=0)
    parser.add_argument("--delivery-pending", action="store_true")
    parser.add_argument("--finish-marker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = publish_actor_pair_terminal(
            attempt_dir=arguments.attempt_dir,
            run_dir=arguments.run_dir,
            cohort_dir=arguments.cohort_dir,
            campaign_id=arguments.campaign_id,
            macro_round=arguments.macro_round,
            cohort_index=arguments.cohort_index,
            rank0_exit_code=arguments.rank0_exit_code,
            rank1_exit_code=arguments.rank1_exit_code,
            lifecycle_exit_code=arguments.lifecycle_exit_code,
            quality_exit_code=arguments.quality_exit_code,
            wandb_wrapper_exit_code=arguments.wandb_wrapper_exit_code,
            delivery_pending=arguments.delivery_pending,
            finish_marker=arguments.finish_marker,
        )
    except Exception as exc:
        detail = str(exc) if isinstance(exc, ActorPairTerminalError) else type(exc).__name__
        print(f"actor-pair terminal publication failed: {detail}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
