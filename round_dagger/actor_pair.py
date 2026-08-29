"""Exact completion gate for the two independent production actor ranks.

The frozen round actors deliberately do not form a Torch distributed process
group.  This module is the filesystem rendezvous between those two processes:
it verifies committed shard bytes, recognizes only an empty or exactly complete
rank, and publishes the cohort seal only after both ranks cover their exact
32-environment partition for 512 steps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    ArtifactError,
    load_strict_json,
    seal_cohort,
    verify_cohort,
    verify_rollout_shard,
)


PRODUCTION_RANKS = [0, 1]
PRODUCTION_ENVIRONMENTS = 64
PRODUCTION_STEPS = 512
PRODUCTION_ENVIRONMENTS_PER_SHARD = 4
PRODUCTION_TRANSITIONS = PRODUCTION_ENVIRONMENTS * PRODUCTION_STEPS
_SHARD_TEMP_RE = re.compile(
    r"^\.rank-(?P<rank>[0-9]{4})_rollout-00000000_"
    r"part-(?P<part>[0-9]{4})\.npz\.[a-z0-9_]{8}$"
)


class ActorPairError(RuntimeError):
    """The independent actor pair is incomplete or violates its contract."""


def _check_path_chain(path: Path, label: str) -> None:
    """Reject every existing symlink or non-directory ancestor."""

    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(parts):
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ActorPairError(f"{label} contains a symbolic link: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ActorPairError(
                f"{label} contains a non-directory ancestor: {current}"
            )


def _root_and_document(cohort_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(cohort_dir))))
    if not root.is_absolute():
        raise ActorPairError("actor cohort path must be absolute")
    _check_path_chain(root, "actor cohort path")
    try:
        root_info = os.lstat(root)
    except OSError as exc:
        raise ActorPairError("actor cohort path is unavailable") from exc
    if not stat.S_ISDIR(root_info.st_mode):
        raise ActorPairError("actor cohort path must be a safe directory")
    try:
        document = load_strict_json(root / "contract.json")
    except Exception as exc:
        raise ActorPairError(f"cannot load actor cohort contract: {exc}") from exc
    return root, document


def _validated_stale_temporary(
    path: Path, *, expected_part_count: int
) -> tuple[int, int] | None:
    match = _SHARD_TEMP_RE.fullmatch(path.name)
    if match is None:
        return None
    rank = int(match.group("rank"))
    part = int(match.group("part"))
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ActorPairError(
            f"cannot inspect stale shard temporary: {path.name}"
        ) from exc
    if (
        rank not in PRODUCTION_RANKS
        or part >= expected_part_count
        or not stat.S_ISREG(info.st_mode)
    ):
        raise ActorPairError(f"unsafe stale shard temporary: {path.name}")
    return rank, part


def prune_stale_shard_temporaries(cohort_dir: str | Path) -> dict[str, Any]:
    """Delete only exporter-owned, never-committed NPZ temporary files.

    ``publish_rollout_shard`` creates a same-directory hidden temporary before
    atomically linking the immutable payload into place.  SIGKILL can prevent
    its ``finally`` block from removing that hidden inode.  A pair supervisor
    calls this operation only while it owns the cohort lifecycle lock, so no
    exporter can still be writing it.  The temporary is never interpreted as
    rollout evidence; unknown hidden names remain in place and fail closed.
    """

    root, _ = _root_and_document(cohort_dir)
    contract = _contract(root)
    ranks = contract.get("expected_ranks")
    envs = contract.get("expected_global_env_ids")
    if ranks != PRODUCTION_RANKS or not isinstance(envs, list):
        raise ActorPairError("stale-temp pruning requires the production rank contract")
    environments_per_rank = len(envs) // len(ranks)
    expected_parts = (
        environments_per_rank + PRODUCTION_ENVIRONMENTS_PER_SHARD - 1
    ) // PRODUCTION_ENVIRONMENTS_PER_SHARD
    shard_root = root / "shards"
    _check_path_chain(shard_root, "actor shard directory")
    try:
        shard_info = os.lstat(shard_root)
    except OSError as exc:
        raise ActorPairError("actor cohort has no safe shards directory") from exc
    if not stat.S_ISDIR(shard_info.st_mode):
        raise ActorPairError("actor cohort has no safe shards directory")

    removable: list[str] = []
    for path in shard_root.iterdir():
        if not path.name.startswith("."):
            continue
        identity = _validated_stale_temporary(
            path, expected_part_count=expected_parts
        )
        if identity is None:
            # Do not delete unrecognized evidence.  The subsequent strict
            # status gate will reject it as ambiguous.
            continue
        removable.append(path.name)

    if removable:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(shard_root, flags)
        try:
            for name in sorted(removable):
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise ActorPairError(
                        f"cannot reverify stale shard temporary: {name}"
                    ) from exc
                if not stat.S_ISREG(info.st_mode):
                    raise ActorPairError(f"unsafe stale shard temporary: {name}")
                os.unlink(name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return {
        "schema": "fast_roborender.round_dagger.actor_shard_temp_cleanup.v1",
        "cohort_dir": str(root),
        "removed": sorted(removable),
        "removed_count": len(removable),
    }


def _contract(root: Path) -> dict[str, Any]:
    _, document = _root_and_document(root)
    cohort_index = document.get("cohort_index")
    if type(cohort_index) is not int or cohort_index < 0:
        raise ActorPairError("actor cohort_index must be a non-negative integer")
    environment_start = cohort_index * PRODUCTION_ENVIRONMENTS
    expected = {
        "purpose": "train",
        "expected_ranks": PRODUCTION_RANKS,
        "expected_global_env_ids": list(
            range(environment_start, environment_start + PRODUCTION_ENVIRONMENTS)
        ),
        "expected_transitions": PRODUCTION_TRANSITIONS,
        "expected_steps_per_env": PRODUCTION_STEPS,
    }
    mismatches = [key for key, value in expected.items() if document.get(key) != value]
    if mismatches:
        raise ActorPairError(
            "actor cohort is not the exact 2x32x512 production contract: "
            + ", ".join(mismatches)
        )
    return document


def _inspect_actor_rank(
    cohort_dir: str | Path,
    rank: int,
    *,
    tolerate_stale_temporaries: bool,
) -> dict[str, Any]:
    """Verify one logical rank, optionally classifying exact stale temporaries."""

    if type(rank) is not int or rank < 0:
        raise ActorPairError("actor rank must be a non-negative integer")
    root, contract = _root_and_document(cohort_dir)
    ranks = contract.get("expected_ranks")
    envs = contract.get("expected_global_env_ids")
    steps = contract.get("expected_steps_per_env")
    if (
        not isinstance(ranks, list)
        or not ranks
        or any(type(item) is not int or item < 0 for item in ranks)
        or ranks != sorted(set(ranks))
    ):
        raise ActorPairError("actor cohort expected_ranks is invalid")
    if rank not in ranks:
        raise ActorPairError("actor rank is absent from the cohort contract")
    if (
        not isinstance(envs, list)
        or not envs
        or any(type(item) is not int or item < 0 for item in envs)
        or envs != sorted(set(envs))
        or len(envs) % len(ranks)
    ):
        raise ActorPairError("actor cohort environment partition is invalid")
    if type(steps) is not int or steps < 1:
        raise ActorPairError("actor cohort expected_steps_per_env is invalid")
    expected_transitions = contract.get("expected_transitions")
    if expected_transitions != len(envs) * steps:
        raise ActorPairError("actor cohort transition contract is inconsistent")

    environments_per_rank = len(envs) // len(ranks)
    rank_index = ranks.index(rank)
    expected_envs = envs[
        rank_index * environments_per_rank : (rank_index + 1) * environments_per_rank
    ]
    expected_part_count = (
        len(expected_envs) + PRODUCTION_ENVIRONMENTS_PER_SHARD - 1
    ) // PRODUCTION_ENVIRONMENTS_PER_SHARD
    prefix = f"rank-{rank:04d}_"
    temporary_prefix = f".rank-{rank:04d}_"
    shard_root = root / "shards"
    _check_path_chain(shard_root, "actor shard directory")
    try:
        shard_info = os.lstat(shard_root)
    except OSError as exc:
        raise ActorPairError("actor cohort has no safe shards directory") from exc
    if not stat.S_ISDIR(shard_info.st_mode):
        raise ActorPairError("actor cohort has no safe shards directory")
    actual_files: set[str] = set()
    sidecars: list[Path] = []
    for path in shard_root.iterdir():
        if path.name.startswith(temporary_prefix):
            identity = _validated_stale_temporary(
                path, expected_part_count=expected_part_count
            )
            if tolerate_stale_temporaries and identity is not None:
                continue
            raise ActorPairError(
                f"rank {rank} has a stale uncommitted shard temporary; "
                "exclusive supervisor cleanup is required"
            )
        if not path.name.startswith(prefix):
            continue
        if path.is_symlink() or not path.is_file():
            raise ActorPairError(f"unsafe rank {rank} shard entry: {path.name}")
        if path.suffix not in {".json", ".npz"} or path.name.startswith("."):
            raise ActorPairError(f"unexpected rank {rank} shard entry: {path.name}")
        actual_files.add(path.name)
        if path.suffix == ".json":
            sidecars.append(path)

    reports: list[dict[str, Any]] = []
    referenced_files: set[str] = set()
    try:
        for sidecar in sorted(sidecars):
            report = verify_rollout_shard(sidecar)
            if report["rank"] != rank:
                raise ActorPairError("rank shard filename and manifest disagree")
            reports.append(report)
            referenced_files.add(Path(report["manifest_path"]).name)
            referenced_files.add(Path(report["payload_path"]).name)
    except ArtifactError as exc:
        raise ActorPairError(f"rank {rank} shard verification failed: {exc}") from exc
    observed_envs: set[int] = set()
    transitions = 0
    identities: set[tuple[int, int]] = set()
    for report in reports:
        identity = (report["rollout_id"], report["part_index"])
        if identity in identities:
            raise ActorPairError(f"rank {rank} repeats a rollout/part identity")
        identities.add(identity)
        report_envs = set(report["global_env_ids"])
        if observed_envs & report_envs:
            raise ActorPairError(f"rank {rank} repeats logical environments")
        observed_envs |= report_envs
        transitions += int(report["transition_count"])
    exact_transitions = len(expected_envs) * steps
    complete = observed_envs == set(expected_envs) and transitions == exact_transitions
    # Historical completed ranks may use one 32-environment shard. Interrupted
    # production ranks, however, are resumable only when every committed part
    # is the exact contiguous 4-environment prefix emitted by the current
    # exporter. This makes the retry decision deterministic and copy-safe.
    if not complete:
        ordered = sorted(reports, key=lambda item: item["part_index"])
        if len(ordered) > expected_part_count or [
            (item["rollout_id"], item["part_index"]) for item in ordered
        ] != [(0, part_index) for part_index in range(len(ordered))]:
            raise ActorPairError(
                f"rank {rank} shards are not an exact contiguous part prefix"
            )
        for part_index, item in enumerate(ordered):
            start = part_index * PRODUCTION_ENVIRONMENTS_PER_SHARD
            part_envs = expected_envs[
                start : start + PRODUCTION_ENVIRONMENTS_PER_SHARD
            ]
            if (
                item["global_env_ids"] != part_envs
                or item["transition_count"] != len(part_envs) * steps
            ):
                raise ActorPairError(
                    f"rank {rank} part {part_index} is not an exact resumable chunk"
                )

    extra_files = actual_files - referenced_files
    missing_files = referenced_files - actual_files
    recoverable_payload: str | None = None
    if missing_files:
        raise ActorPairError(f"rank {rank} is missing a referenced shard file")
    if extra_files:
        next_part = len(reports)
        expected_orphan = (
            f"rank-{rank:04d}_rollout-00000000_part-{next_part:04d}.npz"
        )
        if (
            complete
            or next_part >= expected_part_count
            or extra_files != {expected_orphan}
        ):
            raise ActorPairError(
                f"rank {rank} has an unsafe uncommitted or unreferenced shard payload"
            )
        # Payload publication precedes its immutable JSON commit. A crash in
        # that narrow window is recoverable: the exporter will regenerate this
        # exact part, compare bytes create-only, and only then publish sidecar.
        recoverable_payload = expected_orphan

    state = (
        "complete"
        if complete
        else "empty"
        if not reports and recoverable_payload is None
        else "resumable"
    )
    return {
        "schema": "fast_roborender.round_dagger.actor_rank_status.v1",
        "cohort_dir": str(root),
        "rank": rank,
        "state": state,
        "expected_global_env_ids": expected_envs,
        "global_env_ids": sorted(observed_envs),
        "expected_steps_per_env": steps,
        "expected_transitions": exact_transitions,
        "transition_count": transitions,
        "shard_count": len(reports),
        "expected_part_count": expected_part_count,
        "recoverable_payload": recoverable_payload,
    }


def inspect_actor_rank(cohort_dir: str | Path, rank: int) -> dict[str, Any]:
    """Verify one logical rank without observing a concurrently publishing peer."""

    return _inspect_actor_rank(
        cohort_dir, rank, tolerate_stale_temporaries=False
    )


def require_actor_rank_completion(cohort_dir: str | Path, rank: int) -> dict[str, Any]:
    """Fail unless one trainer invocation published its exact immutable rank."""

    report = inspect_actor_rank(cohort_dir, rank)
    if report["state"] != "complete":
        raise ActorPairError(
            f"actor rank {rank} returned without exact shard coverage "
            f"(state={report['state']}, transitions={report['transition_count']}/"
            f"{report['expected_transitions']})"
        )
    return report


def _verify_report(report: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    expected = {
        "campaign_id": contract["campaign_id"],
        "macro_round": contract["macro_round"],
        "cohort_index": contract["cohort_index"],
        "purpose": "train",
        "sealed": True,
        "expected_ranks": PRODUCTION_RANKS,
        "ranks": PRODUCTION_RANKS,
        "expected_global_env_ids": contract["expected_global_env_ids"],
        "global_env_ids": contract["expected_global_env_ids"],
        "expected_transitions": PRODUCTION_TRANSITIONS,
        "transition_count": PRODUCTION_TRANSITIONS,
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    for key in (
        "policy_sha256",
        "runtime_policy_sha256",
        "teacher_sha256",
        "renderer_sha256",
        "beta_sha256",
        "config_sha256",
    ):
        if report.get(key) != contract.get(key):
            mismatches.append(key)
    if mismatches:
        raise ActorPairError(
            "sealed actor cohort failed exact production verification: "
            + ", ".join(sorted(set(mismatches)))
        )


def _unsealed_rank_states(
    root: Path,
    contract: Mapping[str, Any],
    *,
    tolerate_stale_temporaries: bool = False,
) -> dict[int, str]:
    shard_root = root / "shards"
    _check_path_chain(shard_root, "actor shard directory")
    try:
        shard_info = os.lstat(shard_root)
    except OSError as exc:
        raise ActorPairError("actor cohort has no safe shards directory") from exc
    if not stat.S_ISDIR(shard_info.st_mode):
        raise ActorPairError("actor cohort has no safe shards directory")

    for path in shard_root.iterdir():
        if path.name.startswith(".") and tolerate_stale_temporaries:
            environments_per_rank = len(contract["expected_global_env_ids"]) // len(
                contract["expected_ranks"]
            )
            expected_part_count = (
                environments_per_rank + PRODUCTION_ENVIRONMENTS_PER_SHARD - 1
            ) // PRODUCTION_ENVIRONMENTS_PER_SHARD
            if (
                _validated_stale_temporary(
                    path, expected_part_count=expected_part_count
                )
                is not None
            ):
                continue
        if path.is_symlink() or not path.is_file():
            raise ActorPairError(f"unsafe actor shard entry: {path.name}")
        if path.suffix not in {".json", ".npz"} or path.name.startswith("."):
            raise ActorPairError(f"unexpected actor shard entry: {path.name}")
        if not any(
            path.name.startswith(f"rank-{rank:04d}_") for rank in PRODUCTION_RANKS
        ):
            raise ActorPairError(f"unexpected actor shard rank: {path.name}")

    return {
        rank: _inspect_actor_rank(
            root,
            rank,
            tolerate_stale_temporaries=tolerate_stale_temporaries,
        )["state"]
        for rank in PRODUCTION_RANKS
    }


def inspect_actor_pair(cohort_dir: str | Path) -> dict[str, Any]:
    """Return exact rank completion state without changing cohort artifacts."""

    root, _ = _root_and_document(cohort_dir)
    contract = _contract(root)
    seal_path = root / "seal.json"
    if seal_path.exists() or seal_path.is_symlink():
        try:
            report = verify_cohort(root)
        except ArtifactError as exc:
            raise ActorPairError(f"existing actor cohort seal is invalid: {exc}") from exc
        _verify_report(report, contract)
        states = {0: "complete", 1: "complete"}
        sealed = True
    else:
        states = _unsealed_rank_states(root, contract)
        sealed = False
    return {
        "schema": "fast_roborender.round_dagger.actor_pair_status.v1",
        "cohort_dir": str(root),
        "sealed": sealed,
        "rank_states": {str(rank): states[rank] for rank in PRODUCTION_RANKS},
    }


def inspect_actor_pair_for_recovery(cohort_dir: str | Path) -> dict[str, Any]:
    """Classify a retry frontier without treating hidden temporaries as data.

    Exact exporter-owned hidden NPZ temporaries are reported as cleanup work;
    they never contribute environments or transitions. Unknown hidden entries
    and every other artifact inconsistency still fail closed.
    """

    root, _ = _root_and_document(cohort_dir)
    contract = _contract(root)
    seal_path = root / "seal.json"
    if seal_path.exists() or seal_path.is_symlink():
        report = inspect_actor_pair(root)
        return {**report, "requires_temp_cleanup": False, "stale_temporaries": []}
    states = _unsealed_rank_states(
        root, contract, tolerate_stale_temporaries=True
    )
    environments_per_rank = len(contract["expected_global_env_ids"]) // len(
        contract["expected_ranks"]
    )
    expected_part_count = (
        environments_per_rank + PRODUCTION_ENVIRONMENTS_PER_SHARD - 1
    ) // PRODUCTION_ENVIRONMENTS_PER_SHARD
    stale: list[str] = []
    for path in (root / "shards").iterdir():
        if path.name.startswith("."):
            if (
                _validated_stale_temporary(
                    path, expected_part_count=expected_part_count
                )
                is None
            ):
                raise ActorPairError(f"unexpected actor shard entry: {path.name}")
            stale.append(path.name)
    return {
        "schema": "fast_roborender.round_dagger.actor_pair_recovery_status.v1",
        "cohort_dir": str(root),
        "sealed": False,
        "rank_states": {str(rank): states[rank] for rank in PRODUCTION_RANKS},
        "requires_temp_cleanup": bool(stale),
        "stale_temporaries": sorted(stale),
    }


def seal_actor_pair(cohort_dir: str | Path) -> dict[str, Any]:
    """Seal and reverify the exact cohort only after both ranks are complete."""

    status = inspect_actor_pair(cohort_dir)
    if status["sealed"]:
        return verify_cohort(cohort_dir)
    if status["rank_states"] != {"0": "complete", "1": "complete"}:
        raise ActorPairError("cannot seal before both independent actor ranks are complete")
    root = Path(cohort_dir)
    contract = _contract(root)
    try:
        report = seal_cohort(root)
        report = verify_cohort(root)
    except ArtifactError as exc:
        raise ActorPairError(f"actor pair cohort sealing failed: {exc}") from exc
    _verify_report(report, contract)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("status", "rank", "seal", "prune-temps")
    )
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument("--rank", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.operation == "status":
        if arguments.rank is not None:
            raise ActorPairError("--rank is valid only for the rank operation")
        result = inspect_actor_pair(arguments.cohort_dir)
    elif arguments.operation == "rank":
        if arguments.rank is None:
            raise ActorPairError("rank operation requires --rank")
        result = require_actor_rank_completion(arguments.cohort_dir, arguments.rank)
    elif arguments.operation == "seal":
        if arguments.rank is not None:
            raise ActorPairError("--rank is valid only for the rank operation")
        result = seal_actor_pair(arguments.cohort_dir)
    else:
        if arguments.rank is not None:
            raise ActorPairError("--rank is invalid for prune-temps")
        result = prune_stale_shard_temporaries(arguments.cohort_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ActorPairError, ArtifactError) as error:
        print(f"actor pair gate failed: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)


__all__ = [
    "ActorPairError",
    "inspect_actor_rank",
    "inspect_actor_pair",
    "inspect_actor_pair_for_recovery",
    "prune_stale_shard_temporaries",
    "require_actor_rank_completion",
    "seal_actor_pair",
]
