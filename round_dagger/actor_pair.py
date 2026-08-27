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
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    ArtifactError,
    discover_rollout_shards,
    load_strict_json,
    seal_cohort,
    verify_cohort,
    verify_rollout_shard,
)


PRODUCTION_RANKS = [0, 1]
PRODUCTION_ENVIRONMENTS = 64
PRODUCTION_STEPS = 512
PRODUCTION_TRANSITIONS = PRODUCTION_ENVIRONMENTS * PRODUCTION_STEPS


class ActorPairError(RuntimeError):
    """The independent actor pair is incomplete or violates its contract."""


def _root_and_document(cohort_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(cohort_dir).expanduser()
    if not root.is_absolute():
        raise ActorPairError("actor cohort path must be absolute")
    if root.is_symlink() or not root.is_dir():
        raise ActorPairError("actor cohort path must be a safe directory")
    try:
        document = load_strict_json(root / "contract.json")
    except Exception as exc:
        raise ActorPairError(f"cannot load actor cohort contract: {exc}") from exc
    return root, document


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


def inspect_actor_rank(cohort_dir: str | Path, rank: int) -> dict[str, Any]:
    """Verify one logical rank without observing a concurrently publishing peer."""

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
    prefix = f"rank-{rank:04d}_"
    shard_root = root / "shards"
    if not shard_root.is_dir() or shard_root.is_symlink():
        raise ActorPairError("actor cohort has no safe shards directory")
    actual_files: set[str] = set()
    sidecars: list[Path] = []
    for path in shard_root.iterdir():
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
    if actual_files != referenced_files:
        raise ActorPairError(
            f"rank {rank} has an uncommitted or unreferenced shard payload"
        )

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
    state = (
        "empty"
        if not reports
        else "complete"
        if observed_envs == set(expected_envs) and transitions == exact_transitions
        else "partial"
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
    }


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


def _unsealed_rank_states(root: Path, contract: Mapping[str, Any]) -> dict[int, str]:
    shard_root = root / "shards"
    if not shard_root.is_dir() or shard_root.is_symlink():
        raise ActorPairError("actor cohort has no safe shards directory")

    actual_files: set[str] = set()
    for path in shard_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ActorPairError(f"unsafe actor shard entry: {path.name}")
        if path.suffix not in {".json", ".npz"} or path.name.startswith("."):
            raise ActorPairError(f"unexpected actor shard entry: {path.name}")
        actual_files.add(path.name)

    reports: list[dict[str, Any]] = []
    referenced_files: set[str] = set()
    try:
        sidecars = discover_rollout_shards(root)
        for sidecar in sidecars:
            report = verify_rollout_shard(sidecar)
            reports.append(report)
            referenced_files.add(Path(report["manifest_path"]).name)
            referenced_files.add(Path(report["payload_path"]).name)
    except ArtifactError as exc:
        raise ActorPairError(f"actor shard verification failed: {exc}") from exc
    if actual_files != referenced_files:
        raise ActorPairError(
            "actor shard file set contains an uncommitted or unreferenced payload"
        )

    offset = int(contract["cohort_index"]) * PRODUCTION_ENVIRONMENTS
    expected_envs = {
        0: set(range(offset, offset + PRODUCTION_ENVIRONMENTS // 2)),
        1: set(
            range(
                offset + PRODUCTION_ENVIRONMENTS // 2,
                offset + PRODUCTION_ENVIRONMENTS,
            )
        ),
    }
    rank_reports: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for report in reports:
        rank = report["rank"]
        if rank not in rank_reports:
            raise ActorPairError(f"unexpected actor shard rank: {rank}")
        rank_reports[rank].append(report)

    states: dict[int, str] = {}
    for rank in PRODUCTION_RANKS:
        items = rank_reports[rank]
        if not items:
            states[rank] = "empty"
            continue
        envs: set[int] = set()
        transitions = 0
        identities: set[tuple[int, int]] = set()
        for item in items:
            identity = (item["rollout_id"], item["part_index"])
            if identity in identities:
                raise ActorPairError(f"rank {rank} repeats a rollout/part identity")
            identities.add(identity)
            item_envs = set(item["global_env_ids"])
            if envs & item_envs:
                raise ActorPairError(f"rank {rank} repeats logical environments")
            envs |= item_envs
            transitions += int(item["transition_count"])
        expected_transitions = len(expected_envs[rank]) * PRODUCTION_STEPS
        states[rank] = (
            "complete"
            if envs == expected_envs[rank] and transitions == expected_transitions
            else "partial"
        )
    return states


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
    parser.add_argument("operation", choices=("status", "rank", "seal"))
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
    else:
        if arguments.rank is not None:
            raise ActorPairError("--rank is valid only for the rank operation")
        result = seal_actor_pair(arguments.cohort_dir)
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
    "require_actor_rank_completion",
    "seal_actor_pair",
]
