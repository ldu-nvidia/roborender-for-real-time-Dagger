#!/usr/bin/env python3
"""Build a tiny immutable learner fixture from a sealed real RTX cohort.

The builder deliberately reads, verifies, and derives from an existing cohort;
it never edits that cohort.  It selects the earliest logical environments from
each requested actor rank, keeps the first ``T`` transitions, and republishes
one environment per learner rank through the normal artifact/replay APIs.

``fixture_manifest.json`` is published last and is the only completion marker.
It binds the complete source cohort, every selected source shard, the slicing
policy, the derived cohort, and the replay snapshot by SHA-256.

Held-out replay keeps the historical exact-count behavior.  The sole bounded
exception derives 32 cases from the verified fixed-256 source: it takes the
four lowest global environment IDs from each of exactly eight source shards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger import artifacts, replay  # noqa: E402


FIXTURE_SCHEMA = "fast_roborender.round_dagger.real_rtx_smoke_fixture.v1"
DEFAULT_SOURCE_RANKS = (0, 1)
DEFAULT_ENVIRONMENTS_PER_SOURCE_RANK = 4
DEFAULT_STEPS = 4
DEFAULT_HELDOUT_CASES = 256
EXPECTED_HELDOUT_SOURCE_SHARDS = 8
DOWNSAMPLE_SOURCE_HELDOUT_CASES = 256
DOWNSAMPLE_TARGET_HELDOUT_CASES = 32
DOWNSAMPLE_CASES_PER_SOURCE_SHARD = 4
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_DIGEST_FIELDS = (
    "policy_sha256",
    "runtime_policy_sha256",
    "teacher_sha256",
    "renderer_sha256",
    "beta_sha256",
    "config_sha256",
)


class FixtureError(RuntimeError):
    """The source or requested real-data fixture contract is invalid."""


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _true_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise FixtureError(f"{label} must be a positive integer")
    return value


def _campaign_token(value: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise FixtureError("campaign_id must be a safe artifact token")
    return value


def _source_selection(
    raw_seal: Mapping[str, Any],
    source_ranks: Sequence[int],
    environments_per_source_rank: int,
) -> list[dict[str, int]]:
    if not isinstance(raw_seal, Mapping) or not isinstance(raw_seal.get("shards"), list):
        raise FixtureError("source seal is missing its shard inventory")
    if not source_ranks or any(type(rank) is not int or rank < 0 for rank in source_ranks):
        raise FixtureError("source_ranks must contain non-negative integers")
    if len(source_ranks) != len(set(source_ranks)):
        raise FixtureError("source_ranks cannot contain duplicates")
    environments_per_source_rank = _true_positive_int(
        environments_per_source_rank, "environments_per_source_rank"
    )

    environments_by_rank: dict[int, set[int]] = {rank: set() for rank in source_ranks}
    for item in raw_seal["shards"]:
        if not isinstance(item, Mapping):
            raise FixtureError("source seal contains a malformed shard record")
        rank = item.get("rank")
        if rank not in environments_by_rank:
            continue
        raw_envs = item.get("global_env_ids")
        if not isinstance(raw_envs, list) or any(
            type(env_id) is not int or env_id < 0 for env_id in raw_envs
        ):
            raise FixtureError("source seal contains malformed environment coverage")
        environments_by_rank[rank].update(raw_envs)

    plan: list[dict[str, int]] = []
    learner_rank = 0
    for source_rank in source_ranks:
        available = sorted(environments_by_rank[source_rank])
        if len(available) < environments_per_source_rank:
            raise FixtureError(
                f"source rank {source_rank} has only {len(available)} environments; "
                f"{environments_per_source_rank} are required"
            )
        for global_env_id in available[:environments_per_source_rank]:
            plan.append(
                {
                    "learner_rank": learner_rank,
                    "source_actor_rank": source_rank,
                    "global_env_id": global_env_id,
                }
            )
            learner_rank += 1
    return plan


def _slice_one_environment(
    arrays: Mapping[str, np.ndarray],
    *,
    source_num_steps: int,
    source_num_envs: int,
    env_index: int,
    steps: int,
) -> dict[str, np.ndarray]:
    sliced: dict[str, np.ndarray] = {}
    for key, array in arrays.items():
        if key == "global_env_ids" and tuple(array.shape) == (source_num_envs,):
            value = array[env_index : env_index + 1]
        elif array.ndim >= 2 and tuple(array.shape[:2]) == (
            source_num_steps,
            source_num_envs,
        ):
            value = array[:steps, env_index : env_index + 1, ...]
        else:
            raise FixtureError(
                f"verified source array {key!r} has an unsupported leading shape "
                f"{tuple(array.shape)}"
            )
        sliced[key] = np.ascontiguousarray(value)
    return sliced


def _slice_environment_subset(
    arrays: Mapping[str, np.ndarray],
    *,
    source_num_steps: int,
    source_num_envs: int,
    selected_global_env_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    """Select named environments from every verified array's environment axis."""

    raw_ids = arrays.get("global_env_ids")
    if not isinstance(raw_ids, np.ndarray):
        raise FixtureError("verified source global_env_ids array is missing")
    if tuple(raw_ids.shape) == (source_num_envs,):
        env_vector = raw_ids
    elif tuple(raw_ids.shape) == (source_num_steps, source_num_envs):
        if not np.all(raw_ids == raw_ids[0:1]):
            raise FixtureError("verified source global_env_ids changed over time")
        env_vector = raw_ids[0]
    else:
        raise FixtureError("verified source global_env_ids shape changed")

    index_by_env_id = {
        int(raw_env_id): index
        for index, raw_env_id in enumerate(env_vector.tolist())
    }
    if len(index_by_env_id) != source_num_envs:
        raise FixtureError("verified source global_env_ids are not unique")
    selected = [int(item) for item in selected_global_env_ids]
    if not selected or len(selected) != len(set(selected)):
        raise FixtureError("selected held-out environments are empty or duplicated")
    missing = sorted(set(selected) - set(index_by_env_id))
    if missing:
        raise FixtureError(
            f"selected held-out environments are absent from source shard: {missing}"
        )
    indices = [index_by_env_id[env_id] for env_id in selected]

    sliced: dict[str, np.ndarray] = {}
    for key, array in arrays.items():
        if key == "global_env_ids" and tuple(array.shape) == (source_num_envs,):
            value = np.take(array, indices, axis=0)
        elif array.ndim >= 2 and tuple(array.shape[:2]) == (
            source_num_steps,
            source_num_envs,
        ):
            value = np.take(array, indices, axis=1)
        else:
            raise FixtureError(
                f"verified held-out source array {key!r} has an unsupported "
                f"leading shape {tuple(array.shape)}"
            )
        sliced[key] = np.ascontiguousarray(value)
    return sliced


def _extract_verified_arrays(
    source_cohort: Path,
    selection: Sequence[Mapping[str, int]],
    *,
    steps: int,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, Any]], dict[str, Any]]:
    wanted = {int(item["global_env_id"]): dict(item) for item in selection}
    if len(wanted) != len(selection):
        raise FixtureError("selected source environments are not unique")
    extracted: dict[int, dict[str, np.ndarray]] = {}
    origins: dict[int, dict[str, Any]] = {}

    def visitor(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
        source_steps = int(report["transition_count"]) // len(report["global_env_ids"])
        source_envs = len(report["global_env_ids"])
        if source_steps < steps:
            overlap = sorted(set(report["global_env_ids"]) & set(wanted))
            if overlap:
                raise FixtureError(
                    f"source shard {report['shard_id']} has only {source_steps} steps; "
                    f"{steps} were requested"
                )
        raw_ids = arrays["global_env_ids"]
        if tuple(raw_ids.shape) == (source_envs,):
            env_vector = raw_ids
        elif tuple(raw_ids.shape) == (source_steps, source_envs):
            env_vector = raw_ids[0]
        else:
            raise FixtureError("verified source global_env_ids shape changed")

        for env_index, raw_env_id in enumerate(env_vector.tolist()):
            env_id = int(raw_env_id)
            if env_id not in wanted:
                continue
            expected_source_rank = int(wanted[env_id]["source_actor_rank"])
            if int(report["rank"]) != expected_source_rank:
                raise FixtureError(
                    f"environment {env_id} appeared on source rank {report['rank']} "
                    f"instead of {expected_source_rank}"
                )
            if env_id in extracted:
                raise FixtureError(f"environment {env_id} appeared in multiple source shards")
            extracted[env_id] = _slice_one_environment(
                arrays,
                source_num_steps=source_steps,
                source_num_envs=source_envs,
                env_index=env_index,
                steps=steps,
            )
            origins[env_id] = {
                "source_actor_rank": int(report["rank"]),
                "source_shard_id": report["shard_id"],
                "source_manifest_path": report["manifest_path"],
                "source_manifest_sha256": report["manifest_sha256"],
                "source_payload_path": report["payload_path"],
                "source_payload_sha256": report["payload_sha256"],
                "source_rollout_id": int(report["rollout_id"]),
                "source_part_index": int(report["part_index"]),
                "source_num_steps": source_steps,
                "source_time_slice": [0, steps],
                "global_env_id": env_id,
            }

    verified = artifacts.visit_verified_cohort_shards(source_cohort, visitor)
    missing = sorted(set(wanted) - set(extracted))
    if missing:
        raise FixtureError(f"selected environments were absent after verification: {missing}")
    return extracted, origins, verified


def _file_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise FixtureError(f"derived artifact is not a regular file: {path}")
    return {
        "path": relative_path,
        "sha256": artifacts.sha256_file(path),
        "size_bytes": info.st_size,
    }


def _derive_fixed_heldout(
    *,
    source_snapshot_path: Path,
    output: Path,
    campaign_id: str,
    expected_cases: int,
    expected_source_campaign_id: str,
    expected_view_contract: Mapping[str, Any],
    expected_action_contract: Mapping[str, Any],
    expected_digests: Mapping[str, str],
    training_payload_sha256s: set[str],
    allow_source_eval_schedule_rebinding: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recursively verify and republish eight fixed-eval shards as one cohort."""

    if not source_snapshot_path.is_file() or source_snapshot_path.is_symlink():
        raise FixtureError(
            "source_heldout_snapshot must be an existing non-symlink file"
        )
    expected_cases = _true_positive_int(expected_cases, "expected_heldout_cases")
    source_snapshot_file_sha256 = artifacts.sha256_file(source_snapshot_path)
    source_snapshot = replay.load_heldout_snapshot(
        source_snapshot_path,
        expected_file_sha256=source_snapshot_file_sha256,
    )
    source_transition_count = int(source_snapshot["transition_count"])
    downsample = source_transition_count != expected_cases
    if downsample and (
        source_transition_count != DOWNSAMPLE_SOURCE_HELDOUT_CASES
        or expected_cases != DOWNSAMPLE_TARGET_HELDOUT_CASES
    ):
        raise FixtureError(
            "source held-out transition count differs from the fixed-case contract: "
            f"actual={source_transition_count} expected={expected_cases}; only "
            "deterministic 32-from-256 downsampling is supported"
        )
    if source_snapshot["campaign_id"] != expected_source_campaign_id:
        raise FixtureError("source held-out snapshot belongs to another campaign")
    if source_snapshot["cohort_count"] != EXPECTED_HELDOUT_SOURCE_SHARDS:
        raise FixtureError(
            "source held-out snapshot must contain exactly eight eval cohorts/shards"
        )
    if (
        source_snapshot["view_contract"] != dict(expected_view_contract)
        or source_snapshot["action_contract"] != dict(expected_action_contract)
    ):
        raise FixtureError("source held-out view/action contract differs from training")

    captures: list[dict[str, Any]] = []
    source_cohort_records: list[dict[str, Any]] = []
    source_payload_sha256s: set[str] = set()
    all_source_eval_env_ids: set[int] = set()
    all_selected_eval_env_ids: set[int] = set()
    total_source_transitions = 0
    total_selected_transitions = 0
    for learner_rank, snapshot_cohort in enumerate(source_snapshot["cohorts"]):
        source_manifest = Path(snapshot_cohort["cohort_manifest_path"])
        visited: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []

        def visitor(
            report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
        ) -> None:
            visited.append(
                (
                    dict(report),
                    {
                        key: np.ascontiguousarray(array)
                        for key, array in arrays.items()
                    },
                )
            )

        source_report = artifacts.visit_verified_cohort_shards(
            source_manifest, visitor
        )
        if source_report["purpose"] != "eval" or len(visited) != 1:
            raise FixtureError(
                "each fixed held-out source cohort must contain exactly one eval shard"
            )
        if source_report["cohort_sha256"] != snapshot_cohort["manifest_sha256"]:
            raise FixtureError("held-out snapshot/cohort digest binding changed")
        core_digest_fields = (
            "policy_sha256",
            "runtime_policy_sha256",
            "teacher_sha256",
            "renderer_sha256",
        )
        source_schedule_differs = any(
            source_report[field] != expected_digests[field]
            for field in ("beta_sha256", "config_sha256")
        )
        if (
            source_report["view_contract"] != dict(expected_view_contract)
            or source_report["action_contract"] != dict(expected_action_contract)
            or any(
                source_report[field] != expected_digests[field]
                for field in core_digest_fields
            )
        ):
            raise FixtureError(
                "held-out source model/view/action lineage differs from training"
            )
        if source_schedule_differs and not allow_source_eval_schedule_rebinding:
            raise FixtureError(
                "held-out source beta/config lineage differs from training; "
                "explicit eval schedule rebinding is required"
            )
        source_shard, arrays = visited[0]
        shard_envs = {int(item) for item in source_shard["global_env_ids"]}
        source_shard_transitions = int(source_shard["transition_count"])
        if source_shard_transitions != len(shard_envs):
            raise FixtureError(
                "each fixed held-out environment must contribute exactly one case"
            )
        if all_source_eval_env_ids & shard_envs:
            raise FixtureError("held-out source shards reuse logical environment IDs")
        all_source_eval_env_ids.update(shard_envs)
        source_payload_sha = str(source_shard["payload_sha256"])
        if source_payload_sha in source_payload_sha256s:
            raise FixtureError("held-out source shards reuse a payload digest")
        source_payload_sha256s.add(source_payload_sha)
        if downsample:
            selected_env_ids = sorted(shard_envs)[
                :DOWNSAMPLE_CASES_PER_SOURCE_SHARD
            ]
            if len(selected_env_ids) != DOWNSAMPLE_CASES_PER_SOURCE_SHARD:
                raise FixtureError(
                    "each 256-case held-out source shard must provide at least "
                    f"{DOWNSAMPLE_CASES_PER_SOURCE_SHARD} environments"
                )
            arrays = _slice_environment_subset(
                arrays,
                source_num_steps=1,
                source_num_envs=source_shard_transitions,
                selected_global_env_ids=selected_env_ids,
            )
            selection_policy = "lowest_global_env_id_per_source_shard"
        else:
            selected_env_ids = sorted(shard_envs)
            selection_policy = "all_verified_cases"
        if all_selected_eval_env_ids & set(selected_env_ids):
            raise FixtureError("selected held-out environments are duplicated")
        all_selected_eval_env_ids.update(selected_env_ids)
        total_source_transitions += source_shard_transitions
        total_selected_transitions += len(selected_env_ids)
        captures.append(
            {
                "learner_rank": learner_rank,
                "arrays": arrays,
                "global_env_ids": selected_env_ids,
                "transition_count": len(selected_env_ids),
            }
        )
        source_cohort_records.append(
            {
                "learner_rank": learner_rank,
                "logical_id": source_report["logical_id"],
                "cohort_path": str(source_manifest.parent),
                "cohort_sha256": source_report["cohort_sha256"],
                "contract_sha256": source_report["contract_sha256"],
                "transition_count": source_report["transition_count"],
                "source_transition_count": source_shard_transitions,
                "selected_transition_count": len(selected_env_ids),
                "selection_policy": selection_policy,
                "source_shard_id": source_shard["shard_id"],
                "source_manifest_sha256": source_shard["manifest_sha256"],
                "source_payload_sha256": source_payload_sha,
                "source_beta_sha256": source_report["beta_sha256"],
                "source_config_sha256": source_report["config_sha256"],
                "global_env_ids": sorted(shard_envs),
                "selected_global_env_ids": selected_env_ids,
            }
        )

    if (
        total_source_transitions != source_transition_count
        or total_selected_transitions != expected_cases
        or len(captures) != EXPECTED_HELDOUT_SOURCE_SHARDS
    ):
        raise FixtureError("recursively verified held-out coverage is not exact")

    heldout_cohort = output / "heldout_cohort"
    artifacts.create_cohort(
        heldout_cohort,
        campaign_id=campaign_id,
        macro_round=0,
        cohort_index=1,
        purpose="eval",
        expected_ranks=list(range(len(captures))),
        expected_global_env_ids=sorted(all_selected_eval_env_ids),
        expected_transitions=expected_cases,
        view_contract=expected_view_contract,
        action_contract=expected_action_contract,
        **expected_digests,
    )
    for capture in captures:
        artifacts.publish_rollout_shard(
            heldout_cohort,
            rank=capture["learner_rank"],
            rollout_id=0,
            part_index=0,
            arrays=capture["arrays"],
            campaign_id=campaign_id,
            macro_round=0,
            cohort_index=1,
            purpose="eval",
            **expected_digests,
        )
    heldout_report = artifacts.seal_cohort(heldout_cohort)
    heldout_snapshot_path = output / "heldout_snapshot.json"
    heldout_snapshot = replay.build_heldout_snapshot(
        [heldout_cohort], output=heldout_snapshot_path
    )
    destination_payload_sha256s = {
        str(item["payload_sha256"]) for item in heldout_report["shards"]
    }
    collisions = sorted(training_payload_sha256s & destination_payload_sha256s)
    if collisions:
        raise FixtureError(
            "training and held-out payload digests overlap: " + ",".join(collisions)
        )

    loaded_heldout = replay.load_heldout_snapshot(
        heldout_snapshot_path,
        expected_file_sha256=heldout_snapshot["file_sha256"],
    )
    if (
        loaded_heldout["campaign_id"] != campaign_id
        or loaded_heldout["transition_count"] != expected_cases
        or len(loaded_heldout["resolved_shards"]) != len(captures)
    ):
        raise FixtureError("derived held-out snapshot failed final verification")

    source_record = {
        "snapshot_path": str(source_snapshot_path),
        "snapshot_file_sha256": source_snapshot_file_sha256,
        "snapshot_sha256": source_snapshot["snapshot_sha256"],
        "campaign_id": source_snapshot["campaign_id"],
        "transition_count": source_snapshot["transition_count"],
        "selected_transition_count": total_selected_transitions,
        "cohort_count": source_snapshot["cohort_count"],
        "source_shard_count": len(captures),
        "full_recursive_verification": True,
        "source_eval_schedule_rebinding": {
            "authorized": allow_source_eval_schedule_rebinding,
            "rebound_fields": ["beta_sha256", "config_sha256"],
            "source": {
                field: source_cohort_records[0][f"source_{field}"]
                for field in ("beta_sha256", "config_sha256")
            },
            "destination": {
                field: expected_digests[field]
                for field in ("beta_sha256", "config_sha256")
            },
        },
        "selection": {
            "policy": (
                "lowest_global_env_id_per_source_shard"
                if downsample
                else "all_verified_cases"
            ),
            "source_transition_count": source_transition_count,
            "selected_transition_count": total_selected_transitions,
            "source_shard_count": len(captures),
            "selected_cases_per_source_shard": (
                DOWNSAMPLE_CASES_PER_SOURCE_SHARD if downsample else None
            ),
        },
        "cohorts": source_cohort_records,
    }
    derived_record = {
        "heldout_cohort": _file_record(output, "heldout_cohort/seal.json"),
        "heldout_cohort_contract": _file_record(
            output, "heldout_cohort/contract.json"
        ),
        "heldout_cohort_logical_id": heldout_report["logical_id"],
        "heldout_transition_count": heldout_report["transition_count"],
        "heldout_snapshot": _file_record(output, "heldout_snapshot.json"),
        "heldout_snapshot_sha256": heldout_snapshot["snapshot_sha256"],
        "train_eval_payload_disjointness": {
            "enforced": True,
            "training_payload_sha256s": sorted(training_payload_sha256s),
            "heldout_payload_sha256s": sorted(destination_payload_sha256s),
            "intersection": collisions,
        },
    }
    return source_record, derived_record


def build_fixture(
    *,
    source_cohort: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    campaign_id: str,
    source_ranks: Sequence[int] = DEFAULT_SOURCE_RANKS,
    environments_per_source_rank: int = DEFAULT_ENVIRONMENTS_PER_SOURCE_RANK,
    steps: int = DEFAULT_STEPS,
    source_heldout_snapshot: str | os.PathLike[str] | None = None,
    expected_source_heldout_campaign_id: str | None = None,
    allow_source_eval_schedule_rebinding: bool = False,
    expected_heldout_cases: int = DEFAULT_HELDOUT_CASES,
) -> dict[str, Any]:
    """Build and seal one real-data cohort plus a one-round replay snapshot."""

    source = _absolute(source_cohort)
    output = _absolute(output_root)
    campaign_id = _campaign_token(campaign_id)
    steps = _true_positive_int(steps, "steps")
    if not source.is_dir() or source.is_symlink():
        raise FixtureError("source_cohort must be an existing non-symlink directory")
    if output.exists() or output.is_symlink():
        raise FixtureError("output_root must not already exist")

    raw_seal = artifacts.load_strict_json(source / artifacts.SEAL_FILENAME)
    plan = _source_selection(raw_seal, source_ranks, environments_per_source_rank)
    extracted, origins, source_report = _extract_verified_arrays(
        source, plan, steps=steps
    )
    if source_report["purpose"] != "train":
        raise FixtureError("real smoke fixture source must be a training cohort")

    source_contract = artifacts.load_strict_json(source / artifacts.CONTRACT_FILENAME)
    source_digests = {field: source_contract[field] for field in _DIGEST_FIELDS}
    output.mkdir(parents=True, exist_ok=False)
    cohort = output / "cohort"
    learner_ranks = [int(item["learner_rank"]) for item in plan]
    selected_envs = [int(item["global_env_id"]) for item in plan]
    artifacts.create_cohort(
        cohort,
        campaign_id=campaign_id,
        macro_round=0,
        cohort_index=0,
        purpose="train",
        expected_ranks=learner_ranks,
        expected_global_env_ids=selected_envs,
        expected_transitions=len(plan) * steps,
        view_contract=source_report["view_contract"],
        action_contract=source_report["action_contract"],
        **source_digests,
    )

    for item in plan:
        learner_rank = int(item["learner_rank"])
        env_id = int(item["global_env_id"])
        artifacts.publish_rollout_shard(
            cohort,
            rank=learner_rank,
            rollout_id=0,
            part_index=0,
            arrays=extracted[env_id],
            campaign_id=campaign_id,
            macro_round=0,
            cohort_index=0,
            purpose="train",
            **source_digests,
        )
    derived_report = artifacts.seal_cohort(cohort)

    ledger_path = output / "replay.sqlite3"
    snapshot_path = output / "training_snapshot.json"
    with replay.ReplayLedger(ledger_path) as ledger:
        admission = ledger.ingest_cohort(cohort)
        snapshot = ledger.build_snapshot(
            campaign_id,
            through_macro_round=0,
            expected_train_cohorts=1,
            output=snapshot_path,
        )
    os.chmod(ledger_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    destination_shards = {int(item["rank"]): item for item in derived_report["shards"]}
    selections: list[dict[str, Any]] = []
    for item in plan:
        learner_rank = int(item["learner_rank"])
        env_id = int(item["global_env_id"])
        selections.append(
            {
                "learner_rank": learner_rank,
                **origins[env_id],
                "destination_shard": dict(destination_shards[learner_rank]),
            }
        )

    training_payload_sha256s = {
        str(item["payload_sha256"]) for item in derived_report["shards"]
    }
    source_heldout_record: dict[str, Any] | None = None
    derived_heldout_record: dict[str, Any] | None = None
    if source_heldout_snapshot is not None:
        source_heldout_campaign_id = (
            source_report["campaign_id"]
            if expected_source_heldout_campaign_id is None
            else _campaign_token(expected_source_heldout_campaign_id)
        )
        source_heldout_record, derived_heldout_record = _derive_fixed_heldout(
            source_snapshot_path=_absolute(source_heldout_snapshot),
            output=output,
            campaign_id=campaign_id,
            expected_cases=expected_heldout_cases,
            expected_source_campaign_id=source_heldout_campaign_id,
            expected_view_contract=source_report["view_contract"],
            expected_action_contract=source_report["action_contract"],
            expected_digests=source_digests,
            training_payload_sha256s=training_payload_sha256s,
            allow_source_eval_schedule_rebinding=(
                allow_source_eval_schedule_rebinding
            ),
        )

    manifest: dict[str, Any] = {
        "schema": FIXTURE_SCHEMA,
        "campaign_id": campaign_id,
        "complete": True,
        "source": {
            "cohort_path": str(source),
            "schema": source_report["schema"],
            "logical_id": source_report["logical_id"],
            "cohort_sha256": source_report["cohort_sha256"],
            "contract_sha256": source_report["contract_sha256"],
            "transition_count": source_report["transition_count"],
            "full_recursive_verification": True,
            **(
                {"data_lineage_sha256": source_report["data_lineage_sha256"]}
                if "data_lineage_sha256" in source_report
                else {}
            ),
            **(
                {"heldout": source_heldout_record}
                if source_heldout_record is not None
                else {}
            ),
        },
        "extraction": {
            "source_ranks": list(source_ranks),
            "environments_per_source_rank": environments_per_source_rank,
            "step_slice": [0, steps],
            "learner_world_size": len(plan),
            "one_environment_per_learner_rank": True,
            "global_env_ids": selected_envs,
            "selections": selections,
        },
        "derived": {
            "cohort": _file_record(output, "cohort/seal.json"),
            "cohort_contract": _file_record(output, "cohort/contract.json"),
            "cohort_schema": derived_report["schema"],
            "cohort_logical_id": derived_report["logical_id"],
            "transition_count": derived_report["transition_count"],
            "replay_snapshot": _file_record(output, "training_snapshot.json"),
            "replay_snapshot_sha256": snapshot["snapshot_sha256"],
            "replay_ledger": _file_record(output, "replay.sqlite3"),
            "replay_admission_status": admission["status"],
            **(
                {"heldout": derived_heldout_record}
                if derived_heldout_record is not None
                else {}
            ),
        },
        "builder": {
            "path": str(_absolute(__file__)),
            "sha256": artifacts.sha256_file(_absolute(__file__)),
        },
    }
    manifest["fixture_content_sha256"] = artifacts.canonical_json_sha256(manifest)
    manifest_path = output / "fixture_manifest.json"
    artifacts.publish_immutable_json(manifest_path, manifest)

    # Revalidate the recursively bound output after its completion marker exists.
    final_cohort = artifacts.verify_cohort(cohort)
    loaded_snapshot = replay.load_snapshot(
        snapshot_path, expected_file_sha256=snapshot["file_sha256"]
    )
    if (
        final_cohort["cohort_sha256"] != derived_report["cohort_sha256"]
        or loaded_snapshot["snapshot_sha256"] != snapshot["snapshot_sha256"]
        or loaded_snapshot["transition_count"] != len(plan) * steps
    ):
        raise FixtureError("derived fixture changed during final verification")
    result = dict(manifest)
    result["manifest_path"] = str(manifest_path)
    result["manifest_file_sha256"] = artifacts.sha256_file(manifest_path)
    return result


def _parse_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from exc
    if not ranks or any(rank < 0 for rank in ranks) or len(ranks) != len(set(ranks)):
        raise argparse.ArgumentTypeError("ranks must be unique non-negative integers")
    return ranks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cohort", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--campaign-id", default="roborender-real-smoke")
    parser.add_argument("--source-ranks", type=_parse_ranks, default=DEFAULT_SOURCE_RANKS)
    parser.add_argument(
        "--environments-per-source-rank",
        type=int,
        default=DEFAULT_ENVIRONMENTS_PER_SOURCE_RANK,
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--source-heldout-snapshot", type=Path)
    parser.add_argument(
        "--expected-source-heldout-campaign-id",
        help=(
            "explicit source campaign pinned by a reused fixed held-out pool; "
            "defaults to the training source campaign"
        ),
    )
    parser.add_argument(
        "--allow-source-eval-schedule-rebinding",
        action="store_true",
        help=(
            "allow only beta/config lineage to be rebound when republishing a "
            "fixed eval pool; policy/runtime/teacher/renderer must still match"
        ),
    )
    parser.add_argument(
        "--expected-heldout-cases", type=int, default=DEFAULT_HELDOUT_CASES
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_fixture(
        source_cohort=args.source_cohort,
        output_root=args.output_root,
        campaign_id=args.campaign_id,
        source_ranks=args.source_ranks,
        environments_per_source_rank=args.environments_per_source_rank,
        steps=args.steps,
        source_heldout_snapshot=args.source_heldout_snapshot,
        expected_source_heldout_campaign_id=(
            args.expected_source_heldout_campaign_id
        ),
        allow_source_eval_schedule_rebinding=(
            args.allow_source_eval_schedule_rebinding
        ),
        expected_heldout_cases=args.expected_heldout_cases,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_path": result["manifest_path"],
                "manifest_file_sha256": result["manifest_file_sha256"],
                "source_cohort_sha256": result["source"]["cohort_sha256"],
                "replay_snapshot_sha256": result["derived"][
                    "replay_snapshot_sha256"
                ],
                "learner_world_size": result["extraction"]["learner_world_size"],
                "transition_count": result["derived"]["transition_count"],
                "heldout_transition_count": (
                    result["derived"]["heldout"]["heldout_transition_count"]
                    if "heldout" in result["derived"]
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
