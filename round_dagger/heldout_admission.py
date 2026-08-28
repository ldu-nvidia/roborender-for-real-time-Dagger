"""Admission gate for the fixed 256-case held-out DAgger replay.

This module does not collect simulator data.  It admits only already sealed,
``purpose=eval`` actor cohorts and publishes the ordinary learner snapshot plus
an immutable semantic manifest binding each replay case to the checked-in
motion split and fixed seed assignment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from round_dagger import artifacts, replay


CASE_COUNT = 256
SEEDS = (17, 29, 43, 59, 71, 89, 101, 127)
CASES_PER_SEED = 32
MINIMUM_SHARDS = 8
PLAN_SCHEMA = "fast_roborender.round_dagger.heldout_motion_capture_plan.v2"
SPLIT_SCHEMA = "roborender.motion_split.v1"
MANIFEST_SCHEMA = "fast_roborender.round_dagger.heldout_case_manifest.v1"
PLANNED_CASE_DOMAIN = "fast_roborender.round_dagger.planned_motion_case.v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class HeldoutAdmissionError(RuntimeError):
    """Held-out replay bytes or their semantic assignment are unsafe."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _semantic_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _regular(path: str | os.PathLike[str], label: str) -> Path:
    value = Path(os.path.abspath(Path(path).expanduser()))
    if value.is_symlink() or not value.is_file():
        raise HeldoutAdmissionError(f"{label} must be a regular non-symlink file")
    return value


def _json(path: Path, label: str) -> Any:
    try:
        return artifacts.load_strict_json(path)
    except Exception as exc:
        raise HeldoutAdmissionError(f"cannot load {label}: {exc}") from exc


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise HeldoutAdmissionError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_sources(
    plan_path: Path, split_path: Path, scenarios_path: Path
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], str], dict[str, str]]:
    plan = _json(plan_path, "held-out plan")
    split = _json(split_path, "motion split")
    scenarios = _json(scenarios_path, "scenario contract")
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise HeldoutAdmissionError("held-out plan schema mismatch")
    capture = plan.get("capture")
    source = plan.get("source")
    partition = plan.get("partition")
    if not all(isinstance(item, Mapping) for item in (capture, source, partition)):
        raise HeldoutAdmissionError("held-out plan sections are malformed")
    if (
        capture.get("case_count") != CASE_COUNT
        or capture.get("fixed_seeds") != list(SEEDS)
        or capture.get("motions_per_seed") != CASES_PER_SEED
        or capture.get("seed_count") != len(SEEDS)
        or capture.get("unique_motion_count") != CASE_COUNT
        or partition.get("eval_count") != CASE_COUNT
        or partition.get("collection_training_count") != 39
        or partition.get("path_overlap_count") != 0
        or partition.get("content_overlap_count") != 0
    ):
        raise HeldoutAdmissionError("held-out plan is not the fixed disjoint 8x32 split")
    contract = capture.get("contract")
    if not isinstance(contract, Mapping) or _semantic_sha(contract) != _sha(
        capture.get("contract_sha256"), "capture contract SHA-256"
    ):
        raise HeldoutAdmissionError("capture contract semantic digest mismatch")
    if contract.get("policy_views") != ["head_color", "hand_left", "hand_right"]:
        raise HeldoutAdmissionError("capture contract is not three-view policy RGB")

    if not isinstance(split, Mapping) or split.get("schema") != SPLIT_SCHEMA:
        raise HeldoutAdmissionError("motion split schema mismatch")
    if artifacts.sha256_file(split_path) != _sha(
        source.get("nested_split_manifest_sha256"), "split manifest SHA-256"
    ):
        raise HeldoutAdmissionError("motion split file differs from the held-out plan")
    raw_motions = split.get("motions")
    if not isinstance(raw_motions, list) or split.get("counts") != {
        "total": 295,
        "collection": 39,
        "eval": CASE_COUNT,
    }:
        raise HeldoutAdmissionError("motion split counts are not 39/256")
    roles: dict[str, tuple[str, str]] = {}
    content_roles: dict[str, str] = {}
    for raw in raw_motions:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "split"}:
            raise HeldoutAdmissionError("motion split row is malformed")
        path = raw["path"]
        digest = _sha(raw["sha256"], "motion SHA-256")
        role = raw["split"]
        if not isinstance(path, str) or Path(path).name != path or role not in {"eval", "collection"}:
            raise HeldoutAdmissionError("motion split row is unsafe")
        if path in roles or (digest in content_roles and content_roles[digest] != role):
            raise HeldoutAdmissionError("motion split has duplicate or cross-role content")
        roles[path] = (digest, role)
        content_roles[digest] = role

    raw_cases = capture.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != CASE_COUNT:
        raise HeldoutAdmissionError("held-out plan does not contain 256 cases")
    cases: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping):
            raise HeldoutAdmissionError("held-out case is malformed")
        seed = SEEDS[index // CASES_PER_SEED]
        env_index = index % CASES_PER_SEED
        motion_path = raw.get("motion_relative_path")
        motion_sha = raw.get("motion_sha256")
        if (
            raw.get("global_case_index") != index
            or raw.get("seed") != seed
            or raw.get("env_index") != env_index
            or roles.get(motion_path) != (motion_sha, "eval")
        ):
            raise HeldoutAdmissionError("held-out case assignment differs from the eval split")
        expected_id = _semantic_sha(
            {
                "domain": PLANNED_CASE_DOMAIN,
                "seed": seed,
                "env_index": env_index,
                "global_case_index": index,
                "motion_relative_path": motion_path,
                "motion_sha256": motion_sha,
                "capture_contract_sha256": capture["contract_sha256"],
            }
        )
        if raw.get("planned_case_id") != expected_id:
            raise HeldoutAdmissionError("planned case ID is not content-derived")
        cases.append(dict(raw))

    if not isinstance(scenarios, list) or len(scenarios) != CASE_COUNT:
        raise HeldoutAdmissionError("scenario contract must contain 256 cases")
    scenario_by_key: dict[tuple[int, int], str] = {}
    for raw in scenarios:
        if not isinstance(raw, Mapping) or set(raw) != {"seed", "env_index", "scenario_id"}:
            raise HeldoutAdmissionError("scenario contract row is malformed")
        key = (raw["seed"], raw["env_index"])
        if key in scenario_by_key:
            raise HeldoutAdmissionError("scenario contract contains duplicate cases")
        scenario_by_key[key] = _sha(raw["scenario_id"], "scenario ID")
    if set(scenario_by_key) != {(seed, index) for seed in SEEDS for index in range(32)}:
        raise HeldoutAdmissionError("scenario contract does not cover fixed 8x32 cases")
    return cases, scenario_by_key, {
        "plan_sha256": artifacts.sha256_file(plan_path),
        "split_sha256": artifacts.sha256_file(split_path),
        "scenario_sha256": artifacts.sha256_file(scenarios_path),
        "capture_contract_sha256": capture["contract_sha256"],
    }


def _load_arrays(payload: Path, expected_sha: str) -> dict[str, np.ndarray]:
    payload = _regular(payload, "rollout payload")
    if artifacts.sha256_file(payload) != expected_sha:
        raise HeldoutAdmissionError("rollout payload SHA-256 mismatch")
    try:
        with np.load(payload, allow_pickle=False) as archive:
            return {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    except Exception as exc:
        raise HeldoutAdmissionError(f"cannot load rollout payload: {exc}") from exc


def _case_rows(
    snapshot: Mapping[str, Any],
    cases: list[dict[str, Any]],
    scenario_by_key: Mapping[tuple[int, int], str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if snapshot["transition_count"] != CASE_COUNT or len(snapshot["resolved_shards"]) < MINIMUM_SHARDS:
        raise HeldoutAdmissionError("held-out replay must be 256 one-step cases in at least 8 shards")
    by_env: dict[int, dict[str, Any]] = {}
    observation_keys: list[str] | None = None
    for shard in snapshot["resolved_shards"]:
        arrays = _load_arrays(Path(shard["payload_path"]), shard["payload_sha256"])
        required = {"policy_rgb", "expert_actions", "obs__actor_obs", "global_env_ids", "episode_ids", "episode_steps"}
        if not required <= set(arrays):
            raise HeldoutAdmissionError("held-out shard lacks RGB, actor observations, or expert actions")
        current_obs = sorted(name for name in arrays if name.startswith("obs__"))
        if observation_keys is None:
            observation_keys = current_obs
        elif current_obs != observation_keys:
            raise HeldoutAdmissionError("held-out shard observation keys differ")
        rgb, expert = arrays["policy_rgb"], arrays["expert_actions"]
        if rgb.shape[0] != 1 or expert.shape[:2] != rgb.shape[:2]:
            raise HeldoutAdmissionError("each held-out environment must contribute exactly one transition")
        envs = arrays["global_env_ids"]
        envs = envs[0] if envs.ndim == 2 else envs
        for offset, env in enumerate(envs.tolist()):
            env = int(env)
            if env in by_env or not 0 <= env < CASE_COUNT:
                raise HeldoutAdmissionError("held-out global case coverage overlaps or escapes 0..255")
            if not np.isfinite(expert[0, offset]).all() or any(
                np.issubdtype(arrays[name].dtype, np.inexact)
                and not np.isfinite(arrays[name][0, offset]).all()
                for name in current_obs
            ):
                raise HeldoutAdmissionError("held-out labels or observations are non-finite")
            obs_digests = {
                name: hashlib.sha256(np.ascontiguousarray(arrays[name][0, offset]).tobytes()).hexdigest()
                for name in current_obs
            }
            by_env[env] = {
                "cohort_logical_id": shard["logical_id"],
                "shard_id": shard["shard_id"],
                "shard_manifest_sha256": shard["manifest_sha256"],
                "payload_sha256": shard["payload_sha256"],
                "episode_id": int(arrays["episode_ids"][0, offset]),
                "episode_step": int(arrays["episode_steps"][0, offset]),
                "policy_rgb_sha256": hashlib.sha256(np.ascontiguousarray(rgb[0, offset]).tobytes()).hexdigest(),
                "expert_actions_sha256": hashlib.sha256(np.ascontiguousarray(expert[0, offset]).tobytes()).hexdigest(),
                "actor_observations_sha256": _semantic_sha(obs_digests),
            }
    if set(by_env) != set(range(CASE_COUNT)):
        raise HeldoutAdmissionError("held-out replay does not cover every case exactly once")
    rows = []
    for case in cases:
        index = case["global_case_index"]
        rows.append(
            {
                **case,
                "scenario_id": scenario_by_key[(case["seed"], case["env_index"])],
                **by_env[index],
            }
        )
    return rows, observation_keys or []


def build_fixed_heldout_bundle(
    cohort_paths: Iterable[str | os.PathLike[str]],
    *,
    plan: str | os.PathLike[str],
    motion_split: str | os.PathLike[str],
    scenarios: str | os.PathLike[str],
    snapshot_output: str | os.PathLike[str],
    manifest_output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Admit sealed eval cohorts and publish immutable learner/semantic manifests."""

    plan_path = _regular(plan, "held-out plan")
    split_path = _regular(motion_split, "motion split")
    scenarios_path = _regular(scenarios, "scenario contract")
    cases, scenario_by_key, source_hashes = _validate_sources(plan_path, split_path, scenarios_path)
    paths = [Path(path) for path in cohort_paths]
    if not paths:
        raise HeldoutAdmissionError("at least one eval cohort is required")
    try:
        snapshot = replay.build_heldout_snapshot(paths, output=snapshot_output)
        loaded = replay.load_heldout_snapshot(snapshot["path"], expected_file_sha256=snapshot["file_sha256"])
    except Exception as exc:
        raise HeldoutAdmissionError(f"held-out snapshot admission failed: {exc}") from exc
    rows, observation_keys = _case_rows(loaded, cases, scenario_by_key)
    body = {
        "schema": MANIFEST_SCHEMA,
        "purpose": "eval",
        "campaign_id": loaded["campaign_id"],
        "case_count": CASE_COUNT,
        "transition_count": loaded["transition_count"],
        "shard_count": len(loaded["resolved_shards"]),
        "view_contract": loaded["view_contract"],
        "action_contract": loaded["action_contract"],
        "observation_keys": observation_keys,
        "snapshot_path": str(Path(snapshot["path"])),
        "snapshot_file_sha256": snapshot["file_sha256"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "source_hashes": source_hashes,
        "cases": rows,
    }
    body["case_inventory_sha256"] = _semantic_sha(rows)
    body["manifest_sha256"] = _semantic_sha(body)
    output = Path(os.path.abspath(Path(manifest_output).expanduser()))
    try:
        file_sha = artifacts.publish_immutable_json(output, body)
    except Exception as exc:
        raise HeldoutAdmissionError(f"case manifest publication failed: {exc}") from exc
    return {
        "snapshot": snapshot,
        "case_manifest": {**body, "path": str(output), "file_sha256": file_sha},
    }


def verify_fixed_heldout_bundle(
    snapshot_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    plan: str | os.PathLike[str],
    motion_split: str | os.PathLike[str],
    scenarios: str | os.PathLike[str],
) -> dict[str, Any]:
    """Recursively verify snapshot bytes and their fixed semantic case manifest."""

    manifest_file = _regular(manifest_path, "held-out case manifest")
    document = _json(manifest_file, "held-out case manifest")
    if not isinstance(document, Mapping) or document.get("schema") != MANIFEST_SCHEMA:
        raise HeldoutAdmissionError("held-out case manifest schema mismatch")
    semantic = dict(document)
    declared = semantic.pop("manifest_sha256", None)
    if _semantic_sha(semantic) != _sha(declared, "case manifest semantic SHA-256"):
        raise HeldoutAdmissionError("case manifest semantic digest mismatch")
    loaded = replay.load_heldout_snapshot(
        snapshot_path, expected_file_sha256=document.get("snapshot_file_sha256")
    )
    cases, scenario_by_key, source_hashes = _validate_sources(
        _regular(plan, "held-out plan"),
        _regular(motion_split, "motion split"),
        _regular(scenarios, "scenario contract"),
    )
    rows, observation_keys = _case_rows(loaded, cases, scenario_by_key)
    expected = {
        "campaign_id": loaded["campaign_id"],
        "case_count": CASE_COUNT,
        "transition_count": loaded["transition_count"],
        "shard_count": len(loaded["resolved_shards"]),
        "view_contract": loaded["view_contract"],
        "action_contract": loaded["action_contract"],
        "observation_keys": observation_keys,
        "snapshot_file_sha256": loaded["file_sha256"],
        "snapshot_sha256": loaded["snapshot_sha256"],
        "source_hashes": source_hashes,
        "cases": rows,
        "case_inventory_sha256": _semantic_sha(rows),
    }
    mismatches = [key for key, value in expected.items() if document.get(key) != value]
    if mismatches:
        raise HeldoutAdmissionError("held-out semantic admission mismatch: " + ", ".join(mismatches))
    return {
        "status": "verified",
        "case_count": CASE_COUNT,
        "shard_count": len(loaded["resolved_shards"]),
        "snapshot_sha256": loaded["snapshot_sha256"],
        "manifest_sha256": declared,
        "manifest_file_sha256": artifacts.sha256_file(manifest_file),
    }


__all__ = [
    "HeldoutAdmissionError",
    "build_fixed_heldout_bundle",
    "verify_fixed_heldout_bundle",
]
