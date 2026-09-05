"""Append-only controller for one local native-H200 BC/DAgger experiment.

The controller owns experiment decisions, not model code.  It emits phase-gated
launch plans for the protocol-bound actor/learner/evaluator entrypoints and records immutable,
hash-chained evidence after each phase.  Test data are represented by a digest
only until the final checkpoint has been selected.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from round_dagger.native_actor_protocol import (
    GENERATION_IDS,
    NATIVE_ACTOR_GENERATIONS,
    train_round_relative_root,
)


CONTRACT_SCHEMA = "fast_roborender.round_dagger.native_h200_experiment.v4"
GROUP_REGISTRY_SCHEMA = "fast_roborender.round_dagger.group_registry.v2"
SPLIT_SCHEMA = "fast_roborender.round_dagger.group_split.v1"
EVENT_SCHEMA = "fast_roborender.round_dagger.native_h200_event.v1"
TEST_REGISTRY_SCHEMA = "fast_roborender.round_dagger.sealed_test_registry.v1"
VALIDATION_REGISTRY_SCHEMA = "fast_roborender.round_dagger.validation_panel_registry.v1"
VALIDATION_MSE_RESULT_SCHEMA = "fast_roborender.round_dagger.validation_mse_result.v1"
FINAL_TEST_RESULT_SCHEMA = "fast_roborender.round_dagger.final_test_result.v1"
TRAIN_MANIFEST_SCHEMA = "fast_roborender.round_dagger.train_replay_manifest.v3"
RESOURCE_SHAPE = "local.h200.8gpu"
GPU_NAME = "NVIDIA H200"
COMPUTE_CAPABILITY = "9.0"
BETAS = tuple(item.beta for item in NATIVE_ACTOR_GENERATIONS if item.stage == "dagger")
SPLITS = ("train", "validation", "test")
_ASSIGNMENT_CACHE: dict[str, dict[str, tuple[str, ...]]] = {}
_ASSIGNMENT_CACHE_LIMIT = 8
CANONICAL_SOURCE_ROOT = Path("/workspace/project/roborender")
CANONICAL_ARTIFACT_ROOT = CANONICAL_SOURCE_ROOT / "data"
CANONICAL_CAMPAIGN_ROOT = CANONICAL_ARTIFACT_ROOT
CANONICAL_VALIDATION_ROOT = CANONICAL_ARTIFACT_ROOT / "validation"
CANONICAL_TEST_ROOT = CANONICAL_ARTIFACT_ROOT / "test"
CANONICAL_CAMPAIGN_ID = "dagger-redo-v001"
TRAIN_ROUND_SUBDIRECTORIES = (
    "collection",
    "rendered",
    "replay/shards",
    "replay/manifests",
    "checkpoints/candidates",
    "checkpoints/selected",
    "logs",
    "telemetry",
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CONTRACT_KEYS = {
    "schema", "protocol_id", "campaign_root", "source_root", "group_split",
    "hardware", "topology", "collection_budget", "generations",
    "validation_panel", "optimization", "dagger", "launch",
}
_GROUP_SPLIT_KEYS = {
    "strategy", "seed", "group_count", "train_count", "validation_count",
    "test_count", "stratify_by",
}
_HARDWARE_KEYS = {"actor", "learner", "evaluation"}
_ROLE_HARDWARE_KEYS = {
    "resource_shape", "gpu_name", "compute_capability", "nodes",
    "gpus_per_node", "exclusive",
}
_TOPOLOGY_KEYS = {
    "macro_round_count", "actor_process_count", "gpus_per_actor_process",
    "environments_per_actor_process", "collection_wave_count",
    "total_environment_streams",
    "steps_per_environment", "nominal_bc_target_transitions",
}
_COLLECTION_BUDGET_KEYS = {
    "entrypoint", "generation_id",
    "hard_wall_time_seconds", "probe_budget_seconds", "seal_reserve_seconds",
    "probe_environments_per_actor_process", "probe_steps_per_environment",
    "maximum_peak_memory_fraction", "throughput_tie_fraction",
    "host_physical_gpu_count", "admitted_physical_gpu_indices",
    "physical_execution_concurrency", "virtual_actor_world_size",
    "actor_wave_scheduling", "whole_waves_only",
}
_GENERATION_KEYS = {"generation_id", "stage", "round", "beta"}
_VALIDATION_PANEL_KEYS = {
    "registry_path", "registry_sha256", "paired_seeds", "cases_per_seed",
}
_OPTIMIZATION_KEYS = {"stopping_rule"}
_PLATEAU_KEYS = {
    "minimum_balanced_exposure_units", "maximum_balanced_exposure_units",
    "validation_interval_balanced_exposure_units", "patience",
    "validation_mse_min_delta",
}
_DAGGER_KEYS = {
    "betas", "newest_replay_fraction", "prior_replay_fraction",
    "global_validation_mse_min_delta",
    "stop_after_no_progress_rounds",
}
_LAUNCH_KEYS = {"actor", "learner", "validation", "test"}
_LAUNCH_ROLE_KEYS = {
    "entrypoint", "parallel_processes", "gpus_per_process", "fixed_environment",
    "input_bindings",
}
_POINT_KEYS = {
    "optimizer_examples", "newest_source_examples", "prior_source_examples",
    "newest_source_population", "prior_source_population",
    "balanced_exposure_unit_examples", "balanced_exposure_units",
    "optimizer_examples_per_step", "validation_mse", "checkpoint_sha256",
}
_VALIDATION_MSE_RESULT_KEYS = {
    "schema", "campaign_id", "generation_id", "macro_round",
    "checkpoint_sha256", "validation_artifact_sha256",
    "validation_snapshot_sha256", "split_sha256", "sampling_policy",
    "optimizer_examples", "newest_source_examples", "prior_source_examples",
    "newest_source_population", "prior_source_population",
    "balanced_exposure_unit_examples", "balanced_exposure_units",
    "newest_expected_passes", "prior_expected_passes",
    "optimizer_examples_per_step", "validation_examples", "validation_mse",
}
_FINAL_TEST_RESULT_KEYS = {
    "schema", "campaign_id", "protocol_id", "generation_id",
    "checkpoint_sha256", "test_registry_sha256", "test_cases_sha256",
    "split_sha256", "episode_count", "success_count", "safety_violations",
    "per_stratum",
}
_TRAIN_REPLAY_SNAPSHOT_KEYS = {
    "path", "file_sha256", "snapshot_sha256", "through_macro_round",
}


class NativeH200ExperimentError(RuntimeError):
    """The protocol contract, state transition, or immutable evidence is invalid."""


def _canonical(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NativeH200ExperimentError("value is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise NativeH200ExperimentError(f"{label} must be a lowercase SHA-256")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise NativeH200ExperimentError(f"{label} must be a safe token")
    return value


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise NativeH200ExperimentError(f"{label} key-set differs")
    return dict(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise NativeH200ExperimentError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeH200ExperimentError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise NativeH200ExperimentError(f"{label} is outside its finite range")
    return result


def _absolute(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise NativeH200ExperimentError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or os.path.abspath(value) != value:
        raise NativeH200ExperimentError(f"{label} must be canonical and absolute")
    return path


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise NativeH200ExperimentError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise NativeH200ExperimentError(f"{label} is unsafe")
    return value


def _load_json(path: Path) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise NativeH200ExperimentError(f"cannot open JSON: {path}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise NativeH200ExperimentError(f"JSON is not a regular file: {path}")
        payload = b""
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            payload += chunk
    finally:
        os.close(descriptor)

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise NativeH200ExperimentError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=pairs,
                          parse_constant=lambda raw: (_ for _ in ()).throw(
                              ValueError(f"non-finite number {raw}")))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeH200ExperimentError(f"invalid JSON: {path}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise NativeH200ExperimentError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _safe_launch_executable(source_root: Path, relative: str) -> Path:
    """Resolve one checked-in executable without following a symlink."""

    candidate = source_root / _relative(relative, "fleet entrypoint")
    try:
        resolved = candidate.resolve(strict=True)
        info = os.lstat(candidate)
        resolved.relative_to(source_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise NativeH200ExperimentError(
            f"fleet entrypoint is missing or escaped the source root: {candidate}"
        ) from exc
    if (
        resolved != candidate
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or not (stat.S_IMODE(info.st_mode) & 0o111)
    ):
        raise NativeH200ExperimentError(
            f"fleet entrypoint must be an executable regular file: {candidate}"
        )
    return candidate


def _resolve_checkpoint_path(
    campaign_root: Path,
    generation_root: Path,
    expected_sha256: str,
    *,
    include_bootstrap: bool = False,
) -> Path:
    """Find one digest-pinned checkpoint only in canonical policy namespaces."""

    expected = _sha(expected_sha256, "fleet checkpoint")
    search_roots = [
        generation_root / "checkpoints/selected",
        generation_root / "checkpoints/candidates",
        campaign_root / "final/checkpoint",
    ]
    if include_bootstrap:
        search_roots.append(
            campaign_root / "BC/collection/checkpoints/input"
        )
    matches: list[Path] = []
    for root in search_roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for candidate in sorted(root.rglob("*.pt")):
            try:
                info = os.lstat(candidate)
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root.resolve(strict=True))
            except (OSError, ValueError):
                continue
            if (
                resolved == candidate
                and stat.S_ISREG(info.st_mode)
                and not stat.S_ISLNK(info.st_mode)
                and _sha256_file(candidate) == expected
            ):
                matches.append(candidate)
    unique: dict[tuple[int, int], Path] = {}
    for path in matches:
        info = os.stat(path)
        unique.setdefault((info.st_dev, info.st_ino), path)
    if len(unique) != 1:
        raise NativeH200ExperimentError(
            "fleet checkpoint digest did not resolve to one canonical inode"
        )
    return next(iter(unique.values()))


def _load_pinned_canonical_json(
    path: Path, expected_sha256: str, *, label: str
) -> dict[str, Any]:
    """Read and hash one regular immutable inode, then decode those exact bytes."""

    expected = _sha(expected_sha256, f"{label} file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise NativeH200ExperimentError(f"cannot safely open {label}: {path}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise NativeH200ExperimentError(f"{label} is not a regular file: {path}")
        if stat.S_IMODE(info.st_mode) & 0o222:
            raise NativeH200ExperimentError(f"{label} is not immutable: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise NativeH200ExperimentError(f"{label} file SHA-256 differs")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise NativeH200ExperimentError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite number {raw}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeH200ExperimentError(f"invalid {label}: {path}") from exc
    if not isinstance(document, Mapping) or _canonical(document) != payload:
        raise NativeH200ExperimentError(f"{label} is not canonical JSON")
    return dict(document)


def _publish(path: Path, value: Any) -> str:
    """Create one immutable file atomically and never replace a conflicting file."""

    payload = _canonical(value)
    expected = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    staged = Path(name)
    created = False
    staged_identity: os.stat_result | None = None
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise NativeH200ExperimentError("short write while staging artifact")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        staged_identity = os.fstat(descriptor)
        try:
            os.link(staged, path, follow_symlinks=False)
            created = True
        except FileExistsError:
            existing = _load_json(path)
            if _canonical(existing) != payload:
                raise NativeH200ExperimentError(f"immutable artifact conflicts: {path}")
            return expected
        final_descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            final_identity = os.fstat(final_descriptor)
            if (final_identity.st_dev, final_identity.st_ino) != (
                staged_identity.st_dev, staged_identity.st_ino
            ):
                raise NativeH200ExperimentError("published artifact inode changed")
        finally:
            os.close(final_descriptor)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return expected
    except BaseException:
        if created and staged_identity is not None:
            try:
                current = os.lstat(path)
                if (current.st_dev, current.st_ino) == (
                    staged_identity.st_dev, staged_identity.st_ino
                ):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(descriptor)
        try:
            staged.unlink()
        except FileNotFoundError:
            pass


def validate_contract(value: Any) -> dict[str, Any]:
    contract = _exact(value, _CONTRACT_KEYS, "experiment contract")
    if contract["schema"] != CONTRACT_SCHEMA:
        raise NativeH200ExperimentError("experiment contract schema differs")
    _token(contract["protocol_id"], "protocol ID")
    campaign_root = _absolute(contract["campaign_root"], "campaign root")
    if campaign_root != CANONICAL_CAMPAIGN_ROOT:
        raise NativeH200ExperimentError(
            f"campaign root must be the canonical data root: "
            f"{CANONICAL_CAMPAIGN_ROOT}"
        )
    source_root = _absolute(contract["source_root"], "source root")
    if (
        source_root != CANONICAL_SOURCE_ROOT
        or not source_root.is_dir()
        or source_root.is_symlink()
    ):
        raise NativeH200ExperimentError("source root is not the canonical repository")

    split = _exact(contract["group_split"], _GROUP_SPLIT_KEYS, "group split")
    if split["strategy"] != "deterministic_stratified_group_sha256_v1":
        raise NativeH200ExperimentError("group split strategy differs")
    _token(split["seed"], "split seed")
    count = _integer(split["group_count"], "group_count", minimum=3)
    counts = [_integer(split[f"{name}_count"], f"{name}_count", minimum=1)
              for name in SPLITS]
    if count != 33 or counts != [23, 5, 5]:
        raise NativeH200ExperimentError("group split must be the exact 33 -> 23/5/5 contract")
    strata = split["stratify_by"]
    if not isinstance(strata, list) or not strata or len(set(strata)) != len(strata):
        raise NativeH200ExperimentError("stratify_by must contain unique fields")
    for field in strata:
        _token(field, "stratification field")

    hardware = _exact(contract["hardware"], _HARDWARE_KEYS, "hardware")
    expected_hardware = {
        "resource_shape": RESOURCE_SHAPE, "gpu_name": GPU_NAME,
        "compute_capability": COMPUTE_CAPABILITY, "nodes": 1,
        "gpus_per_node": 8, "exclusive": True,
    }
    for role in _HARDWARE_KEYS:
        observed = _exact(hardware[role], _ROLE_HARDWARE_KEYS, f"{role} hardware")
        if observed != expected_hardware:
            raise NativeH200ExperimentError(f"{role} is not exact one-node/eight-H200")

    topology = _exact(contract["topology"], _TOPOLOGY_KEYS, "topology")
    if topology != {
        "macro_round_count": 5,
        "actor_process_count": 1,
        "gpus_per_actor_process": 1,
        "environments_per_actor_process": 128,
        "collection_wave_count": 2,
        "total_environment_streams": 256,
        "steps_per_environment": 512,
        "nominal_bc_target_transitions": 131072,
    }:
        raise NativeH200ExperimentError("experiment topology differs")

    collection = _exact(
        contract["collection_budget"], _COLLECTION_BUDGET_KEYS, "BC collection budget"
    )
    if (
        _relative(collection["entrypoint"], "BC saturation entrypoint")
        != "scripts/round_dagger/run_native_h200_actor_fleet.py"
        or collection["generation_id"] != GENERATION_IDS[0]
        or collection["hard_wall_time_seconds"] != 5400
        or collection["probe_budget_seconds"] != 900
        or collection["seal_reserve_seconds"] != 600
        or collection["probe_environments_per_actor_process"] != [128]
        or collection["probe_steps_per_environment"] != 128
        or collection["maximum_peak_memory_fraction"] != 0.85
        or collection["throughput_tie_fraction"] != 0.05
        or collection["host_physical_gpu_count"] != 8
        or collection["admitted_physical_gpu_indices"] != [0]
        or collection["physical_execution_concurrency"] != 1
        or collection["virtual_actor_world_size"] != 1
        or collection["actor_wave_scheduling"]
        != "sequential_single_actor_waves_v1"
        or collection["whole_waves_only"] is not True
    ):
        raise NativeH200ExperimentError("BC collection budget/saturation policy differs")
    if (
        collection["probe_budget_seconds"] + collection["seal_reserve_seconds"]
        >= collection["hard_wall_time_seconds"]
    ):
        raise NativeH200ExperimentError("BC collection leaves no rollout time")

    generations = contract["generations"]
    if not isinstance(generations, list) or len(generations) != len(GENERATION_IDS):
        raise NativeH200ExperimentError("generation registry differs")
    expected_generations = [
        {"generation_id": GENERATION_IDS[0], "stage": "bc", "round": None, "beta": 1.0},
        *[
            {
                "generation_id": GENERATION_IDS[index + 1],
                "stage": "dagger",
                "round": index + 1,
                "beta": beta,
            }
            for index, beta in enumerate(BETAS)
        ],
    ]
    for index, raw in enumerate(generations):
        row = _exact(raw, _GENERATION_KEYS, f"generation {index}")
        if row != expected_generations[index]:
            raise NativeH200ExperimentError("generation naming/round mapping differs")
        round_root = campaign_root / train_round_relative_root(row["generation_id"])
        if round_root.parent != campaign_root:
            raise NativeH200ExperimentError("train round root escaped the campaign")

    panel = _exact(
        contract["validation_panel"], _VALIDATION_PANEL_KEYS, "validation MSE panel"
    )
    registry_relative = _relative(panel["registry_path"], "validation registry path")
    _sha(panel["registry_sha256"], "validation registry")
    seeds = panel["paired_seeds"]
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise NativeH200ExperimentError("paired validation seeds are invalid")
    _integer(panel["cases_per_seed"], "validation cases per seed", minimum=1)
    registry_path = source_root / registry_relative
    registry = _exact(_load_json(registry_path), {
        "schema", "panel_id", "paired_seeds", "case_ids", "case_count",
    }, "validation registry")
    if registry["schema"] != VALIDATION_REGISTRY_SCHEMA:
        raise NativeH200ExperimentError("validation registry schema differs")
    _token(registry["panel_id"], "validation panel ID")
    case_ids = registry["case_ids"]
    if (not isinstance(case_ids, list) or not case_ids
            or len(set(case_ids)) != len(case_ids)
            or any(_token(item, "validation case ID") != item for item in case_ids)):
        raise NativeH200ExperimentError("validation case IDs are invalid")
    if (registry["paired_seeds"] != seeds
            or len(case_ids) != panel["cases_per_seed"]
            or registry["case_count"] != len(seeds) * len(case_ids)
            or _digest(registry) != panel["registry_sha256"]):
        raise NativeH200ExperimentError("validation registry differs from panel commitment")

    optimization = _exact(contract["optimization"], _OPTIMIZATION_KEYS, "optimization")
    rule = _exact(optimization["stopping_rule"], _PLATEAU_KEYS, "shared stopping rule")
    minimum = _number(
        rule["minimum_balanced_exposure_units"],
        "minimum balanced exposure units",
        minimum=0.0,
    )
    maximum = _number(
        rule["maximum_balanced_exposure_units"],
        "maximum balanced exposure units",
        minimum=minimum,
    )
    interval = _number(
        rule["validation_interval_balanced_exposure_units"],
        "validation interval balanced exposure units",
        minimum=0.0,
    )
    if minimum != 2.0 or maximum != 10.0 or interval != 0.25:
        raise NativeH200ExperimentError("shared source-exposure budget differs")
    if _integer(rule["patience"], "patience", minimum=1) != 4:
        raise NativeH200ExperimentError("validation patience must be exactly four")
    _number(rule["validation_mse_min_delta"], "validation MSE min_delta", minimum=0.0)
    if interval <= 0 or minimum % interval or maximum % interval:
        raise NativeH200ExperimentError("source-exposure budgets must align to validation cadence")

    dagger = _exact(contract["dagger"], _DAGGER_KEYS, "DAgger")
    if dagger["betas"] != list(BETAS):
        raise NativeH200ExperimentError("DAgger beta schedule differs")
    newest = _number(dagger["newest_replay_fraction"], "newest replay fraction", minimum=0)
    prior = _number(dagger["prior_replay_fraction"], "prior replay fraction", minimum=0)
    if newest != 0.5 or prior != 0.5 or newest + prior != 1.0:
        raise NativeH200ExperimentError("cumulative replay mixture must be exact 50/50")
    _number(
        dagger["global_validation_mse_min_delta"],
        "global validation MSE min_delta",
        minimum=0,
    )
    if dagger["stop_after_no_progress_rounds"] != 2:
        raise NativeH200ExperimentError("outer stopping patience must be two rounds")

    launch = _exact(contract["launch"], _LAUNCH_KEYS, "launch contract")
    expected = {
        "actor": ("scripts/round_dagger/run_native_h200_actor_fleet.py", 1, 8,
                  ["checkpoint", "train_group_split", "generation_conditioning_output"]),
        "learner": ("scripts/round_dagger/run_native_h200_learner_fleet.py", 1, 8,
                    ["train_replay_manifest", "train_replay_snapshot",
                     "parent_checkpoint"]),
        "validation": ("scripts/round_dagger/run_native_h200_mse_fleet.py", 1, 8,
                       ["checkpoint", "validation_registry", "validation_result_output"]),
        "test": ("scripts/round_dagger/run_native_h200_test_fleet.py", 1, 8,
                 ["checkpoint", "test_registry", "test_result_output"]),
    }
    for role, (entrypoint, processes, per_process, input_bindings) in expected.items():
        row = _exact(launch[role], _LAUNCH_ROLE_KEYS, f"{role} launch")
        if (_relative(row["entrypoint"], f"{role} entrypoint") != entrypoint
                or row["parallel_processes"] != processes
                or row["gpus_per_process"] != per_process
                or processes * per_process != 8
                or row["input_bindings"] != input_bindings):
            raise NativeH200ExperimentError(f"{role} launch does not consume all eight H200s")
        _safe_launch_executable(source_root, row["entrypoint"])
        fixed = row["fixed_environment"]
        if not isinstance(fixed, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) and k and "\x00" not in v
            for k, v in fixed.items()
        ):
            raise NativeH200ExperimentError(f"{role} fixed environment is invalid")
        if role in {"actor", "test"} and fixed.get(
            "ROUND_DAGGER_OBSERVATION_SOURCE"
        ) != "native":
            raise NativeH200ExperimentError(f"{role} must select native observations")
        if role == "actor" and fixed.get(
            "ROUND_DAGGER_ACTOR_WORLD_SIZE"
        ) != "1":
            raise NativeH200ExperimentError(
                "actor collection must use the exact world-1 process topology"
            )
        if role == "validation" and fixed.get(
            "ROUND_DAGGER_EVALUATION_PURPOSE"
        ) != "validation_mse":
            raise NativeH200ExperimentError("validation must be offline MSE only")
        if role == "test" and fixed.get(
            "ROUND_DAGGER_EVALUATION_PURPOSE"
        ) != "test_success":
            raise NativeH200ExperimentError("test must report success only")
        if role == "learner" and (
            fixed.get("ROUND_DAGGER_LEARNER_NNODES") != "1"
            or fixed.get("ROUND_DAGGER_LEARNER_NPROC_PER_NODE") != "8"
        ):
            raise NativeH200ExperimentError("learner launch is not eight-rank")
    return contract


def load_contract(path: Path) -> dict[str, Any]:
    return validate_contract(_load_json(path))


def _sha256_regular_file(path: Path, *, expected_size: int) -> str:
    """Hash one no-follow regular file and reject a concurrent replacement."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise NativeH200ExperimentError(f"cannot open motion file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise NativeH200ExperimentError(f"motion file size/type differs: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise NativeH200ExperimentError(f"motion file changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _group_rows(value: Any, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry = _exact(value, {"schema", "groups"}, "group registry")
    if registry["schema"] != GROUP_REGISTRY_SCHEMA or not isinstance(registry["groups"], list):
        raise NativeH200ExperimentError("group registry schema differs")
    keys = contract["group_split"]["stratify_by"]
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    paths: set[str] = set()
    motion_hashes: set[str] = set()
    source_root = Path(contract["source_root"])
    for raw in registry["groups"]:
        row = _exact(raw, {
            "group_id", "relative_path", "size_bytes", "motion_sha256", "strata",
        }, "group")
        group_id = _token(row["group_id"], "group ID")
        relative_path = _relative(row["relative_path"], "group relative path")
        size_bytes = _integer(row["size_bytes"], "group motion size", minimum=1)
        motion_sha256 = _sha(row["motion_sha256"], "motion hash")
        strata = _exact(row["strata"], set(keys), "group strata")
        if (group_id in identities or relative_path in paths
                or motion_sha256 in motion_hashes):
            raise NativeH200ExperimentError(
                "group registry contains duplicate identity, path, or content"
            )
        identities.add(group_id)
        paths.add(relative_path)
        motion_hashes.add(motion_sha256)
        for key, item in strata.items():
            _token(item, f"stratum {key}")
        motion_path = source_root / relative_path
        if motion_path.name != group_id:
            raise NativeH200ExperimentError("group ID must equal the motion basename")
        if _sha256_regular_file(motion_path, expected_size=size_bytes) != motion_sha256:
            raise NativeH200ExperimentError(f"motion content differs: {relative_path}")
        normalized = {
            "group_id": group_id,
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "motion_sha256": motion_sha256,
            "strata": strata,
        }
        rows.append(normalized)
    rows.sort(key=lambda row: row["group_id"])
    if len(rows) != contract["group_split"]["group_count"]:
        raise NativeH200ExperimentError("group registry count differs")
    return rows


def _assign_groups(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, list[str]]:
    """Choose two covariate-balanced held-out sets, then use the remainder.

    With only five validation and five test groups, a greedy 23/5/5 fill can
    consume rare covariates in train before either held-out set receives one.
    Enumerating five-group candidates is small for the frozen 33-group corpus
    and lets us require full marginal coverage whenever it is mathematically
    supportable.  The objective and all ties are deterministic.
    """

    split = contract["group_split"]
    if (split["validation_count"], split["test_count"]) != (5, 5):
        raise NativeH200ExperimentError("balanced assignment requires 5/5 held-out groups")
    seed = split["seed"]
    fields = tuple(split["stratify_by"])
    by_id = {row["group_id"]: row for row in rows}
    ids = tuple(sorted(by_id))
    total = len(ids)
    frequencies = Counter(
        (field, row["strata"][field]) for row in rows for field in fields
    )
    values = {
        field: tuple(sorted({row["strata"][field] for row in rows}))
        for field in fields
    }
    cache_key = _digest({
        "seed": seed,
        "targets": {name: split[f"{name}_count"] for name in SPLITS},
        "fields": fields,
        "groups": [
            {
                "group_id": row["group_id"],
                "strata": {field: row["strata"][field] for field in fields},
            }
            for row in rows
        ],
    })
    cached = _ASSIGNMENT_CACHE.get(cache_key)
    if cached is not None:
        return {name: list(cached[name]) for name in SPLITS}
    # Requiring each held-out split to cover a marginal also leaves at least
    # one such group in train.  Sparse/uncoverable marginals remain optimized
    # by the residual objective rather than being hidden by a false guarantee.
    covered_fields = tuple(
        field for field in fields
        if len(values[field]) <= 5
        and all(frequencies[(field, value)] >= 3 for value in values[field])
    )

    def tokens(group_ids: Sequence[str]) -> Counter[tuple[str, str]]:
        return Counter(
            (field, by_id[group_id]["strata"][field])
            for group_id in group_ids for field in fields
        )

    def covers(counts: Counter[tuple[str, str]]) -> bool:
        return all(
            counts[(field, value)] > 0
            for field in covered_fields for value in values[field]
        )

    def residual_score(counts: Counter[tuple[str, str]], size: int) -> float:
        score = 0.0
        for token, frequency in frequencies.items():
            expected = frequency * size / total
            score += (counts[token] - expected) ** 2 / max(expected, 1.0)
        return score

    candidates: list[tuple[float, str, tuple[str, ...], Counter[tuple[str, str]]]] = []
    for group_ids in itertools.combinations(ids, 5):
        counts = tokens(group_ids)
        if not covers(counts):
            continue
        tie = hashlib.sha256(
            f"{seed}:heldout:{','.join(group_ids)}".encode("utf-8")
        ).hexdigest()
        candidates.append((residual_score(counts, 5), tie, group_ids, counts))
    if not candidates:
        raise NativeH200ExperimentError(
            "no disjoint representative validation/test assignment exists"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))

    # The best few hundred marginal candidates are enough to consider every
    # optimal-quality boundary while keeping synthetic unit fixtures fast.
    pool = candidates[: min(512, len(candidates))]
    best: tuple[
        float, int, str, tuple[str, ...], tuple[str, ...]
    ] | None = None
    all_ids = set(ids)
    for left in pool:
        left_ids = set(left[2])
        for right in pool:
            if left_ids & set(right[2]):
                continue
            train_ids = tuple(sorted(all_ids - left_ids - set(right[2])))
            train_counts = tokens(train_ids)
            if not covers(train_counts):
                continue
            # Optimize balance across all three splits first.  Validation/test
            # symmetry is only a tie-breaker: making their singleton bins the
            # same can needlessly leave the 23-group train remainder at 9/7/7
            # when complementary 2/2/1 panels allow 8/8/7.
            pair_gap = sum(
                abs(left[3][token] - right[3][token]) for token in frequencies
            )
            objective = (
                left[0] + right[0] + residual_score(train_counts, len(train_ids))
            )
            ordered_pair = tuple(sorted((left[2], right[2])))
            tie = hashlib.sha256(
                f"{seed}:pair:{ordered_pair!r}".encode("utf-8")
            ).hexdigest()
            proposal = (objective, pair_gap, tie, ordered_pair[0], ordered_pair[1])
            if best is None or proposal[:3] < best[:3]:
                best = proposal
    if best is None:
        raise NativeH200ExperimentError(
            "no disjoint representative validation/test assignment exists"
        )
    first, second = best[3], best[4]
    role_tie = hashlib.sha256(f"{seed}:validation-role:{first!r}".encode()).digest()
    validation, test = (first, second) if role_tie[0] & 1 else (second, first)
    heldout = set(validation) | set(test)
    result = {
        "train": sorted(all_ids - heldout),
        "validation": sorted(validation),
        "test": sorted(test),
    }
    if len(_ASSIGNMENT_CACHE) >= _ASSIGNMENT_CACHE_LIMIT:
        _ASSIGNMENT_CACHE.pop(next(iter(_ASSIGNMENT_CACHE)))
    _ASSIGNMENT_CACHE[cache_key] = {
        name: tuple(result[name]) for name in SPLITS
    }
    return result


def build_group_split(contract: Mapping[str, Any], registry: Any) -> dict[str, Any]:
    checked = validate_contract(contract)
    rows = _group_rows(registry, checked)
    assignments = _assign_groups(rows, checked)
    split_rule = checked["group_split"]
    target_proportions = {
        name: checked["group_split"][f"{name}_count"] / len(rows) for name in SPLITS
    }
    balance_rows = []
    for field in checked["group_split"]["stratify_by"]:
        values = sorted({row["strata"][field] for row in rows})
        for value in values:
            matching = {row["group_id"] for row in rows if row["strata"][field] == value}
            by_split = {name: len(matching & set(assignments[name])) for name in SPLITS}
            proportions = {name: by_split[name] / len(matching) for name in SPLITS}
            residuals = {
                name: by_split[name] - len(matching) * target_proportions[name]
                for name in SPLITS
            }
            balance_rows.append({"field": field, "value": value, "total": len(matching),
                                 "by_split": by_split, "proportions": proportions,
                                 "residuals": residuals})
    balance = {
        "target_proportions": target_proportions,
        "strata": balance_rows,
        "max_absolute_residual": max(
            abs(item) for row in balance_rows for item in row["residuals"].values()
        ),
    }
    body = {
        "schema": SPLIT_SCHEMA,
        "protocol_id": checked["protocol_id"],
        "strategy": split_rule["strategy"],
        "seed": split_rule["seed"],
        "stratify_by": list(split_rule["stratify_by"]),
        "groups": rows,
        "group_registry_sha256": _digest({"schema": GROUP_REGISTRY_SCHEMA, "groups": rows}),
        "assignments": assignments,
        "counts": {name: len(assignments[name]) for name in SPLITS},
        "balance_report": balance,
    }
    return {**body, "split_sha256": _digest(body)}


def validate_group_split(value: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_contract(contract)
    document = _exact(value, {
        "schema", "protocol_id", "strategy", "seed", "stratify_by", "groups",
        "group_registry_sha256", "assignments", "counts", "balance_report",
        "split_sha256",
    }, "group split contract")
    rebuilt = build_group_split(checked, {"schema": GROUP_REGISTRY_SCHEMA,
                                          "groups": document["groups"]})
    if document != rebuilt:
        raise NativeH200ExperimentError("group split is not the deterministic contract")
    return document


def _train_manifest(
    *, split: Mapping[str, Any], protocol_id: str, replay_sha256: Sequence[str],
    round_number: int | None, train_replay_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if not replay_sha256:
        raise NativeH200ExperimentError("train replay manifest must contain a source")
    sources = [_sha(item, "train replay source") for item in replay_sha256]
    if len(set(sources)) != len(sources):
        raise NativeH200ExperimentError("train replay manifest contains a duplicate source")
    if round_number is None:
        if len(sources) != 1:
            raise NativeH200ExperimentError("BC must have exactly one train source")
        newest, prior, sampling = sources[0], [], {"newest_fraction": 1.0,
                                                   "prior_fraction": 0.0}
        stage = "bc"
        generation_id = GENERATION_IDS[0]
    else:
        _integer(round_number, "DAgger round", minimum=1)
        if round_number > len(BETAS) or len(sources) != round_number + 1:
            raise NativeH200ExperimentError("DAgger train replay lineage is incomplete")
        newest, prior = sources[-1], sources[:-1]
        sampling = {"newest_fraction": 0.5, "prior_fraction": 0.5}
        stage = "dagger"
        generation_id = GENERATION_IDS[round_number]
    snapshot = _exact(
        train_replay_snapshot,
        _TRAIN_REPLAY_SNAPSHOT_KEYS,
        "train replay snapshot binding",
    )
    expected_round = 0 if round_number is None else round_number
    if (
        _absolute(snapshot["path"], "train replay snapshot") != Path(snapshot["path"])
        or _integer(
            snapshot["through_macro_round"], "snapshot through macro round"
        )
        != expected_round
    ):
        raise NativeH200ExperimentError("train replay snapshot round/path differs")
    _sha(snapshot["file_sha256"], "train replay snapshot file")
    _sha(snapshot["snapshot_sha256"], "train replay snapshot semantic identity")
    return {
        "schema": TRAIN_MANIFEST_SCHEMA,
        "protocol_id": _token(protocol_id, "protocol ID"),
        "split_sha256": _sha(split["split_sha256"], "group split"),
        "stage": stage,
        "round": round_number,
        "generation_id": generation_id,
        "generation_relative_root": train_round_relative_root(generation_id),
        "train_group_ids": list(split["assignments"]["train"]),
        "newest_replay_sha256": newest,
        "prior_replay_sha256": prior,
        "sampling": sampling,
        "train_replay_snapshot": snapshot,
    }


def _validation_result_point(
    value: Any,
    *,
    stage: str,
    expected_generation_id: str,
    expected_macro_round: int,
    expected_split_sha256: str,
) -> dict[str, Any]:
    """Validate evaluator lineage/exposure and return the stopping-rule projection."""

    result = _exact(value, _VALIDATION_MSE_RESULT_KEYS, "validation MSE result")
    if result["schema"] != VALIDATION_MSE_RESULT_SCHEMA:
        raise NativeH200ExperimentError("validation MSE result schema differs")
    macro_round = _integer(result["macro_round"], "validation macro round")
    if (
        result["campaign_id"] != CANONICAL_CAMPAIGN_ID
        or result["generation_id"] != expected_generation_id
        or macro_round != expected_macro_round
        or result["split_sha256"] != expected_split_sha256
    ):
        raise NativeH200ExperimentError("validation MSE result lineage differs")
    _sha(result["checkpoint_sha256"], "validation checkpoint")
    _sha(result["validation_artifact_sha256"], "validation artifact")
    _sha(result["validation_snapshot_sha256"], "validation snapshot")
    _sha(result["split_sha256"], "validation split")

    if stage not in {"bc", "dagger"}:
        raise NativeH200ExperimentError("validation stage must be BC or DAgger")
    expected_policy = (
        "bc_train_uniform_v1"
        if stage == "bc"
        else "newest_half_cumulative_prior_half_uniform_v1"
    )
    if result["sampling_policy"] != expected_policy:
        raise NativeH200ExperimentError("validation sampling policy differs")

    examples = _integer(result["optimizer_examples"], "optimizer examples", minimum=1)
    examples_per_step = _integer(
        result["optimizer_examples_per_step"],
        "optimizer examples per step",
        minimum=1,
    )
    if examples % examples_per_step:
        raise NativeH200ExperimentError("optimizer examples do not end on a complete step")
    newest = _integer(result["newest_source_examples"], "newest source examples")
    prior = _integer(result["prior_source_examples"], "prior source examples")
    newest_population = _integer(
        result["newest_source_population"], "newest source population", minimum=1
    )
    prior_population = _integer(
        result["prior_source_population"], "prior source population"
    )
    unit_examples = _integer(
        result["balanced_exposure_unit_examples"],
        "balanced exposure unit examples",
        minimum=1,
    )
    units = _number(
        result["balanced_exposure_units"], "balanced exposure units", minimum=0.0
    )
    newest_passes = _number(
        result["newest_expected_passes"], "newest expected passes", minimum=0.0
    )
    prior_passes = _number(
        result["prior_expected_passes"], "prior expected passes", minimum=0.0
    )
    _integer(result["validation_examples"], "validation examples", minimum=1)
    mse = _number(result["validation_mse"], "validation MSE", minimum=0.0)

    if newest + prior != examples:
        raise NativeH200ExperimentError("source exposure does not equal optimizer examples")
    expected_unit = (
        newest_population
        if stage == "bc"
        else 2 * max(newest_population, prior_population)
    )
    if unit_examples != expected_unit:
        raise NativeH200ExperimentError("balanced exposure unit is derived incorrectly")
    if stage == "bc":
        if prior_population != 0 or newest != examples or prior != 0:
            raise NativeH200ExperimentError("BC exposure must come only from train data")
        expected_prior_passes = 0.0
    else:
        if prior_population <= 0 or newest != prior or newest * 2 != examples:
            raise NativeH200ExperimentError(
                "DAgger exposure must remain exact 50/50 newest/cumulative-prior"
            )
        expected_prior_passes = prior / prior_population
    expected_units = examples / unit_examples
    expected_newest_passes = newest / newest_population
    for observed, expected, label in (
        (units, expected_units, "balanced exposure units"),
        (newest_passes, expected_newest_passes, "newest expected passes"),
        (prior_passes, expected_prior_passes, "prior expected passes"),
    ):
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise NativeH200ExperimentError(f"{label} disagree with source exposure")

    return {
        "optimizer_examples": examples,
        "newest_source_examples": newest,
        "prior_source_examples": prior,
        "newest_source_population": newest_population,
        "prior_source_population": prior_population,
        "balanced_exposure_unit_examples": unit_examples,
        "balanced_exposure_units": units,
        "optimizer_examples_per_step": examples_per_step,
        "validation_mse": mse,
        "checkpoint_sha256": result["checkpoint_sha256"],
    }


def plateau_decision(
    points: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Minimize validation MSE using the same source-exposure rule every stage."""

    checked_rule = _exact(rule, _PLATEAU_KEYS, "plateau rule")
    if stage not in {"bc", "dagger"}:
        raise NativeH200ExperimentError("plateau stage must be BC or DAgger")
    interval = _number(
        checked_rule["validation_interval_balanced_exposure_units"],
        "validation interval balanced exposure units",
        minimum=0.0,
    )
    minimum = _number(
        checked_rule["minimum_balanced_exposure_units"],
        "minimum balanced exposure units",
        minimum=0.0,
    )
    maximum = _number(
        checked_rule["maximum_balanced_exposure_units"],
        "maximum balanced exposure units",
        minimum=minimum,
    )
    if interval <= 0:
        raise NativeH200ExperimentError("validation exposure interval must be positive")
    patience = _integer(checked_rule["patience"], "patience", minimum=1)
    min_delta = _number(checked_rule["validation_mse_min_delta"],
                        "validation MSE min_delta", minimum=0)
    if not points:
        return {"stop": False, "reason": None, "best_checkpoint_sha256": None,
                "best_validation_mse": None,
                "balanced_exposure_units": 0.0,
                "non_improving_evaluations": 0}
    best: dict[str, Any] | None = None
    patience_reference_mse: float | None = None
    stale = 0
    source_contract: tuple[int, int, int, int] | None = None
    for index, raw in enumerate(points, 1):
        point = _exact(raw, _POINT_KEYS, "validation point")
        examples = _integer(point["optimizer_examples"], "optimizer examples", minimum=1)
        examples_per_step = _integer(
            point["optimizer_examples_per_step"], "optimizer examples per step", minimum=1
        )
        newest = _integer(point["newest_source_examples"], "newest source examples")
        prior = _integer(point["prior_source_examples"], "prior source examples")
        newest_population = _integer(
            point["newest_source_population"], "newest source population", minimum=1
        )
        prior_population = _integer(
            point["prior_source_population"], "prior source population"
        )
        unit_examples = _integer(
            point["balanced_exposure_unit_examples"],
            "balanced exposure unit examples",
            minimum=1,
        )
        units = _number(
            point["balanced_exposure_units"], "balanced exposure units", minimum=0.0
        )
        mse = _number(point["validation_mse"], "validation MSE", minimum=0)
        expected_source_contract = (
            newest_population, prior_population, unit_examples, examples_per_step
        )
        if source_contract is None:
            source_contract = expected_source_contract
        elif source_contract != expected_source_contract:
            raise NativeH200ExperimentError(
                "validation source populations changed within one training stage"
            )
        if newest + prior != examples:
            raise NativeH200ExperimentError("source exposure does not equal optimizer examples")
        expected_unit = (
            newest_population
            if stage == "bc"
            else 2 * max(newest_population, prior_population)
        )
        if unit_examples != expected_unit:
            raise NativeH200ExperimentError("balanced exposure unit is derived incorrectly")
        if stage == "bc":
            if prior_population != 0 or newest != examples or prior != 0:
                raise NativeH200ExperimentError("BC exposure must come only from train data")
        elif (
            prior_population <= 0
            or newest != prior
            or newest * 2 != examples
        ):
            raise NativeH200ExperimentError(
                "DAgger exposure must remain exact 50/50 newest/cumulative-prior"
            )
        if not math.isclose(units, examples / unit_examples, rel_tol=0.0, abs_tol=1e-12):
            raise NativeH200ExperimentError("balanced exposure units disagree with examples")
        target_units = index * interval
        expected_examples = (
            math.ceil(target_units * unit_examples / examples_per_step)
            * examples_per_step
        )
        if examples != expected_examples:
            raise NativeH200ExperimentError(
                "validation examples violate the exposure-normalized cadence"
            )
        _sha(point["checkpoint_sha256"], "validation checkpoint")
        # Checkpoint selection is the literal MSE argmin.  ``min_delta`` affects
        # only plateau patience; a real but sub-threshold improvement must not be
        # discarded when the final/global checkpoint is selected.
        if best is None or mse < float(best["validation_mse"]):
            best = point
        if (
            patience_reference_mse is None
            or (
                mse < patience_reference_mse
                and patience_reference_mse - mse >= min_delta
            )
        ):
            patience_reference_mse = mse
            stale = 0
        else:
            stale += 1
    assert best is not None
    last_units = float(points[-1]["balanced_exposure_units"])
    if last_units >= minimum and stale >= patience:
        stop, reason = True, "plateau"
    elif last_units >= maximum:
        stop, reason = True, "capped"
    else:
        stop, reason = False, None
    return {
        "stop": stop, "reason": reason,
        "best_checkpoint_sha256": best["checkpoint_sha256"],
        "best_validation_mse": float(best["validation_mse"]),
        "balanced_exposure_units": last_units,
        "non_improving_evaluations": stale,
    }


def _wilson_95(successes: int, total: int) -> dict[str, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if total < 1 or successes < 0 or successes > total:
        raise NativeH200ExperimentError("Wilson interval counts are invalid")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {"lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def _success_metrics(
    *,
    episode_count: int,
    success_count: int,
    safety_violations: int,
    per_stratum: Mapping[str, Any],
    expected_stratum_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate aggregate final-test results and add exact rates/CIs."""

    episodes = _integer(episode_count, "episode count", minimum=1)
    successes = _integer(success_count, "success count")
    violations = _integer(safety_violations, "safety violations")
    if successes > episodes or violations > episodes:
        raise NativeH200ExperimentError("final-test aggregate counts are inconsistent")
    if not isinstance(per_stratum, Mapping) or not per_stratum:
        raise NativeH200ExperimentError("per-stratum final-test metrics are required")
    expected: dict[str, int] | None = None
    if expected_stratum_counts is not None:
        if not isinstance(expected_stratum_counts, Mapping) or not expected_stratum_counts:
            raise NativeH200ExperimentError("expected stratum counts are invalid")
        expected = {
            _token(key, "stratum"): _integer(value, "stratum case count", minimum=1)
            for key, value in expected_stratum_counts.items()
        }
        if sum(expected.values()) != episodes:
            raise NativeH200ExperimentError("expected stratum counts do not cover all cases")
    rows: dict[str, dict[str, Any]] = {}
    for raw_name, raw_row in per_stratum.items():
        name = _token(raw_name, "stratum")
        row = _exact(raw_row, {"episode_count", "success_count", "safety_violations"},
                     f"stratum {name}")
        count = _integer(row["episode_count"], f"{name} episode count", minimum=1)
        won = _integer(row["success_count"], f"{name} success count")
        unsafe = _integer(row["safety_violations"], f"{name} safety violations")
        if won > count or unsafe > count:
            raise NativeH200ExperimentError(f"stratum {name} counts are inconsistent")
        if expected is not None and (name not in expected or expected[name] != count):
            raise NativeH200ExperimentError(f"stratum {name} differs from sealed registry")
        rows[name] = {
            "episode_count": count,
            "success_count": won,
            "success_rate": won / count,
            "success_rate_wilson_95": _wilson_95(won, count),
            "safety_violations": unsafe,
        }
    if expected is not None and set(rows) != set(expected):
        raise NativeH200ExperimentError("per-stratum result does not cover sealed registry")
    if (sum(row["episode_count"] for row in rows.values()) != episodes
            or sum(row["success_count"] for row in rows.values()) != successes
            or sum(row["safety_violations"] for row in rows.values()) != violations):
        raise NativeH200ExperimentError("per-stratum results do not partition the aggregate")
    return {
        "episode_count": episodes,
        "success_count": successes,
        "success_rate": successes / episodes,
        "success_rate_wilson_95": _wilson_95(successes, episodes),
        "safety_violations": violations,
        "per_stratum": dict(sorted(rows.items())),
    }


def _final_test_result_document(
    value: Any,
    *,
    protocol_id: str,
    generation_id_value: str,
    checkpoint_sha256: str,
    test_registry_sha256: str,
    split_sha256: str,
    expected_episode_count: int | None = None,
    expected_test_cases_sha256: str | None = None,
    expected_stratum_counts: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable, success-only result emitted by the final-test fleet."""

    result = _exact(value, _FINAL_TEST_RESULT_KEYS, "final test result")
    if (
        result["schema"] != FINAL_TEST_RESULT_SCHEMA
        or result["campaign_id"] != CANONICAL_CAMPAIGN_ID
        or result["protocol_id"] != protocol_id
        or result["generation_id"] != generation_id_value
        or result["checkpoint_sha256"] != checkpoint_sha256
        or result["test_registry_sha256"] != test_registry_sha256
        or result["split_sha256"] != split_sha256
    ):
        raise NativeH200ExperimentError("final test result lineage differs")
    _sha(result["checkpoint_sha256"], "final test checkpoint")
    _sha(result["test_registry_sha256"], "final test registry")
    cases_sha256 = _sha(result["test_cases_sha256"], "final test cases")
    if (
        expected_test_cases_sha256 is not None
        and cases_sha256 != expected_test_cases_sha256
    ):
        raise NativeH200ExperimentError("final test cases differ from sealed registry")
    episode_count = _integer(result["episode_count"], "test episode count", minimum=1)
    if expected_episode_count is not None and episode_count != expected_episode_count:
        raise NativeH200ExperimentError("final test episode count differs from sealed registry")
    metrics = _success_metrics(
        episode_count=episode_count,
        success_count=result["success_count"],
        safety_violations=result["safety_violations"],
        per_stratum=result["per_stratum"],
        expected_stratum_counts=expected_stratum_counts,
    )
    return result, metrics


def generation_id(round_number: int | None) -> str:
    """Map BC/DAgger stage numbering to its immutable artifact namespace."""

    if round_number is None:
        return GENERATION_IDS[0]
    number = _integer(round_number, "DAgger round", minimum=1)
    if number > len(BETAS):
        raise NativeH200ExperimentError("DAgger round has no generation namespace")
    return GENERATION_IDS[number]


class Controller:
    """Reconstruct and advance one immutable experiment event journal."""

    def __init__(self, campaign_root: Path):
        self.root = Path(os.path.abspath(campaign_root))
        self.contract_path = self.root / "BC/contracts/native_h200_experiment.json"
        self.split_path = self.root / "BC/contracts/group_split.v1.json"
        self.events = self.root / "events"
        self.learner_inputs = self.root / "learner_inputs"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.root / ".protocol.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _contract(self) -> dict[str, Any]:
        contract = load_contract(self.contract_path)
        if contract["campaign_root"] != str(self.root):
            raise NativeH200ExperimentError("campaign root differs from immutable contract")
        return contract

    def _ensure_artifact_layout(self) -> None:
        """Create only the contract-defined generation tree; reject symlinked paths."""

        relative_directories = [
            *(f"{train_round_relative_root(item)}/{leaf}"
              for item in GENERATION_IDS for leaf in TRAIN_ROUND_SUBDIRECTORIES),
            *(f"validation/results/{train_round_relative_root(item)}/mse"
              for item in GENERATION_IDS),
            "final/checkpoint",
            "final/validation",
            "final/report",
            "test/heldout/results",
            "BC/contracts",
            "events",
            "learner_inputs",
        ]
        for relative in relative_directories:
            path = self.root / relative
            path.mkdir(parents=True, exist_ok=True)
            current = path
            while current != self.root:
                info = os.lstat(current)
                if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    raise NativeH200ExperimentError(
                        f"artifact layout contains an unsafe component: {current}"
                    )
                current = current.parent

    def _validated_train_replay_snapshot(
        self,
        *,
        path: Path,
        file_sha256: str,
        snapshot_sha256: str,
        round_number: int | None,
    ) -> dict[str, Any]:
        """Bind and fully verify one generation-local cumulative train snapshot."""

        expected_round = 0 if round_number is None else _integer(
            round_number, "train snapshot round", minimum=1
        )
        expected_generation = generation_id(round_number)
        expected_parent = (
            self.root
            / train_round_relative_root(expected_generation)
            / "replay/manifests"
        ).resolve(strict=True)
        candidate = _absolute(str(path), "train replay snapshot")
        try:
            resolved = candidate.resolve(strict=True)
            info = os.lstat(candidate)
        except OSError as exc:
            raise NativeH200ExperimentError(
                "train replay snapshot is missing or unsafe"
            ) from exc
        if (
            resolved != candidate
            or candidate.parent != expected_parent
            or not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or stat.S_IMODE(info.st_mode) & 0o222
        ):
            raise NativeH200ExperimentError(
                "train replay snapshot must be immutable in its generation manifest directory"
            )
        expected_file = _sha(file_sha256, "train replay snapshot file")
        expected_semantic = _sha(
            snapshot_sha256, "train replay snapshot semantic identity"
        )
        try:
            from round_dagger.replay import ReplayError, load_snapshot

            loaded = load_snapshot(
                candidate,
                expected_file_sha256=expected_file,
                artifact_root=self.root,
            )
        except (OSError, ReplayError) as exc:
            raise NativeH200ExperimentError(
                f"train replay snapshot failed verification: {exc}"
            ) from exc
        if (
            loaded.get("campaign_id") != CANONICAL_CAMPAIGN_ID
            or loaded.get("purpose") != "train"
            or loaded.get("through_macro_round") != expected_round
            or loaded.get("snapshot_sha256") != expected_semantic
            or loaded.get("file_sha256") != expected_file
        ):
            raise NativeH200ExperimentError(
                "train replay snapshot lineage or semantic identity differs"
            )
        return {
            "path": str(candidate),
            "file_sha256": expected_file,
            "snapshot_sha256": expected_semantic,
            "through_macro_round": expected_round,
        }

    def initialize(
        self,
        *,
        contract: Mapping[str, Any],
        split: Mapping[str, Any],
        test_registry_sha256: str,
        bc_replay_sha256: str,
        initial_checkpoint_sha256: str,
        train_replay_snapshot_path: Path,
        train_replay_snapshot_file_sha256: str,
        train_replay_snapshot_sha256: str,
    ) -> dict[str, Any]:
        checked = validate_contract(contract)
        if Path(checked["campaign_root"]) != self.root:
            raise NativeH200ExperimentError("controller root differs from contract")
        checked_split = validate_group_split(split, checked)
        with self._lock():
            self._ensure_artifact_layout()
            train_replay_snapshot = self._validated_train_replay_snapshot(
                path=train_replay_snapshot_path,
                file_sha256=train_replay_snapshot_file_sha256,
                snapshot_sha256=train_replay_snapshot_sha256,
                round_number=None,
            )
            _publish(self.contract_path, checked)
            _publish(self.split_path, checked_split)
            train_manifest = _train_manifest(
                split=checked_split, protocol_id=checked["protocol_id"],
                replay_sha256=[bc_replay_sha256], round_number=None,
                train_replay_snapshot=train_replay_snapshot,
            )
            train_manifest_sha256 = _digest(train_manifest)
            _publish(
                self.learner_inputs / f"{train_manifest_sha256}.json", train_manifest
            )
            self.events.mkdir(parents=True, exist_ok=True)
            if list(self.events.iterdir()):
                raise NativeH200ExperimentError("campaign journal already exists")
            self._append("initialized", {
                "contract_sha256": _digest(checked),
                "split_sha256": checked_split["split_sha256"],
                "test_registry_sha256": _sha(test_registry_sha256, "test registry"),
                "bc_replay_sha256": _sha(bc_replay_sha256, "BC replay"),
                "initial_checkpoint_sha256": _sha(
                    initial_checkpoint_sha256, "initial checkpoint"
                ),
                "train_replay_snapshot": train_replay_snapshot,
                "train_replay_manifest_sha256": train_manifest_sha256,
            }, locked=True)
        return self.status()

    def _event_documents(self) -> list[dict[str, Any]]:
        if not self.events.is_dir() or self.events.is_symlink():
            raise NativeH200ExperimentError("event journal is missing or unsafe")
        files = sorted(self.events.glob("*.json"))
        documents: list[dict[str, Any]] = []
        previous = "0" * 64
        for index, path in enumerate(files):
            expected_prefix = f"{index:06d}_"
            if not path.name.startswith(expected_prefix):
                raise NativeH200ExperimentError("event sequence has a gap")
            event = _exact(_load_json(path), {
                "schema", "sequence", "kind", "previous_event_sha256", "payload",
                "event_sha256",
            }, "event")
            body = {key: event[key] for key in event if key != "event_sha256"}
            if (event["schema"] != EVENT_SCHEMA or event["sequence"] != index
                    or event["previous_event_sha256"] != previous
                    or event["event_sha256"] != _digest(body)):
                raise NativeH200ExperimentError("event hash chain is invalid")
            previous = event["event_sha256"]
            documents.append(event)
        if not documents:
            raise NativeH200ExperimentError("event journal is empty")
        return documents

    def _append(self, kind: str, payload: Mapping[str, Any], *, locked: bool = False) -> None:
        def write() -> None:
            existing = [] if not any(self.events.glob("*.json")) else self._event_documents()
            sequence = len(existing)
            previous = "0" * 64 if not existing else existing[-1]["event_sha256"]
            body = {
                "schema": EVENT_SCHEMA, "sequence": sequence, "kind": _token(kind, "event kind"),
                "previous_event_sha256": previous, "payload": dict(payload),
            }
            event = {**body, "event_sha256": _digest(body)}
            _publish(self.events / f"{sequence:06d}_{kind}.json", event)
        if locked:
            write()
        else:
            with self._lock():
                write()

    def status(self) -> dict[str, Any]:
        contract = self._contract()
        split = validate_group_split(_load_json(self.split_path), contract)
        state: dict[str, Any] = {
            "phase": None, "history": [], "active_checkpoint_sha256": None,
            "active_generation_id": None,
            "active_validation_mse": None,
            "selected_checkpoint_sha256": None, "selected_validation_mse": None,
            "selected_generation_id": None,
            "current_round": None, "current_beta": None,
            "cumulative_replay_sha256": [], "stage_best": None,
            "train_replay_manifest_sha256": None,
            "train_replay_snapshot": None,
            "no_progress_rounds": 0, "outer_stop_reason": None,
            "test_registry_sha256": None, "final_checkpoint_sha256": None,
            "final_generation_id": None, "initial_checkpoint_sha256": None,
        }
        events = self._event_documents()
        for event in events:
            kind, payload = event["kind"], event["payload"]
            if kind == "initialized":
                if state["phase"] is not None:
                    raise NativeH200ExperimentError("duplicate initialization event")
                initialized = _exact(payload, {
                    "contract_sha256", "split_sha256", "test_registry_sha256",
                    "bc_replay_sha256", "initial_checkpoint_sha256",
                    "train_replay_snapshot",
                    "train_replay_manifest_sha256",
                }, "initialization event")
                if (initialized["contract_sha256"] != _digest(contract)
                        or initialized["split_sha256"] != split["split_sha256"]):
                    raise NativeH200ExperimentError("initialization contract lineage differs")
                state.update(
                    phase="bc_running",
                    test_registry_sha256=_sha(
                        initialized["test_registry_sha256"], "test registry"
                    ),
                    cumulative_replay_sha256=[
                        _sha(initialized["bc_replay_sha256"], "BC replay")
                    ],
                    initial_checkpoint_sha256=_sha(
                        initialized["initial_checkpoint_sha256"],
                        "initial checkpoint",
                    ),
                )
                manifest = _train_manifest(
                    split=split, protocol_id=contract["protocol_id"],
                    replay_sha256=state["cumulative_replay_sha256"], round_number=None,
                    train_replay_snapshot=initialized["train_replay_snapshot"],
                )
                if initialized["train_replay_manifest_sha256"] != _digest(manifest):
                    raise NativeH200ExperimentError("BC train manifest lineage differs")
                state["train_replay_manifest_sha256"] = _digest(manifest)
                state["train_replay_snapshot"] = manifest["train_replay_snapshot"]
            elif kind == "validation_mse":
                if state["phase"] not in {"bc_running", "dagger_running"}:
                    raise NativeH200ExperimentError("validation-MSE event is out of phase")
                stage = "bc" if state["phase"] == "bc_running" else "dagger"
                row = _exact(
                    payload,
                    {"validation_result_file_sha256", "result", "decision"},
                    "validation-MSE event",
                )
                _sha(row["validation_result_file_sha256"], "validation result file")
                expected_round = 0 if stage == "bc" else state["current_round"]
                point = _validation_result_point(
                    row["result"],
                    stage=stage,
                    expected_generation_id=generation_id(
                        None if stage == "bc" else state["current_round"]
                    ),
                    expected_macro_round=expected_round,
                    expected_split_sha256=split["split_sha256"],
                )
                state["history"].append(point)
                decision = plateau_decision(
                    state["history"], contract["optimization"]["stopping_rule"],
                    stage=stage,
                )
                if row["decision"] != decision:
                    raise NativeH200ExperimentError("recorded plateau decision differs")
                if decision["stop"]:
                    state["stage_best"] = decision
                    best_checkpoint = decision["best_checkpoint_sha256"]
                    best_mse = decision["best_validation_mse"]
                    completed_generation = generation_id(
                        None if stage == "bc" else state["current_round"]
                    )
                    state["active_checkpoint_sha256"] = best_checkpoint
                    state["active_generation_id"] = completed_generation
                    state["active_validation_mse"] = best_mse
                    if stage == "bc":
                        state["selected_checkpoint_sha256"] = best_checkpoint
                        state["selected_generation_id"] = completed_generation
                        state["selected_validation_mse"] = best_mse
                        state["phase"] = "bc_complete"
                    else:
                        previous_global_mse = float(state["selected_validation_mse"])
                        min_delta = float(
                            contract["dagger"]["global_validation_mse_min_delta"]
                        )
                        made_progress = (
                            best_mse < previous_global_mse
                            and previous_global_mse - best_mse >= min_delta
                        )
                        if best_mse < previous_global_mse:
                            state["selected_checkpoint_sha256"] = best_checkpoint
                            state["selected_generation_id"] = completed_generation
                            state["selected_validation_mse"] = best_mse
                        state["no_progress_rounds"] = (
                            0 if made_progress else state["no_progress_rounds"] + 1
                        )
                        maxed = state["current_round"] == len(BETAS)
                        stalled = state["no_progress_rounds"] >= contract["dagger"][
                            "stop_after_no_progress_rounds"
                        ]
                        if maxed or stalled:
                            state["phase"] = "training_complete"
                            state["outer_stop_reason"] = (
                                "max_rounds" if maxed else "no_progress"
                            )
                        else:
                            state["phase"] = "round_complete"
            elif kind == "round_started":
                if state["phase"] not in {"bc_complete", "round_complete"}:
                    raise NativeH200ExperimentError("DAgger round started out of phase")
                row = _exact(payload, {
                    "round", "beta", "collection_groups", "new_replay_sha256",
                    "prior_replay_sha256", "sampling", "train_replay_manifest_sha256",
                    "train_replay_snapshot",
                }, "round-start event")
                expected_round = 1 if state["current_round"] is None else state["current_round"] + 1
                if row["round"] != expected_round or row["beta"] != BETAS[expected_round - 1]:
                    raise NativeH200ExperimentError("DAgger round/beta differs")
                if row["prior_replay_sha256"] != state["cumulative_replay_sha256"]:
                    raise NativeH200ExperimentError("DAgger prior replay lineage differs")
                groups = row["collection_groups"]
                if (not isinstance(groups, list) or not groups
                        or groups != sorted(set(groups))
                        or not set(groups) <= set(split["assignments"]["train"])):
                    raise NativeH200ExperimentError("DAgger collection escaped train groups")
                replay = _sha(row["new_replay_sha256"], "new replay")
                if replay in state["cumulative_replay_sha256"]:
                    raise NativeH200ExperimentError("DAgger replay was admitted twice")
                if row["sampling"] != {"newest_fraction": 0.5, "prior_fraction": 0.5}:
                    raise NativeH200ExperimentError("DAgger replay sampling differs")
                state["cumulative_replay_sha256"].append(replay)
                manifest = _train_manifest(
                    split=split, protocol_id=contract["protocol_id"],
                    replay_sha256=state["cumulative_replay_sha256"],
                    round_number=expected_round,
                    train_replay_snapshot=row["train_replay_snapshot"],
                )
                if row["train_replay_manifest_sha256"] != _digest(manifest):
                    raise NativeH200ExperimentError("DAgger train manifest lineage differs")
                state["train_replay_manifest_sha256"] = _digest(manifest)
                state["train_replay_snapshot"] = manifest["train_replay_snapshot"]
                state["current_round"] = expected_round
                state["current_beta"] = row["beta"]
                state["history"] = []
                state["stage_best"] = None
                state["phase"] = "dagger_running"
            elif kind == "final_selected":
                if state["phase"] != "training_complete" or payload != {
                    "checkpoint_sha256": state["selected_checkpoint_sha256"],
                    "generation_id": state["selected_generation_id"],
                    "validation_mse": state["selected_validation_mse"],
                }:
                    raise NativeH200ExperimentError("final selection is invalid")
                state["phase"] = "final_selected"
                state["final_checkpoint_sha256"] = payload["checkpoint_sha256"]
                state["final_generation_id"] = payload["generation_id"]
            elif kind == "test_recorded":
                if state["phase"] != "final_selected":
                    raise NativeH200ExperimentError("test result precedes final selection")
                row = _exact(payload, {
                    "test_result_file_sha256", "result", "metrics",
                }, "test event")
                _sha(row["test_result_file_sha256"], "test result file")
                _result, metrics = _final_test_result_document(
                    row["result"],
                    protocol_id=contract["protocol_id"],
                    generation_id_value=state["final_generation_id"],
                    checkpoint_sha256=state["final_checkpoint_sha256"],
                    test_registry_sha256=state["test_registry_sha256"],
                    split_sha256=split["split_sha256"],
                )
                if row["metrics"] != metrics:
                    raise NativeH200ExperimentError("stored final test metrics differ")
                state["phase"] = "tested"
                state["test_metrics"] = metrics
                state["test_result_file_sha256"] = row["test_result_file_sha256"]
            else:
                raise NativeH200ExperimentError(f"unknown event kind: {kind}")
        return {**state, "protocol_id": contract["protocol_id"],
                "split_sha256": split["split_sha256"],
                "event_count": len(events)}

    def record_validation(
        self,
        *,
        result_path: Path,
        expected_file_sha256: str,
    ) -> dict[str, Any]:
        """Admit one immutable evaluator result; never accept reconstructed scalars."""

        with self._lock():
            state, contract = self.status(), self._contract()
            if state["phase"] not in {"bc_running", "dagger_running"}:
                raise NativeH200ExperimentError("validation MSE is not allowed in this phase")
            stage = "bc" if state["phase"] == "bc_running" else "dagger"
            expected_round = 0 if stage == "bc" else state["current_round"]
            expected_generation = generation_id(
                None if stage == "bc" else state["current_round"]
            )
            expected_directory = (
                self.root
                / "validation/results"
                / train_round_relative_root(expected_generation)
                / "mse"
            )
            path = _absolute(str(result_path), "validation MSE result path")
            if (
                path.parent != expected_directory
                or path.suffix != ".json"
                or _token(path.name, "validation MSE result filename") != path.name
            ):
                raise NativeH200ExperimentError(
                    "validation MSE result escaped its generation-local directory"
                )
            try:
                directory_info = os.lstat(expected_directory)
            except OSError as exc:
                raise NativeH200ExperimentError(
                    "validation MSE result directory is missing"
                ) from exc
            if (
                not stat.S_ISDIR(directory_info.st_mode)
                or stat.S_ISLNK(directory_info.st_mode)
            ):
                raise NativeH200ExperimentError(
                    "validation MSE result directory is unsafe"
                )
            result = _load_pinned_canonical_json(
                path,
                expected_file_sha256,
                label="validation MSE result",
            )
            point = _validation_result_point(
                result,
                stage=stage,
                expected_generation_id=expected_generation,
                expected_macro_round=expected_round,
                expected_split_sha256=state["split_sha256"],
            )
            decision = plateau_decision(
                [*state["history"], point], contract["optimization"]["stopping_rule"],
                stage=stage,
            )
            self._append(
                "validation_mse",
                {
                    "validation_result_file_sha256": _sha(
                        expected_file_sha256, "validation result file"
                    ),
                    "result": result,
                    "decision": decision,
                },
                locked=True,
            )
        return self.status()

    def start_round(
        self,
        *,
        new_replay_sha256: str,
        collection_groups: Sequence[str],
        train_replay_snapshot_path: Path,
        train_replay_snapshot_file_sha256: str,
        train_replay_snapshot_sha256: str,
    ) -> dict[str, Any]:
        with self._lock():
            state, contract = self.status(), self._contract()
            if state["phase"] not in {"bc_complete", "round_complete"}:
                raise NativeH200ExperimentError("a DAgger round cannot start in this phase")
            round_number = 1 if state["current_round"] is None else state["current_round"] + 1
            if round_number > len(BETAS):
                raise NativeH200ExperimentError("DAgger round horizon is exhausted")
            split = validate_group_split(_load_json(self.split_path), contract)
            groups = sorted({_token(item, "collection group") for item in collection_groups})
            if not groups or not set(groups) <= set(split["assignments"]["train"]):
                raise NativeH200ExperimentError("DAgger collection must use train groups only")
            replay_sha = _sha(new_replay_sha256, "new replay")
            if replay_sha in state["cumulative_replay_sha256"]:
                raise NativeH200ExperimentError("new replay identity was already admitted")
            train_replay_snapshot = self._validated_train_replay_snapshot(
                path=train_replay_snapshot_path,
                file_sha256=train_replay_snapshot_file_sha256,
                snapshot_sha256=train_replay_snapshot_sha256,
                round_number=round_number,
            )
            manifest = _train_manifest(
                split=split, protocol_id=contract["protocol_id"],
                replay_sha256=[*state["cumulative_replay_sha256"], replay_sha],
                round_number=round_number,
                train_replay_snapshot=train_replay_snapshot,
            )
            manifest_sha256 = _digest(manifest)
            _publish(self.learner_inputs / f"{manifest_sha256}.json", manifest)
            self._append("round_started", {
                "round": round_number, "beta": BETAS[round_number - 1],
                "collection_groups": groups, "new_replay_sha256": replay_sha,
                "prior_replay_sha256": state["cumulative_replay_sha256"],
                "sampling": {"newest_fraction": 0.5, "prior_fraction": 0.5},
                "train_replay_snapshot": train_replay_snapshot,
                "train_replay_manifest_sha256": manifest_sha256,
            }, locked=True)
        return self.status()

    def select_final(self) -> dict[str, Any]:
        with self._lock():
            state = self.status()
            if state["phase"] != "training_complete":
                raise NativeH200ExperimentError("training has not reached its outer stop")
            self._append("final_selected", {
                "checkpoint_sha256": state["selected_checkpoint_sha256"],
                "generation_id": state["selected_generation_id"],
                "validation_mse": state["selected_validation_mse"],
            }, locked=True)
        return self.status()

    def record_test(
        self,
        *,
        registry_path: Path,
        result_path: Path,
        expected_result_file_sha256: str,
    ) -> dict[str, Any]:
        """Admit one immutable final-test result after final selection is frozen."""

        # Deliberately check the phase before opening either sealed-test path.
        with self._lock():
            state = self.status()
            if state["phase"] != "final_selected":
                raise NativeH200ExperimentError("test registry is sealed until final selection")
            result = _absolute(str(result_path), "final test result path")
            expected_directory = self.root / "test/heldout/results"
            if (
                result.parent != expected_directory
                or result.suffix != ".json"
                or _token(result.name, "final test result filename") != result.name
            ):
                raise NativeH200ExperimentError(
                    "final test result escaped the sealed test output directory"
                )
            try:
                directory_info = os.lstat(expected_directory)
            except OSError as exc:
                raise NativeH200ExperimentError(
                    "sealed test result directory is missing"
                ) from exc
            if (
                not stat.S_ISDIR(directory_info.st_mode)
                or stat.S_ISLNK(directory_info.st_mode)
            ):
                raise NativeH200ExperimentError(
                    "sealed test result directory is unsafe"
                )
            registry_document = _load_pinned_canonical_json(
                registry_path,
                state["test_registry_sha256"],
                label="test registry",
            )
            registry = _exact(
                registry_document,
                {"schema", "groups", "case_count", "cases_sha256",
                 "stratum_case_counts"},
                "test registry",
            )
            if registry["schema"] != TEST_REGISTRY_SCHEMA:
                raise NativeH200ExperimentError("test registry schema differs")
            contract = self._contract()
            split = validate_group_split(_load_json(self.split_path), contract)
            if registry["groups"] != split["assignments"]["test"]:
                raise NativeH200ExperimentError("test registry groups differ from sealed split")
            test_result = _load_pinned_canonical_json(
                result,
                expected_result_file_sha256,
                label="final test result",
            )
            checked_result, metrics = _final_test_result_document(
                test_result,
                protocol_id=contract["protocol_id"],
                generation_id_value=state["final_generation_id"],
                checkpoint_sha256=state["final_checkpoint_sha256"],
                test_registry_sha256=state["test_registry_sha256"],
                split_sha256=split["split_sha256"],
                expected_episode_count=_integer(
                    registry["case_count"], "test case count", minimum=1
                ),
                expected_test_cases_sha256=_sha(
                    registry["cases_sha256"], "test cases"
                ),
                expected_stratum_counts=registry["stratum_case_counts"],
            )
            self._append("test_recorded", {
                "test_result_file_sha256": _sha(
                    expected_result_file_sha256, "final test result file"
                ),
                "result": checked_result,
                "metrics": metrics,
            }, locked=True)
        return self.status()

    def launch_plan(self, role: str, *, checkpoint_sha256: str | None = None) -> dict[str, Any]:
        state, contract = self.status(), self._contract()
        if role not in _LAUNCH_KEYS:
            raise NativeH200ExperimentError("launch role is invalid")
        allowed = {
            "actor": {"bc_complete", "round_complete"},
            "learner": {"bc_running", "dagger_running"},
            "validation": {"bc_running", "dagger_running"},
            "test": {"final_selected"},
        }
        if state["phase"] not in allowed[role]:
            raise NativeH200ExperimentError(f"{role} launch is not allowed in this phase")
        row = contract["launch"][role]
        executable = str(
            _safe_launch_executable(Path(contract["source_root"]), row["entrypoint"])
        )
        split = validate_group_split(_load_json(self.split_path), contract)
        if role == "learner":
            launch_generation_id = generation_id(
                None if state["phase"] == "bc_running" else state["current_round"]
            )
            manifest_path = self.learner_inputs / (
                f"{state['train_replay_manifest_sha256']}.json"
            )
            manifest = _load_json(manifest_path)
            if _digest(manifest) != state["train_replay_manifest_sha256"]:
                raise NativeH200ExperimentError("train replay manifest digest differs")
            bound_inputs: dict[str, Any] = {
                "train_replay_manifest": {
                    "path": str(manifest_path),
                    "sha256": state["train_replay_manifest_sha256"],
                },
                "train_replay_snapshot": dict(state["train_replay_snapshot"]),
                "parent_checkpoint": {
                    "sha256": (
                        state["initial_checkpoint_sha256"]
                        if state["phase"] == "bc_running"
                        else state["active_checkpoint_sha256"]
                    ),
                    "generation_id": (
                        None
                        if state["phase"] == "bc_running"
                        else state["active_generation_id"]
                    ),
                },
            }
        elif role == "actor":
            next_round = 1 if state["current_round"] is None else state["current_round"] + 1
            launch_generation_id = generation_id(next_round)
            bound_inputs = {
                "checkpoint": {"sha256": state["active_checkpoint_sha256"]},
                "train_group_split": {
                    "path": str(self.split_path),
                    "file_sha256": _digest(split),
                    "split_sha256": split["split_sha256"],
                    "group_ids": split["assignments"]["train"],
                },
                "generation_conditioning_output": {
                    "path": str(
                        self.root
                        / train_round_relative_root(launch_generation_id)
                        / "collection"
                    ),
                    "generation_id": launch_generation_id,
                },
            }
        elif role == "validation":
            launch_generation_id = generation_id(
                None if state["phase"].startswith("bc_") else state["current_round"]
            )
            if checkpoint_sha256 is None:
                raise NativeH200ExperimentError(
                    "a candidate checkpoint is required for running-phase validation"
                )
            bound_inputs = {
                "checkpoint": {
                    "sha256": _sha(checkpoint_sha256, "validation checkpoint")
                },
                "validation_registry": {
                    "path": str(Path(contract["source_root"])
                                / contract["validation_panel"]["registry_path"]),
                    "sha256": contract["validation_panel"]["registry_sha256"],
                    "paired_seeds": contract["validation_panel"]["paired_seeds"],
                    "cases_per_seed": contract["validation_panel"]["cases_per_seed"],
                    "group_ids": split["assignments"]["validation"],
                },
                "validation_result_output": {
                    "directory": str(
                        self.root
                        / "validation/results"
                        / train_round_relative_root(launch_generation_id)
                        / "mse"
                    ),
                    "schema": VALIDATION_MSE_RESULT_SCHEMA,
                },
            }
        else:
            launch_generation_id = state["final_generation_id"]
            if launch_generation_id not in GENERATION_IDS:
                raise NativeH200ExperimentError("final checkpoint generation is invalid")
            bound_inputs = {
                "checkpoint": {"sha256": state["final_checkpoint_sha256"]},
                "test_registry": {"sha256": state["test_registry_sha256"]},
                "test_result_output": {
                    "directory": str(self.root / "test/heldout/results"),
                    "schema": FINAL_TEST_RESULT_SCHEMA,
                },
            }
        generation_root = self.root / train_round_relative_root(launch_generation_id)
        if (
            not generation_root.is_dir()
            or generation_root.is_symlink()
        ):
            raise NativeH200ExperimentError(
                f"generation artifact root is missing or unsafe: {generation_root}"
            )
        output_root = (
            self.root / "test/heldout/results" if role == "test" else generation_root
        )
        if not output_root.is_dir() or output_root.is_symlink():
            raise NativeH200ExperimentError(f"artifact output root is missing or unsafe: {output_root}")
        checkpoint_path: Path | None = None
        if role == "learner":
            parent_generation = (
                GENERATION_IDS[0]
                if state["phase"] == "bc_running"
                else state["active_generation_id"]
            )
            if parent_generation not in GENERATION_IDS:
                raise NativeH200ExperimentError("learner parent generation is invalid")
            parent_root = self.root / train_round_relative_root(parent_generation)
            checkpoint_path = _resolve_checkpoint_path(
                self.root,
                parent_root,
                bound_inputs["parent_checkpoint"]["sha256"],
                include_bootstrap=state["phase"] == "bc_running",
            )
            policy_spec_path = Path(f"{checkpoint_path}.policy_spec.json")
            try:
                policy_spec_info = os.lstat(policy_spec_path)
            except OSError as exc:
                raise NativeH200ExperimentError(
                    "learner parent policy specification is missing"
                ) from exc
            if (
                not stat.S_ISREG(policy_spec_info.st_mode)
                or stat.S_ISLNK(policy_spec_info.st_mode)
                or stat.S_IMODE(policy_spec_info.st_mode) & 0o222
            ):
                raise NativeH200ExperimentError(
                    "learner parent policy specification is unsafe"
                )
        elif role in {"validation", "test"}:
            checkpoint_path = _resolve_checkpoint_path(
                self.root,
                generation_root,
                bound_inputs["checkpoint"]["sha256"],
            )
        commands = []
        for index in range(row["parallel_processes"]):
            first = index * row["gpus_per_process"]
            gpu_ids = ",".join(
                str(item) for item in range(first, first + row["gpus_per_process"])
            )
            environment = {**row["fixed_environment"], "CUDA_VISIBLE_DEVICES": gpu_ids}
            if role == "learner":
                environment["NATIVE_H200_TRAIN_REPLAY_MANIFEST"] = str(manifest_path)
                environment["NATIVE_H200_TRAIN_REPLAY_MANIFEST_SHA256"] = state[
                    "train_replay_manifest_sha256"
                ]
                environment["ROUND_DAGGER_PARENT_CHECKPOINT"] = str(checkpoint_path)
                environment["ROUND_DAGGER_PARENT_CHECKPOINT_SHA256"] = bound_inputs[
                    "parent_checkpoint"
                ]["sha256"]
                environment["ROUND_DAGGER_PARENT_POLICY_SPEC"] = str(policy_spec_path)
                environment["ROUND_DAGGER_PARENT_POLICY_SPEC_SHA256"] = _sha256_file(
                    policy_spec_path
                )
                snapshot = bound_inputs["train_replay_snapshot"]
                environment["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT"] = snapshot["path"]
                environment["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_FILE_SHA256"] = snapshot[
                    "file_sha256"
                ]
                environment["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_SHA256"] = snapshot[
                    "snapshot_sha256"
                ]
            elif role == "validation":
                registry = bound_inputs["validation_registry"]
                environment["ROUND_DAGGER_POLICY_CHECKPOINT"] = str(checkpoint_path)
                environment["ROUND_DAGGER_POLICY_SHA256"] = bound_inputs[
                    "checkpoint"
                ]["sha256"]
                environment["ROUND_DAGGER_VALIDATION_REGISTRY"] = registry["path"]
                environment["ROUND_DAGGER_VALIDATION_REGISTRY_SHA256"] = registry[
                    "sha256"
                ]
                environment["ROUND_DAGGER_VALIDATION_RESULT_DIR"] = bound_inputs[
                    "validation_result_output"
                ]["directory"]
            elif role == "test":
                test_root = (
                    self.root / "test/heldout/contracts"
                )
                test_registry = test_root / "test_registry.v1.json"
                test_cases = test_root / "test_cases.private.v1.json"
                test_partition = test_root / "motion_partition.v1.json"
                for path, label in (
                    (test_registry, "test registry"),
                    (test_cases, "private test cases"),
                    (test_partition, "test motion partition"),
                ):
                    try:
                        info = os.lstat(path)
                    except OSError as exc:
                        raise NativeH200ExperimentError(f"{label} is missing") from exc
                    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                        raise NativeH200ExperimentError(f"{label} is unsafe")
                if _sha256_file(test_registry) != bound_inputs["test_registry"]["sha256"]:
                    raise NativeH200ExperimentError("test registry digest differs")
                registry_document = _load_json(test_registry)
                cases_sha256 = _sha(
                    registry_document.get("cases_sha256"), "private test cases"
                )
                if _sha256_file(test_cases) != cases_sha256:
                    raise NativeH200ExperimentError("private test cases digest differs")
                environment["ROUND_DAGGER_POLICY_CHECKPOINT"] = str(checkpoint_path)
                environment["ROUND_DAGGER_POLICY_SHA256"] = bound_inputs[
                    "checkpoint"
                ]["sha256"]
                environment["ROUND_DAGGER_TEST_REGISTRY"] = str(test_registry)
                environment["ROUND_DAGGER_TEST_REGISTRY_SHA256"] = bound_inputs[
                    "test_registry"
                ]["sha256"]
                environment["ROUND_DAGGER_TEST_CASES"] = str(test_cases)
                environment["ROUND_DAGGER_TEST_CASES_SHA256"] = cases_sha256
                environment["ROUND_DAGGER_TEST_PARTITION"] = str(test_partition)
                environment["ROUND_DAGGER_TEST_PARTITION_SHA256"] = _sha256_file(
                    test_partition
                )
                environment["ROUND_DAGGER_TEST_RESULT_DIR"] = bound_inputs[
                    "test_result_output"
                ]["directory"]
                environment["ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT"] = (
                    "/workspace/codex/roborender-agent/runs/"
                    "h200_rollout_qualification_20260903T201136Z"
                )
            environment["ROUND_DAGGER_CAMPAIGN_ROOT"] = str(self.root)
            environment["ROUND_DAGGER_PROTOCOL_ID"] = contract["protocol_id"]
            environment["ROUND_DAGGER_SPLIT_SHA256"] = split["split_sha256"]
            environment["ROUND_DAGGER_GENERATION_ID"] = launch_generation_id
            environment["ROUND_DAGGER_GENERATION_ROOT"] = str(generation_root)
            environment["ROUND_DAGGER_OUTPUT_ROOT"] = str(output_root)
            environment["ROUND_DAGGER_FLEET_ROLE"] = role
            commands.append({"argv": [executable], "environment": environment})
        return {"schema": "fast_roborender.round_dagger.native_h200_launch_plan.v1",
                "role": role, "phase": state["phase"], "resource_shape": RESOURCE_SHAPE,
                "generation_id": launch_generation_id,
                "output_root": str(output_root),
                "input_bindings": row["input_bindings"],
                "bound_inputs": bound_inputs,
                "forbidden_data_bindings": (
                    ["validation_registry", "test_registry"] if role == "learner" else []
                ),
                "commands": commands}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-contract")
    validate.add_argument("--contract", type=Path, required=True)
    split = sub.add_parser("build-split")
    split.add_argument("--contract", type=Path, required=True)
    split.add_argument("--registry", type=Path, required=True)
    initialize = sub.add_parser("initialize")
    initialize.add_argument("--campaign-root", type=Path, required=True)
    initialize.add_argument("--contract", type=Path, required=True)
    initialize.add_argument("--split", type=Path, required=True)
    initialize.add_argument("--test-registry-sha256", required=True)
    initialize.add_argument("--bc-replay-sha256", required=True)
    initialize.add_argument("--initial-checkpoint-sha256", required=True)
    initialize.add_argument("--train-replay-snapshot", type=Path, required=True)
    initialize.add_argument("--train-replay-snapshot-file-sha256", required=True)
    initialize.add_argument("--train-replay-snapshot-sha256", required=True)
    status = sub.add_parser("status")
    status.add_argument("--campaign-root", type=Path, required=True)
    validation = sub.add_parser("record-validation")
    validation.add_argument("--campaign-root", type=Path, required=True)
    validation.add_argument("--result", type=Path, required=True)
    validation.add_argument("--result-file-sha256", required=True)
    start = sub.add_parser("start-round")
    start.add_argument("--campaign-root", type=Path, required=True)
    start.add_argument("--new-replay-sha256", required=True)
    start.add_argument("--collection-group", action="append", required=True)
    start.add_argument("--train-replay-snapshot", type=Path, required=True)
    start.add_argument("--train-replay-snapshot-file-sha256", required=True)
    start.add_argument("--train-replay-snapshot-sha256", required=True)
    select = sub.add_parser("select-final")
    select.add_argument("--campaign-root", type=Path, required=True)
    test = sub.add_parser("record-test")
    test.add_argument("--campaign-root", type=Path, required=True)
    test.add_argument("--registry", type=Path, required=True)
    test.add_argument("--result", type=Path, required=True)
    test.add_argument("--result-file-sha256", required=True)
    launch = sub.add_parser("launch-plan")
    launch.add_argument("--campaign-root", type=Path, required=True)
    launch.add_argument("--role", choices=sorted(_LAUNCH_KEYS), required=True)
    launch.add_argument("--checkpoint-sha256")
    args = parser.parse_args(argv)
    if args.command == "validate-contract":
        document = load_contract(args.contract)
        result = {"status": "valid", "contract_sha256": _digest(document)}
    elif args.command == "build-split":
        result = build_group_split(load_contract(args.contract), _load_json(args.registry))
    elif args.command == "initialize":
        result = Controller(args.campaign_root).initialize(
            contract=load_contract(args.contract), split=_load_json(args.split),
            test_registry_sha256=args.test_registry_sha256,
            bc_replay_sha256=args.bc_replay_sha256,
            initial_checkpoint_sha256=args.initial_checkpoint_sha256,
            train_replay_snapshot_path=args.train_replay_snapshot,
            train_replay_snapshot_file_sha256=(
                args.train_replay_snapshot_file_sha256
            ),
            train_replay_snapshot_sha256=args.train_replay_snapshot_sha256,
        )
    elif args.command == "status":
        result = Controller(args.campaign_root).status()
    elif args.command == "record-validation":
        result = Controller(args.campaign_root).record_validation(
            result_path=args.result,
            expected_file_sha256=args.result_file_sha256,
        )
    elif args.command == "start-round":
        result = Controller(args.campaign_root).start_round(
            new_replay_sha256=args.new_replay_sha256,
            collection_groups=args.collection_group,
            train_replay_snapshot_path=args.train_replay_snapshot,
            train_replay_snapshot_file_sha256=(
                args.train_replay_snapshot_file_sha256
            ),
            train_replay_snapshot_sha256=args.train_replay_snapshot_sha256,
        )
    elif args.command == "select-final":
        result = Controller(args.campaign_root).select_final()
    elif args.command == "record-test":
        result = Controller(args.campaign_root).record_test(
            registry_path=args.registry,
            result_path=args.result,
            expected_result_file_sha256=args.result_file_sha256,
        )
    else:
        result = Controller(args.campaign_root).launch_plan(
            args.role, checkpoint_sha256=args.checkpoint_sha256
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "BETAS", "CONTRACT_SCHEMA", "Controller", "FINAL_TEST_RESULT_SCHEMA",
    "GROUP_REGISTRY_SCHEMA",
    "NativeH200ExperimentError", "SPLIT_SCHEMA", "TEST_REGISTRY_SCHEMA",
    "build_group_split", "load_contract", "main", "plateau_decision",
    "validate_contract", "validate_group_split",
]
