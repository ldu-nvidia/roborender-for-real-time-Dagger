"""Sealed, fixed-panel validation collection on one native eight-H200 host.

The validation collector is deliberately separate from training collection.  It
copies only the five committed validation motions into held-out storage, runs
four independent two-rank cohorts, and publishes a validation-only replay
snapshot.  The 256 registry cases identify the *initial* environment/reset
streams.  Collection is rectangular (512 steps per environment), so simulator
auto-resets after termination remain in the MSE population and are explicitly
not represented as additional registry cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from round_dagger import artifacts, replay
from round_dagger.heldout_contracts import publish_create_only
from round_dagger.native_actor_file_inventory import (
    NativeActorInventoryError,
    verify_native_actor_bundle,
    verify_file_inventory as verify_native_actor_file_inventory,
)
from round_dagger.native_actor_protocol import (
    GENERATION_IDS,
    train_round_relative_root,
)


SCHEMA = "fast_roborender.round_dagger.native_h200_validation_assignment.v1"
HARDWARE_SCHEMA = "fast_roborender.round_dagger.native_h200_validation_hardware.v1"
SEAL_SCHEMA = "fast_roborender.round_dagger.native_h200_validation_dataset.v1"
CASE_DOMAIN = "fast_roborender.round_dagger.native_h200_validation_case.v1"
HOST_ATTESTATION_SCHEMA = (
    "fast_roborender.round_dagger.native_actor_host_attestation.v1"
)

CAMPAIGN_ID = "dagger-redo-v001"
CANONICAL_ARTIFACT_ROOT = Path("/workspace/project/roborender/data")
CANONICAL_CAMPAIGN_ROOT = CANONICAL_ARTIFACT_ROOT
CANONICAL_VALIDATION_ROOT = CANONICAL_ARTIFACT_ROOT / "validation/heldout"
CANONICAL_TEST_ROOT = CANONICAL_ARTIFACT_ROOT / "test"
VALIDATION_PARTITION = CANONICAL_VALIDATION_ROOT / "contracts/motion_partition.v1.json"
VALIDATION_REGISTRY = (
    CANONICAL_VALIDATION_ROOT / "contracts/validation_case_registry.v1.json"
)
VALIDATION_PARTITION_SHA256 = (
    "2af11e01e52349b3df524c098cd3d00f1407dd23fe02b6bf1c02c5cc4a8e5855"
)
VALIDATION_REGISTRY_SHA256 = (
    "2d9561f3a157116aee706473a58582b808dd61a8888d52eab1aa44acb7b071e7"
)
SPLIT_SHA256 = "ddb5d8ac9157d0ecb4eb89f966fef4d17bf73157c8b63be0571f389318b34bdf"
PAIRED_SEEDS = (1103, 2207, 3301, 4409)
GPU_COUNT = 8
COHORT_COUNT = 4
RANKS_PER_COHORT = 2
ENVIRONMENTS_PER_RANK = 32
ENVIRONMENTS_PER_COHORT = RANKS_PER_COHORT * ENVIRONMENTS_PER_RANK
CASE_COUNT = COHORT_COUNT * ENVIRONMENTS_PER_COHORT
STEPS_PER_ENVIRONMENT = 512
TRANSITION_COUNT = CASE_COUNT * STEPS_PER_ENVIRONMENT
ENVIRONMENTS_PER_SHARD = 4
VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COLLECTION_ID = re.compile(r"gen-[0-9]{3}-[a-z0-9-]+__validation-v[0-9]{3}\Z")
_GENERATION_ID = re.compile(r"gen-[0-9]{3}-[a-z0-9-]+\Z")
SAFE_EXECUTABLE_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
_BC_NAMING_ENVIRONMENT = {
    "ROUND_DAGGER_CAMPAIGN_ROOT",
    "ROUND_DAGGER_GENERATION_ROOT",
    "ROUND_DAGGER_GENERATION_ID",
    "ROUND_DAGGER_SATURATION_PHASE",
    "ROUND_DAGGER_SATURATION_WAVE_INDEX",
    "ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON",
}
_DISTRIBUTED_ENVIRONMENT = {
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "ROLE_RANK", "MASTER_ADDR", "MASTER_PORT",
}
_UNSAFE_HANDOFF_ENVIRONMENT = {
    "ROUND_DAGGER_PRIVATE_MOUNT_STAGE",
    "ROUND_DAGGER_PARENT_PID_NAMESPACE",
    "ROUND_DAGGER_ORIGINAL_TRAIN_PARTITION",
    "ROUND_DAGGER_JAIL_HANDLE_ROOT",
    "BASH_ENV",
    "ENV",
    "CDPATH",
    "GLOBIGNORE",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONINSPECT",
    "PYTHONSTARTUP",
    "PYTHONWARNINGS",
    "PYTHONBREAKPOINT",
    "PYTHONSAFEPATH",
    "PATH",
    "BASHOPTS",
    "SHELLOPTS",
    "BASH_XTRACEFD",
    "PS4",
    "PROMPT_COMMAND",
    "IFS",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "LD_DEBUG",
    "LD_PROFILE",
    "LD_SHOW_AUXV",
    "LD_BIND_NOW",
    "GCONV_PATH",
    "LOCPATH",
    "NLSPATH",
    "MALLOC_TRACE",
    "GLIBC_TUNABLES",
}


class NativeValidationError(RuntimeError):
    """The fixed validation set cannot be safely collected or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise NativeValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _regular(path: Path, label: str, *, expected_sha256: str | None = None) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise NativeValidationError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise NativeValidationError(f"{label} must be a regular non-symlink file")
    if expected_sha256 is not None and sha256_file(path) != _sha(expected_sha256, label):
        raise NativeValidationError(f"{label} SHA-256 mismatch")
    return path


def _directory(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise NativeValidationError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise NativeValidationError(f"{label} must be a real directory")
    return path


def _inside(path: Path, root: Path, label: str, *, may_not_exist: bool = False) -> Path:
    resolved_root = _directory(root, f"{label} root").resolve(strict=True)
    resolved = Path(os.path.abspath(path.expanduser())).resolve(
        strict=not may_not_exist and path.exists()
    )
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise NativeValidationError(f"{label} lies outside {resolved_root}") from exc
    if not relative.parts:
        raise NativeValidationError(f"{label} cannot be the storage root itself")
    return resolved


def _strict_json(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    path = _regular(path, label, expected_sha256=expected_sha256)
    try:
        value = artifacts.load_strict_json(path)
    except Exception as exc:
        raise NativeValidationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise NativeValidationError(f"{label} must contain a JSON object")
    return dict(value)


def _canonical_sha(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return artifacts.canonical_json_sha256(value)


@dataclass(frozen=True)
class ValidationSources:
    partition: dict[str, Any]
    registry: dict[str, Any]
    groups: tuple[dict[str, Any], ...]
    cases_by_group: dict[str, tuple[str, ...]]


def load_validation_sources(
    partition_path: Path = VALIDATION_PARTITION,
    registry_path: Path = VALIDATION_REGISTRY,
) -> ValidationSources:
    """Load the exact committed validation split without touching test cases."""

    partition = _strict_json(
        partition_path, "validation partition", VALIDATION_PARTITION_SHA256
    )
    if set(partition) != {"schema", "split_sha256", "purpose", "groups"}:
        raise NativeValidationError("validation partition fields differ")
    if (
        partition["schema"] != "fast_roborender.round_dagger.motion_partition.v1"
        or partition["purpose"] != "validation"
        or partition["split_sha256"] != SPLIT_SHA256
    ):
        raise NativeValidationError("validation partition identity differs")
    raw_groups = partition["groups"]
    if not isinstance(raw_groups, list) or len(raw_groups) != 5:
        raise NativeValidationError("validation partition must contain five groups")
    groups: list[dict[str, Any]] = []
    group_ids: list[str] = []
    for index, raw in enumerate(raw_groups):
        if not isinstance(raw, Mapping) or set(raw) != {
            "group_id", "relative_path", "size_bytes", "motion_sha256"
        }:
            raise NativeValidationError(f"validation group {index} fields differ")
        row = dict(raw)
        group_id = row["group_id"]
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in group_ids
            or row["relative_path"] != group_id
            or Path(group_id).name != group_id
            or Path(group_id).suffix != ".npz"
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] <= 0
        ):
            raise NativeValidationError("validation motion group is malformed")
        _sha(row["motion_sha256"], "validation motion")
        group_ids.append(group_id)
        groups.append(row)

    registry = _strict_json(
        registry_path, "validation registry", VALIDATION_REGISTRY_SHA256
    )
    if set(registry) != {
        "schema", "panel_id", "paired_seeds", "case_ids", "case_count"
    }:
        raise NativeValidationError("validation registry fields differ")
    raw_case_ids = registry["case_ids"]
    if (
        registry["schema"]
        != "fast_roborender.round_dagger.validation_panel_registry.v1"
        or registry["paired_seeds"] != list(PAIRED_SEEDS)
        or registry["case_count"] != CASE_COUNT
        or not isinstance(raw_case_ids, list)
        or len(raw_case_ids) != ENVIRONMENTS_PER_COHORT
        or len(set(raw_case_ids)) != len(raw_case_ids)
    ):
        raise NativeValidationError("validation registry is not the fixed 64x4 panel")
    cases: dict[str, list[tuple[int, str]]] = {group_id: [] for group_id in group_ids}
    for case_id in raw_case_ids:
        if not isinstance(case_id, str) or "__reset_" not in case_id:
            raise NativeValidationError("validation panel case ID is malformed")
        group_id, reset_raw = case_id.rsplit("__reset_", 1)
        if group_id not in cases or not re.fullmatch(r"0|[1-9][0-9]*|0[0-9]+", reset_raw):
            raise NativeValidationError("validation panel case leaves the validation split")
        reset_index = int(reset_raw)
        if any(previous == reset_index for previous, _ in cases[group_id]):
            raise NativeValidationError("validation panel repeats a group reset index")
        cases[group_id].append((reset_index, case_id))
    if sorted(len(values) for values in cases.values()) != [12, 13, 13, 13, 13]:
        raise NativeValidationError("validation panel group balance must be 13/13/13/13/12")
    ordered_cases: dict[str, tuple[str, ...]] = {}
    for group_id, values in cases.items():
        ordered = sorted(values)
        if [index for index, _ in ordered] != list(range(len(ordered))):
            raise NativeValidationError("validation reset indices must be contiguous from zero")
        ordered_cases[group_id] = tuple(case_id for _, case_id in ordered)
    return ValidationSources(
        partition=partition,
        registry=registry,
        groups=tuple(groups),
        cases_by_group=ordered_cases,
    )


def build_case_assignments(sources: ValidationSources) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map every fixed panel case to one initial simulator environment."""

    group_ids = [row["group_id"] for row in sources.groups]
    cases: list[dict[str, Any]] = []
    cohorts: list[dict[str, Any]] = []
    for seed_index, paired_seed in enumerate(PAIRED_SEEDS):
        consumed = {group_id: 0 for group_id in group_ids}
        cohort_cases: list[dict[str, Any]] = []
        for cohort_env_index in range(ENVIRONMENTS_PER_COHORT):
            local_rank = cohort_env_index // ENVIRONMENTS_PER_RANK
            local_env_index = cohort_env_index % ENVIRONMENTS_PER_RANK
            physical_gpu = seed_index * RANKS_PER_COHORT + local_rank
            group_id = group_ids[cohort_env_index % len(group_ids)]
            reset_index = consumed[group_id]
            try:
                panel_case_id = sources.cases_by_group[group_id][reset_index]
            except IndexError as exc:
                raise NativeValidationError(
                    "cyclic simulator group assignment exceeds the fixed panel"
                ) from exc
            consumed[group_id] += 1
            global_env_id = seed_index * ENVIRONMENTS_PER_COHORT + cohort_env_index
            body = {
                "domain": CASE_DOMAIN,
                "validation_registry_sha256": VALIDATION_REGISTRY_SHA256,
                "paired_seed": paired_seed,
                "panel_case_id": panel_case_id,
                "global_env_id": global_env_id,
            }
            row = {
                "case_sha256": _canonical_sha(body),
                "panel_case_id": panel_case_id,
                "paired_seed": paired_seed,
                "seed_index": seed_index,
                "cohort_index": seed_index,
                "physical_gpu": physical_gpu,
                "cohort_rank": local_rank,
                "local_env_index": local_env_index,
                "cohort_env_index": cohort_env_index,
                "global_env_id": global_env_id,
                "motion_group_id": group_id,
                "reset_index": reset_index,
            }
            cases.append(row)
            cohort_cases.append(row)
        if consumed != {group_id: len(sources.cases_by_group[group_id]) for group_id in group_ids}:
            raise NativeValidationError("one paired-seed cohort does not cover all panel cases")
        cohorts.append(
            {
                "cohort_index": seed_index,
                "paired_seed": paired_seed,
                "physical_gpus": [seed_index * 2, seed_index * 2 + 1],
                "expected_ranks": [0, 1],
                "expected_global_env_ids": [
                    row["global_env_id"] for row in cohort_cases
                ],
                "expected_transitions": ENVIRONMENTS_PER_COHORT * STEPS_PER_ENVIRONMENT,
                "ranks": [
                    {
                        "physical_gpu": seed_index * 2 + rank,
                        "cohort_rank": rank,
                        "hydra_base_seed": paired_seed - rank,
                        "effective_simulator_seed": paired_seed,
                        "motion_assignment_offset": rank * ENVIRONMENTS_PER_RANK,
                        "global_env_offset": seed_index * ENVIRONMENTS_PER_COHORT,
                        "global_env_ids": [
                            row["global_env_id"]
                            for row in cohort_cases
                            if row["cohort_rank"] == rank
                        ],
                    }
                    for rank in range(RANKS_PER_COHORT)
                ],
            }
        )
    if len(cases) != CASE_COUNT or {row["global_env_id"] for row in cases} != set(range(CASE_COUNT)):
        raise NativeValidationError("validation assignments do not cover 256 unique environments")
    if len({row["case_sha256"] for row in cases}) != CASE_COUNT:
        raise NativeValidationError("validation case identities are not unique")
    return cases, cohorts


@dataclass(frozen=True)
class CollectionConfig:
    repository_root: Path
    gr00t_root: Path
    gr00t_python: Path
    source_motion_root: Path
    source_generation_id: str
    macro_round: int
    collection_id: str
    policy_checkpoint: Path
    policy_sha256: str
    policy_spec: Path
    policy_spec_sha256: str
    runtime_policy_sha256: str
    teacher_checkpoint: Path
    teacher_sha256: str
    beta_sha256: str
    config_path: Path
    config_sha256: str
    observation_provenance: dict[str, Any]
    runtime_environment: dict[str, str]
    replay_action_dim: int

    @property
    def validation_root(self) -> Path:
        return CANONICAL_VALIDATION_ROOT

    @property
    def motion_root(self) -> Path:
        return self.validation_root / "inputs/motions-v001"

    @property
    def collection_root(self) -> Path:
        return self.validation_root / "replay/collections" / self.collection_id

    @property
    def assignment_path(self) -> Path:
        return self.validation_root / "contracts" / f"{self.collection_id}.assignment.json"

    @property
    def hardware_path(self) -> Path:
        return self.validation_root / "contracts" / f"{self.collection_id}.hardware.json"

    @property
    def snapshot_path(self) -> Path:
        return self.validation_root / "replay" / f"{self.collection_id}.snapshot.json"

    @property
    def seal_path(self) -> Path:
        return self.validation_root / "seals" / f"{self.collection_id}.seal.json"

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "CollectionConfig":
        values = os.environ if environment is None else environment

        def required(name: str) -> str:
            value = values.get(name)
            if not value:
                raise NativeValidationError(f"missing required environment: {name}")
            return value

        repository_root = Path(__file__).resolve().parents[1]
        gr00t_root = _directory(
            Path(values.get("GR00T_ROOT", repository_root / "GR00T-VisualSim2Real")),
            "GR00T root",
        )
        if gr00t_root.resolve(strict=True) != (
            repository_root / "GR00T-VisualSim2Real"
        ).resolve(strict=True):
            raise NativeValidationError(
                "validation GR00T source must be the repository checkout"
            )
        gr00t_python = Path(
            os.path.abspath(Path(required("GR00T_PYTHON")).expanduser())
        )
        try:
            python_link_info = os.lstat(gr00t_python)
            python_link_target = os.readlink(gr00t_python)
            resolved_python = gr00t_python.resolve(strict=True)
            resolved_python_info = os.lstat(resolved_python)
        except OSError as exc:
            raise NativeValidationError(
                f"qualified GR00T Python is unavailable: {exc}"
            ) from exc
        if (
            not stat.S_ISLNK(python_link_info.st_mode)
            or not Path(python_link_target).is_absolute()
            or Path(python_link_target) != resolved_python
            or not stat.S_ISREG(resolved_python_info.st_mode)
            or not os.access(gr00t_python, os.X_OK)
        ):
            raise NativeValidationError(
                "GR00T_PYTHON must be the qualified absolute symlink to an "
                "executable regular target"
            )
        source_motion_root = _directory(
            gr00t_root / "gr00t/rl/data/motions/g1_wsdpt/33demos_675_775",
            "source all-motion root",
        )
        generation = required("ROUND_DAGGER_VALIDATION_SOURCE_GENERATION_ID")
        if _GENERATION_ID.fullmatch(generation) is None:
            raise NativeValidationError("validation source generation ID is malformed")
        macro_raw = required("ROUND_DAGGER_VALIDATION_SOURCE_MACRO_ROUND")
        if re.fullmatch(r"0|[1-9][0-9]*", macro_raw) is None:
            raise NativeValidationError("validation source macro round is malformed")
        macro_round = int(macro_raw)
        if not generation.startswith(f"gen-{macro_round:03d}-"):
            raise NativeValidationError("validation generation and macro round disagree")
        collection_id = values.get(
            "ROUND_DAGGER_VALIDATION_COLLECTION_ID",
            f"{generation}__validation-v001",
        )
        if _COLLECTION_ID.fullmatch(collection_id) is None or not collection_id.startswith(generation + "__"):
            raise NativeValidationError("validation collection ID is not generation-bound")

        policy_sha = _sha(required("ROUND_DAGGER_POLICY_SHA256"), "policy SHA-256")
        policy = _regular(
            Path(required("ROUND_DAGGER_POLICY_CHECKPOINT")),
            "policy checkpoint",
            expected_sha256=policy_sha,
        )
        _inside(
            policy,
            CANONICAL_CAMPAIGN_ROOT / train_round_relative_root(generation),
            "policy checkpoint",
        )
        policy_spec_sha = _sha(
            required("ROUND_DAGGER_POLICY_SPEC_SHA256"),
            "policy specification",
        )
        policy_spec = _regular(
            Path(required("ROUND_DAGGER_POLICY_SPEC")),
            "policy specification",
            expected_sha256=policy_spec_sha,
        )
        _inside(
            policy_spec,
            CANONICAL_CAMPAIGN_ROOT / train_round_relative_root(generation),
            "policy specification",
        )
        policy_document = _strict_json(policy_spec, "policy specification")
        runtime_policy_sha = _sha(
            required("ROUND_DAGGER_RUNTIME_POLICY_SHA256"),
            "runtime policy SHA-256",
        )
        if (
            policy_document.get("checkpoint_sha256") != policy_sha
            or policy_document.get("runtime_policy_sha256") != runtime_policy_sha
        ):
            raise NativeValidationError("policy specification identity mismatch")
        output_contract = policy_document.get("output_contract")
        replay_action_dim = (
            output_contract.get("replay_action_dim")
            if isinstance(output_contract, Mapping)
            else None
        )
        if type(replay_action_dim) is not int or replay_action_dim <= 0:
            raise NativeValidationError("policy specification replay action dimension is invalid")

        teacher_sha = _sha(required("ROUND_DAGGER_TEACHER_SHA256"), "teacher SHA-256")
        teacher = _regular(
            Path(required("ROUND_DAGGER_TEACHER_CHECKPOINT")),
            "teacher checkpoint",
            expected_sha256=teacher_sha,
        )
        _inside(
            teacher,
            CANONICAL_CAMPAIGN_ROOT / train_round_relative_root(generation),
            "teacher checkpoint",
        )
        beta_sha = _sha(required("ROUND_DAGGER_BETA_SHA256"), "beta SHA-256")
        if beta_sha != _canonical_sha({"beta": 1.0}):
            raise NativeValidationError("validation beta SHA-256 must bind beta=1.0")

        config_path = _regular(
            gr00t_root / "gr00t/rl/config/exp/loco_manip/wsdpt_round_actor_native_3view.yaml",
            "native actor config",
        )
        config_sha = _sha(required("ROUND_DAGGER_CONFIG_SHA256"), "config SHA-256")
        if sha256_file(config_path) != config_sha:
            raise NativeValidationError("native actor config SHA-256 mismatch")
        try:
            raw_inventory = json.loads(
                required("ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON")
            )
            physical_inventory = artifacts.validate_physical_gpu_inventory(
                raw_inventory
            )
        except Exception as exc:
            if isinstance(exc, NativeValidationError):
                raise
            raise NativeValidationError(
                f"physical H200 inventory is invalid: {exc}"
            ) from exc
        physical_inventory_sha = _sha(
            required("ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256"),
            "physical GPU inventory",
        )
        if _canonical_sha(physical_inventory) != physical_inventory_sha:
            raise NativeValidationError("physical GPU inventory digest mismatch")
        provenance = {
            "observation_source": required("ROUND_DAGGER_OBSERVATION_SOURCE"),
            "observation_runtime_sha256": _sha(
                required("ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256"), "observation runtime"
            ),
            "isaac_runtime_sha256": _sha(
                required("ROUND_DAGGER_ISAAC_RUNTIME_SHA256"), "Isaac runtime"
            ),
            "actor_python_freeze_sha256": _sha(
                required("ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256"), "actor Python freeze"
            ),
            "actor_source_inventory_sha256": _sha(
                required("ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256"), "actor source inventory"
            ),
            "gr00t_revision": required("ROUND_DAGGER_GR00T_REVISION"),
            "gr00t_source_tree_sha256": _sha(
                required("ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"), "GR00T source tree"
            ),
            "gpu_name": required("ROUND_DAGGER_GPU_NAME"),
            "compute_capability": required("ROUND_DAGGER_COMPUTE_CAPABILITY"),
            "gpu_count": int(required("ROUND_DAGGER_GPU_COUNT")),
            "physical_gpu_inventory": physical_inventory,
            "physical_gpu_inventory_sha256": physical_inventory_sha,
        }
        if (
            provenance["observation_source"] != "native"
            or provenance["gr00t_revision"] == ""
            or re.fullmatch(r"[0-9a-f]{40}", provenance["gr00t_revision"]) is None
            or provenance["gpu_name"] != "NVIDIA H200"
            or provenance["compute_capability"] != "9.0"
            or provenance["gpu_count"] != GPU_COUNT
        ):
            raise NativeValidationError("native validation requires exact 8x H200 provenance")
        runtime_body = {
            "schema": "fast_roborender.round_dagger.native_h200_observation_runtime.v1",
            "observation_source": "native",
            "isaac_runtime_sha256": provenance["isaac_runtime_sha256"],
            "actor_python_freeze_sha256": provenance["actor_python_freeze_sha256"],
            "config_sha256": config_sha,
            "actor_source_inventory_sha256": provenance["actor_source_inventory_sha256"],
            "gr00t_revision": provenance["gr00t_revision"],
            "gr00t_source_tree_sha256": provenance["gr00t_source_tree_sha256"],
            "gpu_name": provenance["gpu_name"],
            "compute_capability": provenance["compute_capability"],
            "gpu_count": provenance["gpu_count"],
            "physical_gpu_inventory": physical_inventory,
            "physical_gpu_inventory_sha256": physical_inventory_sha,
        }
        if _canonical_sha(runtime_body) != provenance["observation_runtime_sha256"]:
            raise NativeValidationError("native observation runtime digest mismatch")
        runtime_environment: dict[str, str] = {}
        manifest_digests = {
            "ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST": provenance["isaac_runtime_sha256"],
            "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST": provenance[
                "actor_python_freeze_sha256"
            ],
            "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY": provenance[
                "actor_source_inventory_sha256"
            ],
            "ROUND_DAGGER_GR00T_SOURCE_INVENTORY": provenance[
                "gr00t_source_tree_sha256"
            ],
        }
        for name, expected_digest in manifest_digests.items():
            path = _regular(
                Path(required(name)), name, expected_sha256=expected_digest
            )
            runtime_environment[name] = str(path)
        runtime_environment["ROUND_DAGGER_HOST_RUNTIME_ATTESTATION_SHA256"] = (
            _canonical_sha(
                {
                    "schema": HOST_ATTESTATION_SCHEMA,
                    "manifests": {
                        "actor_source": provenance[
                            "actor_source_inventory_sha256"
                        ],
                        "gr00t_source": provenance["gr00t_source_tree_sha256"],
                        "isaac_runtime": provenance["isaac_runtime_sha256"],
                        "python_runtime": provenance[
                            "actor_python_freeze_sha256"
                        ],
                    },
                }
            )
        )
        directory_names = (
            "ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT",
            "ROUND_DAGGER_ISAAC_RUNTIME_ROOT",
            "ROUND_DAGGER_PYTHON_RUNTIME_ROOT",
            "ROUND_DAGGER_CUDNN_LIBRARY_DIR",
            "ROUND_DAGGER_ISAACLAB_SOURCE_ROOT",
            "ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT",
            "ROUND_DAGGER_SYSTEM_LIBRARY_DIR",
        )
        for name in directory_names:
            runtime_environment[name] = str(_directory(Path(required(name)), name))
        expected_python_link = (
            Path(runtime_environment["ROUND_DAGGER_ISAAC_RUNTIME_ROOT"])
            / "bin/python"
        )
        python_runtime_root = Path(
            runtime_environment["ROUND_DAGGER_PYTHON_RUNTIME_ROOT"]
        ).resolve(strict=True)
        qualified_runtime_root = Path(
            runtime_environment["ROUND_DAGGER_QUALIFIED_RUNTIME_ROOT"]
        ).resolve(strict=True)
        isaac_runtime_root = Path(
            runtime_environment["ROUND_DAGGER_ISAAC_RUNTIME_ROOT"]
        ).resolve(strict=True)
        isaaclab_source_root = Path(
            runtime_environment["ROUND_DAGGER_ISAACLAB_SOURCE_ROOT"]
        ).resolve(strict=True)
        dependency_root = Path(
            runtime_environment["ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT"]
        ).resolve(strict=True)
        system_library_root = Path(
            runtime_environment["ROUND_DAGGER_SYSTEM_LIBRARY_DIR"]
        ).resolve(strict=True)
        cudnn_root = Path(
            runtime_environment["ROUND_DAGGER_CUDNN_LIBRARY_DIR"]
        ).resolve(strict=True)
        if (
            python_runtime_root != qualified_runtime_root
            or isaac_runtime_root
            != qualified_runtime_root / "runtime/isaacsim-5.1"
            or isaaclab_source_root
            != qualified_runtime_root / "runtime/IsaacLab/source"
            or dependency_root
            != qualified_runtime_root / "runtime/actor-python-site-packages"
            or system_library_root
            != qualified_runtime_root / "packages/root/usr/lib/x86_64-linux-gnu"
            or cudnn_root
            != isaac_runtime_root
            / "lib/python3.11/site-packages/nvidia/cudnn/lib"
        ):
            raise NativeValidationError(
                "validation runtime roots differ from the exact qualified QROOT topology"
            )
        try:
            resolved_python.relative_to(python_runtime_root)
        except ValueError as exc:
            raise NativeValidationError(
                "qualified GR00T Python target leaves the attested Python runtime"
            ) from exc
        if gr00t_python != expected_python_link:
            raise NativeValidationError(
                "GR00T_PYTHON must be <Isaac runtime root>/bin/python"
            )
        if resolved_python != (
            qualified_runtime_root
            / "runtime/python/cpython-3.11.13-linux-x86_64-gnu/bin/python3.11"
        ):
            raise NativeValidationError(
                "GR00T_PYTHON target differs from the qualified CPython 3.11.13 binary"
            )
        forbidden_dependency_entries = [
            entry.name
            for entry in dependency_root.iterdir()
            if entry.name == "sitecustomize.py"
            or entry.name.endswith(".pth")
            or "editable" in entry.name.lower()
        ]
        if forbidden_dependency_entries:
            raise NativeValidationError(
                "qualified Python dependency root contains path-injection metadata"
            )
        for name in (
            "ROUND_DAGGER_ISAAC_SETUP_SCRIPT",
            "ROBORENDER_HEADLESS_EXPERIENCE",
        ):
            runtime_environment[name] = str(_regular(Path(required(name)), name))
        if Path(runtime_environment["ROUND_DAGGER_ISAAC_SETUP_SCRIPT"]) != (
            repository_root / "scripts/round_dagger/setup_native_h200_isaac_runtime.sh"
        ):
            raise NativeValidationError("validation must use the repository Isaac setup")
        if Path(runtime_environment["ROBORENDER_HEADLESS_EXPERIENCE"]) != (
            qualified_runtime_root
            / "runtime/IsaacLab/apps/phc.isaaclab.python.headless.rendering.kit"
        ):
            raise NativeValidationError("validation headless experience differs from QROOT")
        for name, expected in (
            ("OMNI_KIT_ACCEPT_EULA", "YES"),
            ("OMNI_KIT_ALLOW_ROOT", "1"),
            ("ACCEPT_EULA", "Y"),
            ("PRIVACY_CONSENT", "Y"),
        ):
            actual = required(name)
            if actual != expected:
                raise NativeValidationError(f"{name} must be exactly {expected}")
            runtime_environment[name] = actual
        expected_library_path = ":".join(
            (
                runtime_environment["ROUND_DAGGER_CUDNN_LIBRARY_DIR"],
                str(system_library_root),
                "/usr/lib/x86_64-linux-gnu",
            )
        )
        if required("LD_LIBRARY_PATH") != expected_library_path:
            raise NativeValidationError(
                "LD_LIBRARY_PATH must be the exact qualified cuDNN/system search path"
            )
        return cls(
            repository_root=repository_root,
            gr00t_root=gr00t_root,
            gr00t_python=gr00t_python,
            source_motion_root=source_motion_root,
            source_generation_id=generation,
            macro_round=macro_round,
            collection_id=collection_id,
            policy_checkpoint=policy,
            policy_sha256=policy_sha,
            policy_spec=policy_spec,
            policy_spec_sha256=policy_spec_sha,
            runtime_policy_sha256=runtime_policy_sha,
            teacher_checkpoint=teacher,
            teacher_sha256=teacher_sha,
            beta_sha256=beta_sha,
            config_path=config_path,
            config_sha256=config_sha,
            observation_provenance=provenance,
            runtime_environment=runtime_environment,
            replay_action_dim=replay_action_dim,
        )


def _exclusive_copy(source: Path, destination: Path, *, expected_size: int, expected_sha256: str) -> None:
    source_fd = destination_fd = None
    created: tuple[int, int] | None = None
    digest = hashlib.sha256()
    size = 0
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            raise NativeValidationError("source validation motion is not a regular file")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        info = os.fstat(destination_fd)
        created = (info.st_dev, info.st_ino)
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise NativeValidationError("short validation motion write")
                view = view[written:]
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise NativeValidationError("source validation motion content differs from partition")
        os.fchmod(destination_fd, 0o400)
        os.fsync(destination_fd)
    except Exception:
        if created is not None:
            try:
                current = os.lstat(destination)
                if (current.st_dev, current.st_ino) == created:
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)


def validate_source_motions(config: CollectionConfig, sources: ValidationSources) -> list[dict[str, Any]]:
    inventory = []
    for row in sources.groups:
        source = _regular(
            config.source_motion_root / row["relative_path"],
            f"source validation motion {row['group_id']}",
            expected_sha256=row["motion_sha256"],
        )
        if source.stat().st_size != row["size_bytes"]:
            raise NativeValidationError("source validation motion size mismatch")
        inventory.append({
            "group_id": row["group_id"],
            "relative_path": row["relative_path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["motion_sha256"],
        })
    return inventory


def _ensure_new_targets(config: CollectionConfig) -> None:
    for path in (
        config.motion_root,
        config.collection_root,
        config.assignment_path,
        config.hardware_path,
        config.snapshot_path,
        config.seal_path,
        config.validation_root / "logs" / config.collection_id,
    ):
        if path.exists() or path.is_symlink():
            raise NativeValidationError(f"refusing to reuse validation output: {path}")


def query_h200_inventory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,mig.mode.current",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NativeValidationError(f"cannot inventory H200 GPUs: {exc}") from exc
    rows = []
    for line in result.stdout.splitlines():
        tokens = [token.strip() for token in line.split(",")]
        if len(tokens) != 7:
            raise NativeValidationError("nvidia-smi inventory row is malformed")
        index_raw, uuid, pci_bus_id, name, compute_capability, memory_raw, mig_mode = tokens
        if not index_raw.isdigit() or not memory_raw.isdigit():
            raise NativeValidationError("nvidia-smi inventory values are malformed")
        rows.append({
            "rank": int(index_raw),
            "physical_index": int(index_raw),
            "name": name,
            "uuid": uuid.lower().replace("gpu-", "GPU-", 1),
            "pci_bus_id": pci_bus_id.upper(),
            "compute_capability": compute_capability,
            "total_memory_mib": int(memory_raw),
            "mig_mode": mig_mode,
        })
    try:
        return artifacts.validate_physical_gpu_inventory(rows)
    except Exception as exc:
        raise NativeValidationError(
            f"validation host is not exact 8xH200: {exc}"
        ) from exc


def _verify_file_inventory(
    manifest: Path,
    *,
    kind: str,
    allowed_roots: Sequence[Path],
    required_paths: Sequence[Path],
    required_symlink_paths: Sequence[Path] = (),
) -> int:
    roots = tuple(
        _directory(root, f"{kind} allowed root").resolve(strict=True)
        for root in allowed_roots
    )
    symlinks: dict[Path, Path] = {}
    for raw_path in required_symlink_paths:
        link = Path(os.path.abspath(raw_path))
        try:
            raw_target = Path(os.readlink(link))
            target = (
                raw_target if raw_target.is_absolute() else link.parent / raw_target
            ).resolve(strict=True)
        except OSError as exc:
            raise NativeValidationError(
                f"{kind} required symlink is unavailable: {link}: {exc}"
            ) from exc
        symlinks[link] = target
    try:
        return verify_native_actor_file_inventory(
            manifest,
            kind=kind,
            allowed_roots=roots,
            required_paths=tuple(Path(path) for path in required_paths),
            allowed_symlinks=symlinks,
        )
    except (NativeActorInventoryError, OSError) as exc:
        raise NativeValidationError(
            f"{kind} inventory verification failed: {exc}"
        ) from exc


def verify_runtime_inventories(
    config: CollectionConfig, *, full_runtime_trees: bool = True
) -> dict[str, Any]:
    """Verify the shared actor bundle plus validation-specific entry points.

    The trusted fleet controller performs the expensive full QROOT/dependency
    rehash once.  A rank whose source and QROOT are already read-only reparses
    the same manifests and rehashes only the live runtime entry points.
    """

    actor_root = config.repository_root.resolve(strict=True)
    gr00t_root = config.gr00t_root.resolve(strict=True)
    isaac_root = Path(
        config.runtime_environment["ROUND_DAGGER_ISAAC_RUNTIME_ROOT"]
    ).resolve(strict=True)
    python_root = Path(
        config.runtime_environment["ROUND_DAGGER_PYTHON_RUNTIME_ROOT"]
    ).resolve(strict=True)
    cudnn_root = Path(
        config.runtime_environment["ROUND_DAGGER_CUDNN_LIBRARY_DIR"]
    ).resolve(strict=True)
    isaaclab_source_root = Path(
        config.runtime_environment["ROUND_DAGGER_ISAACLAB_SOURCE_ROOT"]
    ).resolve(strict=True)
    dependency_root = Path(
        config.runtime_environment["ROUND_DAGGER_PYTHON_DEPENDENCY_ROOT"]
    ).resolve(strict=True)
    system_library_root = Path(
        config.runtime_environment["ROUND_DAGGER_SYSTEM_LIBRARY_DIR"]
    ).resolve(strict=True)
    isaac_setup = Path(
        config.runtime_environment["ROUND_DAGGER_ISAAC_SETUP_SCRIPT"]
    ).resolve(strict=True)
    headless = Path(
        config.runtime_environment["ROBORENDER_HEADLESS_EXPERIENCE"]
    ).resolve(strict=True)
    gr00t_python_link = config.gr00t_python
    actor_required = (
        actor_root / "round_dagger/actor.py",
        actor_root / "round_dagger/actor_checkpoint_compat.py",
        actor_root / "round_dagger/actor_identity.py",
        actor_root / "round_dagger/artifacts.py",
        actor_root / "round_dagger/gr00t_policy.py",
        actor_root / "round_dagger/native_actor_file_inventory.py",
        actor_root / "round_dagger/native_h200_validation.py",
        actor_root / "round_dagger/policy_identity.py",
        actor_root / "scripts/round_dagger/actor_training_entrypoint.py",
        actor_root / "scripts/round_dagger/run_native_h200_validation_fleet.py",
        actor_root / "scripts/round_dagger/run_native_h200_validation_rank.sh",
        isaac_setup,
    )
    try:
        counts = verify_native_actor_bundle(
            actor_manifest=Path(
                config.runtime_environment["ROUND_DAGGER_ACTOR_SOURCE_INVENTORY"]
            ),
            gr00t_manifest=Path(
                config.runtime_environment["ROUND_DAGGER_GR00T_SOURCE_INVENTORY"]
            ),
            isaac_manifest=Path(
                config.runtime_environment["ROUND_DAGGER_ISAAC_RUNTIME_MANIFEST"]
            ),
            python_manifest=Path(
                config.runtime_environment[
                    "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_MANIFEST"
                ]
            ),
            actor_root=actor_root,
            gr00t_root=gr00t_root,
            gr00t_python=gr00t_python_link,
            isaac_root=isaac_root,
            python_root=python_root,
            isaac_setup=isaac_setup,
            headless_experience=headless,
            cudnn_root=cudnn_root,
            isaaclab_source_root=isaaclab_source_root,
            dependency_root=dependency_root,
            system_library_root=system_library_root,
            full_runtime_trees=full_runtime_trees,
        )
        # The shared actor contract deliberately knows only the train actor.
        # Bind every additional repository module/script executed by validation.
        validation_entry_count = _verify_file_inventory(
            Path(
                config.runtime_environment["ROUND_DAGGER_ACTOR_SOURCE_INVENTORY"]
            ),
            kind="actor_source",
            allowed_roots=(actor_root,),
            required_paths=actor_required
            + (
                actor_root / "round_dagger/__init__.py",
                actor_root / "round_dagger/heldout_contracts.py",
                actor_root / "round_dagger/native_h200_experiment.py",
                actor_root / "round_dagger/replay.py",
            ),
        )
    except (NativeActorInventoryError, OSError) as exc:
        raise NativeValidationError(
            f"native validation runtime inventory verification failed: {exc}"
        ) from exc
    if validation_entry_count != counts["actor_source"]:
        raise NativeValidationError(
            "validation/shared actor inventory entry counts disagree"
        )
    expected_attestation = _canonical_sha(
        {
            "schema": HOST_ATTESTATION_SCHEMA,
            "manifests": {
                "actor_source": config.observation_provenance[
                    "actor_source_inventory_sha256"
                ],
                "gr00t_source": config.observation_provenance[
                    "gr00t_source_tree_sha256"
                ],
                "isaac_runtime": config.observation_provenance[
                    "isaac_runtime_sha256"
                ],
                "python_runtime": config.observation_provenance[
                    "actor_python_freeze_sha256"
                ],
            },
        }
    )
    if config.runtime_environment.get(
        "ROUND_DAGGER_HOST_RUNTIME_ATTESTATION_SHA256"
    ) != expected_attestation:
        raise NativeValidationError("host runtime attestation binding differs")
    return {
        "status": "verified",
        "full_runtime_trees": full_runtime_trees,
        "entry_counts": counts,
        "host_runtime_attestation_sha256": expected_attestation,
    }


def _cohort_path(config: CollectionConfig, cohort_index: int, paired_seed: int) -> Path:
    return config.collection_root / "cohorts" / f"seed-{cohort_index:02d}-{paired_seed}"


def prepare_collection(config: CollectionConfig, hardware: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create validation-only inputs, cohort contracts, and launch assignment."""

    _directory(config.validation_root, "canonical validation root")
    _directory(CANONICAL_TEST_ROOT, "sealed test root")  # metadata only; never enumerate/read
    _ensure_new_targets(config)
    sources = load_validation_sources()
    source_inventory = validate_source_motions(config, sources)
    if list(hardware) != query_h200_inventory():
        raise NativeValidationError("H200 inventory changed during validation preparation")
    if list(hardware) != config.observation_provenance["physical_gpu_inventory"]:
        raise NativeValidationError("live H200 inventory differs from runtime provenance")
    runtime_inventory_report = verify_runtime_inventories(config)

    for parent in (
        config.validation_root / "inputs",
        config.validation_root / "replay/collections",
        config.validation_root / "seals",
    ):
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _directory(parent, "validation output parent")
    config.motion_root.mkdir(mode=0o700)
    for row in sources.groups:
        _exclusive_copy(
            config.source_motion_root / row["relative_path"],
            config.motion_root / row["relative_path"],
            expected_size=row["size_bytes"],
            expected_sha256=row["motion_sha256"],
        )
    if sorted(path.name for path in config.motion_root.iterdir()) != sorted(
        row["relative_path"] for row in sources.groups
    ):
        raise NativeValidationError("materialized validation motion inventory is not exact")
    os.chmod(config.motion_root, 0o500)

    config.collection_root.mkdir(mode=0o700)
    (config.collection_root / "cohorts").mkdir(mode=0o700)
    (config.collection_root / "runtime").mkdir(mode=0o700)
    cases, cohorts = build_case_assignments(sources)
    cohort_documents = []
    for cohort in cohorts:
        cohort_path = _cohort_path(
            config, cohort["cohort_index"], cohort["paired_seed"]
        )
        contract = artifacts.create_cohort(
            cohort_path,
            campaign_id=CAMPAIGN_ID,
            macro_round=config.macro_round,
            cohort_index=cohort["cohort_index"],
            purpose="validation",
            expected_ranks=cohort["expected_ranks"],
            expected_global_env_ids=cohort["expected_global_env_ids"],
            expected_transitions=cohort["expected_transitions"],
            view_contract=VIEW_CONTRACT,
            action_contract={"dtype": "float32", "shape": [config.replay_action_dim]},
            policy_sha256=config.policy_sha256,
            runtime_policy_sha256=config.runtime_policy_sha256,
            teacher_sha256=config.teacher_sha256,
            beta_sha256=config.beta_sha256,
            config_sha256=config.config_sha256,
            **config.observation_provenance,
        )
        contract_path = cohort_path / artifacts.CONTRACT_FILENAME
        cohort_documents.append({
            **cohort,
            "cohort_path": str(cohort_path),
            "contract_path": str(contract_path),
            "contract_file_sha256": sha256_file(contract_path),
            "contract_semantic_sha256": _canonical_sha(contract),
        })

    assignment = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_generation_id": config.source_generation_id,
        "source_macro_round": config.macro_round,
        "collection_id": config.collection_id,
        "role": "validation_collect",
        "purpose": "validation",
        "beta": 1.0,
        "beta_sha256": config.beta_sha256,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "environments_per_rank": ENVIRONMENTS_PER_RANK,
        "environments_per_shard": ENVIRONMENTS_PER_SHARD,
        "case_count": CASE_COUNT,
        "transition_count": TRANSITION_COUNT,
        "environment_semantics": {
            "registry_scope": "initial_environment_reset_stream",
            "trajectory_shape": "rectangular_steps_by_environment",
            "automatic_resets": "included_after_initial_case",
            "exactly_one_episode_claimed": False,
            "mse_population": "all_512_labeled_transitions_per_environment",
        },
        "validation_partition": {
            "path": str(VALIDATION_PARTITION),
            "file_sha256": VALIDATION_PARTITION_SHA256,
            "split_sha256": SPLIT_SHA256,
            "group_ids": [row["group_id"] for row in sources.groups],
        },
        "validation_registry": {
            "path": str(VALIDATION_REGISTRY),
            "file_sha256": VALIDATION_REGISTRY_SHA256,
            "panel_id": sources.registry["panel_id"],
            "paired_seeds": list(PAIRED_SEEDS),
        },
        "motion_materialization": {
            "root": str(config.motion_root),
            "source_root_masked_at_runtime": str(config.source_motion_root),
            "inventory": source_inventory,
            "inventory_sha256": _canonical_sha(source_inventory),
        },
        "policy": {
            "checkpoint_path": str(config.policy_checkpoint),
            "checkpoint_sha256": config.policy_sha256,
            "runtime_policy_sha256": config.runtime_policy_sha256,
            "policy_spec_path": str(config.policy_spec),
            "policy_spec_file_sha256": config.policy_spec_sha256,
        },
        "teacher": {
            "checkpoint_path": str(config.teacher_checkpoint),
            "checkpoint_sha256": config.teacher_sha256,
        },
        "runtime": {
            "config_path": str(config.config_path),
            "config_sha256": config.config_sha256,
            **config.observation_provenance,
            "manifest_paths": dict(config.runtime_environment),
            "inventory_verification": runtime_inventory_report,
        },
        "isolation": {
            "private_mount_namespace": True,
            "no_new_privileges": True,
            "dropped_capabilities": ["dac_override", "dac_read_search", "sys_admin"],
            "masked_roots": [
                str(CANONICAL_TEST_ROOT),
                str(config.source_motion_root),
                *[
                    str(
                        CANONICAL_CAMPAIGN_ROOT
                        / train_round_relative_root(generation)
                        / "replay"
                    )
                    for generation in GENERATION_IDS
                ],
            ],
        },
        "cohorts": cohort_documents,
        "cases": cases,
    }
    assignment_sha = publish_create_only(config.assignment_path, assignment, mode=0o444)
    hardware_document = {
        "schema": HARDWARE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "collection_id": config.collection_id,
        "expected_gpu_count": GPU_COUNT,
        "expected_gpu_name": "NVIDIA H200",
        "expected_compute_capability": "9.0",
        "gpus": [dict(row) for row in hardware],
    }
    hardware_sha = publish_create_only(config.hardware_path, hardware_document, mode=0o444)
    return {
        **assignment,
        "path": str(config.assignment_path),
        "file_sha256": assignment_sha,
        "hardware_path": str(config.hardware_path),
        "hardware_file_sha256": hardware_sha,
    }


def _load_assignment(config: CollectionConfig, expected_sha256: str | None = None) -> dict[str, Any]:
    assignment = _strict_json(
        config.assignment_path, "validation assignment", expected_sha256
    )
    if (
        assignment.get("schema") != SCHEMA
        or assignment.get("campaign_id") != CAMPAIGN_ID
        or assignment.get("source_generation_id") != config.source_generation_id
        or assignment.get("source_macro_round") != config.macro_round
        or assignment.get("collection_id") != config.collection_id
        or assignment.get("role") != "validation_collect"
        or assignment.get("purpose") != "validation"
        or assignment.get("beta") != 1.0
        or assignment.get("case_count") != CASE_COUNT
        or assignment.get("transition_count") != TRANSITION_COUNT
    ):
        raise NativeValidationError("validation assignment identity mismatch")
    return assignment


def rank_environment(
    config: CollectionConfig,
    assignment: Mapping[str, Any],
    *,
    physical_gpu: int,
    base_environment: Mapping[str, str] | None = None,
    assignment_file_sha256: str | None = None,
) -> dict[str, str]:
    """Build one independent actor environment without BC naming leakage."""

    if type(physical_gpu) is not int or not 0 <= physical_gpu < GPU_COUNT:
        raise NativeValidationError("physical GPU must be in [0, 7]")
    matches = [
        (cohort, rank)
        for cohort in assignment["cohorts"]
        for rank in cohort["ranks"]
        if rank["physical_gpu"] == physical_gpu
    ]
    if len(matches) != 1:
        raise NativeValidationError("validation assignment does not map one physical GPU")
    cohort, rank = matches[0]
    physical_inventory = config.observation_provenance["physical_gpu_inventory"]
    physical = physical_inventory[physical_gpu]
    if (
        physical["rank"] != physical_gpu
        or physical["physical_index"] != physical_gpu
    ):
        raise NativeValidationError("physical H200 assignment is not rank ordered")
    values = dict(os.environ if base_environment is None else base_environment)
    for name in (
        _BC_NAMING_ENVIRONMENT
        | _DISTRIBUTED_ENVIRONMENT
        | _UNSAFE_HANDOFF_ENVIRONMENT
    ):
        values.pop(name, None)
    for name in tuple(values):
        if name.startswith(("ROUND_DAGGER_TEST_", "BASH_FUNC_")):
            values.pop(name, None)
    values.update({
        "CUDA_VISIBLE_DEVICES": physical["uuid"],
        "ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON": json.dumps(
            physical, sort_keys=True, separators=(",", ":")
        ),
        "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON": json.dumps(
            physical_inventory, sort_keys=True, separators=(",", ":")
        ),
        "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256": config.observation_provenance[
            "physical_gpu_inventory_sha256"
        ],
        "ROUND_DAGGER_ACTOR_RANK": str(rank["cohort_rank"]),
        "ROUND_DAGGER_ACTOR_WORLD_SIZE": str(RANKS_PER_COHORT),
        "ROUND_DAGGER_ENVS_PER_GPU": str(ENVIRONMENTS_PER_RANK),
        "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": str(STEPS_PER_ENVIRONMENT),
        "ROUND_DAGGER_ENVS_PER_SHARD": str(ENVIRONMENTS_PER_SHARD),
        "ROUND_DAGGER_ACTOR_ROLE": "validation_collect",
        "ROUND_DAGGER_PURPOSE": "validation",
        "ROUND_DAGGER_BETA": "1.0",
        "ROUND_DAGGER_CAMPAIGN_ID": CAMPAIGN_ID,
        "ROUND_DAGGER_MACRO_ROUND": str(config.macro_round),
        "ROUND_DAGGER_COHORT_INDEX": str(cohort["cohort_index"]),
        "ROUND_DAGGER_ROLLOUT_ID": "0",
        "ROUND_DAGGER_COHORT_DIR": cohort["cohort_path"],
        "ROUND_DAGGER_GLOBAL_ENV_OFFSET": str(rank["global_env_offset"]),
        "ROUND_DAGGER_VALIDATION_ROOT": str(config.validation_root),
        "ROUND_DAGGER_VALIDATION_REPLAY_ROOT": str(config.validation_root / "replay"),
        "ROUND_DAGGER_VALIDATION_INPUT_ROOT": str(config.validation_root / "inputs"),
        "ROUND_DAGGER_VALIDATION_TEST_ROOT": str(CANONICAL_TEST_ROOT),
        "ROUND_DAGGER_VALIDATION_CAMPAIGN_ROOT": str(CANONICAL_CAMPAIGN_ROOT),
        "ROUND_DAGGER_VALIDATION_SOURCE_MOTION_ROOT": str(config.source_motion_root),
        "ROUND_DAGGER_VALIDATION_MOTION_ROOT": str(config.motion_root),
        "ROUND_DAGGER_VALIDATION_PARTITION": str(VALIDATION_PARTITION),
        "ROUND_DAGGER_VALIDATION_PARTITION_SHA256": VALIDATION_PARTITION_SHA256,
        "ROUND_DAGGER_VALIDATION_REGISTRY": str(VALIDATION_REGISTRY),
        "ROUND_DAGGER_VALIDATION_REGISTRY_SHA256": VALIDATION_REGISTRY_SHA256,
        "ROUND_DAGGER_SPLIT_SHA256": SPLIT_SHA256,
        "ROUND_DAGGER_VALIDATION_GROUP_IDS_JSON": json.dumps(
            assignment["validation_partition"]["group_ids"], separators=(",", ":")
        ),
        "ROUND_DAGGER_VALIDATION_ASSIGNMENT": str(config.assignment_path),
        "ROUND_DAGGER_VALIDATION_ASSIGNMENT_SHA256": (
            sha256_file(config.assignment_path)
            if assignment_file_sha256 is None
            else _sha(assignment_file_sha256, "validation assignment")
        ),
        "ROUND_DAGGER_VALIDATION_COLLECTION_ID": config.collection_id,
        "ROUND_DAGGER_VALIDATION_SOURCE_GENERATION_ID": config.source_generation_id,
        "ROUND_DAGGER_VALIDATION_SOURCE_MACRO_ROUND": str(config.macro_round),
        "ROUND_DAGGER_VALIDATION_PHYSICAL_GPU": str(physical_gpu),
        "ROUND_DAGGER_VALIDATION_PAIRED_SEED": str(cohort["paired_seed"]),
        "ROUND_DAGGER_VALIDATION_HYDRA_BASE_SEED": str(rank["hydra_base_seed"]),
        "ROUND_DAGGER_EFFECTIVE_SIMULATOR_SEED": str(rank["effective_simulator_seed"]),
        "ROUND_DAGGER_VALIDATION_MOTION_ASSIGNMENT_OFFSET": str(
            rank["motion_assignment_offset"]
        ),
        "ROUND_DAGGER_VALIDATION_RUN_DIR": str(
            config.collection_root / "runtime" / f"gpu-{physical_gpu:02d}"
        ),
        "ROUND_DAGGER_POLICY_CHECKPOINT": str(config.policy_checkpoint),
        "ROUND_DAGGER_POLICY_SHA256": config.policy_sha256,
        "ROUND_DAGGER_POLICY_SPEC": str(config.policy_spec),
        "ROUND_DAGGER_POLICY_SPEC_SHA256": config.policy_spec_sha256,
        "ROUND_DAGGER_RUNTIME_POLICY_SHA256": config.runtime_policy_sha256,
        "ROUND_DAGGER_TEACHER_CHECKPOINT": str(config.teacher_checkpoint),
        "ROUND_DAGGER_TEACHER_SHA256": config.teacher_sha256,
        "ROUND_DAGGER_BETA_SHA256": config.beta_sha256,
        "ROUND_DAGGER_CONFIG_SHA256": config.config_sha256,
        "GR00T_ROOT": str(config.gr00t_root),
        "GR00T_PYTHON": str(config.gr00t_python),
    })
    values.update(config.runtime_environment)
    values.pop("LD_LIBRARY_PATH", None)
    # The rank script is entered through /usr/bin/env, so this fixed path and
    # removal of exported Bash functions must happen at the fleet handoff,
    # before the script can execute even its first external utility.
    values["PATH"] = SAFE_EXECUTABLE_PATH
    for field, value in config.observation_provenance.items():
        # These two values are serialized and bound above because the inventory
        # is structured data rather than a scalar provenance field.
        if field in {
            "physical_gpu_inventory",
            "physical_gpu_inventory_sha256",
        }:
            continue
        env_name = {
            "observation_source": "ROUND_DAGGER_OBSERVATION_SOURCE",
            "observation_runtime_sha256": "ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256",
            "isaac_runtime_sha256": "ROUND_DAGGER_ISAAC_RUNTIME_SHA256",
            "actor_python_freeze_sha256": "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256",
            "actor_source_inventory_sha256": "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256",
            "gr00t_revision": "ROUND_DAGGER_GR00T_REVISION",
            "gr00t_source_tree_sha256": "ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256",
            "gpu_name": "ROUND_DAGGER_GPU_NAME",
            "compute_capability": "ROUND_DAGGER_COMPUTE_CAPABILITY",
            "gpu_count": "ROUND_DAGGER_GPU_COUNT",
        }[field]
        values[env_name] = str(value)
    return values


def rank_preflight(environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Recompute one rank's assignment immediately before entering Isaac."""

    values = os.environ if environment is None else environment
    forbidden = sorted(name for name in _BC_NAMING_ENVIRONMENT if values.get(name))
    if forbidden:
        raise NativeValidationError(
            "validation rank inherited BC generation naming: " + ", ".join(forbidden)
        )
    config = CollectionConfig.from_environment(values)
    expected_assignment_sha = _sha(
        values.get("ROUND_DAGGER_VALIDATION_ASSIGNMENT_SHA256"),
        "validation assignment",
    )
    assignment = _load_assignment(config, expected_assignment_sha)
    physical_raw = values.get("ROUND_DAGGER_VALIDATION_PHYSICAL_GPU", "")
    if not physical_raw.isdigit():
        raise NativeValidationError("validation physical GPU is malformed")
    physical_gpu = int(physical_raw)
    expected = rank_environment(
        config, assignment, physical_gpu=physical_gpu, base_environment={}
    )
    compared = {
        key: expected[key]
        for key in expected
        if key.startswith("ROUND_DAGGER_") or key == "CUDA_VISIBLE_DEVICES"
    }
    mismatch = sorted(key for key, value in compared.items() if values.get(key) != value)
    if mismatch:
        raise NativeValidationError(
            "validation rank environment differs from assignment: " + ", ".join(mismatch)
        )
    return {
        "status": "authorized",
        "collection_id": config.collection_id,
        "physical_gpu": physical_gpu,
        "cohort_index": int(values["ROUND_DAGGER_COHORT_INDEX"]),
        "cohort_rank": int(values["ROUND_DAGGER_ACTOR_RANK"]),
        "paired_seed": int(values["ROUND_DAGGER_VALIDATION_PAIRED_SEED"]),
        "effective_simulator_seed": int(values["ROUND_DAGGER_EFFECTIVE_SIMULATOR_SEED"]),
    }


def _verify_labeled_arrays(
    assignment: Mapping[str, Any], cohort_path: Path
) -> dict[str, Any]:
    expected_cases = {
        row["global_env_id"]: row for row in assignment["cases"]
        if row["cohort_index"] == int(cohort_path.name.split("-", 2)[1])
    }
    seen: set[int] = set()

    def visitor(_shard: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
        forbidden = sorted(name for name in arrays if "teacher_obs" in name)
        if forbidden:
            raise NativeValidationError("validation replay exposes teacher observations")
        required = {
            "policy_rgb", "expert_actions", "student_actions", "behavior_actions",
            "expert_mask", "dones", "global_env_ids", "episode_ids",
            "episode_steps", "obs__actor_obs",
        }
        if not required <= set(arrays):
            raise NativeValidationError("validation shard lacks native labeled replay arrays")
        expert = arrays["expert_actions"]
        behavior = arrays["behavior_actions"]
        mask = arrays["expert_mask"]
        global_ids = arrays["global_env_ids"]
        episode_ids = arrays["episode_ids"]
        episode_steps = arrays["episode_steps"]
        if (
            expert.shape[0] != STEPS_PER_ENVIRONMENT
            or behavior.shape != expert.shape
            or mask.shape[:2] != expert.shape[:2]
            or global_ids.shape[:2] != expert.shape[:2]
            or episode_ids.shape[:2] != expert.shape[:2]
            or episode_steps.shape[:2] != expert.shape[:2]
            or not np.all(mask)
            or not np.array_equal(behavior, expert)
            or not np.isfinite(expert).all()
            or not np.isfinite(arrays["obs__actor_obs"]).all()
        ):
            raise NativeValidationError("validation replay is not exact beta=1 native supervision")
        if global_ids.ndim != 2 or not np.all(global_ids == global_ids[0:1]):
            raise NativeValidationError("validation global environment IDs are unstable")
        if not np.all(episode_ids[0] == 0) or not np.all(episode_steps[0] == 0):
            raise NativeValidationError("validation shard does not begin at the bound initial cases")
        for raw in global_ids[0].tolist():
            global_id = int(raw)
            if global_id not in expected_cases or global_id in seen:
                raise NativeValidationError("validation shard case coverage overlaps or escapes")
            seen.add(global_id)

    try:
        report = artifacts.visit_verified_cohort_shards(cohort_path, visitor)
    except NativeValidationError:
        raise
    except Exception as exc:
        raise NativeValidationError(f"validation cohort verification failed: {exc}") from exc
    if seen != set(expected_cases):
        raise NativeValidationError("validation cohort does not cover all assigned initial cases")
    return report


def finalize_collection(
    config: CollectionConfig,
    *,
    rank_logs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal all cohorts, build the frozen MSE snapshot, and publish a root seal."""

    assignment_sha = sha256_file(config.assignment_path)
    assignment = _load_assignment(config, assignment_sha)
    hardware = _strict_json(config.hardware_path, "validation hardware")
    if (
        hardware.get("schema") != HARDWARE_SCHEMA
        or hardware.get("collection_id") != config.collection_id
        or len(hardware.get("gpus", [])) != GPU_COUNT
    ):
        raise NativeValidationError("validation hardware binding differs")
    if len(rank_logs) != GPU_COUNT or {
        row.get("physical_gpu") for row in rank_logs
    } != set(range(GPU_COUNT)):
        raise NativeValidationError("validation completion logs do not cover all H200s")
    sealed_logs = []
    for raw in sorted(rank_logs, key=lambda row: row["physical_gpu"]):
        path = _inside(Path(raw["path"]), config.validation_root, "validation rank log")
        path = _regular(path, "validation rank log")
        marker = (
            f"NATIVE_VALIDATION_RANK_OK physical_gpu={raw['physical_gpu']} "
            f"paired_seed={raw['paired_seed']}"
        )
        if marker not in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raise NativeValidationError("validation rank log lacks exact terminal marker")
        os.chmod(path, 0o444)
        sealed_logs.append({**dict(raw), "file_sha256": sha256_file(path)})

    cohort_seals = []
    for cohort in assignment["cohorts"]:
        cohort_path = Path(cohort["cohort_path"])
        try:
            artifacts.seal_cohort(cohort_path)
        except Exception as exc:
            raise NativeValidationError(f"cannot seal validation cohort: {exc}") from exc
        report = _verify_labeled_arrays(assignment, cohort_path)
        if (
            report["purpose"] != "validation"
            or report["expected_ranks"] != [0, 1]
            or report["global_env_ids"] != cohort["expected_global_env_ids"]
            or report["transition_count"] != cohort["expected_transitions"]
        ):
            raise NativeValidationError("sealed validation cohort coverage differs")
        cohort_seals.append({
            "cohort_index": cohort["cohort_index"],
            "paired_seed": cohort["paired_seed"],
            "path": report["manifest_path"],
            "file_sha256": report["cohort_sha256"],
            "logical_id": report["logical_id"],
            "transition_count": report["transition_count"],
        })

    try:
        snapshot = replay.build_validation_snapshot(
            [row["path"] for row in cohort_seals],
            split_sha256=SPLIT_SHA256,
            output=config.snapshot_path,
        )
        loaded = replay.load_validation_snapshot(
            config.snapshot_path,
            expected_file_sha256=snapshot["file_sha256"],
            artifact_root=config.validation_root,
        )
    except Exception as exc:
        raise NativeValidationError(f"cannot publish validation snapshot: {exc}") from exc
    if (
        loaded["purpose"] != "validation"
        or loaded["campaign_id"] != CAMPAIGN_ID
        or loaded["split_sha256"] != SPLIT_SHA256
        or loaded["cohort_count"] != COHORT_COUNT
        or loaded["transition_count"] != TRANSITION_COUNT
    ):
        raise NativeValidationError("validation snapshot identity differs")

    document: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_generation_id": config.source_generation_id,
        "source_macro_round": config.macro_round,
        "collection_id": config.collection_id,
        "role": "validation_collect",
        "purpose": "validation",
        "beta": 1.0,
        "case_count": CASE_COUNT,
        "transition_count": TRANSITION_COUNT,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "exactly_one_episode_claimed": False,
        "automatic_resets_included": True,
        "assignment_path": str(config.assignment_path),
        "assignment_file_sha256": assignment_sha,
        "hardware_path": str(config.hardware_path),
        "hardware_file_sha256": sha256_file(config.hardware_path),
        "validation_partition_path": str(VALIDATION_PARTITION),
        "validation_partition_file_sha256": VALIDATION_PARTITION_SHA256,
        "validation_registry_path": str(VALIDATION_REGISTRY),
        "validation_registry_file_sha256": VALIDATION_REGISTRY_SHA256,
        "motion_inventory_sha256": assignment["motion_materialization"]["inventory_sha256"],
        "cohort_seals": cohort_seals,
        "snapshot_path": str(config.snapshot_path),
        "snapshot_file_sha256": snapshot["file_sha256"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "rank_logs": sealed_logs,
    }
    document["seal_sha256"] = _canonical_sha(document)
    file_sha = publish_create_only(config.seal_path, document, mode=0o444)
    result = verify_collection(config.seal_path)
    if result["seal_file_sha256"] != file_sha:
        raise NativeValidationError("published validation seal digest changed")
    for cohort in assignment["cohorts"]:
        root = Path(cohort["cohort_path"])
        for directory in (root / "shards", root):
            os.chmod(directory, 0o555)
    os.chmod(config.collection_root / "cohorts", 0o555)
    return result


def verify_collection(seal_path: Path) -> dict[str, Any]:
    """Recursively verify the exact heldout validation root and final seal."""

    seal_path = _inside(
        seal_path, CANONICAL_VALIDATION_ROOT, "validation dataset seal"
    )
    seal = _strict_json(seal_path, "validation dataset seal")
    if seal.get("schema") != SEAL_SCHEMA:
        raise NativeValidationError("validation dataset seal schema mismatch")
    expected_semantic = seal.get("seal_sha256")
    body = dict(seal)
    body.pop("seal_sha256", None)
    if expected_semantic != _canonical_sha(body):
        raise NativeValidationError("validation dataset semantic seal mismatch")
    if (
        seal.get("campaign_id") != CAMPAIGN_ID
        or seal.get("purpose") != "validation"
        or seal.get("role") != "validation_collect"
        or seal.get("beta") != 1.0
        or seal.get("case_count") != CASE_COUNT
        or seal.get("transition_count") != TRANSITION_COUNT
        or seal.get("steps_per_environment") != STEPS_PER_ENVIRONMENT
        or seal.get("exactly_one_episode_claimed") is not False
        or seal.get("automatic_resets_included") is not True
        or seal.get("validation_partition_path") != str(VALIDATION_PARTITION)
        or seal.get("validation_partition_file_sha256") != VALIDATION_PARTITION_SHA256
        or seal.get("validation_registry_path") != str(VALIDATION_REGISTRY)
        or seal.get("validation_registry_file_sha256") != VALIDATION_REGISTRY_SHA256
    ):
        raise NativeValidationError("validation dataset seal identity differs")
    collection_id = seal.get("collection_id")
    if (
        not isinstance(collection_id, str)
        or _COLLECTION_ID.fullmatch(collection_id) is None
        or seal_path
        != CANONICAL_VALIDATION_ROOT / "seals" / f"{collection_id}.seal.json"
    ):
        raise NativeValidationError("validation dataset seal path is non-canonical")
    _regular(VALIDATION_PARTITION, "validation partition", expected_sha256=VALIDATION_PARTITION_SHA256)
    _regular(VALIDATION_REGISTRY, "validation registry", expected_sha256=VALIDATION_REGISTRY_SHA256)
    assignment_path = _inside(
        Path(seal["assignment_path"]), CANONICAL_VALIDATION_ROOT, "validation assignment"
    )
    if assignment_path != (
        CANONICAL_VALIDATION_ROOT / "contracts" / f"{collection_id}.assignment.json"
    ):
        raise NativeValidationError("sealed validation assignment path is non-canonical")
    assignment = _strict_json(
        assignment_path,
        "validation assignment",
        expected_sha256=seal["assignment_file_sha256"],
    )
    if (
        assignment.get("schema") != SCHEMA
        or assignment.get("campaign_id") != CAMPAIGN_ID
        or assignment.get("collection_id") != collection_id
        or assignment.get("purpose") != "validation"
        or assignment.get("role") != "validation_collect"
        or assignment.get("beta") != 1.0
        or assignment.get("case_count") != CASE_COUNT
        or assignment.get("transition_count") != TRANSITION_COUNT
    ):
        raise NativeValidationError("sealed validation assignment differs")
    motion = assignment["motion_materialization"]
    motion_root = _directory(Path(motion["root"]), "materialized validation motions")
    if motion_root != CANONICAL_VALIDATION_ROOT / "inputs/motions-v001":
        raise NativeValidationError("materialized motion root is not the exact heldout root")
    if motion.get("inventory_sha256") != seal.get("motion_inventory_sha256") or _canonical_sha(
        motion.get("inventory", [])
    ) != motion.get("inventory_sha256"):
        raise NativeValidationError("validation motion inventory digest mismatch")
    expected_names = []
    for row in motion["inventory"]:
        expected_names.append(row["relative_path"])
        path = _regular(
            motion_root / row["relative_path"],
            "materialized validation motion",
            expected_sha256=row["sha256"],
        )
        if path.stat().st_size != row["size_bytes"]:
            raise NativeValidationError("materialized validation motion size changed")
    if sorted(path.name for path in motion_root.iterdir()) != sorted(expected_names):
        raise NativeValidationError("materialized validation motion root has extra files")
    hardware_path = _inside(
        Path(seal["hardware_path"]), CANONICAL_VALIDATION_ROOT, "validation hardware"
    )
    if hardware_path != (
        CANONICAL_VALIDATION_ROOT / "contracts" / f"{collection_id}.hardware.json"
    ):
        raise NativeValidationError("sealed validation hardware path is non-canonical")
    hardware = _strict_json(
        hardware_path, "validation hardware", seal["hardware_file_sha256"]
    )
    try:
        hardware_inventory = artifacts.validate_physical_gpu_inventory(
            hardware.get("gpus")
        )
    except Exception as exc:
        raise NativeValidationError(f"sealed validation hardware differs: {exc}") from exc
    if (
        hardware.get("schema") != HARDWARE_SCHEMA
        or hardware.get("campaign_id") != CAMPAIGN_ID
        or hardware.get("collection_id") != collection_id
        or hardware.get("expected_gpu_count") != GPU_COUNT
        or hardware.get("expected_gpu_name") != "NVIDIA H200"
        or hardware.get("expected_compute_capability") != "9.0"
        or hardware_inventory
        != assignment.get("runtime", {}).get("physical_gpu_inventory")
    ):
        raise NativeValidationError("sealed validation hardware identity differs")
    cohort_seals = seal.get("cohort_seals", [])
    if (
        not isinstance(cohort_seals, list)
        or len(cohort_seals) != COHORT_COUNT
        or any(not isinstance(row, Mapping) for row in cohort_seals)
        or {row.get("cohort_index") for row in cohort_seals}
        != set(range(COHORT_COUNT))
        or len({row.get("path") for row in cohort_seals}) != COHORT_COUNT
    ):
        raise NativeValidationError("validation seal does not bind four unique cohorts")
    assignment_cohorts = {
        row["cohort_index"]: row for row in assignment.get("cohorts", [])
    }
    if set(assignment_cohorts) != set(range(COHORT_COUNT)):
        raise NativeValidationError("validation assignment cohort coverage differs")
    for cohort in cohort_seals:
        assigned = assignment_cohorts[cohort["cohort_index"]]
        if (
            cohort.get("paired_seed") != assigned.get("paired_seed")
            or cohort.get("path")
            != str(Path(assigned["cohort_path"]) / artifacts.SEAL_FILENAME)
        ):
            raise NativeValidationError("validation cohort seal leaves its assignment")
        path = _inside(
            Path(cohort["path"]), CANONICAL_VALIDATION_ROOT, "validation cohort seal"
        )
        report = artifacts.verify_cohort(path)
        if (
            report["cohort_sha256"] != cohort["file_sha256"]
            or report["logical_id"] != cohort["logical_id"]
            or report["purpose"] != "validation"
            or report["transition_count"] != cohort["transition_count"]
        ):
            raise NativeValidationError("validation cohort seal changed")
    snapshot_path = _inside(
        Path(seal["snapshot_path"]), CANONICAL_VALIDATION_ROOT, "validation snapshot"
    )
    if snapshot_path != (
        CANONICAL_VALIDATION_ROOT / "replay" / f"{collection_id}.snapshot.json"
    ):
        raise NativeValidationError("sealed validation snapshot path is non-canonical")
    loaded = replay.load_validation_snapshot(
        snapshot_path,
        expected_file_sha256=seal["snapshot_file_sha256"],
        artifact_root=CANONICAL_VALIDATION_ROOT,
    )
    if (
        loaded["snapshot_sha256"] != seal["snapshot_sha256"]
        or loaded["purpose"] != "validation"
        or loaded["transition_count"] != TRANSITION_COUNT
        or loaded["cohort_count"] != COHORT_COUNT
    ):
        raise NativeValidationError("validation snapshot changed")
    snapshot_bindings = {
        (row["cohort_index"], row["manifest_sha256"])
        for row in loaded["cohorts"]
    }
    seal_bindings = {
        (row["cohort_index"], row["file_sha256"]) for row in cohort_seals
    }
    if snapshot_bindings != seal_bindings:
        raise NativeValidationError("validation snapshot/cohort seal binding differs")
    rank_logs = seal.get("rank_logs", [])
    if (
        not isinstance(rank_logs, list)
        or len(rank_logs) != GPU_COUNT
        or any(not isinstance(row, Mapping) for row in rank_logs)
        or {row.get("physical_gpu") for row in rank_logs} != set(range(GPU_COUNT))
    ):
        raise NativeValidationError("validation seal lacks eight rank logs")
    for row in rank_logs:
        path = _inside(Path(row["path"]), CANONICAL_VALIDATION_ROOT, "rank log")
        if path != (
            CANONICAL_VALIDATION_ROOT
            / "logs"
            / collection_id
            / f"gpu-{row['physical_gpu']:02d}.log"
        ):
            raise NativeValidationError("validation rank log path is non-canonical")
        _regular(path, "rank log", expected_sha256=row["file_sha256"])
    return {
        "status": "verified",
        "seal_path": str(seal_path),
        "seal_file_sha256": sha256_file(seal_path),
        "seal_sha256": expected_semantic,
        "snapshot_path": str(snapshot_path),
        "snapshot_file_sha256": loaded["file_sha256"],
        "snapshot_sha256": loaded["snapshot_sha256"],
        "case_count": CASE_COUNT,
        "transition_count": TRANSITION_COUNT,
    }


def dry_run_plan(config: CollectionConfig, hardware: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _ensure_new_targets(config)
    sources = load_validation_sources()
    validate_source_motions(config, sources)
    if list(hardware) != config.observation_provenance["physical_gpu_inventory"]:
        raise NativeValidationError("dry-run H200 inventory differs from runtime provenance")
    runtime_inventory_report = verify_runtime_inventories(config)
    cases, cohorts = build_case_assignments(sources)
    launches = []
    placeholder = {
        "cohorts": [
            {
                **cohort,
                "cohort_path": str(_cohort_path(config, cohort["cohort_index"], cohort["paired_seed"])),
            }
            for cohort in cohorts
        ],
        "validation_partition": {
            "group_ids": [row["group_id"] for row in sources.groups]
        },
    }
    for physical_gpu in range(GPU_COUNT):
        env = rank_environment(
            config,
            placeholder,
            physical_gpu=physical_gpu,
            base_environment={},
            assignment_file_sha256="0" * 64,
        )
        launches.append({
            "physical_gpu": physical_gpu,
            "cohort_index": int(env["ROUND_DAGGER_COHORT_INDEX"]),
            "cohort_rank": int(env["ROUND_DAGGER_ACTOR_RANK"]),
            "paired_seed": int(env["ROUND_DAGGER_VALIDATION_PAIRED_SEED"]),
            "environments": ENVIRONMENTS_PER_RANK,
            "steps": STEPS_PER_ENVIRONMENT,
            "cohort_dir": env["ROUND_DAGGER_COHORT_DIR"],
        })
    return {
        "status": "dry-run",
        "mutated": False,
        "collection_id": config.collection_id,
        "hardware": [dict(row) for row in hardware],
        "case_count": len(cases),
        "transition_count": TRANSITION_COUNT,
        "runtime_inventory_verification": runtime_inventory_report,
        "launches": launches,
        "masked_test_root": str(CANONICAL_TEST_ROOT),
        "masked_source_motion_root": str(config.source_motion_root),
        "output_root": str(config.validation_root),
    }


def run_fleet(
    config: CollectionConfig,
    *,
    rank_command: Sequence[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Launch all eight ranks concurrently, then seal only after all succeed."""

    if not rank_command:
        raise NativeValidationError("validation rank command is empty")
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise NativeValidationError("validation timeout must be positive")
    hardware = query_h200_inventory()
    assignment = prepare_collection(config, hardware)
    processes: list[subprocess.Popen[bytes]] = []
    handles = []
    logs: list[dict[str, Any]] = []
    log_root = config.validation_root / "logs" / config.collection_id
    log_root.mkdir(mode=0o700)
    started = time.monotonic()
    try:
        for physical_gpu in range(GPU_COUNT):
            environment = rank_environment(
                config, assignment, physical_gpu=physical_gpu
            )
            run_dir = Path(environment["ROUND_DAGGER_VALIDATION_RUN_DIR"])
            run_dir.mkdir(mode=0o700)
            log_path = log_root / f"gpu-{physical_gpu:02d}.log"
            handle = log_path.open("xb")
            process = subprocess.Popen(
                list(rank_command),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            handles.append(handle)
            processes.append(process)
            logs.append({
                "physical_gpu": physical_gpu,
                "paired_seed": int(environment["ROUND_DAGGER_VALIDATION_PAIRED_SEED"]),
                "cohort_index": int(environment["ROUND_DAGGER_COHORT_INDEX"]),
                "cohort_rank": int(environment["ROUND_DAGGER_ACTOR_RANK"]),
                "path": str(log_path),
            })
        while any(process.poll() is None for process in processes):
            if time.monotonic() - started > timeout_seconds:
                raise NativeValidationError("validation fleet exceeded its wall-clock timeout")
            time.sleep(1.0)
        exit_codes = [process.returncode for process in processes]
        if exit_codes != [0] * GPU_COUNT:
            raise NativeValidationError(f"validation ranks failed: {exit_codes}")
    except BaseException:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and any(
            process.poll() is None for process in processes
        ):
            time.sleep(0.1)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        raise
    finally:
        for handle in handles:
            handle.close()
    return finalize_collection(config, rank_logs=logs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("dry-run")
    launch = commands.add_parser("launch")
    launch.add_argument(
        "--rank-command",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "scripts/round_dagger/run_native_h200_validation_rank.sh",
    )
    launch.add_argument("--timeout-seconds", type=float, default=3600.0)
    verify = commands.add_parser("verify")
    verify.add_argument("--seal", type=Path, required=True)
    commands.add_parser("rank-preflight")
    verify_runtime = commands.add_parser("verify-runtime")
    verify_runtime.add_argument(
        "--critical-only",
        action="store_true",
        help="rehash only live entry points after source/QROOT are read-only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "verify":
        report = verify_collection(arguments.seal)
    elif arguments.command == "rank-preflight":
        report = rank_preflight()
    elif arguments.command == "verify-runtime":
        report = verify_runtime_inventories(
            CollectionConfig.from_environment(),
            full_runtime_trees=not arguments.critical_only,
        )
    else:
        config = CollectionConfig.from_environment()
        if arguments.command == "dry-run":
            report = dry_run_plan(config, query_h200_inventory())
        else:
            rank_command = _regular(arguments.rank_command, "validation rank launcher")
            if not os.access(rank_command, os.X_OK):
                raise NativeValidationError("validation rank launcher must be executable")
            report = run_fleet(
                config,
                rank_command=[str(rank_command)],
                timeout_seconds=arguments.timeout_seconds,
            )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NativeValidationError as exc:
        print(f"native H200 validation failed: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
