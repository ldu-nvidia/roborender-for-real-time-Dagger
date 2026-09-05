#!/usr/bin/env python3
"""Deterministic, leakage-safe split of raw BC conditioning artifacts.

The collection actor writes several environments into each physical shard.  A
physical shard therefore cannot be assigned directly to train/validation/test:
doing so would make the split depend on rank and shard boundaries.  This module
uses one complete ``global_env_id`` stream as the indivisible group, including
every episode and every frame in that stream, and republishes each group into a
split-specific raw-conditioning artifact.

Raw conditioning is still not learner replay.  The later RoboRender projection
must consume the appropriate sealed split, and a training consumer is admitted
only to the train seal.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from round_dagger.artifacts import (
    ArtifactError,
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
    validate_physical_gpu_inventory,
)
from round_dagger.conditioning_artifacts import (
    ARTIFACT_KIND,
    ConditioningArtifactError,
    VIEWS,
    load_raw_conditioning_artifact,
    publish_conditioning_shard,
)

CAMPAIGN_ID = "dagger-redo-v001"
GENERATION_ID = "gen-000-bc"
CANONICAL_CAMPAIGN_ROOT = Path(
    "/workspace/project/roborender/data"
)
SOURCE_RELATIVE_ROOT = Path("BC/collection/conditioning/waves")
FROZEN_BC_SOURCE_RELATIVE_ROOT = Path(
    f"generations/{GENERATION_ID}/conditioning/waves"
)
SPLITS = ("train", "validation", "test")
SPLIT_SEED = 42
DEFAULT_WEIGHTS = {"train": 80, "validation": 10, "test": 10}
PROTOCOL_RELATIVE_PATH = Path(
    "configs/round_dagger/native_h200_experiment_protocol.json"
)
# This value is part of the already-sealed BC manifest bytes. Operational
# verification resolves the plan through the canonical data root below rather
# than treating this historical relative string as a live filesystem path.
SEALED_BC_PLAN_PATH_FIELD = "../../../contracts/bc_collection_split.v1.json"

PLAN_SCHEMA = "fast_roborender.round_dagger.bc_conditioning_split_plan.v1"
MANIFEST_SCHEMA = "fast_roborender.round_dagger.bc_conditioning_split_manifest.v1"
SEAL_SCHEMA = "fast_roborender.round_dagger.bc_conditioning_split_seal.v1"
STRATEGY = "deterministic_stratified_global_env_sha256_v1"
SATURATION_SCHEMA = "fast_roborender.round_dagger.native_actor_saturation.v1"
COHORT_SCHEMA = "fast_roborender.round_dagger.isaac_conditioning_cohort.v1"
COHORT_SEAL_SCHEMA = "fast_roborender.round_dagger.isaac_conditioning_cohort_seal.v1"
PRE_RENDER_AUDIT_SCHEMA = "fast_roborender.round_dagger.pre_render_audit.v1"
EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV = 512
EXPECTED_BC_ACTOR_WORLD_SIZE = 1
EXPECTED_BC_ENVIRONMENTS_PER_ACTOR = 128
# The saturation manifest retains its established ``environments_per_gpu``
# field; with one GPU per actor process it is exactly the per-actor count.
EXPECTED_BC_ENVIRONMENTS_PER_GPU = EXPECTED_BC_ENVIRONMENTS_PER_ACTOR
EXPECTED_BC_ENVIRONMENTS_PER_WAVE = 128
EXPECTED_BC_WAVE_COUNT = 2
EXPECTED_BC_TRANSITIONS_PER_WAVE = (
    EXPECTED_BC_ENVIRONMENTS_PER_WAVE
    * EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV
)
EXPECTED_BC_ENVIRONMENTS_PER_SHARD = 4
EXPECTED_BC_SHARDS_PER_WAVE = (
    EXPECTED_BC_ENVIRONMENTS_PER_WAVE // EXPECTED_BC_ENVIRONMENTS_PER_SHARD
)
EXPECTED_BC_TOTAL_SHARDS = EXPECTED_BC_WAVE_COUNT * EXPECTED_BC_SHARDS_PER_WAVE
EXPECTED_BC_ADMITTED_PHYSICAL_GPU_INDICES = [0]
EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS = 256
EXPECTED_BC_NOMINAL_TRANSITIONS = (
    EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS * EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV
)
EXPECTED_BC_GROUP_COUNTS = {"train": 178, "validation": 39, "test": 39}
EXPECTED_BC_PROTOCOL_SHA256 = (
    "e6a1b47f8e803c45751890799b00d005566a98e802603785a7583a3b08842b24"
)

POSTPROCESS_PROVENANCE_SCHEMA = (
    "fast_roborender.round_dagger.bc_postprocess_provenance.v1"
)
_POSTPROCESS_SOURCE_RELATIVE_PATHS = (
    "round_dagger/artifacts.py",
    "round_dagger/conditioning_artifacts.py",
    "round_dagger/conditioning_split.py",
    "scripts/round_dagger/split_native_h200_bc_collection.py",
    "scripts/round_dagger/finalize_native_h200_bc_collection.py",
)

_ANCHOR_IDS = {"anchor_global_env_ids", "anchor_episode_ids"}
_ANCHOR_VIEWS = {f"rgb_anchor__{view}" for view in VIEWS}
_IMAGE_IDS = {
    "image_global_env_ids",
    "image_episode_ids",
    "image_episode_steps",
    "image_reference_source_steps",
    "image_reference_indices",
}
_IMAGE_VIEWS = {
    *(f"depth_ctrl__{view}" for view in VIEWS),
    *(f"rgb_robot__{view}" for view in VIEWS),
}
_PER_ENV_METADATA = {
    "task_ids": "task",
    "scenario_ids": "scenario",
    "motion_ids": "motion",
    "group_ids": "group",
}


class ConditioningSplitError(RuntimeError):
    """A BC split is incomplete, overlapping, mutable, or inconsistent."""


def _regular_file_descriptor(path: Path, label: str) -> dict[str, Any]:
    _require_safe_tree(path)
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ConditioningSplitError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ConditioningSplitError(f"{label} must be a regular non-symlink file")
    try:
        digest = sha256_file(path)
    except ArtifactError as exc:
        raise ConditioningSplitError(f"cannot hash {label}: {exc}") from exc
    return {
        "path": path.name,
        "size_bytes": info.st_size,
        "sha256": digest,
    }


def build_postprocess_provenance(
    source_root: str | os.PathLike[str],
    protocol_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Hash the exact split/finalize implementation and protocol inputs."""

    root = _absolute(source_root)
    protocol = _absolute(protocol_path)
    expected_protocol = root / PROTOCOL_RELATIVE_PATH
    if protocol != expected_protocol:
        raise ConditioningSplitError(
            f"postprocess protocol must be exactly {expected_protocol}"
        )
    sources: list[dict[str, Any]] = []
    for relative in _POSTPROCESS_SOURCE_RELATIVE_PATHS:
        path = root / relative
        descriptor = _regular_file_descriptor(path, f"postprocess source {relative}")
        descriptor["path"] = relative
        sources.append(descriptor)
    protocol_descriptor = _regular_file_descriptor(protocol, "postprocess protocol")
    protocol_descriptor["path"] = PROTOCOL_RELATIVE_PATH.as_posix()
    bundle = {"sources": sources, "protocol": protocol_descriptor}
    return {
        "schema": POSTPROCESS_PROVENANCE_SCHEMA,
        "source_root": str(root),
        **bundle,
        "bundle_sha256": canonical_json_sha256(bundle),
    }


def _validate_postprocess_provenance(value: Any) -> dict[str, Any]:
    keys = {"schema", "source_root", "sources", "protocol", "bundle_sha256"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConditioningSplitError("postprocess provenance fields differ")
    if (
        value["schema"] != POSTPROCESS_PROVENANCE_SCHEMA
        or not isinstance(value["source_root"], str)
        or not Path(value["source_root"]).is_absolute()
        or not isinstance(value["sources"], list)
        or len(value["sources"]) != len(_POSTPROCESS_SOURCE_RELATIVE_PATHS)
    ):
        raise ConditioningSplitError("postprocess provenance identity differs")

    expected_paths = list(_POSTPROCESS_SOURCE_RELATIVE_PATHS)
    observed_paths: list[str] = []
    for descriptor in value["sources"]:
        _validate_file_descriptor(descriptor, "postprocess source")
        observed_paths.append(descriptor["path"])
    _validate_file_descriptor(value["protocol"], "postprocess protocol")
    if (
        observed_paths != expected_paths
        or value["protocol"]["path"] != PROTOCOL_RELATIVE_PATH.as_posix()
    ):
        raise ConditioningSplitError("postprocess provenance file inventory differs")
    bundle = {"sources": value["sources"], "protocol": value["protocol"]}
    if value["bundle_sha256"] != canonical_json_sha256(bundle):
        raise ConditioningSplitError("postprocess provenance bundle SHA-256 differs")
    return dict(value)


def _validate_file_descriptor(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ConditioningSplitError(f"{label} descriptor fields differ")
    path = value["path"]
    digest = value["sha256"]
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(path).parts)
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ConditioningSplitError(f"{label} descriptor is malformed")


@contextmanager
def _generation_lock(generation_root: str | os.PathLike[str]):
    """Hold the same directory-inode flock used by the actor controller."""

    generation = _absolute(generation_root)
    _require_safe_tree(generation)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(generation, flags)
    except OSError as exc:
        raise ConditioningSplitError(
            f"cannot open generation lock target: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        current = os.stat(generation, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ConditioningSplitError("generation lock target changed identity")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConditioningSplitError(
                "another collection or postprocess controller is active"
            ) from exc
        current = os.stat(generation, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (
            current.st_dev,
            current.st_ino,
        ):
            raise ConditioningSplitError(
                "generation lock target changed identity while acquiring lock"
            )
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def generation_lock(generation_root: str | os.PathLike[str]):
    with _generation_lock(generation_root):
        yield


@contextmanager
def canonical_generation_lock():
    generation = CANONICAL_CAMPAIGN_ROOT / "BC/collection"
    with generation_lock(generation):
        yield


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _resolve_sealed_cohort_path(
    source_root: Path,
    value: Any,
) -> tuple[Path, Path, Path]:
    """Resolve a sealed cohort after the campaign directory has been moved.

    Collection-time manifests intentionally retain their original absolute
    paths as immutable provenance.  Operational verification binds those
    references to the canonical local cohort by their strict semantic suffix;
    it never follows or requires the historical prefix.
    """

    if not isinstance(value, (str, os.PathLike)):
        raise ConditioningSplitError(
            "saturation cohort escapes conditioning waves"
        )
    reported_value = Path(os.path.expanduser(os.fspath(value)))
    if not reported_value.is_absolute() or ".." in reported_value.parts:
        raise ConditioningSplitError(
            "saturation cohort escapes conditioning waves"
        )
    reported = _absolute(reported_value)
    name = reported.name
    if (
        len(name) != len("wave-000")
        or not name.startswith("wave-")
        or not name.removeprefix("wave-").isdigit()
        or tuple(reported.parts[-3:]) != ("conditioning", "waves", name)
    ):
        raise ConditioningSplitError(
            "saturation cohort escapes conditioning waves"
        )
    canonical_source = _absolute(source_root)
    canonical = canonical_source / name
    if reported == canonical:
        identity_root = SOURCE_RELATIVE_ROOT
    elif (
        canonical_source == CANONICAL_CAMPAIGN_ROOT / SOURCE_RELATIVE_ROOT
        and tuple(reported.parts[-5:])
        == tuple((FROZEN_BC_SOURCE_RELATIVE_ROOT / name).parts)
    ):
        identity_root = FROZEN_BC_SOURCE_RELATIVE_ROOT
    else:
        raise ConditioningSplitError(
            "saturation cohort escapes conditioning waves"
        )
    return reported, canonical, identity_root / name


def _require_safe_tree(path: Path, *, allow_missing_leaf: bool = False) -> None:
    target = _absolute(path)
    current = Path(target.anchor)
    parts = target.parts[1:] if target.anchor else target.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf or index < len(parts):
                continue
            raise
        if stat.S_ISLNK(info.st_mode):
            raise ConditioningSplitError(f"symbolic links are forbidden: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ConditioningSplitError(f"path ancestor is not a directory: {current}")


def _relative(path: Path, root: Path, label: str) -> str:
    try:
        value = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ConditioningSplitError(f"{label} escapes the campaign root") from exc
    if not value or value.startswith("../"):
        raise ConditioningSplitError(f"{label} is not a safe relative path")
    return value


def _load_weights(protocol_path: Path | None) -> tuple[dict[str, int], str]:
    if protocol_path is None:
        return dict(DEFAULT_WEIGHTS), "default_80_10_10"
    try:
        document = load_strict_json(protocol_path)
        group_split = document["group_split"]
        weights = {split: group_split[f"{split}_count"] for split in SPLITS}
    except (ArtifactError, KeyError, TypeError) as exc:
        raise ConditioningSplitError(
            f"cannot load split ratio from protocol: {exc}"
        ) from exc
    if any(type(value) is not int or value <= 0 for value in weights.values()):
        raise ConditioningSplitError("protocol split weights must all be positive")
    return weights, "native_h200_experiment_protocol.group_split"


def _require_exact_protocol(protocol_path: Path) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[1]
    expected = source_root / PROTOCOL_RELATIVE_PATH
    if protocol_path != expected:
        raise ConditioningSplitError(f"BC split protocol must be exactly {expected}")
    descriptor = _regular_file_descriptor(protocol_path, "BC split protocol")
    descriptor["path"] = PROTOCOL_RELATIVE_PATH.as_posix()
    if descriptor["sha256"] != EXPECTED_BC_PROTOCOL_SHA256:
        raise ConditioningSplitError("BC split protocol SHA-256 differs")
    return descriptor


def _target_counts(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    if total < len(SPLITS):
        raise ConditioningSplitError("at least three environment groups are required")
    denominator = sum(weights.values())
    result = {split: total * weights[split] // denominator for split in SPLITS}
    remainder = total - sum(result.values())
    order = sorted(
        SPLITS,
        key=lambda split: (
            -(total * weights[split] % denominator),
            SPLITS.index(split),
        ),
    )
    for split in order[:remainder]:
        result[split] += 1
    if any(result[split] == 0 for split in SPLITS):
        raise ConditioningSplitError(
            "collection is too small for three nonempty splits"
        )
    return result


def _metadata_stratum(
    metadata: Mapping[str, Any], env_index: int, num_envs: int
) -> dict[str, str]:
    """Use only metadata that is unambiguously aligned with environment axes."""

    result: dict[str, str] = {}
    explicit = metadata.get("strata_by_env_index")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        if len(explicit) == num_envs and isinstance(explicit[env_index], Mapping):
            for key, value in sorted(explicit[env_index].items()):
                if isinstance(key, str) and isinstance(value, (str, int, bool)):
                    result[key] = str(value)
    for source_key, target_key in _PER_ENV_METADATA.items():
        values = metadata.get(source_key)
        if (
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
            and len(values) == num_envs
            and isinstance(values[env_index], (str, int, bool))
        ):
            result[target_key] = str(values[env_index])
    return result


def _stratum_key(strata: Mapping[str, str]) -> str:
    if not strata:
        return "__metadata_unavailable__"
    return json.dumps(
        dict(sorted(strata.items())), sort_keys=True, separators=(",", ":")
    )


def _immutable_regular(
    path: Path,
    label: str,
    *,
    exact_mode: int | None = None,
) -> tuple[int, str]:
    _require_safe_tree(path)
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ConditioningSplitError(f"cannot inspect {label}: {exc}") from exc
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISREG(info.st_mode)
        or mode & 0o222
        or (exact_mode is not None and mode != exact_mode)
    ):
        mode_contract = (
            "a read-only regular file"
            if exact_mode is None
            else f"a mode-{exact_mode:04o} read-only regular file"
        )
        raise ConditioningSplitError(f"{label} must be {mode_contract}")
    try:
        return info.st_size, sha256_file(path)
    except ArtifactError as exc:
        raise ConditioningSplitError(f"cannot hash {label}: {exc}") from exc


def _descriptor(
    value: Any, *, parent: Path, expected_path: str, label: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ConditioningSplitError(f"{label} descriptor fields differ")
    if value["path"] != expected_path:
        raise ConditioningSplitError(f"{label} path differs")
    path = parent / expected_path
    size, digest = _immutable_regular(path, label, exact_mode=0o444)
    if value["size_bytes"] != size or value["sha256"] != digest:
        raise ConditioningSplitError(f"{label} descriptor does not match its file")
    return path, dict(value)


def _verify_cohort(
    campaign_root: Path,
    source_root: Path,
    reference: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path], list[dict[str, Any]]]:
    required_reference = {
        "path",
        "seal_path",
        "seal_sha256",
        "transition_count",
        "image_count",
        "purpose",
    }
    if not isinstance(reference, Mapping) or set(reference) != required_reference:
        raise ConditioningSplitError("saturation cohort reference fields differ")
    reported_cohort, cohort, _ = _resolve_sealed_cohort_path(
        source_root, reference["path"]
    )
    if cohort.parent != source_root or not cohort.name.startswith("wave-"):
        raise ConditioningSplitError(
            "saturation cohort is not an immediate wave directory"
        )
    reported_seal = _absolute(reference["seal_path"])
    seal_path = cohort / "seal.json"
    if (
        reported_seal != reported_cohort / "seal.json"
        or reference["purpose"] != "train"
    ):
        raise ConditioningSplitError("saturation cohort seal identity differs")
    _, seal_sha = _immutable_regular(
        seal_path,
        "conditioning cohort seal",
        exact_mode=0o444,
    )
    if reference["seal_sha256"] != seal_sha:
        raise ConditioningSplitError("saturation cohort seal SHA-256 differs")
    try:
        seal = load_strict_json(seal_path)
    except ArtifactError as exc:
        raise ConditioningSplitError(
            f"cannot load conditioning cohort seal: {exc}"
        ) from exc
    seal_keys = {
        "schema",
        "artifact_kind",
        "sealed",
        "learner_replay_ready",
        "campaign_id",
        "generation_id",
        "macro_round",
        "cohort_index",
        "purpose",
        "contract_sha256",
        "split_manifest_sha256",
        "split_sha256",
        "train_group_ids",
        "group_rotation",
        "transition_count",
        "episode_count",
        "image_count",
        "global_env_ids",
        "shard_count",
        "shards",
    }
    if (
        not isinstance(seal, Mapping)
        or set(seal) != seal_keys
        or seal["schema"] != COHORT_SEAL_SCHEMA
        or seal["artifact_kind"] != ARTIFACT_KIND
        or seal["sealed"] is not True
        or seal["learner_replay_ready"] is not False
        or seal["campaign_id"] != CAMPAIGN_ID
        or seal["generation_id"] != GENERATION_ID
        or seal["purpose"] != "train"
        or type(seal["macro_round"]) is not int
        or seal["macro_round"] < 0
        or type(seal["cohort_index"]) is not int
        or seal["cohort_index"] < 0
        or not _is_sha256(seal["contract_sha256"])
        or not _is_sha256(seal["split_manifest_sha256"])
        or not _is_sha256(seal["split_sha256"])
        or type(seal["transition_count"]) is not int
        or seal["transition_count"] <= 0
        or type(seal["episode_count"]) is not int
        or seal["episode_count"] <= 0
        or type(seal["image_count"]) is not int
        or seal["image_count"] <= 0
        or not isinstance(seal["global_env_ids"], list)
        or not seal["global_env_ids"]
        or any(
            type(global_env_id) is not int or global_env_id < 0
            for global_env_id in seal["global_env_ids"]
        )
        or type(seal["shard_count"]) is not int
        or seal["shard_count"] <= 0
        or not isinstance(seal["shards"], list)
        or reference["transition_count"] != seal["transition_count"]
        or reference["image_count"] != seal["image_count"]
    ):
        raise ConditioningSplitError("conditioning cohort seal contract differs")

    contract_path = cohort / "contract.json"
    _, contract_sha = _immutable_regular(
        contract_path,
        "conditioning cohort contract",
        exact_mode=0o444,
    )
    if seal["contract_sha256"] != contract_sha:
        raise ConditioningSplitError("conditioning cohort contract SHA-256 differs")
    try:
        contract = load_strict_json(contract_path)
    except ArtifactError as exc:
        raise ConditioningSplitError(
            f"cannot load conditioning cohort contract: {exc}"
        ) from exc
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema") != COHORT_SCHEMA
        or contract.get("artifact_kind") != ARTIFACT_KIND
        or contract.get("status") != "prepared"
        or contract.get("learner_replay_ready") is not False
        or contract.get("campaign_id") != CAMPAIGN_ID
        or contract.get("generation_id") != GENERATION_ID
        or contract.get("purpose") != "train"
        or contract.get("macro_round") != seal["macro_round"]
        or contract.get("cohort_index") != seal["cohort_index"]
        or contract.get("expected_transitions") != seal["transition_count"]
        or contract.get("expected_global_env_ids") != seal["global_env_ids"]
        or contract.get("split_manifest_sha256")
        != seal["split_manifest_sha256"]
        or contract.get("split_sha256") != seal["split_sha256"]
        or contract.get("train_group_ids") != seal["train_group_ids"]
        or contract.get("group_rotation") != seal["group_rotation"]
    ):
        raise ConditioningSplitError(
            "conditioning cohort contract differs from its seal"
        )
    expected_ranks = contract.get("expected_ranks")
    environments_per_shard = contract.get("environments_per_shard")
    train_group_ids = contract.get("train_group_ids")
    rotation = contract.get("group_rotation")
    if (
        not isinstance(expected_ranks, list)
        or not expected_ranks
        or len(expected_ranks) != len(set(expected_ranks))
        or any(type(rank) is not int or rank < 0 for rank in expected_ranks)
        or type(environments_per_shard) is not int
        or environments_per_shard <= 0
        or not _is_sha256(contract.get("split_manifest_sha256"))
        or not _is_sha256(contract.get("split_sha256"))
        or not isinstance(train_group_ids, list)
        or not train_group_ids
        or len(train_group_ids) != len(set(train_group_ids))
        or any(not isinstance(group_id, str) or not group_id for group_id in train_group_ids)
        or not isinstance(rotation, Mapping)
        or set(rotation) != {"start_group_offset", "rank_group_offsets", "counts"}
        or type(rotation.get("start_group_offset")) is not int
        or rotation["start_group_offset"] not in range(len(train_group_ids))
        or not isinstance(rotation.get("rank_group_offsets"), list)
        or len(rotation["rank_group_offsets"]) != len(expected_ranks)
        or any(
            type(offset) is not int or offset not in range(len(train_group_ids))
            for offset in rotation["rank_group_offsets"]
        )
        or not isinstance(rotation.get("counts"), Mapping)
        or set(rotation["counts"]) != set(train_group_ids)
        or any(type(count) is not int or count < 0 for count in rotation["counts"].values())
        or sum(rotation["counts"].values()) != len(seal["global_env_ids"])
    ):
        raise ConditioningSplitError(
            "conditioning cohort partition/rotation contract is malformed"
        )
    if rotation["rank_group_offsets"][0] != rotation["start_group_offset"]:
        raise ConditioningSplitError(
            "conditioning cohort first rank/group rotation offset differs"
        )
    expected_names = contract.get("expected_shard_names")
    if (
        not isinstance(expected_names, list)
        or not expected_names
        or len(expected_names) != len(set(expected_names))
        or any(
            not isinstance(name, str) or Path(name).name != name
            for name in expected_names
        )
    ):
        raise ConditioningSplitError(
            "conditioning cohort expected shard inventory is malformed"
        )
    shard_root = cohort / "shards"
    _require_safe_tree(shard_root)
    if not shard_root.is_dir():
        raise ConditioningSplitError("conditioning cohort shard directory is missing")
    actual_names = sorted(path.name for path in shard_root.iterdir())
    if actual_names != sorted(expected_names):
        raise ConditioningSplitError("conditioning cohort shard inventory differs")

    manifest_paths: list[Path] = []
    flat_inventory: list[dict[str, Any]] = []
    observed_ids: list[int] = []
    transitions = 0
    episodes = 0
    images = 0
    observed_motion_ids: list[str] = []
    observed_parts_by_rank: dict[int, list[int]] = defaultdict(list)
    shards = seal["shards"]
    if not isinstance(shards, list) or len(shards) != seal["shard_count"]:
        raise ConditioningSplitError("conditioning cohort seal shard count differs")
    expected_from_seal: set[str] = set()
    for row in shards:
        row_keys = {
            "shard_id",
            "rank",
            "part_index",
            "global_env_ids",
            "motion_ids",
            "transition_count",
            "episode_count",
            "image_count",
            "manifest",
            "payload",
        }
        if not isinstance(row, Mapping) or set(row) != row_keys:
            raise ConditioningSplitError(
                "conditioning cohort sealed shard fields differ"
            )
        stem = row["shard_id"]
        if not isinstance(stem, str) or Path(stem).name != stem:
            raise ConditioningSplitError("conditioning cohort shard ID is unsafe")
        rank = row["rank"]
        part_index = row["part_index"]
        motion_ids = row["motion_ids"]
        if (
            type(rank) is not int
            or rank not in expected_ranks
            or type(part_index) is not int
            or part_index < 0
            or not isinstance(motion_ids, list)
            or len(motion_ids) != environments_per_shard
            or any(
                not isinstance(motion_id, str)
                or motion_id not in train_group_ids
                for motion_id in motion_ids
            )
        ):
            raise ConditioningSplitError(
                "conditioning cohort sealed shard rank/part/motion fields differ"
            )
        manifest_path, manifest_descriptor = _descriptor(
            row["manifest"],
            parent=cohort,
            expected_path=f"shards/{stem}.json",
            label="conditioning shard manifest",
        )
        _, payload_descriptor = _descriptor(
            row["payload"],
            parent=cohort,
            expected_path=f"shards/{stem}.npz",
            label="conditioning shard payload",
        )
        expected_from_seal.update({f"{stem}.json", f"{stem}.npz"})
        try:
            report, arrays = load_raw_conditioning_artifact(manifest_path)
        except ConditioningArtifactError as exc:
            raise ConditioningSplitError(
                f"sealed conditioning shard is invalid: {exc}"
            ) from exc
        raw_ids = arrays["global_env_ids"]
        if not np.array_equal(raw_ids, np.broadcast_to(raw_ids[0:1], raw_ids.shape)):
            raise ConditioningSplitError(
                "sealed conditioning shard environment identities changed"
            )
        ids = [int(value) for value in raw_ids[0].tolist()]
        metadata = report["render"]["metadata"]
        rank_ordinal = expected_ranks.index(rank)
        expected_motion_ids = [
            train_group_ids[
                (
                    rotation["rank_group_offsets"][rank_ordinal]
                    + part_index * environments_per_shard
                    + local_index
                )
                % len(train_group_ids)
            ]
            for local_index in range(environments_per_shard)
        ]
        if (
            report["manifest_sha256"] != manifest_descriptor["sha256"]
            or report["payload"]["sha256"] != payload_descriptor["sha256"]
            or report["num_envs"] != environments_per_shard
            or ids != row["global_env_ids"]
            or motion_ids != expected_motion_ids
            or metadata.get("rank") != rank
            or metadata.get("part_index") != part_index
            or metadata.get("split_manifest_sha256")
            != seal["split_manifest_sha256"]
            or metadata.get("split_sha256") != seal["split_sha256"]
            or metadata.get("group_ids") != train_group_ids
            or metadata.get("motion_ids") != motion_ids
            or report["transition_count"] != row["transition_count"]
            or report["episode_count"] != row["episode_count"]
            or report["image_count"] != row["image_count"]
        ):
            raise ConditioningSplitError("sealed conditioning shard evidence differs")
        manifest_paths.append(manifest_path)
        observed_ids.extend(ids)
        observed_motion_ids.extend(motion_ids)
        observed_parts_by_rank[rank].append(part_index)
        transitions += report["transition_count"]
        episodes += report["episode_count"]
        images += report["image_count"]
        flat_inventory.extend(
            [
                {
                    "name": f"{stem}.json",
                    **{
                        key: manifest_descriptor[key]
                        for key in ("size_bytes", "sha256")
                    },
                },
                {
                    "name": f"{stem}.npz",
                    **{
                        key: payload_descriptor[key] for key in ("size_bytes", "sha256")
                    },
                },
            ]
        )
    if expected_from_seal != set(expected_names):
        raise ConditioningSplitError(
            "cohort seal and contract shard inventories differ"
        )
    if set(observed_parts_by_rank) != set(expected_ranks) or any(
        sorted(parts) != list(range(len(parts)))
        for parts in observed_parts_by_rank.values()
    ):
        raise ConditioningSplitError(
            "conditioning cohort shard part coverage differs"
        )
    if Counter(observed_motion_ids) != Counter(rotation["counts"]):
        raise ConditioningSplitError(
            "conditioning cohort group rotation counts differ from shard motions"
        )
    if (
        len(observed_ids) != len(set(observed_ids))
        or sorted(observed_ids) != seal["global_env_ids"]
        or transitions != seal["transition_count"]
        or episodes != seal["episode_count"]
        or images != seal["image_count"]
    ):
        raise ConditioningSplitError("conditioning cohort sealed coverage differs")
    top_level = sorted(path.name for path in cohort.iterdir())
    if top_level != ["contract.json", "seal.json", "shards"]:
        raise ConditioningSplitError("conditioning cohort contains unexpected entries")
    return (
        dict(seal),
        sorted(manifest_paths),
        sorted(flat_inventory, key=lambda row: row["name"]),
    )


def _exact_wave_shard_names(wave_index: int) -> list[str]:
    names: list[str] = []
    for part_index in range(EXPECTED_BC_SHARDS_PER_WAVE):
        shard_index = wave_index * EXPECTED_BC_SHARDS_PER_WAVE + part_index
        stem = (
            f"{GENERATION_ID}__wave-{wave_index:03d}__gpu-00"
            f"__shard-{shard_index:06d}"
        )
        names.extend((f"{stem}.npz", f"{stem}.json"))
    return names


def _validate_exact_bc_saturation(saturation: Any) -> list[dict[str, Any]]:
    """Require exact 8xH200 inventory and the admitted one-actor BC run."""

    if not isinstance(saturation, Mapping):
        raise ConditioningSplitError("generation saturation manifest must be an object")
    try:
        physical_inventory = validate_physical_gpu_inventory(
            saturation.get("physical_gpu_inventory")
        )
    except ArtifactError as exc:
        raise ConditioningSplitError(
            f"generation saturation physical GPU evidence differs: {exc}"
        ) from exc
    selection = saturation.get("selection")
    waves = saturation.get("waves")
    references = saturation.get("train_conditioning_cohorts")
    if (
        saturation.get("schema") != SATURATION_SCHEMA
        or saturation.get("generation_id") != GENERATION_ID
        or saturation.get("status") != "complete"
        or saturation.get("stop_reason") != "nominal_target_met"
        or saturation.get("nominal_transitions") != EXPECTED_BC_NOMINAL_TRANSITIONS
        or saturation.get("completed_transitions") != EXPECTED_BC_NOMINAL_TRANSITIONS
        or saturation.get("actual_overshoot") != 0
        or saturation.get("physical_gpu_inventory_sha256")
        != canonical_json_sha256(physical_inventory)
        or saturation.get("virtual_actor_world_size")
        != EXPECTED_BC_ACTOR_WORLD_SIZE
        or saturation.get("admitted_physical_gpu_indices")
        != EXPECTED_BC_ADMITTED_PHYSICAL_GPU_INDICES
        or saturation.get("physical_execution_concurrency") != 1
        or saturation.get("virtual_rank_cohorts")
        != [[{"virtual_rank": 0, "physical_gpu_index": 0}]]
        or not isinstance(selection, Mapping)
        or selection.get("environments_per_gpu")
        != EXPECTED_BC_ENVIRONMENTS_PER_ACTOR
        or selection.get("total_environments") != EXPECTED_BC_ENVIRONMENTS_PER_WAVE
        or selection.get("transitions_per_wave") != EXPECTED_BC_TRANSITIONS_PER_WAVE
        or selection.get("planned_waves") != EXPECTED_BC_WAVE_COUNT
        or selection.get("planned_transitions") != EXPECTED_BC_NOMINAL_TRANSITIONS
        or selection.get("planned_overshoot") != 0
        or not isinstance(waves, list)
        or len(waves) != EXPECTED_BC_WAVE_COUNT
        or not isinstance(references, list)
        or len(references) != EXPECTED_BC_WAVE_COUNT
    ):
        raise ConditioningSplitError(
            "generation saturation differs from exact 8xH200-inventory, "
            "one-actor, two-wave 256x512 production"
        )
    flattened_inventory: list[dict[str, Any]] = []
    for wave_index, wave in enumerate(waves):
        expected_names = _exact_wave_shard_names(wave_index)
        seal_evidence = wave.get("seal") if isinstance(wave, Mapping) else None
        inventory = (
            seal_evidence.get("shards")
            if isinstance(seal_evidence, Mapping)
            else None
        )
        if (
            not isinstance(wave, Mapping)
            or wave.get("wave_index") != wave_index
            or wave.get("transition_count") != EXPECTED_BC_TRANSITIONS_PER_WAVE
            or wave.get("shard_names") != expected_names
            or not isinstance(seal_evidence, Mapping)
            or not isinstance(inventory, list)
            or len(inventory) != 2 * EXPECTED_BC_SHARDS_PER_WAVE
            or any(
                not isinstance(row, Mapping)
                or set(row) != {"name", "size_bytes", "sha256"}
                or not isinstance(row["name"], str)
                or Path(row["name"]).name != row["name"]
                or type(row["size_bytes"]) is not int
                or row["size_bytes"] <= 0
                or not _is_sha256(row["sha256"])
                for row in inventory
            )
            or sorted(row["name"] for row in inventory) != sorted(expected_names)
        ):
            raise ConditioningSplitError(
                f"generation saturation wave {wave_index} differs from exact "
                "rank-0 128x512/32-shard contract"
            )
        flattened_inventory.extend(dict(row) for row in inventory)
    if saturation.get("shard_inventory") != flattened_inventory:
        raise ConditioningSplitError(
            "generation saturation global 64-shard inventory differs"
        )
    return physical_inventory


def _sealed_source_manifests(
    campaign_root: Path,
    source_root: Path,
    *,
    require_exact_production: bool = False,
) -> tuple[list[Path], dict[str, Any], Path]:
    _require_safe_tree(source_root)
    if not source_root.is_dir():
        raise ConditioningSplitError(
            f"conditioning collection does not exist: {source_root}"
        )
    saturation_path = campaign_root / "BC/collection/saturation_manifest.json"
    saturation_size, saturation_sha = _immutable_regular(
        saturation_path,
        "generation saturation manifest",
        exact_mode=0o444,
    )
    try:
        saturation = load_strict_json(saturation_path)
    except ArtifactError as exc:
        raise ConditioningSplitError(
            f"cannot load generation saturation manifest: {exc}"
        ) from exc
    physical_inventory: list[dict[str, Any]] | None = None
    if require_exact_production:
        physical_inventory = _validate_exact_bc_saturation(saturation)
    if (
        not isinstance(saturation, Mapping)
        or saturation.get("schema") != SATURATION_SCHEMA
        or saturation.get("generation_id") != GENERATION_ID
        or saturation.get("status") != "complete"
        or saturation.get("stop_reason") != "nominal_target_met"
        or type(saturation.get("nominal_transitions")) is not int
        or saturation.get("nominal_transitions", 0) <= 0
        or saturation.get("completed_transitions", 0)
        < saturation.get("nominal_transitions", 0)
        or type(saturation.get("completed_image_count")) is not int
        or saturation.get("completed_image_count", 0) <= 0
        or type(saturation.get("completed_episode_count")) is not int
        or saturation.get("completed_episode_count", 0) <= 0
    ):
        raise ConditioningSplitError("generation saturation manifest is not complete")
    references = saturation.get("train_conditioning_cohorts")
    waves = saturation.get("waves")
    if (
        not isinstance(references, list)
        or not references
        or not isinstance(waves, list)
    ):
        raise ConditioningSplitError(
            "generation saturation manifest lacks sealed cohorts"
        )

    manifests: list[Path] = []
    cohort_evidence: list[dict[str, Any]] = []
    inventory_by_cohort: dict[str, list[dict[str, Any]]] = {}
    seal_by_cohort: dict[str, dict[str, Any]] = {}
    reported_paths_by_cohort: dict[str, tuple[Path, Path]] = {}
    identity_paths_by_cohort: dict[str, Path] = {}
    for reference in references:
        seal, paths, inventory = _verify_cohort(campaign_root, source_root, reference)
        reported_cohort, cohort, identity_path = _resolve_sealed_cohort_path(
            source_root, reference["path"]
        )
        reported_seal = _absolute(reference["seal_path"])
        cohort_path = str(cohort)
        manifests.extend(paths)
        inventory_by_cohort[cohort_path] = inventory
        seal_by_cohort[cohort_path] = seal
        reported_paths_by_cohort[cohort_path] = (
            reported_cohort,
            reported_seal,
        )
        identity_paths_by_cohort[cohort_path] = identity_path
        cohort_evidence.append(
            {
                "path": identity_path.as_posix(),
                "seal_path": (identity_path / "seal.json").as_posix(),
                "seal_sha256": reference["seal_sha256"],
                "transition_count": seal["transition_count"],
                "episode_count": seal["episode_count"],
                "image_count": seal["image_count"],
                "global_env_ids": seal["global_env_ids"],
                "shard_count": seal["shard_count"],
            }
        )
    if len(inventory_by_cohort) != len(references):
        raise ConditioningSplitError("generation saturation manifest repeats a cohort")
    identity_roots = {
        path.parent for path in identity_paths_by_cohort.values()
    }
    if len(identity_roots) != 1:
        raise ConditioningSplitError(
            "conditioning cohorts mix source path identities"
        )
    source_identity_root = next(iter(identity_roots))

    wave_paths: list[str] = []
    flat_from_waves: list[dict[str, Any]] = []
    for wave in waves:
        if not isinstance(wave, Mapping) or not isinstance(wave.get("seal"), Mapping):
            raise ConditioningSplitError("generation wave lacks seal evidence")
        reported_cohort, cohort, identity_path = _resolve_sealed_cohort_path(
            source_root, wave.get("cohort_dir")
        )
        cohort_path = str(cohort)
        seal_evidence = wave["seal"]
        cohort_seal = seal_by_cohort.get(cohort_path)
        reported_pair = reported_paths_by_cohort.get(cohort_path)
        reported_seal = _absolute(seal_evidence.get("path", ""))
        if (
            cohort_path not in inventory_by_cohort
            or not isinstance(cohort_seal, Mapping)
            or reported_pair != (reported_cohort, reported_seal)
            or identity_paths_by_cohort.get(cohort_path) != identity_path
            or reported_seal != reported_cohort / "seal.json"
            or seal_evidence.get("sha256")
            != sha256_file(Path(cohort_path) / "seal.json")
            or seal_evidence.get("shards") != inventory_by_cohort[cohort_path]
            or wave.get("image_count") != seal_evidence.get("image_count")
            or wave.get("episode_count") != seal_evidence.get("episode_count")
            or wave.get("transition_count") != cohort_seal.get("transition_count")
            or wave.get("wave_index") != cohort_seal.get("cohort_index")
            or wave.get("group_rotation") != cohort_seal.get("group_rotation")
            or not isinstance(wave.get("shard_names"), list)
            or sorted(wave.get("shard_names", ()))
            != sorted(row["name"] for row in seal_evidence["shards"])
        ):
            raise ConditioningSplitError(
                "generation wave seal/inventory evidence differs"
            )
        wave_paths.append(cohort_path)
        flat_from_waves.extend(seal_evidence["shards"])
    if sorted(wave_paths) != sorted(inventory_by_cohort):
        raise ConditioningSplitError("generation waves and cohort references differ")
    if saturation.get("shard_inventory") != flat_from_waves:
        raise ConditioningSplitError("generation flattened shard inventory differs")
    if require_exact_production:
        exact_cohorts = sorted(cohort_evidence, key=lambda row: row["path"])
        exact_seals = sorted(
            seal_by_cohort.values(), key=lambda row: row["cohort_index"]
        )
        partition_bindings = {
            canonical_json_sha256(
                {
                    "split_manifest_sha256": seal["split_manifest_sha256"],
                    "split_sha256": seal["split_sha256"],
                    "train_group_ids": seal["train_group_ids"],
                }
            )
            for seal in exact_seals
        }
        if len(partition_bindings) != 1:
            raise ConditioningSplitError(
                "production waves do not share one immutable train partition"
            )
        for wave_index, cohort_row in enumerate(exact_cohorts):
            seal = exact_seals[wave_index]
            group_ids = seal["train_group_ids"]
            start = (
                wave_index * EXPECTED_BC_ENVIRONMENTS_PER_WAVE
            ) % len(group_ids)
            counts = {group_id: 0 for group_id in group_ids}
            for slot in range(EXPECTED_BC_ENVIRONMENTS_PER_WAVE):
                counts[group_ids[(start + slot) % len(group_ids)]] += 1
            exact_rotation = {
                "start_group_offset": start,
                "rank_group_offsets": [start],
                "counts": counts,
            }
            shard_rows = sorted(seal["shards"], key=lambda row: row["part_index"])
            exact_stems = [
                name[:-5]
                for name in _exact_wave_shard_names(wave_index)
                if name.endswith(".json")
            ]
            if (
                cohort_row["path"]
                != (
                    f"{source_identity_root.as_posix()}"
                    f"/wave-{wave_index:03d}"
                )
                or cohort_row["transition_count"]
                != EXPECTED_BC_TRANSITIONS_PER_WAVE
                or cohort_row["global_env_ids"]
                != list(
                    range(
                        wave_index * EXPECTED_BC_ENVIRONMENTS_PER_WAVE,
                        (wave_index + 1) * EXPECTED_BC_ENVIRONMENTS_PER_WAVE,
                    )
                )
                or cohort_row["shard_count"] != EXPECTED_BC_SHARDS_PER_WAVE
                or seal["cohort_index"] != wave_index
                or seal["group_rotation"] != exact_rotation
                or [row["shard_id"] for row in shard_rows] != exact_stems
                or [row["rank"] for row in shard_rows]
                != [0] * EXPECTED_BC_SHARDS_PER_WAVE
                or [row["part_index"] for row in shard_rows]
                != list(range(EXPECTED_BC_SHARDS_PER_WAVE))
                or any(
                    row["global_env_ids"]
                    != list(
                        range(
                            wave_index * EXPECTED_BC_ENVIRONMENTS_PER_WAVE
                            + row["part_index"] * EXPECTED_BC_ENVIRONMENTS_PER_SHARD,
                            wave_index * EXPECTED_BC_ENVIRONMENTS_PER_WAVE
                            + (row["part_index"] + 1)
                            * EXPECTED_BC_ENVIRONMENTS_PER_SHARD,
                        )
                    )
                    for row in shard_rows
                )
            ):
                raise ConditioningSplitError(
                    f"conditioning wave {wave_index} differs from exact "
                    "128-stream/32-shard coverage"
                )
    actual_waves = sorted(
        str(path)
        for path in source_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    )
    if sorted(path.name for path in source_root.iterdir()) != sorted(
        Path(path).name for path in inventory_by_cohort
    ):
        raise ConditioningSplitError(
            "conditioning waves root contains unexpected entries"
        )
    if actual_waves != sorted(inventory_by_cohort):
        raise ConditioningSplitError(
            "conditioning wave directory inventory differs from saturation"
        )
    actual_manifests = sorted(source_root.glob("wave-[0-9][0-9][0-9]/shards/*.json"))
    if sorted(manifests) != actual_manifests:
        raise ConditioningSplitError(
            "source manifests do not exactly match sealed cohorts"
        )
    completed = sum(row["transition_count"] for row in cohort_evidence)
    completed_images = sum(row["image_count"] for row in cohort_evidence)
    completed_episodes = sum(row["episode_count"] for row in cohort_evidence)
    if completed != saturation["completed_transitions"]:
        raise ConditioningSplitError(
            "generation completed transition count differs from cohort seals"
        )
    if (
        completed_images != saturation["completed_image_count"]
        or completed_episodes != saturation["completed_episode_count"]
    ):
        raise ConditioningSplitError(
            "generation completed image/episode counts differ from cohort seals"
        )
    return sorted(manifests), {
        "path": (
            source_identity_root.parents[1] / "saturation_manifest.json"
        ).as_posix(),
        "sha256": saturation_sha,
        "size_bytes": saturation_size,
        "status": saturation["status"],
        "stop_reason": saturation["stop_reason"],
        "nominal_transitions": saturation["nominal_transitions"],
        "completed_transitions": saturation["completed_transitions"],
        "completed_image_count": saturation["completed_image_count"],
        "completed_episode_count": saturation["completed_episode_count"],
        "cohorts": sorted(cohort_evidence, key=lambda row: row["path"]),
        "physical_gpu_inventory": physical_inventory,
    }, source_identity_root


def _inventory(
    source_root: Path,
    source_identity_root: Path,
    manifest_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_global_ids: set[int] = set()
    for manifest_path in manifest_paths:
        try:
            report, arrays = load_raw_conditioning_artifact(manifest_path)
        except ConditioningArtifactError as exc:
            raise ConditioningSplitError(
                f"invalid source artifact {manifest_path}: {exc}"
            ) from exc
        if report["purpose"] != "train":
            raise ConditioningSplitError("source BC collection must have purpose=train")
        metadata = report["render"]["metadata"]
        if metadata.get("campaign_id") != CAMPAIGN_ID:
            raise ConditioningSplitError("source artifact campaign identity differs")
        motion_ids = metadata.get("motion_ids")
        if (
            not isinstance(motion_ids, list)
            or len(motion_ids) != report["num_envs"]
            or any(not isinstance(value, str) or not value for value in motion_ids)
        ):
            raise ConditioningSplitError(
                "source artifact lacks one exact motion_id per environment"
            )
        allowed_motion_ids = metadata.get("group_ids")
        if (
            not isinstance(allowed_motion_ids, list)
            or not allowed_motion_ids
            or len(set(allowed_motion_ids)) != len(allowed_motion_ids)
            or any(value not in allowed_motion_ids for value in motion_ids)
        ):
            raise ConditioningSplitError(
                "source artifact motion_ids differ from its motion allowlist"
            )
        try:
            source_suffix = manifest_path.relative_to(source_root)
        except ValueError as exc:
            raise ConditioningSplitError(
                "source manifest escapes conditioning waves"
            ) from exc
        source_relative = (source_identity_root / source_suffix).as_posix()
        sources.append(
            {
                "manifest_path": source_relative,
                "manifest_sha256": report["manifest_sha256"],
                "payload_sha256": report["payload"]["sha256"],
                "num_envs": report["num_envs"],
                "transition_count": report["transition_count"],
                "episode_count": report["episode_count"],
                "image_count": report["image_count"],
            }
        )
        for env_index, raw_global_id in enumerate(arrays["global_env_ids"][0]):
            global_id = int(raw_global_id)
            if global_id in seen_global_ids:
                raise ConditioningSplitError(
                    f"global_env_id {global_id} appears in more than one source artifact"
                )
            seen_global_ids.add(global_id)
            episode_ids = sorted(
                set(
                    int(value) for value in arrays["episode_ids"][:, env_index].tolist()
                )
            )
            strata = _metadata_stratum(metadata, env_index, report["num_envs"])
            units.append(
                {
                    "group_id": f"global-env-{global_id:012d}",
                    "global_env_id": global_id,
                    "episode_ids": episode_ids,
                    "episode_count": len(episode_ids),
                    "transition_count": report["num_steps"],
                    "image_count": int(
                        np.unique(arrays["transition_image_indices"][:, env_index]).size
                    ),
                    "source_manifest_path": source_relative,
                    "source_manifest_sha256": report["manifest_sha256"],
                    "source_payload_sha256": report["payload"]["sha256"],
                    "source_env_index": env_index,
                    "strata": strata,
                }
            )
    units.sort(key=lambda row: row["global_env_id"])
    sources.sort(key=lambda row: row["manifest_path"])
    return units, sources


def _assign(
    units: Sequence[Mapping[str, Any]], targets: Mapping[str, int], seed: int
) -> dict[str, list[int]]:
    """Deterministically balance discovered strata while meeting exact quotas."""

    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        buckets[_stratum_key(unit["strata"])].append(unit)
    for stratum, rows in buckets.items():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{stratum}:{row['group_id']}".encode("utf-8")
            ).hexdigest()
        )

    # Interleave strata, rarest first, so a large stratum cannot exhaust a
    # held-out quota before smaller task/scenario strata are represented.
    ordered_strata = sorted(
        buckets,
        key=lambda value: (
            len(buckets[value]),
            hashlib.sha256(f"{seed}:stratum:{value}".encode()).hexdigest(),
        ),
    )
    stream: list[tuple[str, Mapping[str, Any]]] = []
    depth = 0
    while len(stream) < len(units):
        for stratum in ordered_strata:
            if depth < len(buckets[stratum]):
                stream.append((stratum, buckets[stratum][depth]))
        depth += 1

    assigned = {split: [] for split in SPLITS}
    by_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    stratum_sizes = Counter(_stratum_key(unit["strata"]) for unit in units)
    total = len(units)
    for stratum, unit in stream:
        candidates = [
            split for split in SPLITS if len(assigned[split]) < targets[split]
        ]
        if not candidates:  # pragma: no cover - arithmetic invariant
            raise ConditioningSplitError("split assignment exhausted all quotas early")

        def score(split: str) -> tuple[float, float, str]:
            desired_stratum = stratum_sizes[stratum] * targets[split] / total
            stratum_deficit = desired_stratum - by_stratum[stratum][split]
            global_deficit = (targets[split] - len(assigned[split])) / targets[split]
            tie = hashlib.sha256(
                f"{seed}:assign:{unit['group_id']}:{split}".encode()
            ).hexdigest()
            return (-stratum_deficit, -global_deficit, tie)

        chosen = min(candidates, key=score)
        assigned[chosen].append(int(unit["global_env_id"]))
        by_stratum[stratum][chosen] += 1
    for split in SPLITS:
        assigned[split].sort()
        if len(assigned[split]) != targets[split]:
            raise ConditioningSplitError(
                f"{split} assignment does not meet exact quota"
            )
    flat = [value for split in SPLITS for value in assigned[split]]
    if len(flat) != len(set(flat)) or set(flat) != {
        int(unit["global_env_id"]) for unit in units
    }:
        raise ConditioningSplitError("split assignment lacks exact disjoint coverage")
    return assigned


def _slice_environment(
    arrays: Mapping[str, np.ndarray], env_index: int, global_env_id: int
) -> dict[str, np.ndarray]:
    anchor_mask = arrays["anchor_global_env_ids"] == global_env_id
    if not bool(np.any(anchor_mask)):
        raise ConditioningSplitError("environment has no episode anchors")
    source_mapping = arrays["transition_image_indices"][:, env_index]
    image_count = int(arrays["image_global_env_ids"].size)
    if (
        source_mapping.dtype != np.int64
        or np.any(source_mapping < 0)
        or np.any(source_mapping >= image_count)
    ):
        raise ConditioningSplitError("environment image mapping is out of bounds")
    used = np.unique(source_mapping)
    if not np.all(arrays["image_global_env_ids"][used] == global_env_id):
        raise ConditioningSplitError("environment image rows cross global identities")
    remap = np.full(image_count, -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)

    output: dict[str, np.ndarray] = {}
    for key, array in arrays.items():
        if key in _ANCHOR_IDS or key in _ANCHOR_VIEWS:
            output[key] = np.ascontiguousarray(array[anchor_mask])
        elif key == "transition_image_indices":
            output[key] = np.ascontiguousarray(
                remap[source_mapping][:, None], dtype=np.int64
            )
        elif key in _IMAGE_IDS or key in _IMAGE_VIEWS:
            selected = np.ascontiguousarray(array[used])
            if key == "image_reference_indices":
                valid = selected >= 0
                rebased = np.full(selected.shape, -1, dtype=np.int64)
                rebased[valid] = remap[selected[valid]]
                if np.any(rebased[valid] < 0):
                    raise ConditioningSplitError(
                        "environment image references are not closed under split"
                    )
                selected = rebased
            output[key] = selected
        else:
            output[key] = np.ascontiguousarray(array[:, env_index : env_index + 1])
    if set(output["transition_image_indices"].ravel().tolist()) != set(
        range(used.size)
    ):
        raise ConditioningSplitError("split image rows are not an exact referenced set")
    return output


def _output_root(campaign_root: Path, split: str) -> Path:
    return {
        "train": campaign_root / "BC/train",
        "validation": campaign_root / "validation/BC",
        "test": campaign_root / "test/BC",
    }[split]


def audit_pre_render_replay_root(
    generation_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Prove that the canonical render/replay output root is still empty."""

    generation = _absolute(generation_root)
    replay_root = generation / "replay"
    _require_safe_tree(replay_root)
    if not replay_root.is_dir():
        raise ConditioningSplitError("pre-render replay root is missing")
    expected_directories = ("manifests", "shards")
    entries = sorted(replay_root.iterdir(), key=lambda path: path.name)
    for path in entries:
        _require_safe_tree(path)
    if [path.name for path in entries] != list(expected_directories) or any(
        not path.is_dir() for path in entries
    ):
        raise ConditioningSplitError(
            "pre-render replay root must contain only empty manifests/shards directories"
        )

    observed: list[dict[str, Any]] = []
    for directory in entries:
        for path in sorted(directory.rglob("*")):
            _require_safe_tree(path)
            info = os.stat(path, follow_symlinks=False)
            observed.append(
                {
                    "path": path.relative_to(generation).as_posix(),
                    "kind": "directory" if stat.S_ISDIR(info.st_mode) else "file",
                    "size_bytes": info.st_size,
                }
            )
    rendering_started = bool(observed)
    if rendering_started:
        raise ConditioningSplitError(
            "render/replay output exists before BC collection completion"
        )
    return {
        "schema": PRE_RENDER_AUDIT_SCHEMA,
        "root": "replay",
        "expected_empty_directories": [
            f"replay/{name}" for name in expected_directories
        ],
        "observed_inventory": observed,
        "observed_entry_count": len(observed),
        "observed_inventory_sha256": canonical_json_sha256(observed),
        "rendering_started": rendering_started,
    }


def _plan_document(
    *,
    campaign_root: Path,
    units: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Sequence[int]],
    weights: Mapping[str, int],
    ratio_source: str,
    seed: int,
    completion: Mapping[str, Any],
    source_identity_root: Path,
    expected_transitions_per_group: int | None,
    postprocess_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generation_id": GENERATION_ID,
        "source_root": source_identity_root.as_posix(),
        "strategy": STRATEGY,
        "seed": seed,
        "grouping": "complete_global_environment_stream",
        "ratio_source": ratio_source,
        "weight_semantics": (
            "post_collection_global_environment_stream_quota_not_motion_holdout"
        ),
        "expected_transitions_per_group": expected_transitions_per_group,
        "postprocess_provenance": (
            None if postprocess_provenance is None else dict(postprocess_provenance)
        ),
        "source_completion": dict(completion),
        "weights": dict(weights),
        "target_counts": {split: len(assignments[split]) for split in SPLITS},
        "source_inventory": list(sources),
        "source_inventory_sha256": canonical_json_sha256(list(sources)),
        "units": list(units),
        "assignments": {split: list(assignments[split]) for split in SPLITS},
        "assignment_sha256": canonical_json_sha256(
            {split: list(assignments[split]) for split in SPLITS}
        ),
    }


def _materialize(
    campaign_root: Path,
    source_root: Path,
    source_identity_root: Path,
    units: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Sequence[int]],
    plan_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    by_global_id = {int(unit["global_env_id"]): unit for unit in units}
    split_by_global_id = {
        int(global_id): split for split in SPLITS for global_id in assignments[split]
    }
    entries = {split: [] for split in SPLITS}
    by_source: dict[str, list[int]] = defaultdict(list)
    for global_id, unit in by_global_id.items():
        by_source[str(unit["source_manifest_path"])].append(global_id)

    for source_relative in sorted(by_source):
        try:
            source_suffix = Path(source_relative).relative_to(source_identity_root)
        except ValueError as exc:
            raise ConditioningSplitError(
                "source manifest escapes its sealed identity root"
            ) from exc
        source_path = source_root / source_suffix
        _require_safe_tree(source_path)
        report, arrays = load_raw_conditioning_artifact(source_path)
        for global_id in sorted(by_source[source_relative]):
            unit = by_global_id[global_id]
            env_index = int(unit["source_env_index"])
            if int(arrays["global_env_ids"][0, env_index]) != global_id:
                raise ConditioningSplitError(
                    "source environment identity changed after planning"
                )
            split = split_by_global_id[global_id]
            split_root = _output_root(campaign_root, split)
            artifact_dir = split_root / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _require_safe_tree(artifact_dir)
            stem = str(unit["group_id"])
            original_metadata = report["render"]["metadata"]
            if "bc_collection_split" in original_metadata:
                raise ConditioningSplitError(
                    "source render metadata already contains a split record"
                )
            metadata = {
                **original_metadata,
                "bc_collection_split": {
                    "schema": PLAN_SCHEMA,
                    "plan_sha256": plan_sha256,
                    "split": split,
                    "seed": SPLIT_SEED,
                    "grouping": "complete_global_environment_stream",
                    "global_env_id": global_id,
                    "source_manifest_path": source_relative,
                    "source_manifest_sha256": report["manifest_sha256"],
                    "source_payload_sha256": report["payload"]["sha256"],
                    "source_env_index": env_index,
                },
            }
            try:
                published = publish_conditioning_shard(
                    artifact_dir / f"{stem}.npz",
                    _slice_environment(arrays, env_index, global_id),
                    contract=report["control_contract"],
                    purpose=split,
                    prompt=report["render"]["prompt"],
                    seed=report["render"]["seed"],
                    reference_lag=report["render"]["reference_lag"],
                    render_metadata=metadata,
                    manifest_path=artifact_dir / f"{stem}.json",
                )
            except ConditioningArtifactError as exc:
                raise ConditioningSplitError(
                    f"cannot publish {split} group {stem}: {exc}"
                ) from exc
            entries[split].append(
                {
                    "group_id": stem,
                    "global_env_id": global_id,
                    "episode_ids": list(unit["episode_ids"]),
                    "episode_count": published["episode_count"],
                    "transition_count": published["transition_count"],
                    "image_count": published["image_count"],
                    "strata": dict(unit["strata"]),
                    "manifest_path": _relative(
                        Path(published["manifest_path"]), split_root, "output manifest"
                    ),
                    "manifest_sha256": published["manifest_sha256"],
                    "payload_path": _relative(
                        Path(published["payload_path"]), split_root, "output payload"
                    ),
                    "payload_sha256": published["payload"]["sha256"],
                    "payload_size_bytes": published["payload"]["size_bytes"],
                    "source_manifest_path": source_relative,
                    "source_manifest_sha256": report["manifest_sha256"],
                }
            )
    for split in SPLITS:
        entries[split].sort(key=lambda row: row["global_env_id"])
    return entries


def _split_manifest(
    split: str,
    plan: Mapping[str, Any],
    plan_sha256: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    strata = Counter(_stratum_key(row["strata"]) for row in entries)
    return {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generation_id": GENERATION_ID,
        "split": split,
        "purpose": split,
        "seed": plan["seed"],
        "strategy": plan["strategy"],
        "grouping": plan["grouping"],
        "weight_semantics": plan["weight_semantics"],
        "plan_path": SEALED_BC_PLAN_PATH_FIELD,
        "plan_sha256": plan_sha256,
        "group_count": len(entries),
        "episode_count": sum(int(row["episode_count"]) for row in entries),
        "transition_count": sum(int(row["transition_count"]) for row in entries),
        "image_count": sum(int(row["image_count"]) for row in entries),
        "global_env_ids": [int(row["global_env_id"]) for row in entries],
        "strata_counts": dict(sorted(strata.items())),
        "artifacts": list(entries),
        "learner_replay_ready": False,
        "training_access": (
            "allowed_after_roborender_projection" if split == "train" else "forbidden"
        ),
    }


_PLAN_KEYS = {
    "schema",
    "campaign_id",
    "generation_id",
    "source_root",
    "strategy",
    "seed",
    "grouping",
    "ratio_source",
    "weight_semantics",
    "expected_transitions_per_group",
    "postprocess_provenance",
    "source_completion",
    "weights",
    "target_counts",
    "source_inventory",
    "source_inventory_sha256",
    "units",
    "assignments",
    "assignment_sha256",
}
_SPLIT_MANIFEST_KEYS = {
    "schema",
    "campaign_id",
    "generation_id",
    "split",
    "purpose",
    "seed",
    "strategy",
    "grouping",
    "weight_semantics",
    "plan_path",
    "plan_sha256",
    "group_count",
    "episode_count",
    "transition_count",
    "image_count",
    "global_env_ids",
    "strata_counts",
    "artifacts",
    "learner_replay_ready",
    "training_access",
}
_SPLIT_ARTIFACT_KEYS = {
    "group_id",
    "global_env_id",
    "episode_ids",
    "episode_count",
    "transition_count",
    "image_count",
    "strata",
    "manifest_path",
    "manifest_sha256",
    "payload_path",
    "payload_sha256",
    "payload_size_bytes",
    "source_manifest_path",
    "source_manifest_sha256",
}
_SPLIT_SEAL_KEYS = {
    "schema",
    "campaign_id",
    "generation_id",
    "split",
    "sealed",
    "manifest_sha256",
    "plan_sha256",
    "group_count",
    "episode_count",
    "transition_count",
    "image_count",
    "global_env_ids",
    "artifact_manifest_sha256",
    "artifact_payload_sha256",
    "learner_replay_ready",
    "training_access",
}
_SOURCE_INVENTORY_KEYS = {
    "manifest_path",
    "manifest_sha256",
    "payload_sha256",
    "num_envs",
    "transition_count",
    "episode_count",
    "image_count",
}
_UNIT_KEYS = {
    "group_id",
    "global_env_id",
    "episode_ids",
    "episode_count",
    "transition_count",
    "image_count",
    "source_manifest_path",
    "source_manifest_sha256",
    "source_payload_sha256",
    "source_env_index",
    "strata",
}
_EMBEDDED_SPLIT_KEYS = {
    "schema",
    "plan_sha256",
    "split",
    "seed",
    "grouping",
    "global_env_id",
    "source_manifest_path",
    "source_manifest_sha256",
    "source_payload_sha256",
    "source_env_index",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_strict_relative_descendant(value: Any, root: Path) -> bool:
    if not isinstance(value, str):
        return False
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return False
    try:
        suffix = path.relative_to(root)
    except ValueError:
        return False
    return suffix != Path(".")


def _validate_plan_source_bindings(
    source_inventory: Sequence[Any],
    units: Sequence[Any],
) -> dict[int, Mapping[str, Any]]:
    """Bind every planned environment to one exact sealed source descriptor."""

    sources: dict[str, Mapping[str, Any]] = {}
    source_order: list[str] = []
    for source in source_inventory:
        if not isinstance(source, Mapping) or set(source) != _SOURCE_INVENTORY_KEYS:
            raise ConditioningSplitError("global plan source descriptor fields differ")
        source_path = source.get("manifest_path")
        relative = Path(source_path) if isinstance(source_path, str) else Path("/")
        if (
            not isinstance(source_path, str)
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or source_path in sources
            or not _is_sha256(source.get("manifest_sha256"))
            or not _is_sha256(source.get("payload_sha256"))
            or any(
                type(source.get(key)) is not int or source[key] <= 0
                for key in (
                    "num_envs",
                    "transition_count",
                    "episode_count",
                    "image_count",
                )
            )
        ):
            raise ConditioningSplitError("global plan source descriptor is malformed")
        sources[source_path] = source
        source_order.append(source_path)
    if source_order != sorted(source_order):
        raise ConditioningSplitError("global plan source inventory is not canonical")

    by_id: dict[int, Mapping[str, Any]] = {}
    units_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unit_order: list[int] = []
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != _UNIT_KEYS:
            raise ConditioningSplitError("global plan unit descriptor fields differ")
        global_id = unit.get("global_env_id")
        source_path = unit.get("source_manifest_path")
        source = sources.get(source_path) if isinstance(source_path, str) else None
        episode_ids = unit.get("episode_ids")
        strata = unit.get("strata")
        if (
            type(global_id) is not int
            or global_id < 0
            or global_id in by_id
            or unit.get("group_id") != f"global-env-{global_id:012d}"
            or not isinstance(episode_ids, list)
            or any(type(value) is not int or value < 0 for value in episode_ids)
            or episode_ids != sorted(set(episode_ids))
            or not isinstance(strata, Mapping)
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or not key
                or not value
                for key, value in strata.items()
            )
            or source is None
            or unit.get("source_manifest_sha256") != source["manifest_sha256"]
            or unit.get("source_payload_sha256") != source["payload_sha256"]
            or type(unit.get("source_env_index")) is not int
            or not 0 <= unit["source_env_index"] < source["num_envs"]
            or any(
                type(unit.get(key)) is not int or unit[key] <= 0
                for key in ("episode_count", "transition_count", "image_count")
            )
            or unit["episode_count"] != len(episode_ids)
        ):
            raise ConditioningSplitError("global plan unit/source binding differs")
        by_id[global_id] = unit
        unit_order.append(global_id)
        units_by_source[source_path].append(unit)
    if unit_order != sorted(unit_order):
        raise ConditioningSplitError("global plan units are not canonical")

    for source_path, source in sources.items():
        source_units = units_by_source.get(source_path, [])
        if (
            sorted(unit["source_env_index"] for unit in source_units)
            != list(range(source["num_envs"]))
            or sum(unit["transition_count"] for unit in source_units)
            != source["transition_count"]
            or sum(unit["episode_count"] for unit in source_units)
            != source["episode_count"]
            or sum(unit["image_count"] for unit in source_units)
            != source["image_count"]
        ):
            raise ConditioningSplitError("global plan source coverage differs")
    return by_id


def _expected_files(manifest: Mapping[str, Any], *, sealed: bool) -> set[str]:
    files = {"manifest.json"}
    if sealed:
        files.add("seal.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ConditioningSplitError("split artifact inventory must be a list")
    for row in artifacts:
        if not isinstance(row, Mapping) or set(row) != _SPLIT_ARTIFACT_KEYS:
            raise ConditioningSplitError("split artifact descriptor fields differ")
        files.add(row["manifest_path"])
        files.add(row["payload_path"])
    return files


def _verify_visible_tree(root: Path, expected_files: set[str], label: str) -> None:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        _require_safe_tree(path)
        relative = path.relative_to(root).as_posix()
        if path.name.startswith("."):
            raise ConditioningSplitError(f"hidden entry in {label}: {path}")
        info = os.stat(path, follow_symlinks=False)
        if stat.S_ISREG(info.st_mode):
            files.add(relative)
        elif stat.S_ISDIR(info.st_mode):
            directories.add(relative)
        else:
            raise ConditioningSplitError(f"special filesystem entry in {label}: {path}")

    expected_directories: set[str] = set()
    for relative in expected_files:
        candidate = Path(relative)
        if candidate.is_absolute() or any(
            part in {"", ".", ".."} for part in candidate.parts
        ):
            raise ConditioningSplitError(f"unsafe expected path in {label}: {relative}")
        expected_directories.update(
            parent.as_posix() for parent in candidate.parents if parent != Path(".")
        )
    if files != expected_files or directories != expected_directories:
        raise ConditioningSplitError(f"{label} filesystem inventory differs")


def _verify_manifest_artifacts(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_transitions_per_group: int | None = None,
    expected_units_by_id: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ConditioningSplitError("split artifact inventory must be a list")
    global_ids: list[int] = []
    episode_count = 0
    transition_count = 0
    image_count = 0
    for row in artifacts:
        if not isinstance(row, Mapping) or set(row) != _SPLIT_ARTIFACT_KEYS:
            raise ConditioningSplitError("split artifact descriptor fields differ")
        global_id = row["global_env_id"]
        expected_group_id = (
            f"global-env-{global_id:012d}" if type(global_id) is int else None
        )
        expected_unit = (
            expected_units_by_id.get(global_id)
            if expected_units_by_id is not None and type(global_id) is int
            else None
        )
        if (
            type(global_id) is not int
            or global_id < 0
            or row["group_id"] != expected_group_id
            or row["manifest_path"] != f"artifacts/{expected_group_id}.json"
            or row["payload_path"] != f"artifacts/{expected_group_id}.npz"
            or not isinstance(row["episode_ids"], list)
            or any(
                type(episode_id) is not int or episode_id < 0
                for episode_id in row["episode_ids"]
            )
            or row["episode_ids"] != sorted(set(row["episode_ids"]))
            or (
                expected_units_by_id is not None
                and (
                    expected_unit is None
                    or any(
                        row[key] != expected_unit[key]
                        for key in (
                            "group_id",
                            "global_env_id",
                            "episode_ids",
                            "episode_count",
                            "transition_count",
                            "image_count",
                            "strata",
                            "source_manifest_path",
                            "source_manifest_sha256",
                        )
                    )
                )
            )
        ):
            raise ConditioningSplitError("split artifact identity/path differs")
        manifest_path = root / row["manifest_path"]
        payload_path = root / row["payload_path"]
        manifest_size, manifest_sha = _immutable_regular(
            manifest_path,
            "split artifact manifest",
            exact_mode=0o444,
        )
        payload_size, payload_sha = _immutable_regular(
            payload_path,
            "split artifact payload",
            exact_mode=0o444,
        )
        report, arrays = load_raw_conditioning_artifact(manifest_path)
        if report["purpose"] != manifest["split"]:
            raise ConditioningSplitError("artifact purpose differs from split")
        if report["num_envs"] != 1:
            raise ConditioningSplitError(
                "materialized split artifact must contain exactly one environment"
            )
        if expected_transitions_per_group is not None and (
            type(row.get("transition_count")) is not int
            or row["transition_count"] != expected_transitions_per_group
            or report["num_steps"] != expected_transitions_per_group
            or report["transition_count"] != expected_transitions_per_group
        ):
            raise ConditioningSplitError(
                "split artifact transition count differs from the exact per-environment target"
            )
        if manifest_sha != row["manifest_sha256"] or manifest_size <= 0:
            raise ConditioningSplitError("artifact manifest hash differs")
        if (
            report["payload"]["sha256"] != row["payload_sha256"]
            or payload_sha != row["payload_sha256"]
            or payload_size != row["payload_size_bytes"]
            or report["payload"]["size_bytes"] != row["payload_size_bytes"]
            or Path(report["payload_path"]) != payload_path
        ):
            raise ConditioningSplitError("artifact payload hash differs")
        if (
            report["episode_count"] != row["episode_count"]
            or report["transition_count"] != row["transition_count"]
            or report["image_count"] != row["image_count"]
        ):
            raise ConditioningSplitError("artifact count evidence differs")
        split_metadata = report["render"]["metadata"].get("bc_collection_split")
        if (
            not isinstance(split_metadata, Mapping)
            or set(split_metadata) != _EMBEDDED_SPLIT_KEYS
            or split_metadata.get("schema") != PLAN_SCHEMA
            or split_metadata.get("plan_sha256") != manifest["plan_sha256"]
            or split_metadata.get("split") != manifest["split"]
            or split_metadata.get("seed") != SPLIT_SEED
            or split_metadata.get("grouping") != "complete_global_environment_stream"
            or split_metadata.get("global_env_id") != global_id
            or split_metadata.get("source_manifest_path") != row["source_manifest_path"]
            or split_metadata.get("source_manifest_sha256")
            != row["source_manifest_sha256"]
            or (
                expected_unit is not None
                and (
                    split_metadata.get("source_payload_sha256")
                    != expected_unit["source_payload_sha256"]
                    or split_metadata.get("source_env_index")
                    != expected_unit["source_env_index"]
                )
            )
        ):
            raise ConditioningSplitError("artifact split/source provenance differs")
        ids = sorted(set(int(value) for value in arrays["global_env_ids"].ravel()))
        if ids != [global_id]:
            raise ConditioningSplitError("artifact contains a mixed environment group")
        actual_episodes = sorted(
            set(int(value) for value in arrays["episode_ids"].ravel())
        )
        if actual_episodes != row["episode_ids"]:
            raise ConditioningSplitError("artifact episode coverage differs")
        global_ids.extend(ids)
        episode_count += report["episode_count"]
        transition_count += report["transition_count"]
        image_count += report["image_count"]
    if len(global_ids) != len(set(global_ids)):
        raise ConditioningSplitError(
            "split contains duplicate global environment groups"
        )
    if sorted(global_ids) != manifest["global_env_ids"]:
        raise ConditioningSplitError("split global environment coverage differs")
    if (
        len(global_ids) != manifest["group_count"]
        or episode_count != manifest["episode_count"]
        or transition_count != manifest["transition_count"]
        or image_count != manifest["image_count"]
    ):
        raise ConditioningSplitError("split count totals differ")
    strata_counts = dict(
        sorted(Counter(_stratum_key(row["strata"]) for row in artifacts).items())
    )
    if manifest.get("strata_counts") != strata_counts:
        raise ConditioningSplitError("split stratum totals differ")
    return {
        "group_count": len(global_ids),
        "episode_count": episode_count,
        "transition_count": transition_count,
        "image_count": image_count,
        "global_env_ids": sorted(global_ids),
    }


def verify_split(
    split_root: str | os.PathLike[str],
    *,
    consumer: str | None = None,
    expected_transitions_per_group: int | None = None,
    expected_source_evidence: Mapping[str, Any] | None = None,
    expected_postprocess_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recursively verify a sealed split and enforce consumer isolation."""

    root = _absolute(split_root)
    if expected_transitions_per_group is not None and (
        type(expected_transitions_per_group) is not int
        or expected_transitions_per_group <= 0
    ):
        raise ConditioningSplitError(
            "expected transitions per environment must be a positive integer"
        )
    _require_safe_tree(root)
    try:
        _, manifest_sha = _immutable_regular(
            root / "manifest.json", "split manifest", exact_mode=0o444
        )
        _, seal_sha = _immutable_regular(
            root / "seal.json", "split seal", exact_mode=0o444
        )
        manifest = load_strict_json(root / "manifest.json")
        seal = load_strict_json(root / "seal.json")
    except ArtifactError as exc:
        raise ConditioningSplitError(f"cannot load split manifest/seal: {exc}") from exc
    split = manifest.get("split") if isinstance(manifest, Mapping) else None
    expected_access = (
        "allowed_after_roborender_projection" if split == "train" else "forbidden"
    )
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != _SPLIT_MANIFEST_KEYS
        or split not in SPLITS
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("generation_id") != GENERATION_ID
        or manifest.get("purpose") != split
        or manifest.get("seed") != SPLIT_SEED
        or manifest.get("strategy") != STRATEGY
        or manifest.get("grouping") != "complete_global_environment_stream"
        or manifest.get("weight_semantics")
        != "post_collection_global_environment_stream_quota_not_motion_holdout"
        or manifest.get("learner_replay_ready") is not False
        or manifest.get("training_access") != expected_access
    ):
        raise ConditioningSplitError("split manifest identity differs")
    if consumer in {"training", "renderer"} and split != "train":
        raise ConditioningSplitError(
            f"{consumer} cannot consume validation or test data"
        )
    if consumer not in {None, "training", "validation", "test", "renderer"}:
        raise ConditioningSplitError("unknown split consumer")
    if consumer in {"validation", "test"} and split != consumer:
        raise ConditioningSplitError(f"{consumer} consumer received the wrong split")
    if (
        not isinstance(seal, Mapping)
        or set(seal) != _SPLIT_SEAL_KEYS
        or seal.get("schema") != SEAL_SCHEMA
        or seal.get("campaign_id") != CAMPAIGN_ID
        or seal.get("generation_id") != GENERATION_ID
        or seal.get("split") != split
        or seal.get("sealed") is not True
        or seal.get("learner_replay_ready") is not False
        or seal.get("training_access") != expected_access
    ):
        raise ConditioningSplitError("split seal identity differs")
    if seal.get("manifest_sha256") != manifest_sha:
        raise ConditioningSplitError("split seal does not bind its manifest")
    try:
        campaign_root = root.parents[1]
    except IndexError as exc:  # pragma: no cover - absolute paths have ancestors
        raise ConditioningSplitError("split root has no campaign ancestor") from exc
    plan_path = campaign_root / "BC/contracts/bc_collection_split.v1.json"
    if (
        root != _output_root(campaign_root, split)
        or manifest.get("plan_path") != SEALED_BC_PLAN_PATH_FIELD
    ):
        raise ConditioningSplitError(
            "split root/plan is outside the canonical campaign layout"
        )
    try:
        _, plan_sha = _immutable_regular(
            plan_path, "global split plan", exact_mode=0o444
        )
        plan = load_strict_json(plan_path)
    except (ArtifactError, OSError) as exc:
        raise ConditioningSplitError(f"cannot verify global split plan: {exc}") from exc
    assignments = plan.get("assignments") if isinstance(plan, Mapping) else None
    source_inventory = (
        plan.get("source_inventory") if isinstance(plan, Mapping) else None
    )
    units = plan.get("units") if isinstance(plan, Mapping) else None
    provenance = (
        plan.get("postprocess_provenance") if isinstance(plan, Mapping) else None
    )
    plan_source_root = plan.get("source_root") if isinstance(plan, Mapping) else None
    if (
        not isinstance(plan, Mapping)
        or set(plan) != _PLAN_KEYS
        or manifest.get("plan_sha256") != plan_sha
        or seal.get("plan_sha256") != plan_sha
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("generation_id") != GENERATION_ID
        or plan_source_root
        not in {
            SOURCE_RELATIVE_ROOT.as_posix(),
            FROZEN_BC_SOURCE_RELATIVE_ROOT.as_posix(),
        }
        or plan.get("seed") != SPLIT_SEED
        or plan.get("strategy") != STRATEGY
        or plan.get("grouping") != "complete_global_environment_stream"
        or plan.get("ratio_source") != "native_h200_experiment_protocol.group_split"
        or plan.get("weight_semantics")
        != "post_collection_global_environment_stream_quota_not_motion_holdout"
        or not isinstance(assignments, Mapping)
        or set(assignments) != set(SPLITS)
        or any(not isinstance(assignments[candidate], list) for candidate in SPLITS)
        or not isinstance(source_inventory, list)
        or plan.get("source_inventory_sha256")
        != canonical_json_sha256(source_inventory)
        or not isinstance(units, list)
        or plan.get("assignment_sha256")
        != canonical_json_sha256(
            {candidate: assignments[candidate] for candidate in SPLITS}
        )
        or (
            expected_transitions_per_group is not None
            and plan.get("expected_transitions_per_group")
            != expected_transitions_per_group
        )
        or assignments.get(split) != manifest.get("global_env_ids")
        or plan.get("target_counts")
        != {candidate: len(assignments[candidate]) for candidate in SPLITS}
    ):
        raise ConditioningSplitError(
            "split manifest/seal does not bind the global plan"
        )
    units_by_id = _validate_plan_source_bindings(source_inventory, units)
    if not isinstance(plan_source_root, str):  # guarded above; keep -O safe
        raise ConditioningSplitError("global plan source root is malformed")
    if any(
        not _is_strict_relative_descendant(
            source.get("manifest_path") if isinstance(source, Mapping) else None,
            Path(plan_source_root),
        )
        for source in source_inventory
    ):
        raise ConditioningSplitError(
            "global plan source inventory escapes its declared root"
        )
    if {candidate: list(assignments[candidate]) for candidate in SPLITS} != _assign(
        units, plan["target_counts"], SPLIT_SEED
    ):
        raise ConditioningSplitError(
            "global split plan assignment is not deterministic"
        )
    if provenance is not None:
        provenance = _validate_postprocess_provenance(provenance)
    if expected_postprocess_provenance is not None and provenance != dict(
        expected_postprocess_provenance
    ):
        raise ConditioningSplitError("split plan postprocess provenance differs")
    if expected_source_evidence is not None:
        completion = plan.get("source_completion")
        if (
            not isinstance(completion, Mapping)
            or completion.get("sha256")
            != expected_source_evidence.get("saturation_manifest_sha256")
            or plan.get("source_inventory_sha256")
            != expected_source_evidence.get("source_inventory_sha256")
            or len(units) != expected_source_evidence.get("source_group_count")
            or plan.get("target_counts")
            != expected_source_evidence.get("target_counts")
            or plan.get("weights") != expected_source_evidence.get("weights")
        ):
            raise ConditioningSplitError("split plan source evidence differs")
    all_assignment_ids = [
        value for candidate in SPLITS for value in assignments[candidate]
    ]
    unit_ids = [
        unit.get("global_env_id") if isinstance(unit, Mapping) else None
        for unit in units
    ]
    if (
        any(type(value) is not int for value in all_assignment_ids)
        or any(type(value) is not int for value in unit_ids)
        or len(all_assignment_ids) != len(set(all_assignment_ids))
        or sorted(all_assignment_ids) != sorted(unit_ids)
        or (
            expected_source_evidence is not None
            and (
                unit_ids != list(range(EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS))
                or any(
                    not isinstance(unit, Mapping)
                    or unit.get("transition_count")
                    != EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV
                    for unit in units
                )
            )
        )
    ):
        raise ConditioningSplitError("global split plan coverage differs")
    expected = _expected_files(manifest, sealed=True)
    _verify_visible_tree(root, expected, "sealed split")
    totals = _verify_manifest_artifacts(
        root,
        manifest,
        expected_transitions_per_group=expected_transitions_per_group,
        expected_units_by_id=units_by_id,
    )
    for key in (
        "group_count",
        "episode_count",
        "transition_count",
        "image_count",
        "global_env_ids",
    ):
        if seal.get(key) != totals[key]:
            raise ConditioningSplitError(f"split seal {key} differs")
    expected_manifest_digests = [
        row["manifest_sha256"] for row in manifest["artifacts"]
    ]
    expected_payload_digests = [row["payload_sha256"] for row in manifest["artifacts"]]
    if (
        seal["artifact_manifest_sha256"] != expected_manifest_digests
        or seal["artifact_payload_sha256"] != expected_payload_digests
    ):
        raise ConditioningSplitError("split seal artifact digest inventory differs")
    return {
        **seal,
        "root": str(root),
        "split": split,
        "seal_sha256": seal_sha,
        "verified_plan_sha256": plan_sha,
    }


def verify_training_projection_source(
    split_root: str | os.PathLike[str],
    *,
    campaign_root: str | os.PathLike[str] = CANONICAL_CAMPAIGN_ROOT,
) -> dict[str, Any]:
    """Admit only the canonical sealed train split to BC render projection."""

    campaign = _absolute(campaign_root)
    root = _absolute(split_root)
    expected = _output_root(campaign, "train")
    if root != expected:
        raise ConditioningSplitError(
            "training render projection requires the canonical sealed train split"
        )
    source_evidence: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None
    if campaign == CANONICAL_CAMPAIGN_ROOT:
        protocol = Path(__file__).resolve().parents[1] / PROTOCOL_RELATIVE_PATH
        source_evidence = verify_exact_bc_collection_source(
            campaign,
            protocol_path=protocol,
        )
        plan_path = campaign / "BC/contracts/bc_collection_split.v1.json"
        try:
            _immutable_regular(
                plan_path, "global split plan", exact_mode=0o444
            )
            plan = load_strict_json(plan_path)
        except ArtifactError as exc:
            raise ConditioningSplitError(
                f"cannot load sealed split-plan provenance: {exc}"
            ) from exc
        provenance = _validate_postprocess_provenance(
            plan.get("postprocess_provenance")
            if isinstance(plan, Mapping)
            else None
        )
    return verify_split(
        root,
        consumer="renderer",
        expected_transitions_per_group=EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV,
        expected_source_evidence=source_evidence,
        expected_postprocess_provenance=provenance,
    )


def _validate_exact_bc_prepublication(
    *,
    units: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    targets: Mapping[str, int],
    weights: Mapping[str, int],
    completion: Mapping[str, Any],
) -> dict[str, int]:
    """Reject any noncanonical collection before a create-only path is written."""

    global_ids = [int(unit["global_env_id"]) for unit in units]
    malformed = [
        global_id
        for global_id, unit in zip(global_ids, units, strict=True)
        if unit.get("transition_count") != EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV
    ]
    source_counts = {
        "episode_count": sum(int(unit["episode_count"]) for unit in units),
        "transition_count": sum(int(unit["transition_count"]) for unit in units),
        "image_count": sum(int(unit["image_count"]) for unit in units),
    }
    completion_counts = {
        "episode_count": completion.get("completed_episode_count"),
        "transition_count": completion.get("completed_transitions"),
        "image_count": completion.get("completed_image_count"),
    }
    if (
        len(units) != EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS
        or global_ids != list(range(EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS))
        or malformed
        or dict(weights) != {"train": 23, "validation": 5, "test": 5}
        or dict(targets) != EXPECTED_BC_GROUP_COUNTS
        or source_counts["transition_count"] != EXPECTED_BC_NOMINAL_TRANSITIONS
        or source_counts != completion_counts
        or len(sources) == 0
        or len(completion.get("cohorts", ())) != EXPECTED_BC_WAVE_COUNT
        or len(completion.get("physical_gpu_inventory") or ()) != 8
    ):
        raise ConditioningSplitError(
            "BC source differs from exact 8xH200-inventory, one-actor, "
            "two-wave 256x512, IDs 0..255, 178/39/39 prepublication contract"
        )
    return source_counts


def verify_exact_bc_collection_source(
    campaign_root: str | os.PathLike[str],
    *,
    protocol_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Deeply reverify the canonical unsplit collection without publishing."""

    root = _absolute(campaign_root)
    if root != CANONICAL_CAMPAIGN_ROOT:
        raise ConditioningSplitError("campaign root differs from canonical data root")
    _require_safe_tree(root)
    if not root.is_dir():
        raise ConditioningSplitError("campaign root does not exist")
    protocol = _absolute(protocol_path)
    _require_exact_protocol(protocol)
    provenance = build_postprocess_provenance(
        Path(__file__).resolve().parents[1], protocol
    )
    weights, ratio_source = _load_weights(protocol)
    manifests, completion, source_identity_root = _sealed_source_manifests(
        root,
        root / SOURCE_RELATIVE_ROOT,
        require_exact_production=True,
    )
    units, sources = _inventory(
        root / SOURCE_RELATIVE_ROOT,
        source_identity_root,
        manifests,
    )
    targets = _target_counts(len(units), weights)
    source_counts = _validate_exact_bc_prepublication(
        units=units,
        sources=sources,
        targets=targets,
        weights=weights,
        completion=completion,
    )
    return {
        "saturation_manifest_sha256": completion["sha256"],
        "source_inventory_sha256": canonical_json_sha256(list(sources)),
        "source_group_count": len(units),
        **source_counts,
        "target_counts": dict(targets),
        "weights": dict(weights),
        "ratio_source": ratio_source,
        "postprocess_provenance": provenance,
    }


def _split_collection_unlocked(
    campaign_root: str | os.PathLike[str],
    *,
    protocol_path: str | os.PathLike[str] | None,
    seed: int = SPLIT_SEED,
    enforce_canonical_root: bool = False,
    expected_transitions_per_group: int | None = None,
    postprocess_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify all source shards, materialize the split, and seal exact coverage."""

    root = _absolute(campaign_root)
    if enforce_canonical_root and root != CANONICAL_CAMPAIGN_ROOT:
        raise ConditioningSplitError("campaign root differs from canonical data root")
    exact_production = root == CANONICAL_CAMPAIGN_ROOT
    if type(seed) is not int or seed != SPLIT_SEED:
        raise ConditioningSplitError("BC collection split seed must be exactly 42")
    if expected_transitions_per_group is not None and (
        type(expected_transitions_per_group) is not int
        or expected_transitions_per_group <= 0
    ):
        raise ConditioningSplitError(
            "expected transitions per environment must be a positive integer"
        )
    _require_safe_tree(root)
    if not root.is_dir():
        raise ConditioningSplitError("campaign root does not exist")
    protocol = None if protocol_path is None else _absolute(protocol_path)
    if exact_production:
        if (
            protocol is None
            or expected_transitions_per_group != EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV
        ):
            raise ConditioningSplitError(
                "canonical BC split requires the pinned protocol and 512 transitions"
            )
        _require_exact_protocol(protocol)
        current_provenance = build_postprocess_provenance(
            Path(__file__).resolve().parents[1], protocol
        )
        if (
            postprocess_provenance is not None
            and dict(postprocess_provenance) != current_provenance
        ):
            raise ConditioningSplitError("caller postprocess provenance differs")
        postprocess_provenance = current_provenance
    elif postprocess_provenance is not None:
        postprocess_provenance = _validate_postprocess_provenance(
            postprocess_provenance
        )
    weights, ratio_source = _load_weights(protocol)
    source_root = root / SOURCE_RELATIVE_ROOT
    manifest_paths, completion, source_identity_root = _sealed_source_manifests(
        root,
        source_root,
        require_exact_production=exact_production,
    )
    units, sources = _inventory(
        source_root,
        source_identity_root,
        manifest_paths,
    )
    if expected_transitions_per_group is not None:
        malformed = [
            int(unit["global_env_id"])
            for unit in units
            if unit["transition_count"] != expected_transitions_per_group
        ]
        if malformed:
            raise ConditioningSplitError(
                "source global environment transition count differs from exact target; "
                f"global_env_ids={malformed}"
            )
    targets = _target_counts(len(units), weights)
    if exact_production:
        _validate_exact_bc_prepublication(
            units=units,
            sources=sources,
            targets=targets,
            weights=weights,
            completion=completion,
        )
    assignments = _assign(units, targets, seed)
    plan = _plan_document(
        campaign_root=root,
        units=units,
        sources=sources,
        assignments=assignments,
        weights=weights,
        ratio_source=ratio_source,
        seed=seed,
        completion=completion,
        source_identity_root=source_identity_root,
        expected_transitions_per_group=expected_transitions_per_group,
        postprocess_provenance=postprocess_provenance,
    )
    plan_path = root / "BC/contracts/bc_collection_split.v1.json"
    try:
        published_plan_sha = publish_immutable_json(plan_path, plan)
    except ArtifactError as exc:
        raise ConditioningSplitError(f"cannot publish split plan: {exc}") from exc
    if published_plan_sha != sha256_file(plan_path):
        raise ConditioningSplitError("published split plan hash differs")
    _immutable_regular(plan_path, "published split plan", exact_mode=0o444)
    entries = _materialize(
        root,
        source_root,
        source_identity_root,
        units,
        assignments,
        published_plan_sha,
    )

    source_evidence = {
        "saturation_manifest_sha256": completion["sha256"],
        "source_inventory_sha256": canonical_json_sha256(list(sources)),
        "source_group_count": len(units),
        "target_counts": dict(targets),
        "weights": dict(weights),
    }
    verified_source_evidence = source_evidence if exact_production else None

    reports: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_root = _output_root(root, split)
        manifest = _split_manifest(split, plan, published_plan_sha, entries[split])
        try:
            manifest_sha = publish_immutable_json(
                split_root / "manifest.json", manifest
            )
        except ArtifactError as exc:
            raise ConditioningSplitError(
                f"cannot publish {split} manifest: {exc}"
            ) from exc
        _, verified_manifest_sha = _immutable_regular(
            split_root / "manifest.json",
            f"published {split} manifest",
            exact_mode=0o444,
        )
        if verified_manifest_sha != manifest_sha:
            raise ConditioningSplitError(f"published {split} manifest hash differs")
        seal_path = split_root / "seal.json"
        already_sealed = seal_path.exists()
        _verify_visible_tree(
            split_root,
            _expected_files(manifest, sealed=already_sealed),
            f"unsealed {split} split",
        )
        if already_sealed:
            reports[split] = verify_split(
                split_root,
                expected_transitions_per_group=expected_transitions_per_group,
                expected_source_evidence=verified_source_evidence,
                expected_postprocess_provenance=postprocess_provenance,
            )
            continue
        totals = _verify_manifest_artifacts(
            split_root,
            manifest,
            expected_transitions_per_group=expected_transitions_per_group,
        )
        seal = {
            "schema": SEAL_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "generation_id": GENERATION_ID,
            "split": split,
            "sealed": True,
            "manifest_sha256": manifest_sha,
            "plan_sha256": published_plan_sha,
            **totals,
            "artifact_manifest_sha256": [
                row["manifest_sha256"] for row in entries[split]
            ],
            "artifact_payload_sha256": [
                row["payload_sha256"] for row in entries[split]
            ],
            "learner_replay_ready": False,
            "training_access": (
                "allowed_after_roborender_projection"
                if split == "train"
                else "forbidden"
            ),
        }
        try:
            publish_immutable_json(seal_path, seal)
        except ArtifactError as exc:
            raise ConditioningSplitError(f"cannot publish {split} seal: {exc}") from exc
        reports[split] = verify_split(
            split_root,
            expected_transitions_per_group=expected_transitions_per_group,
            expected_source_evidence=verified_source_evidence,
            expected_postprocess_provenance=postprocess_provenance,
        )

    all_ids = [value for split in SPLITS for value in reports[split]["global_env_ids"]]
    source_ids = sorted(int(unit["global_env_id"]) for unit in units)
    if len(all_ids) != len(set(all_ids)) or sorted(all_ids) != source_ids:
        raise ConditioningSplitError(
            "sealed train/validation/test sets overlap or lack coverage"
        )
    source_counts = {
        "episode_count": sum(int(unit["episode_count"]) for unit in units),
        "transition_count": sum(int(unit["transition_count"]) for unit in units),
        "image_count": sum(int(unit["image_count"]) for unit in units),
    }
    split_counts = {
        key: sum(int(reports[split][key]) for split in SPLITS) for key in source_counts
    }
    completion_counts = {
        "episode_count": int(completion["completed_episode_count"]),
        "transition_count": int(completion["completed_transitions"]),
        "image_count": int(completion["completed_image_count"]),
    }
    if split_counts != source_counts:
        raise ConditioningSplitError(
            "sealed train/validation/test counts differ from source coverage"
        )
    if source_counts != completion_counts:
        raise ConditioningSplitError(
            "source counts differ from generation saturation completion"
        )
    return {
        "schema": SEAL_SCHEMA,
        "campaign_root": str(root),
        "plan_path": str(plan_path),
        "plan_sha256": published_plan_sha,
        "seed": seed,
        "weights": weights,
        "ratio_source": ratio_source,
        "source_group_count": len(units),
        "source_episode_count": source_counts["episode_count"],
        "source_transition_count": source_counts["transition_count"],
        "source_image_count": source_counts["image_count"],
        "expected_transitions_per_group": expected_transitions_per_group,
        "splits": reports,
        "exact_coverage": True,
        "pairwise_disjoint": True,
    }


def split_collection(
    campaign_root: str | os.PathLike[str],
    *,
    protocol_path: str | os.PathLike[str] | None,
    seed: int = SPLIT_SEED,
    enforce_canonical_root: bool = False,
    expected_transitions_per_group: int | None = None,
    postprocess_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a split, owning the canonical generation lock when it can publish."""

    root = _absolute(campaign_root)
    arguments = {
        "protocol_path": protocol_path,
        "seed": seed,
        "enforce_canonical_root": enforce_canonical_root,
        "expected_transitions_per_group": expected_transitions_per_group,
        "postprocess_provenance": postprocess_provenance,
    }
    if root == CANONICAL_CAMPAIGN_ROOT:
        with canonical_generation_lock():
            return _split_collection_unlocked(root, **arguments)
    return _split_collection_unlocked(root, **arguments)


__all__ = [
    "CANONICAL_CAMPAIGN_ROOT",
    "ConditioningSplitError",
    "EXPECTED_BC_ACTOR_WORLD_SIZE",
    "EXPECTED_BC_ENVIRONMENTS_PER_ACTOR",
    "EXPECTED_BC_ENVIRONMENTS_PER_GPU",
    "EXPECTED_BC_GROUP_COUNTS",
    "EXPECTED_BC_NOMINAL_TRANSITIONS",
    "EXPECTED_BC_PROTOCOL_SHA256",
    "EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS",
    "EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV",
    "EXPECTED_BC_TRANSITIONS_PER_WAVE",
    "EXPECTED_BC_WAVE_COUNT",
    "PRE_RENDER_AUDIT_SCHEMA",
    "POSTPROCESS_PROVENANCE_SCHEMA",
    "SPLIT_SEED",
    "SPLITS",
    "audit_pre_render_replay_root",
    "build_postprocess_provenance",
    "canonical_generation_lock",
    "generation_lock",
    "split_collection",
    "verify_exact_bc_collection_source",
    "verify_split",
    "verify_training_projection_source",
]
