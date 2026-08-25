#!/usr/bin/env python3
"""Immutable actor-side artifacts for round-based RoboRender DAgger.

The RTX actor publishes one deterministic NPZ payload followed by a small JSON
sidecar.  Consumers discover sidecars, never bare payloads, so a process dying
between those two publications cannot expose an incomplete shard.  A cohort is
visible to the learner only after its immutable seal proves exact rank, logical
environment, and transition coverage.

This module deliberately depends only on the Python standard library and
NumPy.  It does not import Isaac, Torch, GR00T, or the learner stack.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np


COHORT_CONTRACT_SCHEMA = "fast_roborender.round_dagger.cohort_contract.v1"
SHARD_SCHEMA = "fast_roborender.round_dagger.rollout_shard.v1"
COHORT_SCHEMA = "fast_roborender.round_dagger.cohort.v1"

CONTRACT_FILENAME = "contract.json"
SEAL_FILENAME = "seal.json"
SHARD_DIRECTORY = "shards"

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ARRAY_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")

# Actor containers commonly run as root while the transfer and learner stages
# run as the submitting user.  mkstemp(3) deliberately creates mode 0600
# files, which made otherwise valid cohorts unreadable after the container
# exited.  Published artifacts are content-addressed and create-only, so make
# their final inode read-only and readable by every handoff stage.
_PUBLISHED_FILE_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH

_DIGEST_FIELDS = (
    "policy_sha256",
    "runtime_policy_sha256",
    "teacher_sha256",
    "renderer_sha256",
    "beta_sha256",
    "config_sha256",
)

_REQUIRED_ARRAYS = {
    "policy_rgb",
    "expert_actions",
    "student_actions",
    "behavior_actions",
    "expert_mask",
    "dones",
    "observation_is_raytraced",
    "global_env_ids",
    "episode_ids",
    "episode_steps",
}

_CONTRACT_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "cohort_index",
    "purpose",
    "expected_ranks",
    "expected_global_env_ids",
    "expected_transitions",
    "expected_steps_per_env",
    "view_contract",
    "action_contract",
    *_DIGEST_FIELDS,
}

_SHARD_KEYS = {
    "schema",
    "shard_id",
    "campaign_id",
    "macro_round",
    "cohort_index",
    "purpose",
    "rank",
    "rollout_id",
    "part_index",
    "cohort_contract_sha256",
    "transition_count",
    "num_steps",
    "num_envs",
    "global_env_ids",
    "view_contract",
    "action_contract",
    "arrays",
    "payload",
    *_DIGEST_FIELDS,
}

_SEAL_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "cohort_index",
    "purpose",
    "logical_id",
    "sealed",
    "transition_count",
    "view_contract",
    "action_contract",
    "expected_ranks",
    "expected_global_env_ids",
    "expected_transitions",
    "ranks",
    "global_env_ids",
    "contract_sha256",
    "shards",
    *_DIGEST_FIELDS,
}

_SEAL_SHARD_KEYS = {
    "shard_id",
    "manifest_path",
    "manifest_sha256",
    "payload_path",
    "payload_sha256",
    "transition_count",
    "rank",
    "rollout_id",
    "part_index",
    "global_env_ids",
}


class ArtifactError(RuntimeError):
    """An artifact is malformed, incomplete, conflicting, or unsafe."""


def _canonical_json_bytes(document: Any) -> bytes:
    _validate_json_value(document, "document")
    try:
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"document is not canonical JSON: {exc}") from exc


def canonical_json_sha256(document: Any) -> str:
    """Return the SHA-256 of the canonical, whitespace-free JSON encoding."""

    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _validate_json_value(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ArtifactError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactError(f"{label} contains a non-string JSON key")
            _validate_json_value(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{label}[{index}]")
        return
    raise ArtifactError(f"{label} contains unsupported JSON type {type(value).__name__}")


def _reject_constant(value: str) -> None:
    raise ArtifactError(f"JSON contains forbidden non-finite constant {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _check_path_chain(path: Path) -> None:
    """Reject every existing symlink or non-directory ancestor of *path*."""

    path = _absolute(path)
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(parts):
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ArtifactError(f"symbolic links are forbidden: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ArtifactError(f"path ancestor is not a directory: {current}")


def _ensure_directory(path: Path) -> Path:
    path = _absolute(path)
    _check_path_chain(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactError(f"cannot create artifact directory {path}: {exc}") from exc
    _check_path_chain(path)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ArtifactError(f"cannot inspect artifact directory {path}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ArtifactError(f"artifact directory is not a regular directory: {path}")
    return path


def _open_regular_read(path: Path) -> BinaryIO:
    path = _absolute(path)
    _check_path_chain(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactError(f"cannot open regular artifact {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactError(f"artifact is not a regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a regular file without following a symbolic link."""

    digest = hashlib.sha256()
    with _open_regular_read(_absolute(path)) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strict_json(path: str | os.PathLike[str]) -> Any:
    """Load UTF-8 JSON while rejecting duplicate keys, NaN, and symlinks."""

    target = _absolute(path)
    try:
        with _open_regular_read(target) as handle:
            raw = handle.read()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON artifact {target}: {exc}") from exc
    _validate_json_value(document, str(target))
    return document


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_file_bytes(first: Path, second: Path) -> bool:
    try:
        first_stat = os.stat(first, follow_symlinks=False)
        second_stat = os.stat(second, follow_symlinks=False)
    except OSError:
        return False
    if not stat.S_ISREG(first_stat.st_mode) or not stat.S_ISREG(second_stat.st_mode):
        return False
    return first_stat.st_size == second_stat.st_size and sha256_file(first) == sha256_file(second)


def _publish_temp_create_only(temporary: Path, target: Path) -> str:
    """Atomically link a fully fsynced same-directory temp into place."""

    target = _absolute(target)
    temporary = _absolute(temporary)
    if temporary.parent != target.parent:
        raise ArtifactError("atomic publication requires a same-directory temporary file")
    _check_path_chain(target)
    try:
        os.link(temporary, target, follow_symlinks=False)
        _fsync_directory(target.parent)
    except FileExistsError:
        if not _same_file_bytes(temporary, target):
            raise ArtifactError(f"immutable artifact conflict at {target}")
    except OSError as exc:
        if exc.errno == errno.EEXIST and _same_file_bytes(temporary, target):
            pass
        else:
            raise ArtifactError(f"cannot publish immutable artifact {target}: {exc}") from exc
    return sha256_file(target)


def _publish_bytes(path: Path, payload: bytes) -> str:
    target = _absolute(path)
    parent = _ensure_directory(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), _PUBLISHED_FILE_MODE)
            os.fsync(handle.fileno())
        return _publish_temp_create_only(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_immutable_json(
    path: str | os.PathLike[str], document: Mapping[str, Any]
) -> str:
    """Create a canonical JSON artifact, or accept an identical retry.

    Existing different bytes are never replaced.  The return value is the
    SHA-256 of the published file.
    """

    if not isinstance(document, Mapping):
        raise ArtifactError("immutable JSON document must be a mapping")
    return _publish_bytes(_absolute(path), _canonical_json_bytes(dict(document)))


# Compatibility spelling used by an early replay prototype.
atomic_publish_json = publish_immutable_json


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ArtifactError(f"{label} has missing={missing} unknown={unknown}")
    return dict(value)


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ArtifactError(f"{label} must be an integer >= {minimum}")
    return value


def _require_token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ArtifactError(f"{label} is not a safe token")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactError(f"{label} must be an exact lowercase SHA-256")
    return value


def _require_sorted_unique_ints(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ArtifactError(f"{label} must be a non-empty list")
    parsed = [_require_int(item, f"{label}[]") for item in value]
    if parsed != sorted(set(parsed)):
        raise ArtifactError(f"{label} must be sorted and unique")
    return parsed


def _safe_relative_path(value: Any, label: str, *, components: int | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactError(f"{label} is not a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ArtifactError(f"{label} is not a canonical relative POSIX path")
    if any(part in {"", ".", ".."} or _TOKEN_RE.fullmatch(part) is None for part in path.parts):
        raise ArtifactError(f"{label} is not a safe relative POSIX path")
    if components is not None and len(path.parts) != components:
        raise ArtifactError(f"{label} must contain exactly {components} path component(s)")
    return value


def _validate_view_contract(value: Any) -> dict[str, Any]:
    contract = _require_exact_keys(
        value, {"dtype", "layout", "shape", "views"}, "view_contract"
    )
    if contract["dtype"] != "uint8" or contract["layout"] != "vertical":
        raise ArtifactError("view_contract must use uint8 vertical layout")
    shape = contract["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(item) is not int or item <= 0 for item in shape)
        or shape[2] != 3
    ):
        raise ArtifactError("view_contract.shape must be positive [H,W,3]")
    views = contract["views"]
    if not isinstance(views, list) or len(views) != 3:
        raise ArtifactError("view_contract.views must contain exactly three views")
    parsed_views = [_require_token(view, "view_contract.views[]") for view in views]
    if len(set(parsed_views)) != 3:
        raise ArtifactError("view_contract.views must be unique")
    if shape[0] % 3:
        raise ArtifactError("vertical three-view image height must be divisible by three")
    return contract


def _validate_action_contract(value: Any) -> dict[str, Any]:
    contract = _require_exact_keys(value, {"dtype", "shape"}, "action_contract")
    try:
        dtype = np.dtype(contract["dtype"])
    except (TypeError, ValueError) as exc:
        raise ArtifactError("action_contract.dtype is invalid") from exc
    if not np.issubdtype(dtype, np.floating) or str(dtype) != contract["dtype"]:
        raise ArtifactError("action_contract.dtype must name a canonical floating dtype")
    shape = contract["shape"]
    if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0:
        raise ArtifactError("action_contract.shape must be [positive action_dim]")
    return contract


def _validate_contract(document: Any) -> dict[str, Any]:
    contract = _require_exact_keys(document, _CONTRACT_KEYS, "cohort contract")
    if contract["schema"] != COHORT_CONTRACT_SCHEMA:
        raise ArtifactError("cohort contract schema mismatch")
    _require_token(contract["campaign_id"], "campaign_id")
    _require_int(contract["macro_round"], "macro_round")
    _require_int(contract["cohort_index"], "cohort_index")
    if contract["purpose"] not in {"train", "eval"}:
        raise ArtifactError("purpose must be exactly 'train' or 'eval'")
    ranks = _require_sorted_unique_ints(contract["expected_ranks"], "expected_ranks")
    envs = _require_sorted_unique_ints(
        contract["expected_global_env_ids"], "expected_global_env_ids"
    )
    transitions = _require_int(
        contract["expected_transitions"], "expected_transitions", minimum=1
    )
    steps = _require_int(
        contract["expected_steps_per_env"], "expected_steps_per_env", minimum=1
    )
    if transitions != len(envs) * steps:
        raise ArtifactError(
            "expected_transitions must equal environment coverage times expected_steps_per_env"
        )
    if not ranks:
        raise ArtifactError("expected_ranks cannot be empty")
    _validate_view_contract(contract["view_contract"])
    _validate_action_contract(contract["action_contract"])
    for field in _DIGEST_FIELDS:
        _require_sha256(contract[field], field)
    return contract


def create_cohort(
    path: str | os.PathLike[str],
    *,
    campaign_id: str,
    macro_round: int,
    cohort_index: int,
    purpose: str,
    expected_ranks: Sequence[int],
    expected_global_env_ids: Sequence[int],
    expected_transitions: int,
    view_contract: Mapping[str, Any],
    action_contract: Mapping[str, Any],
    policy_sha256: str,
    runtime_policy_sha256: str,
    teacher_sha256: str,
    renderer_sha256: str,
    beta_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Create an immutable cohort contract and its empty shard directory."""

    cohort_dir = _ensure_directory(_absolute(path))
    raw_ranks = [_require_int(item, "expected_ranks[]") for item in expected_ranks]
    raw_envs = [
        _require_int(item, "expected_global_env_ids[]")
        for item in expected_global_env_ids
    ]
    if len(raw_ranks) != len(set(raw_ranks)):
        raise ArtifactError("expected_ranks cannot contain duplicates")
    if len(raw_envs) != len(set(raw_envs)):
        raise ArtifactError("expected_global_env_ids cannot contain duplicates")
    ranks = sorted(raw_ranks)
    envs = sorted(raw_envs)
    transitions = _require_int(expected_transitions, "expected_transitions", minimum=1)
    if not envs or transitions % len(envs):
        raise ArtifactError("expected_transitions must divide evenly across expected environments")
    contract = {
        "schema": COHORT_CONTRACT_SCHEMA,
        "campaign_id": _require_token(campaign_id, "campaign_id"),
        "macro_round": _require_int(macro_round, "macro_round"),
        "cohort_index": _require_int(cohort_index, "cohort_index"),
        "purpose": purpose,
        "expected_ranks": ranks,
        "expected_global_env_ids": envs,
        "expected_transitions": transitions,
        "expected_steps_per_env": transitions // len(envs),
        "view_contract": dict(view_contract),
        "action_contract": dict(action_contract),
        "policy_sha256": policy_sha256,
        "runtime_policy_sha256": runtime_policy_sha256,
        "teacher_sha256": teacher_sha256,
        "renderer_sha256": renderer_sha256,
        "beta_sha256": beta_sha256,
        "config_sha256": config_sha256,
    }
    contract = _validate_contract(contract)
    publish_immutable_json(cohort_dir / CONTRACT_FILENAME, contract)
    _ensure_directory(cohort_dir / SHARD_DIRECTORY)
    return contract


def _load_contract(cohort_dir: Path) -> tuple[dict[str, Any], str]:
    contract_path = cohort_dir / CONTRACT_FILENAME
    contract = _validate_contract(load_strict_json(contract_path))
    return contract, sha256_file(contract_path)


def _global_env_vector(array: np.ndarray, steps: int, envs: int) -> np.ndarray:
    if array.shape == (envs,):
        result = array
    elif array.shape == (steps, envs):
        if not np.all(array == array[0:1]):
            raise ArtifactError("global_env_ids must remain constant over the rollout")
        result = array[0]
    else:
        raise ArtifactError("global_env_ids must have shape [B] or constant [T,B]")
    if len(set(int(item) for item in result.tolist())) != envs:
        raise ArtifactError("global_env_ids must be unique within a shard")
    return result


def _finite_numeric(array: np.ndarray, label: str) -> None:
    if array.dtype.hasobject or not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ArtifactError(f"{label} must have a numeric or bool dtype")
    if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
        raise ArtifactError(f"{label} contains non-finite values")


def _validate_arrays(
    arrays: Mapping[str, np.ndarray], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(arrays, Mapping):
        raise ArtifactError("arrays must be a mapping")
    keys = set(arrays)
    missing = sorted(_REQUIRED_ARRAYS - keys)
    unknown = sorted(
        key
        for key in keys
        if key not in _REQUIRED_ARRAYS and not (isinstance(key, str) and key.startswith("obs__"))
    )
    if missing or unknown:
        raise ArtifactError(f"rollout arrays have missing={missing} unknown={unknown}")
    canonical: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or _ARRAY_KEY_RE.fullmatch(key) is None:
            raise ArtifactError(f"unsafe array key: {key!r}")
        if not isinstance(value, np.ndarray):
            raise ArtifactError(f"{key} must be a NumPy array")
        if value.dtype.hasobject:
            raise ArtifactError(f"{key} object arrays are forbidden")
        canonical[key] = np.ascontiguousarray(value)

    rgb = canonical["policy_rgb"]
    expected_image = tuple(contract["view_contract"]["shape"])
    if rgb.dtype != np.uint8 or rgb.ndim != 5 or tuple(rgb.shape[2:]) != expected_image:
        raise ArtifactError("policy_rgb must be uint8 [T,B,H,W,3] matching view_contract")
    steps, envs = int(rgb.shape[0]), int(rgb.shape[1])
    if steps <= 0 or envs <= 0:
        raise ArtifactError("rollout shard must contain positive T and B")
    if steps != contract["expected_steps_per_env"]:
        raise ArtifactError("rollout T does not match cohort expected_steps_per_env")

    action_shape = (steps, envs, int(contract["action_contract"]["shape"][0]))
    action_dtype = np.dtype(contract["action_contract"]["dtype"])
    for key in ("expert_actions", "student_actions", "behavior_actions"):
        array = canonical[key]
        if tuple(array.shape) != action_shape or array.dtype != action_dtype:
            raise ArtifactError(
                f"{key} must have dtype {action_dtype} and shape [T,B,A] {action_shape}"
            )
        if not np.isfinite(array).all():
            raise ArtifactError(f"{key} contains non-finite values")

    for key in ("expert_mask", "dones", "observation_is_raytraced"):
        array = canonical[key]
        if array.dtype != np.bool_ or tuple(array.shape) != (steps, envs):
            raise ArtifactError(f"{key} must be bool [T,B]")

    expected_behavior = np.where(
        canonical["expert_mask"][..., None],
        canonical["expert_actions"],
        canonical["student_actions"],
    )
    if not np.array_equal(canonical["behavior_actions"], expected_behavior):
        raise ArtifactError(
            "behavior_actions must exactly select expert_actions or student_actions "
            "according to expert_mask"
        )

    for key in ("global_env_ids", "episode_ids", "episode_steps"):
        array = canonical[key]
        if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
            raise ArtifactError(f"{key} must have an integer dtype")
        if np.any(array < 0):
            raise ArtifactError(f"{key} cannot contain negative values")
    env_vector = _global_env_vector(canonical["global_env_ids"], steps, envs)
    for key in ("episode_ids", "episode_steps"):
        if tuple(canonical[key].shape) != (steps, envs):
            raise ArtifactError(f"{key} must have shape [T,B]")

    raytraced = canonical["observation_is_raytraced"]
    episodes = canonical["episode_ids"]
    for env_index in range(envs):
        values_by_episode: dict[int, bool] = {}
        for time_index in range(steps):
            episode = int(episodes[time_index, env_index])
            value = bool(raytraced[time_index, env_index])
            if episode in values_by_episode and values_by_episode[episode] != value:
                raise ArtifactError(
                    "observation_is_raytraced must be constant within each episode"
                )
            values_by_episode[episode] = value

    expected_envs = set(contract["expected_global_env_ids"])
    actual_envs = {int(item) for item in env_vector.tolist()}
    if not actual_envs <= expected_envs:
        raise ArtifactError("shard contains global_env_ids outside the cohort contract")

    transition_keys: set[tuple[int, int, int]] = set()
    episode_steps = canonical["episode_steps"]
    for time_index in range(steps):
        for env_index in range(envs):
            key = (
                int(env_vector[env_index]),
                int(episodes[time_index, env_index]),
                int(episode_steps[time_index, env_index]),
            )
            if key in transition_keys:
                raise ArtifactError(f"duplicate transition identity inside shard: {key}")
            transition_keys.add(key)

    for key, array in canonical.items():
        if key.startswith("obs__"):
            if key == "obs__":
                raise ArtifactError("observation array key must have a non-empty obs__ suffix")
            if array.ndim < 2 or tuple(array.shape[:2]) != (steps, envs):
                raise ArtifactError(f"{key} must begin with dimensions [T,B]")
            if any(dimension <= 0 for dimension in array.shape):
                raise ArtifactError(f"{key} cannot contain an empty dimension")
            _finite_numeric(array, key)

    return {
        "arrays": canonical,
        "num_steps": steps,
        "num_envs": envs,
        "transition_count": steps * envs,
        "global_env_ids": sorted(actual_envs),
        "transition_keys": transition_keys,
    }


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        key: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for key, array in sorted(arrays.items())
    }


def _write_npz_temporary(target: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    target = _absolute(target)
    parent = _ensure_directory(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            with zipfile.ZipFile(
                handle, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                for key in sorted(arrays):
                    info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED
                    info.create_system = 3
                    info.external_attr = 0o100444 << 16
                    with archive.open(info, mode="w", force_zip64=True) as entry:
                        np.lib.format.write_array(
                            entry,
                            np.ascontiguousarray(arrays[key]),
                            version=(2, 0),
                            allow_pickle=False,
                        )
            handle.flush()
            os.fchmod(handle.fileno(), _PUBLISHED_FILE_MODE)
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _provenance_from_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: contract[field]
        for field in (
            "campaign_id",
            "macro_round",
            "cohort_index",
            "purpose",
            *_DIGEST_FIELDS,
        )
    }


def _require_exact_provenance(contract: Mapping[str, Any], supplied: Mapping[str, Any]) -> None:
    expected = _provenance_from_contract(contract)
    if dict(supplied) != expected:
        mismatches = sorted(key for key in expected if supplied.get(key) != expected[key])
        raise ArtifactError(f"rollout provenance does not match cohort contract: {mismatches}")


def _shard_id(rank: int, rollout_id: int, part_index: int) -> str:
    return f"rank-{rank:04d}_rollout-{rollout_id:08d}_part-{part_index:04d}"


def publish_rollout_shard(
    cohort_dir: str | os.PathLike[str],
    *,
    rank: int,
    rollout_id: int,
    part_index: int,
    arrays: Mapping[str, np.ndarray],
    campaign_id: str,
    macro_round: int,
    cohort_index: int,
    purpose: str,
    policy_sha256: str,
    runtime_policy_sha256: str,
    teacher_sha256: str,
    renderer_sha256: str,
    beta_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Publish one immutable rollout part, payload first and sidecar second."""

    root = _absolute(cohort_dir)
    _check_path_chain(root)
    contract, contract_sha = _load_contract(root)
    rank = _require_int(rank, "rank")
    rollout_id = _require_int(rollout_id, "rollout_id")
    part_index = _require_int(part_index, "part_index")
    if rank not in contract["expected_ranks"]:
        raise ArtifactError("rank is not listed in the cohort contract")
    supplied = {
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "cohort_index": cohort_index,
        "purpose": purpose,
        "policy_sha256": policy_sha256,
        "runtime_policy_sha256": runtime_policy_sha256,
        "teacher_sha256": teacher_sha256,
        "renderer_sha256": renderer_sha256,
        "beta_sha256": beta_sha256,
        "config_sha256": config_sha256,
    }
    _require_exact_provenance(contract, supplied)
    validated = _validate_arrays(arrays, contract)

    shard_id = _shard_id(rank, rollout_id, part_index)
    shard_dir = _ensure_directory(root / SHARD_DIRECTORY)
    payload_path = shard_dir / f"{shard_id}.npz"
    sidecar_path = shard_dir / f"{shard_id}.json"
    if (root / SEAL_FILENAME).exists() and not sidecar_path.exists():
        raise ArtifactError("cannot add a new shard after the cohort is sealed")

    temporary = _write_npz_temporary(payload_path, validated["arrays"])
    try:
        payload_sha = _publish_temp_create_only(temporary, payload_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    payload_size = os.stat(payload_path, follow_symlinks=False).st_size
    manifest = {
        "schema": SHARD_SCHEMA,
        "shard_id": shard_id,
        **supplied,
        "rank": rank,
        "rollout_id": rollout_id,
        "part_index": part_index,
        "cohort_contract_sha256": contract_sha,
        "transition_count": validated["transition_count"],
        "num_steps": validated["num_steps"],
        "num_envs": validated["num_envs"],
        "global_env_ids": validated["global_env_ids"],
        "view_contract": contract["view_contract"],
        "action_contract": contract["action_contract"],
        "arrays": _array_metadata(validated["arrays"]),
        "payload": {
            "path": payload_path.name,
            "sha256": payload_sha,
            "size_bytes": payload_size,
        },
    }
    _validate_shard_manifest(manifest)
    publish_immutable_json(sidecar_path, manifest)
    return verify_rollout_shard(sidecar_path)


def _validate_array_metadata(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ArtifactError("arrays metadata must be a non-empty mapping")
    parsed: dict[str, dict[str, Any]] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or _ARRAY_KEY_RE.fullmatch(key) is None:
            raise ArtifactError(f"unsafe arrays metadata key: {key!r}")
        item = _require_exact_keys(raw, {"dtype", "shape"}, f"arrays.{key}")
        try:
            dtype = np.dtype(item["dtype"])
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"arrays.{key}.dtype is invalid") from exc
        if dtype.hasobject or str(dtype) != item["dtype"]:
            raise ArtifactError(f"arrays.{key}.dtype is not canonical or is object")
        shape = item["shape"]
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ArtifactError(f"arrays.{key}.shape is invalid")
        parsed[key] = item
    return parsed


def _validate_shard_manifest(document: Any) -> dict[str, Any]:
    manifest = _require_exact_keys(document, _SHARD_KEYS, "rollout shard manifest")
    if manifest["schema"] != SHARD_SCHEMA:
        raise ArtifactError("rollout shard schema mismatch")
    shard_id = _require_token(manifest["shard_id"], "shard_id")
    _require_token(manifest["campaign_id"], "campaign_id")
    for field in ("macro_round", "cohort_index", "rank", "rollout_id", "part_index"):
        _require_int(manifest[field], field)
    if shard_id != _shard_id(manifest["rank"], manifest["rollout_id"], manifest["part_index"]):
        raise ArtifactError("shard_id does not match rank/rollout_id/part_index")
    if manifest["purpose"] not in {"train", "eval"}:
        raise ArtifactError("shard purpose must be exactly 'train' or 'eval'")
    for field in (*_DIGEST_FIELDS, "cohort_contract_sha256"):
        _require_sha256(manifest[field], field)
    _require_int(manifest["transition_count"], "transition_count", minimum=1)
    steps = _require_int(manifest["num_steps"], "num_steps", minimum=1)
    envs = _require_int(manifest["num_envs"], "num_envs", minimum=1)
    if manifest["transition_count"] != steps * envs:
        raise ArtifactError("shard transition_count must equal num_steps * num_envs")
    listed_envs = _require_sorted_unique_ints(manifest["global_env_ids"], "global_env_ids")
    if len(listed_envs) != envs:
        raise ArtifactError("shard global_env_ids length must equal num_envs")
    _validate_view_contract(manifest["view_contract"])
    _validate_action_contract(manifest["action_contract"])
    metadata = _validate_array_metadata(manifest["arrays"])
    if not _REQUIRED_ARRAYS <= set(metadata) or any(
        key not in _REQUIRED_ARRAYS and not key.startswith("obs__") for key in metadata
    ):
        raise ArtifactError("shard arrays metadata has missing or unknown keys")
    payload = _require_exact_keys(
        manifest["payload"], {"path", "sha256", "size_bytes"}, "payload"
    )
    payload_path = _safe_relative_path(payload["path"], "payload.path", components=1)
    if not payload_path.endswith(".npz"):
        raise ArtifactError("payload.path must name an NPZ file")
    _require_sha256(payload["sha256"], "payload.sha256")
    _require_int(payload["size_bytes"], "payload.size_bytes", minimum=1)
    return manifest


def _read_npz_arrays(path: Path, expected_sha: str) -> dict[str, np.ndarray]:
    with _open_regular_read(path) as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha:
            raise ArtifactError(f"payload SHA-256 mismatch: {path}")
        handle.seek(0)
        try:
            with zipfile.ZipFile(handle, mode="r") as archive:
                names = [item.filename for item in archive.infolist()]
                if len(names) != len(set(names)):
                    raise ArtifactError("NPZ contains duplicate ZIP members")
                if any(
                    not name.endswith(".npy")
                    or "/" in name
                    or _ARRAY_KEY_RE.fullmatch(name[:-4]) is None
                    for name in names
                ):
                    raise ArtifactError("NPZ contains unsafe members")
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as bundle:
                arrays = {key: np.ascontiguousarray(bundle[key]) for key in bundle.files}
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactError(f"cannot load NPZ payload {path}: {exc}") from exc
    return arrays


def _verify_rollout_shard(sidecar_path: Path) -> dict[str, Any]:
    sidecar_path = _absolute(sidecar_path)
    manifest = _validate_shard_manifest(load_strict_json(sidecar_path))
    if sidecar_path.name != f"{manifest['shard_id']}.json":
        raise ArtifactError("sidecar filename does not match shard_id")
    cohort_dir = sidecar_path.parent.parent
    contract, contract_sha = _load_contract(cohort_dir)
    if manifest["cohort_contract_sha256"] != contract_sha:
        raise ArtifactError("shard cohort contract digest mismatch")
    _require_exact_provenance(contract, _provenance_from_contract(manifest))
    if manifest["rank"] not in contract["expected_ranks"]:
        raise ArtifactError("shard rank is not expected by cohort")
    if manifest["view_contract"] != contract["view_contract"]:
        raise ArtifactError("shard view contract mismatch")
    if manifest["action_contract"] != contract["action_contract"]:
        raise ArtifactError("shard action contract mismatch")

    payload_name = _safe_relative_path(
        manifest["payload"]["path"], "payload.path", components=1
    )
    payload_path = sidecar_path.parent / payload_name
    _check_path_chain(payload_path)
    info = os.stat(payload_path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_size != manifest["payload"]["size_bytes"]:
        raise ArtifactError("payload type or size does not match manifest")
    arrays = _read_npz_arrays(payload_path, manifest["payload"]["sha256"])
    validated = _validate_arrays(arrays, contract)
    if _array_metadata(validated["arrays"]) != manifest["arrays"]:
        raise ArtifactError("payload array metadata does not match manifest")
    for key in ("transition_count", "num_steps", "num_envs", "global_env_ids"):
        if validated[key] != manifest[key]:
            raise ArtifactError(f"payload {key} does not match manifest")
    relative_sidecar = f"{SHARD_DIRECTORY}/{sidecar_path.name}"
    relative_payload = f"{SHARD_DIRECTORY}/{payload_path.name}"
    return {
        "shard_id": manifest["shard_id"],
        "manifest_path": relative_sidecar,
        "manifest_sha256": sha256_file(sidecar_path),
        "payload_path": relative_payload,
        "payload_sha256": manifest["payload"]["sha256"],
        "transition_count": manifest["transition_count"],
        "rank": manifest["rank"],
        "rollout_id": manifest["rollout_id"],
        "part_index": manifest["part_index"],
        "global_env_ids": manifest["global_env_ids"],
        "_transition_keys": validated["transition_keys"],
        "_manifest": manifest,
    }


def verify_rollout_shard(sidecar_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Fully verify one sidecar and its NPZ, returning safe relative paths."""

    report = _verify_rollout_shard(_absolute(sidecar_path))
    return {key: value for key, value in report.items() if not key.startswith("_")}


def _assert_tree_has_no_symlinks(root: Path) -> None:
    root = _absolute(root)
    _check_path_chain(root)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            path = Path(directory) / name
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise ArtifactError(f"symbolic links are forbidden in cohort: {path}")


def discover_rollout_shards(cohort_dir: str | os.PathLike[str]) -> list[Path]:
    """Discover committed shard sidecars; payload-only partials are invisible."""

    root = _absolute(cohort_dir)
    _assert_tree_has_no_symlinks(root)
    shard_dir = root / SHARD_DIRECTORY
    if not shard_dir.is_dir():
        raise ArtifactError(f"missing shard directory: {shard_dir}")
    paths = []
    for path in shard_dir.iterdir():
        if path.name.startswith(".") or path.suffix != ".json":
            continue
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise ArtifactError(f"shard sidecar is not a regular file: {path}")
        paths.append(path)
    return sorted(paths, key=lambda item: item.name)


def _cohort_logical_id(contract: Mapping[str, Any]) -> str:
    return (
        f"{contract['campaign_id']}:r{contract['macro_round']:09d}:"
        f"c{contract['cohort_index']:04d}:{contract['purpose']}"
    )


def _validate_and_summarize_shards(
    cohort_dir: Path, contract: Mapping[str, Any], contract_sha: str
) -> tuple[list[dict[str, Any]], set[tuple[int, int, int]]]:
    reports = [_verify_rollout_shard(path) for path in discover_rollout_shards(cohort_dir)]
    if not reports:
        raise ArtifactError("cannot seal an empty cohort")
    identities: set[tuple[int, int, int]] = set()
    envs_seen: set[int] = set()
    transition_keys: set[tuple[int, int, int]] = set()
    ranks: set[int] = set()
    transitions = 0
    public_reports: list[dict[str, Any]] = []
    for report in reports:
        identity = (report["rank"], report["rollout_id"], report["part_index"])
        if identity in identities:
            raise ArtifactError(f"duplicate rank/rollout/part identity: {identity}")
        identities.add(identity)
        shard_envs = set(report["global_env_ids"])
        overlap = envs_seen & shard_envs
        if overlap:
            raise ArtifactError(
                "global environments appear in multiple shard parts: "
                f"{sorted(overlap)}"
            )
        envs_seen |= shard_envs
        duplicate_transitions = transition_keys & report["_transition_keys"]
        if duplicate_transitions:
            raise ArtifactError("duplicate transition identities across shards")
        transition_keys |= report["_transition_keys"]
        ranks.add(report["rank"])
        transitions += report["transition_count"]
        public_reports.append(
            {key: value for key, value in report.items() if not key.startswith("_")}
        )

    if sorted(ranks) != contract["expected_ranks"]:
        raise ArtifactError(
            f"cohort rank coverage mismatch: actual={sorted(ranks)} "
            f"expected={contract['expected_ranks']}"
        )
    if sorted(envs_seen) != contract["expected_global_env_ids"]:
        raise ArtifactError("cohort global environment coverage is not exact")
    if transitions != contract["expected_transitions"]:
        raise ArtifactError(
            f"cohort transition coverage mismatch: actual={transitions} "
            f"expected={contract['expected_transitions']}"
        )
    if len(transition_keys) != transitions:
        raise ArtifactError("cohort transition identities are not unique")
    public_reports.sort(key=lambda item: item["shard_id"])
    return public_reports, transition_keys


def _build_seal(
    contract: Mapping[str, Any], contract_sha: str, shards: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    transition_count = sum(int(shard["transition_count"]) for shard in shards)
    ranks = sorted({int(shard["rank"]) for shard in shards})
    envs = sorted({int(item) for shard in shards for item in shard["global_env_ids"]})
    return {
        "schema": COHORT_SCHEMA,
        "campaign_id": contract["campaign_id"],
        "macro_round": contract["macro_round"],
        "cohort_index": contract["cohort_index"],
        "purpose": contract["purpose"],
        "logical_id": _cohort_logical_id(contract),
        "sealed": True,
        "transition_count": transition_count,
        "view_contract": contract["view_contract"],
        "action_contract": contract["action_contract"],
        "expected_ranks": contract["expected_ranks"],
        "expected_global_env_ids": contract["expected_global_env_ids"],
        "expected_transitions": contract["expected_transitions"],
        "ranks": ranks,
        "global_env_ids": envs,
        "contract_sha256": contract_sha,
        "shards": [dict(shard) for shard in shards],
        **{field: contract[field] for field in _DIGEST_FIELDS},
    }


def _validate_seal(document: Any) -> dict[str, Any]:
    seal = _require_exact_keys(document, _SEAL_KEYS, "cohort seal")
    if seal["schema"] != COHORT_SCHEMA or seal["sealed"] is not True:
        raise ArtifactError("cohort seal schema or sealed flag is invalid")
    _require_token(seal["campaign_id"], "campaign_id")
    _require_int(seal["macro_round"], "macro_round")
    _require_int(seal["cohort_index"], "cohort_index")
    if seal["purpose"] not in {"train", "eval"}:
        raise ArtifactError("cohort seal purpose is invalid")
    if seal["logical_id"] != (
        f"{seal['campaign_id']}:r{seal['macro_round']:09d}:"
        f"c{seal['cohort_index']:04d}:{seal['purpose']}"
    ):
        raise ArtifactError("cohort logical_id is inconsistent")
    _require_int(seal["transition_count"], "transition_count", minimum=1)
    _validate_view_contract(seal["view_contract"])
    _validate_action_contract(seal["action_contract"])
    for field in (*_DIGEST_FIELDS, "contract_sha256"):
        _require_sha256(seal[field], field)
    _require_sorted_unique_ints(seal["expected_ranks"], "expected_ranks")
    _require_sorted_unique_ints(
        seal["expected_global_env_ids"], "expected_global_env_ids"
    )
    _require_int(seal["expected_transitions"], "expected_transitions", minimum=1)
    _require_sorted_unique_ints(seal["ranks"], "ranks")
    _require_sorted_unique_ints(seal["global_env_ids"], "global_env_ids")
    if not isinstance(seal["shards"], list) or not seal["shards"]:
        raise ArtifactError("cohort seal must contain shards")
    shard_ids: list[str] = []
    for index, raw in enumerate(seal["shards"]):
        shard = _require_exact_keys(raw, _SEAL_SHARD_KEYS, f"shards[{index}]")
        shard_ids.append(_require_token(shard["shard_id"], "shard_id"))
        manifest_path = _safe_relative_path(
            shard["manifest_path"], "manifest_path", components=2
        )
        payload_path = _safe_relative_path(
            shard["payload_path"], "payload_path", components=2
        )
        if not manifest_path.startswith(
            f"{SHARD_DIRECTORY}/"
        ) or not manifest_path.endswith(".json"):
            raise ArtifactError("manifest_path must be a JSON file in shards/")
        if not payload_path.startswith(
            f"{SHARD_DIRECTORY}/"
        ) or not payload_path.endswith(".npz"):
            raise ArtifactError("payload_path must be an NPZ file in shards/")
        for field in ("manifest_sha256", "payload_sha256"):
            _require_sha256(shard[field], field)
        for field in ("transition_count", "rank", "rollout_id", "part_index"):
            _require_int(shard[field], field, minimum=1 if field == "transition_count" else 0)
        _require_sorted_unique_ints(shard["global_env_ids"], "shard.global_env_ids")
    if shard_ids != sorted(set(shard_ids)):
        raise ArtifactError("cohort seal shards must be sorted with unique IDs")
    return seal


def seal_cohort(cohort_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Seal an exactly complete cohort; identical retries are idempotent."""

    root = _absolute(cohort_dir)
    _assert_tree_has_no_symlinks(root)
    if (root / SEAL_FILENAME).exists():
        return verify_cohort(root)
    contract, contract_sha = _load_contract(root)
    reports, _ = _validate_and_summarize_shards(root, contract, contract_sha)
    seal = _build_seal(contract, contract_sha, reports)
    _validate_seal(seal)
    publish_immutable_json(root / SEAL_FILENAME, seal)
    return verify_cohort(root)


def _cohort_directory(path: str | os.PathLike[str]) -> Path:
    target = _absolute(path)
    _check_path_chain(target)
    if target.is_dir():
        return target
    if target.name == SEAL_FILENAME:
        return target.parent
    raise ArtifactError("verify_cohort expects a cohort directory or its seal.json")


def verify_cohort(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify a sealed cohort and every referenced shard byte-for-byte."""

    root = _cohort_directory(path)
    _assert_tree_has_no_symlinks(root)
    seal_path = root / SEAL_FILENAME
    seal = _validate_seal(load_strict_json(seal_path))
    contract, contract_sha = _load_contract(root)
    if seal["contract_sha256"] != contract_sha:
        raise ArtifactError("cohort seal references a different contract")
    reports, _ = _validate_and_summarize_shards(root, contract, contract_sha)
    expected_seal = _build_seal(contract, contract_sha, reports)
    if seal != expected_seal:
        raise ArtifactError("cohort seal does not exactly match verified shard contents")
    report = dict(seal)
    report["cohort_sha256"] = sha256_file(seal_path)
    report["manifest_path"] = str(seal_path)
    return report


__all__ = [
    "ArtifactError",
    "COHORT_CONTRACT_SCHEMA",
    "SHARD_SCHEMA",
    "COHORT_SCHEMA",
    "canonical_json_sha256",
    "load_strict_json",
    "sha256_file",
    "publish_immutable_json",
    "atomic_publish_json",
    "create_cohort",
    "publish_rollout_shard",
    "verify_rollout_shard",
    "discover_rollout_shards",
    "seal_cohort",
    "verify_cohort",
]
