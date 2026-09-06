"""Deterministic, fail-closed Image3F first-condition bundles.

One bundle contains the exact uint8 Image3F request arrays for one held-out
seed block.  The file is a deterministic, uncompressed NPZ whose first member
is canonical JSON encoded as a one-dimensional uint8 NPY array.  Callers must
always supply the bundle SHA-256 out of band when loading it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


BUNDLE_SCHEMA = "fast_roborender.round_dagger.first_condition_bundle.v1"
VIEWS = ("head_color", "hand_left", "hand_right")
GROUPS = ("depth", "mask", "previous_rgb")
CONTROL_HEIGHT = 240
CONTROL_WIDTH = 416
MAX_ENV_COUNT = 32
MAX_BUNDLE_BYTES = 96 * 1024 * 1024

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
_PLANNED_CASE_KEYS = {
    "seed",
    "env_index",
    "global_case_index",
    "planned_case_id",
    "motion_relative_path",
    "motion_sha256",
}
_BUNDLE_CASE_KEYS = _PLANNED_CASE_KEYS | {
    "first_condition_sha256",
    "scenario_id",
}
_METADATA_KEYS = {
    "schema",
    "seed",
    "generation_seed",
    "prompt",
    "views",
    "case_count",
    "plan_sha256",
    "source_capture_run_id",
    "source_capture_nonce",
    "source_provenance",
    "cases",
    "array_sha256",
    "first_condition_sha256_by_env",
    "live_condition_sha256_by_env",
    "scenario_id_by_env",
    "live_reset_state_sha256_by_env",
}


class FirstConditionBundleError(RuntimeError):
    """A first-condition bundle or its out-of-band identity is invalid."""


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise FirstConditionBundleError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FirstConditionBundleError(f"{label} must be an integer >= {minimum}")
    return value


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise FirstConditionBundleError(f"{label} fields are not exact")
    return dict(value)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FirstConditionBundleError("bundle metadata is not canonical JSON") from exc


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FirstConditionBundleError(f"duplicate metadata key: {key!r}")
        result[key] = value
    return result


def image3f_payload_sha256(
    *,
    views: Sequence[str],
    groups: Sequence[tuple[str, Mapping[str, np.ndarray]]],
    prompt: str | None = None,
    seed: int | None = None,
) -> str:
    """Hash the exact ordered Image3F payload byte representation."""

    digest = hashlib.sha256()
    if prompt is not None:
        encoded = str(prompt).encode("utf-8")
        digest.update(b"prompt\0" + len(encoded).to_bytes(8, "big") + encoded)
    if seed is not None:
        digest.update(b"seed\0" + str(int(seed)).encode("ascii"))
    for group_name, arrays in groups:
        digest.update(str(group_name).encode("utf-8") + b"\0")
        for view in views:
            try:
                value = np.ascontiguousarray(arrays[view])
            except (KeyError, TypeError) as exc:
                raise FirstConditionBundleError(
                    f"Image3F group {group_name!r} is missing view {view!r}"
                ) from exc
            name = str(view).encode("utf-8")
            shape = ",".join(str(int(v)) for v in value.shape).encode("ascii")
            dtype = value.dtype.str.encode("ascii")
            digest.update(len(name).to_bytes(8, "big") + name)
            digest.update(len(shape).to_bytes(8, "big") + shape)
            digest.update(len(dtype).to_bytes(8, "big") + dtype)
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _scenario_id(seed: int, env_index: int, condition_sha256: str) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "seed": seed,
                "env_index": env_index,
                "first_condition_sha256": condition_sha256,
            }
        )
    ).hexdigest()


def _member_names(views: Sequence[str]) -> tuple[str, ...]:
    return ("metadata",) + tuple(
        f"{group}__{view}" for group in GROUPS for view in views
    )


def _expected_shape(group: str, env_count: int) -> tuple[int, ...]:
    if group == "depth":
        return (env_count, CONTROL_HEIGHT, CONTROL_WIDTH)
    return (env_count, CONTROL_HEIGHT, CONTROL_WIDTH, 3)


def _normalize_hash_map(value: Any, env_count: int, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise FirstConditionBundleError(f"{label} must be an environment hash map")
    normalized = {str(key): item for key, item in value.items()}
    if len(normalized) != len(value):
        raise FirstConditionBundleError(f"{label} has colliding environment keys")
    expected = {str(index) for index in range(env_count)}
    if set(normalized) != expected:
        raise FirstConditionBundleError(f"{label} environment coverage is not exact")
    return {
        str(index): _sha(normalized[str(index)], f"{label}[{index}]")
        for index in range(env_count)
    }


def _normalize_cases(
    value: Any, *, seed: int, env_count: int, with_identities: bool
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FirstConditionBundleError("bundle cases must be a list")
    if len(value) != env_count:
        raise FirstConditionBundleError("bundle case coverage is incomplete")
    keys = _BUNDLE_CASE_KEYS if with_identities else _PLANNED_CASE_KEYS
    normalized: list[dict[str, Any]] = []
    for env_index, raw in enumerate(value):
        case = _exact(raw, keys, f"bundle cases[{env_index}]")
        if case["seed"] != seed or case["env_index"] != env_index:
            raise FirstConditionBundleError("bundle case seed/environment order differs")
        _integer(case["global_case_index"], "bundle global case index")
        _sha(case["planned_case_id"], "bundle planned case ID")
        _sha(case["motion_sha256"], "bundle motion SHA-256")
        if (
            not isinstance(case["motion_relative_path"], str)
            or not case["motion_relative_path"]
            or "\x00" in case["motion_relative_path"]
        ):
            raise FirstConditionBundleError("bundle motion path is malformed")
        if with_identities:
            _sha(case["first_condition_sha256"], "bundle condition SHA-256")
            _sha(case["scenario_id"], "bundle scenario ID")
        normalized.append(case)
    return normalized


def _normalize_requests(
    requests_by_env: Any, *, env_count: int, views: tuple[str, ...]
) -> tuple[dict[int, tuple[dict[str, np.ndarray], ...]], dict[str, np.ndarray]]:
    if not isinstance(requests_by_env, Mapping):
        raise FirstConditionBundleError("requests_by_env must be a mapping")
    by_key = {str(key): value for key, value in requests_by_env.items()}
    if len(by_key) != len(requests_by_env):
        raise FirstConditionBundleError("requests_by_env has colliding environment keys")
    if set(by_key) != {str(index) for index in range(env_count)}:
        raise FirstConditionBundleError("request environment coverage is not exact")
    stacked: dict[str, list[np.ndarray]] = {
        f"{group}__{view}": [] for group in GROUPS for view in views
    }
    normalized: dict[int, tuple[dict[str, np.ndarray], ...]] = {}
    for env_index in range(env_count):
        request = by_key[str(env_index)]
        if not isinstance(request, Sequence) or len(request) != len(GROUPS):
            raise FirstConditionBundleError(
                f"request {env_index} must be a depth/mask/previous_rgb tuple"
            )
        request_groups: list[dict[str, np.ndarray]] = []
        for group, raw_arrays in zip(GROUPS, request, strict=True):
            if not isinstance(raw_arrays, Mapping) or set(raw_arrays) != set(views):
                raise FirstConditionBundleError(
                    f"request {env_index} {group} view coverage is not exact"
                )
            arrays: dict[str, np.ndarray] = {}
            expected = _expected_shape(group, 1)[1:]
            for view in views:
                value = np.asarray(raw_arrays[view])
                if value.dtype != np.uint8 or value.shape != expected:
                    raise FirstConditionBundleError(
                        f"request {env_index} {group}/{view} must be uint8{expected}"
                    )
                value = np.ascontiguousarray(value)
                arrays[view] = value
                stacked[f"{group}__{view}"].append(value)
            request_groups.append(arrays)
        normalized[env_index] = tuple(request_groups)
    return normalized, {
        name: np.ascontiguousarray(np.stack(values, axis=0))
        for name, values in stacked.items()
    }


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(
        output,
        np.ascontiguousarray(value),
        version=(1, 0),
        allow_pickle=False,
    )
    return output.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o444) << 16
    return info


def _bundle_bytes(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> bytes:
    metadata_array = np.frombuffer(_canonical_bytes(metadata), dtype=np.uint8).copy()
    members = {"metadata": metadata_array, **arrays}
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in _member_names(metadata["views"]):
            archive.writestr(_zip_info(name), _npy_bytes(members[name]))
    payload = output.getvalue()
    if not payload or len(payload) > MAX_BUNDLE_BYTES:
        raise FirstConditionBundleError("serialized first-condition bundle is oversized")
    return payload


def _snapshot_file(path: Path, expected_sha256: str) -> tuple[bytes, os.stat_result]:
    expected = _sha(expected_sha256, "first-condition bundle SHA-256")
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise FirstConditionBundleError("bundle path must be normalized and absolute")
    try:
        if path.parent.resolve(strict=True) != path.parent:
            raise FirstConditionBundleError("bundle path has a symlinked ancestor")
    except OSError as exc:
        raise FirstConditionBundleError("bundle parent directory is unavailable") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FirstConditionBundleError("cannot open first-condition bundle safely") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise FirstConditionBundleError("first-condition bundle is not a regular file")
        if info.st_mode & 0o222:
            raise FirstConditionBundleError("first-condition bundle must be immutable 0444 data")
        if info.st_size <= 0 or info.st_size > MAX_BUNDLE_BYTES:
            raise FirstConditionBundleError("first-condition bundle size is invalid")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise FirstConditionBundleError("first-condition bundle changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise FirstConditionBundleError("first-condition bundle grew while reading")
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
        ):
            raise FirstConditionBundleError("first-condition bundle changed while reading")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise FirstConditionBundleError("first-condition bundle byte identity mismatch")
    return payload, info


def _load_npy(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    info = archive.getinfo(f"{name}.npy")
    if (
        info.compress_type != zipfile.ZIP_STORED
        or info.flag_bits & 0x1
        or info.file_size <= 0
        or info.file_size > MAX_BUNDLE_BYTES
    ):
        raise FirstConditionBundleError(f"bundle member {name!r} is unsafe")
    try:
        with archive.open(info, "r") as handle:
            value = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise FirstConditionBundleError(f"cannot decode bundle member {name!r}") from exc
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise FirstConditionBundleError(f"bundle member {name!r} is not C-contiguous")
    return value


def load_first_condition_bundle(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_seed: int,
    expected_env_count: int,
    expected_views: Sequence[str],
    expected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and deeply verify one immutable first-condition bundle snapshot."""

    target = Path(path)
    seed = _integer(expected_seed, "expected seed")
    env_count = _integer(expected_env_count, "expected environment count", minimum=1)
    if env_count > MAX_ENV_COUNT:
        raise FirstConditionBundleError("expected environment count exceeds bundle bound")
    views = tuple(expected_views)
    if views != VIEWS:
        raise FirstConditionBundleError("expected bundle view order is not production exact")
    payload, info = _snapshot_file(target, expected_sha256)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise FirstConditionBundleError("first-condition bundle is not a valid NPZ") from exc
    with archive:
        expected_names = [f"{name}.npy" for name in _member_names(views)]
        observed_names = [member.filename for member in archive.infolist()]
        if observed_names != expected_names or len(set(observed_names)) != len(observed_names):
            raise FirstConditionBundleError("bundle member names/order are not exact")
        metadata_array = _load_npy(archive, "metadata")
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
            raise FirstConditionBundleError("bundle metadata member must be one-dimensional uint8")
        metadata_payload = metadata_array.tobytes(order="C")
        try:
            metadata = json.loads(
                metadata_payload.decode("utf-8"), object_pairs_hook=_strict_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FirstConditionBundleError("bundle metadata is not strict UTF-8 JSON") from exc
        if metadata_payload != _canonical_bytes(metadata):
            raise FirstConditionBundleError("bundle metadata bytes are not canonical JSON")
        arrays = {
            name: _load_npy(archive, name) for name in _member_names(views)[1:]
        }

    metadata = _exact(metadata, _METADATA_KEYS, "bundle metadata")
    if metadata["schema"] != BUNDLE_SCHEMA:
        raise FirstConditionBundleError("first-condition bundle schema mismatch")
    if metadata["seed"] != seed:
        raise FirstConditionBundleError("first-condition bundle seed mismatch")
    generation_seed = _integer(metadata["generation_seed"], "bundle generation seed")
    if generation_seed != seed:
        raise FirstConditionBundleError("bundle simulator/generation seeds differ")
    if metadata["views"] != list(views):
        raise FirstConditionBundleError("first-condition bundle views mismatch")
    if metadata["case_count"] != env_count:
        raise FirstConditionBundleError("first-condition bundle case count mismatch")
    plan_sha = _sha(metadata["plan_sha256"], "bundle plan SHA-256")
    if expected_plan_sha256 is not None and plan_sha != _sha(
        expected_plan_sha256, "expected plan SHA-256"
    ):
        raise FirstConditionBundleError("first-condition bundle plan mismatch")
    prompt = metadata["prompt"]
    if not isinstance(prompt, str) or not prompt:
        raise FirstConditionBundleError("bundle prompt must be a non-empty string")
    run_id = metadata["source_capture_run_id"]
    nonce = metadata["source_capture_nonce"]
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise FirstConditionBundleError("bundle source capture run ID is unsafe")
    if not isinstance(nonce, str) or _NONCE_RE.fullmatch(nonce) is None:
        raise FirstConditionBundleError("bundle source capture nonce is malformed")
    provenance = metadata["source_provenance"]
    if not isinstance(provenance, Mapping) or not provenance:
        raise FirstConditionBundleError("bundle source provenance must be a non-empty object")
    _canonical_bytes(provenance)

    array_hashes = _exact(
        metadata["array_sha256"], set(_member_names(views)[1:]), "bundle array hashes"
    )
    for name, value in arrays.items():
        expected_shape = _expected_shape(name.split("__", 1)[0], env_count)
        if value.dtype != np.uint8 or value.shape != expected_shape:
            raise FirstConditionBundleError(
                f"bundle member {name!r} must be uint8{expected_shape}"
            )
        if hashlib.sha256(value.tobytes(order="C")).hexdigest() != _sha(
            array_hashes[name], f"bundle array {name} SHA-256"
        ):
            raise FirstConditionBundleError(f"bundle array {name!r} hash mismatch")

    first_hashes = _normalize_hash_map(
        metadata["first_condition_sha256_by_env"], env_count, "bundle first conditions"
    )
    live_hashes = _normalize_hash_map(
        metadata["live_condition_sha256_by_env"], env_count, "bundle live conditions"
    )
    reset_hashes = _normalize_hash_map(
        metadata["live_reset_state_sha256_by_env"], env_count, "bundle reset states"
    )
    scenario_hashes = _normalize_hash_map(
        metadata["scenario_id_by_env"], env_count, "bundle scenarios"
    )
    cases = _normalize_cases(metadata["cases"], seed=seed, env_count=env_count, with_identities=True)
    requests: dict[int, tuple[dict[str, np.ndarray], ...]] = {}
    for env_index in range(env_count):
        request_groups: list[dict[str, np.ndarray]] = []
        for group in GROUPS:
            request_groups.append(
                {
                    view: np.ascontiguousarray(arrays[f"{group}__{view}"][env_index]).copy()
                    for view in views
                }
            )
        request = tuple(request_groups)
        requests[env_index] = request
        derived_condition = image3f_payload_sha256(
            views=views,
            groups=tuple(zip(GROUPS, request, strict=True)),
            prompt=prompt,
            seed=generation_seed,
        )
        derived_scenario = _scenario_id(seed, env_index, derived_condition)
        case = cases[env_index]
        if (
            first_hashes[str(env_index)] != derived_condition
            or live_hashes[str(env_index)] != derived_condition
            or scenario_hashes[str(env_index)] != derived_scenario
            or case["first_condition_sha256"] != derived_condition
            or case["scenario_id"] != derived_scenario
        ):
            raise FirstConditionBundleError(
                "bundle condition/scenario identities do not match array bytes"
            )

    return {
        "path": str(target),
        "sha256": expected_sha256,
        "size": info.st_size,
        "seed": seed,
        "generation_seed": generation_seed,
        "prompt": prompt,
        "views": views,
        "case_count": env_count,
        "plan_sha256": plan_sha,
        "capture_run_id": run_id,
        "capture_nonce": nonce,
        "provenance": dict(provenance),
        "requests_by_env": requests,
        "first_condition_sha256_by_env": first_hashes,
        "live_condition_sha256_by_env": live_hashes,
        "scenario_id_by_env": scenario_hashes,
        "live_reset_state_sha256_by_env": reset_hashes,
        "cases": cases,
        "metadata": metadata,
    }


def publish_first_condition_bundle(
    path: str | os.PathLike[str],
    *,
    seed: int,
    generation_seed: int,
    prompt: str,
    views: Sequence[str],
    requests_by_env: Mapping[int | str, Sequence[Mapping[str, np.ndarray]]],
    cases: Sequence[Mapping[str, Any]],
    plan_sha256: str,
    capture_run_id: str,
    capture_nonce: str,
    provenance: Mapping[str, Any],
    live_condition_sha256_by_env: Mapping[int | str, str],
    live_reset_state_sha256_by_env: Mapping[int | str, str],
) -> dict[str, Any]:
    """Atomically publish one create-once 0444 first-condition NPZ."""

    target = Path(path)
    if not target.is_absolute() or any(part in {".", ".."} for part in target.parts):
        raise FirstConditionBundleError("bundle output path must be normalized and absolute")
    if target.exists() or target.is_symlink():
        raise FirstConditionBundleError("bundle output must be a fresh path")
    try:
        parent_info = target.parent.lstat()
    except OSError as exc:
        raise FirstConditionBundleError("bundle output directory is unavailable") from exc
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise FirstConditionBundleError("bundle output directory must be a non-symlink directory")
    try:
        if target.parent.resolve(strict=True) != target.parent:
            raise FirstConditionBundleError("bundle output path has a symlinked ancestor")
    except OSError as exc:
        raise FirstConditionBundleError("bundle output directory is unavailable") from exc

    seed = _integer(seed, "bundle seed")
    generation_seed = _integer(generation_seed, "bundle generation seed")
    if generation_seed != seed:
        raise FirstConditionBundleError("bundle simulator/generation seeds differ")
    views = tuple(views)
    if views != VIEWS:
        raise FirstConditionBundleError("bundle view order is not production exact")
    if not isinstance(prompt, str) or not prompt:
        raise FirstConditionBundleError("bundle prompt must be a non-empty string")
    plan_sha256 = _sha(plan_sha256, "bundle plan SHA-256")
    if not isinstance(capture_run_id, str) or _RUN_ID_RE.fullmatch(capture_run_id) is None:
        raise FirstConditionBundleError("bundle capture run ID is unsafe")
    if not isinstance(capture_nonce, str) or _NONCE_RE.fullmatch(capture_nonce) is None:
        raise FirstConditionBundleError("bundle capture nonce is malformed")
    if not isinstance(provenance, Mapping) or not provenance:
        raise FirstConditionBundleError("bundle provenance must be a non-empty object")
    normalized_provenance = json.loads(_canonical_bytes(dict(provenance)).decode("utf-8"))
    env_count = len(cases)
    if env_count < 1 or env_count > MAX_ENV_COUNT:
        raise FirstConditionBundleError("bundle case count is outside the supported bound")
    planned_cases = _normalize_cases(cases, seed=seed, env_count=env_count, with_identities=False)
    requests, arrays = _normalize_requests(
        requests_by_env, env_count=env_count, views=views
    )
    live_hashes = _normalize_hash_map(
        live_condition_sha256_by_env, env_count, "live conditions"
    )
    reset_hashes = _normalize_hash_map(
        live_reset_state_sha256_by_env, env_count, "live reset states"
    )
    first_hashes: dict[str, str] = {}
    scenario_hashes: dict[str, str] = {}
    bundle_cases: list[dict[str, Any]] = []
    for env_index, planned in enumerate(planned_cases):
        request = requests[env_index]
        condition = image3f_payload_sha256(
            views=views,
            groups=tuple(zip(GROUPS, request, strict=True)),
            prompt=prompt,
            seed=generation_seed,
        )
        if live_hashes[str(env_index)] != condition:
            raise FirstConditionBundleError(
                "live condition identity differs from captured request arrays"
            )
        scenario = _scenario_id(seed, env_index, condition)
        first_hashes[str(env_index)] = condition
        scenario_hashes[str(env_index)] = scenario
        bundle_cases.append(
            {
                **planned,
                "first_condition_sha256": condition,
                "scenario_id": scenario,
            }
        )
    array_hashes = {
        name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
        for name, value in arrays.items()
    }
    metadata = {
        "schema": BUNDLE_SCHEMA,
        "seed": seed,
        "generation_seed": generation_seed,
        "prompt": prompt,
        "views": list(views),
        "case_count": env_count,
        "plan_sha256": plan_sha256,
        "source_capture_run_id": capture_run_id,
        "source_capture_nonce": capture_nonce,
        "source_provenance": normalized_provenance,
        "cases": bundle_cases,
        "array_sha256": array_hashes,
        "first_condition_sha256_by_env": first_hashes,
        "live_condition_sha256_by_env": live_hashes,
        "scenario_id_by_env": scenario_hashes,
        "live_reset_state_sha256_by_env": reset_hashes,
    }
    payload = _bundle_bytes(metadata, arrays)
    digest = hashlib.sha256(payload).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        # Validate the exact temporary bytes before making them visible.
        load_first_condition_bundle(
            temporary,
            expected_sha256=digest,
            expected_seed=seed,
            expected_env_count=env_count,
            expected_views=views,
            expected_plan_sha256=plan_sha256,
        )
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise FirstConditionBundleError("bundle output ceased to be fresh") from exc
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return load_first_condition_bundle(
        target,
        expected_sha256=digest,
        expected_seed=seed,
        expected_env_count=env_count,
        expected_views=views,
        expected_plan_sha256=plan_sha256,
    )


__all__ = [
    "BUNDLE_SCHEMA",
    "FirstConditionBundleError",
    "GROUPS",
    "MAX_BUNDLE_BYTES",
    "VIEWS",
    "image3f_payload_sha256",
    "load_first_condition_bundle",
    "publish_first_condition_bundle",
]
