#!/usr/bin/env python3
"""Sealed raw Isaac inputs for the offline RoboRender projection.

This is deliberately a different artifact type from learner replay.  A raw
conditioning artifact contains the compact Image3F controls and expert-labelled
state/action transitions, but it never contains ``policy_rgb``.  The offline
renderer must consume this artifact and publish a separate, sanitized learner
replay before any optimizer is allowed to see it.

The JSON sidecar is the commit record.  The NPZ is published first and is not a
discoverable artifact until the immutable sidecar has been published last.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

import numpy as np

from round_dagger.artifacts import (
    ArtifactError,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)

SCHEMA = "fast_roborender.round_dagger.isaac_roborender_conditioning.v2"
CONFIG_SCHEMA = "fast_roborender.round_dagger.isaac_conditioning.v2"
ARTIFACT_KIND = "raw_isaac_roborender_conditioning"
REPLAY_ADMISSION = "forbidden_pending_roborender_projection"
RENDER_MODE = "image3f_sparse_frame_hold"
DEPTH_ENCODING = "h5_v4_uint8"
VIEWS = ("head_color", "hand_left", "hand_right")
CONTROL_HEIGHT = 240
CONTROL_WIDTH = 416
SENSOR_HEIGHT = 480
SENSOR_WIDTH = 832
REFERENCE_POLICY = "lagged_generated_with_episode_anchor"
REFERENCE_LAG = 256
REFERENCE_LAG_UNIT = "control_steps"
CONTROL_HZ = 50
RENDER_HZ = 5
IMAGE_STRIDE = CONTROL_HZ // RENDER_HZ
PURPOSES = ("train", "validation", "test")

_H5_V4_PARAMETERS = {
    "head_color": {
        "raw_clip": (0.05, 10.0),
        "inv_clip": (1.0, 40.0),
        "norm_range": (0.05, 2.6),
    },
    "hand_left": {
        "raw_clip": (0.0, 1.5),
        "inv_clip": (0.7, 100.0),
        "raw_pct": (1.0, 99.0),
    },
    "hand_right": {
        "raw_clip": (0.0, 1.5),
        "inv_clip": (0.7, 100.0),
        "raw_pct": (1.0, 99.0),
    },
}
H5_V4_PARAMETERS = MappingProxyType(
    {
        view: MappingProxyType(dict(parameters))
        for view, parameters in _H5_V4_PARAMETERS.items()
    }
)

_TRANSITION_ARRAYS = {
    "expert_actions",
    "student_actions",
    "behavior_actions",
    "expert_mask",
    "dones",
    "global_env_ids",
    "episode_ids",
    "episode_steps",
    "anchor_valid",
    "transition_image_indices",
    "obs__actor_obs",
}
_SPARSE_ANCHOR_ARRAYS = {"anchor_global_env_ids", "anchor_episode_ids"}
_SPARSE_IMAGE_ARRAYS = {
    "image_global_env_ids",
    "image_episode_ids",
    "image_episode_steps",
    "image_reference_source_steps",
    "image_reference_indices",
}
_VIEW_ARRAYS = {
    *(f"depth_ctrl__{view}" for view in VIEWS),
    *(f"rgb_robot__{view}" for view in VIEWS),
    *(f"rgb_anchor__{view}" for view in VIEWS),
}
REQUIRED_ARRAYS = frozenset(
    _TRANSITION_ARRAYS | _SPARSE_ANCHOR_ARRAYS | _SPARSE_IMAGE_ARRAYS | _VIEW_ARRAYS
)
FORBIDDEN_LEARNER_ARRAYS = frozenset(
    {"policy_rgb", "observation_is_raytraced", "teacher_obs"}
)

_ARRAY_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOP_LEVEL_KEYS = {
    "schema",
    "artifact_kind",
    "sealed",
    "learner_replay_ready",
    "replay_admission",
    "purpose",
    "num_steps",
    "num_envs",
    "transition_count",
    "episode_count",
    "image_count",
    "action_dim",
    "state_dim",
    "control_contract",
    "render",
    "arrays",
    "payload",
}
_RENDER_KEYS = {
    "mode",
    "prompt",
    "seed",
    "reference_lag",
    "reference_lag_unit",
    "control_hz",
    "render_hz",
    "image_stride",
    "metadata",
}
_PAYLOAD_KEYS = {"path", "sha256", "size_bytes"}
_ARRAY_METADATA_KEYS = {"dtype", "shape", "sha256"}
_READ_ONLY_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


class ConditioningArtifactError(RuntimeError):
    """A raw Isaac-to-RoboRender artifact violates its sealed contract."""


def _plain_json_value(value: Any, label: str) -> Any:
    """Materialize Mapping/Sequence implementations such as OmegaConf safely."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ConditioningArtifactError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ConditioningArtifactError(
                    f"{label} contains a non-string JSON key"
                )
            result[key] = _plain_json_value(nested, f"{label}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _plain_json_value(nested, f"{label}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise ConditioningArtifactError(
        f"{label} is not canonical JSON: unsupported type {type(value).__name__}"
    )


def h5_v4_recipe_document() -> dict[str, dict[str, list[float]]]:
    """Return a JSON-safe copy of the exact production per-view recipe."""

    return {
        view: {name: list(values) for name, values in parameters.items()}
        for view, parameters in _H5_V4_PARAMETERS.items()
    }


def validate_conditioning_contract(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the one canonical compact Image3F conditioning contract.

    Callers may pass either the complete returned contract or the small YAML
    projection ``views``, ``control_resolution``, ``depth_mode``, and
    ``depth_h5v4_per_view`` and the fixed 50-Hz-control/5-Hz-image cadence.
    Every supplied value is compared exactly; this is a drift detector, not a
    mechanism for defining alternate formats.
    """

    canonical = _control_contract()
    if config is None:
        return canonical
    if not isinstance(config, Mapping):
        raise ConditioningArtifactError("conditioning config must be a mapping")
    supplied = _plain_json_value(config, "conditioning config")
    if supplied == canonical:
        return canonical

    full_config_keys = {
        "schema",
        "view_names",
        "sensor_resolution",
        "control_resolution",
        "depth_mode",
        "depth_h5v4_per_view",
        "prompt",
        "seed",
        "reference_policy",
        "reference_lag",
        "reference_lag_unit",
        "control_hz",
        "render_hz",
        "image_stride",
    }
    if set(supplied) == full_config_keys:
        fixed = {
            "schema": CONFIG_SCHEMA,
            "view_names": list(VIEWS),
            "sensor_resolution": [SENSOR_HEIGHT, SENSOR_WIDTH],
            "control_resolution": [CONTROL_HEIGHT, CONTROL_WIDTH],
            "depth_mode": "h5_v4",
            "depth_h5v4_per_view": h5_v4_recipe_document(),
            "reference_policy": REFERENCE_POLICY,
            "reference_lag_unit": REFERENCE_LAG_UNIT,
            "control_hz": CONTROL_HZ,
            "render_hz": RENDER_HZ,
            "image_stride": IMAGE_STRIDE,
        }
        for key, expected in fixed.items():
            if supplied[key] != expected:
                raise ConditioningArtifactError(f"conditioning config {key} differs")
        _validate_render_fields(
            prompt=supplied["prompt"],
            seed=supplied["seed"],
            reference_lag=supplied["reference_lag"],
            metadata={},
        )
        if supplied["reference_lag"] != REFERENCE_LAG:
            raise ConditioningArtifactError("conditioning config reference_lag differs")
        return canonical

    allowed = {
        "views",
        "control_resolution",
        "depth_mode",
        "depth_h5v4_per_view",
        "raw_metric_depth_retained",
        "high_resolution_rasters_retained",
        "control_hz",
        "render_hz",
        "image_stride",
        "reference_lag_unit",
    }
    if set(supplied) - allowed:
        raise ConditioningArtifactError(
            f"conditioning config has unknown keys: {sorted(set(supplied) - allowed)}"
        )
    expected_projection = {
        "views": list(VIEWS),
        "control_resolution": [CONTROL_HEIGHT, CONTROL_WIDTH],
        "depth_mode": "h5_v4",
        "depth_h5v4_per_view": h5_v4_recipe_document(),
        "raw_metric_depth_retained": False,
        "high_resolution_rasters_retained": False,
        "control_hz": CONTROL_HZ,
        "render_hz": RENDER_HZ,
        "image_stride": IMAGE_STRIDE,
        "reference_lag_unit": REFERENCE_LAG_UNIT,
    }
    for key, value in supplied.items():
        if value != expected_projection[key]:
            raise ConditioningArtifactError(f"conditioning config {key} differs")
    for required in (
        "views",
        "control_resolution",
        "depth_mode",
        "depth_h5v4_per_view",
        "control_hz",
        "render_hz",
        "image_stride",
        "reference_lag_unit",
    ):
        if required not in supplied:
            raise ConditioningArtifactError(f"conditioning config lacks {required}")
    return canonical


def h5_v4_control(
    metric_depth: np.ndarray,
    view: str,
    contract: Mapping[str, Any] | None = None,
    *,
    invalid_fill: str = "far",
) -> np.ndarray:
    """Convert metric depth to the exact uint8 h5_v4 Image3F control.

    ``metric_depth`` is either ``[T,B,H,W]`` or ``[T,H,W]``.  Percentiles are
    computed over the complete time sequence independently for every
    environment, matching the production trainer's per-video implementation.
    The function intentionally does not resize; apply :func:`resize_uint8_to_control`
    after conversion so the operation order remains byte-identical.
    """

    canonical_contract = validate_conditioning_contract(contract)
    if view not in VIEWS:
        raise ConditioningArtifactError(f"unknown RoboRender view: {view!r}")
    if invalid_fill not in {"far", "black"}:
        raise ConditioningArtifactError("invalid_fill must be 'far' or 'black'")
    if not isinstance(metric_depth, np.ndarray) or metric_depth.dtype.hasobject:
        raise ConditioningArtifactError("metric depth must be a non-object NumPy array")
    if not (
        np.issubdtype(metric_depth.dtype, np.integer)
        or np.issubdtype(metric_depth.dtype, np.floating)
    ):
        raise ConditioningArtifactError("metric depth must have a real numeric dtype")
    if metric_depth.ndim not in {3, 4}:
        raise ConditioningArtifactError(
            "metric depth must have shape [T,H,W] or [T,B,H,W]"
        )
    if any(int(size) <= 0 for size in metric_depth.shape):
        raise ConditioningArtifactError("metric depth dimensions must be positive")

    squeeze = metric_depth.ndim == 3
    depth = metric_depth[:, None] if squeeze else metric_depth
    parameters = canonical_contract["h5_v4_parameters"][view]
    raw_low, raw_high = parameters["raw_clip"]
    inverse_low, inverse_high = parameters["inv_clip"]
    raw_percentiles = parameters.get("raw_pct")
    normalization_range = parameters.get("norm_range")

    output = np.zeros(depth.shape, dtype=np.uint8)
    for env_index in range(depth.shape[1]):
        env_depth = depth[:, env_index]
        finite = np.isfinite(env_depth) & (env_depth > 0)
        if not finite.any():
            continue

        clip_low, clip_high = float(raw_low), float(raw_high)
        if raw_percentiles is not None:
            valid_raw = env_depth[finite]
            inside_hard_clip = (valid_raw >= raw_low) & (valid_raw <= raw_high)
            if inside_hard_clip.any():
                valid_raw = valid_raw[inside_hard_clip]
            clip_low = float(np.percentile(valid_raw, raw_percentiles[0]))
            clip_high = float(np.percentile(valid_raw, raw_percentiles[1]))
            if clip_high <= clip_low:
                clip_high = clip_low + 1e-3

        clipped = np.where(finite, np.clip(env_depth, clip_low, clip_high), clip_high)
        valid_clipped = clipped[finite]
        if normalization_range is not None:
            normalize_low, normalize_high = (
                float(normalization_range[0]),
                float(normalization_range[1]),
            )
        else:
            normalize_low = float(valid_clipped.min())
            normalize_high = float(valid_clipped.max())

        if normalize_high <= normalize_low:
            encoded = np.full(env_depth.shape, 128, dtype=np.uint8)
            encoded[~finite] = 0
            output[:, env_index] = encoded
            continue

        normalized_depth = np.clip(
            (clipped - normalize_low) / (normalize_high - normalize_low),
            0.0,
            1.0,
        )
        inverse = np.clip(1.0 / (normalized_depth + 1e-3), inverse_low, inverse_high)
        log_inverse = np.log(inverse)
        log_low, log_high = np.log(inverse_low), np.log(inverse_high)
        normalized_inverse = np.clip(
            (log_inverse - log_low) / (log_high - log_low), 0.0, 1.0
        )
        encoded = (normalized_inverse * 255.0).astype(np.uint8)
        if invalid_fill == "black":
            encoded[~finite] = 0
        output[:, env_index] = encoded

    return output[:, 0] if squeeze else output


def resize_uint8_batch(
    array: np.ndarray,
    output_hw: tuple[int, int] | list[int] = (CONTROL_HEIGHT, CONTROL_WIDTH),
) -> np.ndarray:
    """Area-downsample uint8 controls from 480x832 to 240x416 exactly.

    The production cameras use precisely a 2x supersampled raster.  Restricting
    this helper to identity or exact 2x area reduction avoids silently changing
    interpolation semantics.  The integer expression is byte-identical to
    OpenCV ``INTER_AREA`` for this fixed reduction, without adding OpenCV as an
    artifact-format dependency.
    """

    try:
        requested_hw = tuple(output_hw)
    except TypeError as exc:
        raise ConditioningArtifactError("output resolution must be [240,416]") from exc
    if requested_hw != (CONTROL_HEIGHT, CONTROL_WIDTH):
        raise ConditioningArtifactError(
            "the only supported output resolution is 240x416"
        )
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
        raise ConditioningArtifactError("resize input must be a uint8 NumPy array")
    if array.ndim < 2:
        raise ConditioningArtifactError("resize input lacks image dimensions")
    is_rgb = array.ndim >= 3 and array.shape[-1] == 3
    height_axis, width_axis = (-3, -2) if is_rgb else (-2, -1)
    height, width = int(array.shape[height_axis]), int(array.shape[width_axis])
    if (height, width) == (CONTROL_HEIGHT, CONTROL_WIDTH):
        return np.ascontiguousarray(array)
    if (height, width) != (2 * CONTROL_HEIGHT, 2 * CONTROL_WIDTH):
        raise ConditioningArtifactError(
            "resize input must be 480x832 (or already 240x416)"
        )

    if is_rgb:
        leading = array.shape[:-3]
        reshaped = array.reshape(*leading, CONTROL_HEIGHT, 2, CONTROL_WIDTH, 2, 3)
        summed = reshaped.astype(np.uint16).sum(axis=(-4, -2), dtype=np.uint16)
    else:
        leading = array.shape[:-2]
        reshaped = array.reshape(*leading, CONTROL_HEIGHT, 2, CONTROL_WIDTH, 2)
        summed = reshaped.astype(np.uint16).sum(axis=(-3, -1), dtype=np.uint16)
    # OpenCV INTER_AREA rounds half upward for an exact 2x uint8 reduction.
    return np.ascontiguousarray(((summed + 2) // 4).astype(np.uint8))


# Descriptive compatibility spelling used by early callers.
resize_uint8_to_control = resize_uint8_batch


def _array_digest(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    for field in (
        canonical.dtype.str.encode("ascii"),
        json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"),
        canonical.tobytes(order="C"),
    ):
        digest.update(len(field).to_bytes(8, "big"))
        digest.update(field)
    return digest.hexdigest()


def _require_exact_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    *,
    dtype: np.dtype[Any] | type[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    value = arrays[key]
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype or tuple(value.shape) != shape:
        raise ConditioningArtifactError(
            f"{key} must have dtype {expected_dtype} and shape {shape}"
        )
    return value


def _require_raster_content(array: np.ndarray, key: str, *, black_background: bool) -> None:
    """Reject malformed renderer inputs one image at a time.

    All-black textured-robot rows are valid when a particular camera does not
    see the robot.  The public validator applies the cross-view nonempty rule
    after checking each view's dtype, shape, and background semantics.
    """

    flattened = array.reshape(array.shape[0], -1)
    minimum = flattened.min(axis=1)
    maximum = flattened.max(axis=1)
    if black_background:
        bad = (maximum > 0) & (minimum != 0)
    else:
        bad = maximum <= minimum
    if bool(np.any(bad)):
        indices = np.flatnonzero(bad)[:8].tolist()
        problem = (
            "has nonempty image rows without the required black background"
            if black_background
            else "has empty or spatially constant image rows"
        )
        raise ConditioningArtifactError(
            f"{key} {problem}: {indices}"
        )


def derive_sparse_image_schedule(
    global_env_ids: np.ndarray,
    episode_ids: np.ndarray,
    episode_steps: np.ndarray,
    *,
    reference_lag: int = REFERENCE_LAG,
) -> dict[str, np.ndarray]:
    """Derive the one canonical 50-Hz-transition/5-Hz-image schedule.

    Reference lag is measured in source control steps. A lagged source
    transition resolves through the observation-hold map, so it always names
    an earlier generated sparse image row from the same episode.
    """

    values = tuple(np.asarray(value) for value in (global_env_ids, episode_ids, episode_steps))
    if any(value.ndim != 2 for value in values) or not (
        values[0].shape == values[1].shape == values[2].shape
    ):
        raise ConditioningArtifactError("schedule identities must share shape [T,B]")
    if any(value.dtype.kind not in "iu" for value in values):
        raise ConditioningArtifactError("schedule identities must be integer arrays")
    if type(reference_lag) is not int or reference_lag <= 0:
        raise ConditioningArtifactError("schedule reference lag must be positive")
    global_ids, episodes, steps = values
    transition_count, env_count = global_ids.shape
    if transition_count <= 0 or env_count <= 0:
        raise ConditioningArtifactError("schedule identities cannot be empty")

    rows: list[tuple[int, int, int]] = []
    locations: list[tuple[int, int]] = []
    mapping = np.empty((transition_count, env_count), dtype=np.int64)
    latest = np.full(env_count, -1, dtype=np.int64)
    for time_index in range(transition_count):
        for env_index in range(env_count):
            if int(steps[time_index, env_index]) % IMAGE_STRIDE == 0:
                latest[env_index] = len(rows)
                rows.append(
                    (
                        int(global_ids[time_index, env_index]),
                        int(episodes[time_index, env_index]),
                        int(steps[time_index, env_index]),
                    )
                )
                locations.append((time_index, env_index))
            if latest[env_index] < 0:
                raise ConditioningArtifactError("transition lacks an initial image row")
            mapping[time_index, env_index] = latest[env_index]

    row_array = np.asarray(rows, dtype=np.int64)
    transition_lookup = {
        (
            int(global_ids[time_index, env_index]),
            int(episodes[time_index, env_index]),
            int(steps[time_index, env_index]),
        ): int(mapping[time_index, env_index])
        for time_index in range(transition_count)
        for env_index in range(env_count)
    }
    source_steps = np.empty(len(rows), dtype=np.int64)
    source_indices = np.empty(len(rows), dtype=np.int64)
    for image_index, (time_index, env_index) in enumerate(locations):
        episode_step = int(steps[time_index, env_index])
        if episode_step == 0:
            source_steps[image_index] = -1
            source_indices[image_index] = -1
            continue
        source_step = max(0, episode_step - reference_lag)
        source_key = (
            int(global_ids[time_index, env_index]),
            int(episodes[time_index, env_index]),
            source_step,
        )
        try:
            source_index = transition_lookup[source_key]
        except KeyError as exc:
            raise ConditioningArtifactError(
                "image reference source transition is absent"
            ) from exc
        if source_index >= image_index:
            raise ConditioningArtifactError(
                "generated image reference must precede its consumer"
            )
        source_steps[image_index] = source_step
        source_indices[image_index] = source_index
    return {
        "transition_image_indices": mapping,
        "image_global_env_ids": row_array[:, 0],
        "image_episode_ids": row_array[:, 1],
        "image_episode_steps": row_array[:, 2],
        "image_reference_source_steps": source_steps,
        "image_reference_indices": source_indices,
    }


def validate_conditioning_arrays(
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any] | None = None,
    *,
    purpose: str = "train",
) -> dict[str, Any]:
    """Validate and canonicalize one compact raw conditioning payload."""

    canonical_contract = validate_conditioning_contract(contract)
    if purpose not in PURPOSES:
        raise ConditioningArtifactError(
            f"conditioning purpose must be one of {PURPOSES}"
        )
    if not isinstance(arrays, Mapping):
        raise ConditioningArtifactError("conditioning arrays must be a mapping")
    invalid_keys = [
        key
        for key in arrays
        if not isinstance(key, str) or _ARRAY_KEY.fullmatch(key) is None
    ]
    if invalid_keys:
        raise ConditioningArtifactError(
            f"unsafe conditioning array names: {[repr(key) for key in invalid_keys]}"
        )
    keys = set(arrays)
    missing = sorted(REQUIRED_ARRAYS - keys)
    unknown = sorted(keys - REQUIRED_ARRAYS)
    if missing or unknown:
        forbidden = sorted(keys & FORBIDDEN_LEARNER_ARRAYS)
        suffix = f" forbidden_learner_arrays={forbidden}" if forbidden else ""
        raise ConditioningArtifactError(
            f"conditioning arrays have missing={missing} unknown={unknown}{suffix}"
        )

    canonical: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ConditioningArtifactError(f"{key} must be a non-object NumPy array")
        canonical[key] = np.ascontiguousarray(value)

    expert_actions = canonical["expert_actions"]
    if expert_actions.dtype != np.float32 or expert_actions.ndim != 3:
        raise ConditioningArtifactError("expert_actions must be float32 [T,B,A]")
    steps, envs, action_dim = (int(size) for size in expert_actions.shape)
    if min(steps, envs, action_dim) <= 0:
        raise ConditioningArtifactError(
            "conditioning T, B, and action dimension must be positive"
        )
    action_shape = (steps, envs, action_dim)
    for key in ("expert_actions", "student_actions", "behavior_actions"):
        action = _require_exact_array(
            canonical, key, dtype=np.float32, shape=action_shape
        )
        if not np.isfinite(action).all():
            raise ConditioningArtifactError(f"{key} contains non-finite values")

    state = canonical["obs__actor_obs"]
    if (
        state.dtype != np.float32
        or state.ndim != 3
        or tuple(state.shape[:2]) != (steps, envs)
        or int(state.shape[2]) <= 0
        or not np.isfinite(state).all()
    ):
        raise ConditioningArtifactError("obs__actor_obs must be finite float32 [T,B,S]")
    state_dim = int(state.shape[2])

    transition_shape = (steps, envs)
    for key in ("expert_mask", "dones", "anchor_valid"):
        _require_exact_array(canonical, key, dtype=np.bool_, shape=transition_shape)
    for key in ("global_env_ids", "episode_ids", "episode_steps"):
        ids = _require_exact_array(
            canonical, key, dtype=np.int64, shape=transition_shape
        )
        if np.any(ids < 0):
            raise ConditioningArtifactError(f"{key} cannot contain negative values")

    expected_behavior = np.where(
        canonical["expert_mask"][..., None],
        canonical["expert_actions"],
        canonical["student_actions"],
    )
    if not np.array_equal(canonical["behavior_actions"], expected_behavior):
        raise ConditioningArtifactError(
            "behavior_actions must exactly select expert or student actions using expert_mask"
        )

    global_ids = canonical["global_env_ids"]
    if len(set(global_ids[0].tolist())) != envs:
        raise ConditioningArtifactError(
            "global_env_ids must be unique across environments"
        )
    if not np.array_equal(global_ids, np.broadcast_to(global_ids[0], global_ids.shape)):
        raise ConditioningArtifactError(
            "global_env_ids must remain constant through time"
        )

    episode_ids = canonical["episode_ids"]
    episode_steps = canonical["episode_steps"]
    dones = canonical["dones"]
    if np.any(episode_steps[0] != 0):
        raise ConditioningArtifactError(
            "every independently renderable shard must begin at step 0"
        )
    for env_index in range(envs):
        for time_index in range(1, steps):
            previous_episode = int(episode_ids[time_index - 1, env_index])
            current_episode = int(episode_ids[time_index, env_index])
            previous_step = int(episode_steps[time_index - 1, env_index])
            current_step = int(episode_steps[time_index, env_index])
            if bool(dones[time_index - 1, env_index]):
                if current_episode != previous_episode + 1 or current_step != 0:
                    raise ConditioningArtifactError(
                        "episode identity must increment and reset after done"
                    )
            elif (
                current_episode != previous_episode or current_step != previous_step + 1
            ):
                raise ConditioningArtifactError(
                    "episode steps must be contiguous unless the prior transition is done"
                )

    expected_anchor_valid = episode_steps == 0
    if not np.array_equal(canonical["anchor_valid"], expected_anchor_valid):
        raise ConditioningArtifactError(
            "anchor_valid must be true exactly at episode step 0"
        )

    transition_image_indices = _require_exact_array(
        canonical,
        "transition_image_indices",
        dtype=np.int64,
        shape=transition_shape,
    )
    image_vectors: dict[str, np.ndarray] = {}
    image_count: int | None = None
    for key in (
        "image_global_env_ids",
        "image_episode_ids",
        "image_episode_steps",
        "image_reference_source_steps",
        "image_reference_indices",
    ):
        value = canonical[key]
        if value.dtype != np.int64 or value.ndim != 1 or value.size == 0:
            raise ConditioningArtifactError(f"{key} must be nonempty int64 [F]")
        if image_count is None:
            image_count = int(value.size)
        elif int(value.size) != image_count:
            raise ConditioningArtifactError("sparse image vectors disagree on image count")
        image_vectors[key] = value
    assert image_count is not None  # guarded by the nonempty vectors above
    for key in ("image_global_env_ids", "image_episode_ids", "image_episode_steps"):
        if np.any(image_vectors[key] < 0):
            raise ConditioningArtifactError(f"{key} cannot contain negative values")
    for key in ("image_reference_source_steps", "image_reference_indices"):
        if np.any(image_vectors[key] < -1):
            raise ConditioningArtifactError(f"{key} cannot contain values below -1")

    expected_schedule = derive_sparse_image_schedule(
        global_ids, episode_ids, episode_steps, reference_lag=REFERENCE_LAG
    )
    for key in (
        "image_global_env_ids",
        "image_episode_ids",
        "image_episode_steps",
    ):
        if not np.array_equal(image_vectors[key], expected_schedule[key]):
            raise ConditioningArtifactError(
                "sparse image rows must exactly follow nominal cadence plus episode resets"
            )
    if not np.array_equal(
        transition_image_indices, expected_schedule["transition_image_indices"]
    ):
        raise ConditioningArtifactError(
            "transition_image_indices must hold the latest same-episode image row"
        )
    if set(transition_image_indices.ravel().tolist()) != set(range(image_count)):
        raise ConditioningArtifactError("sparse image rows must all be referenced")
    if not np.array_equal(
        image_vectors["image_reference_source_steps"],
        expected_schedule["image_reference_source_steps"],
    ):
        raise ConditioningArtifactError(
            "image reference source steps must use the 256-control-step policy"
        )
    if not np.array_equal(
        image_vectors["image_reference_indices"],
        expected_schedule["image_reference_indices"],
    ):
        raise ConditioningArtifactError(
            "image reference indices must resolve through the transition hold mapping"
        )

    anchor_global_ids = canonical["anchor_global_env_ids"]
    anchor_episode_ids = canonical["anchor_episode_ids"]
    if (
        anchor_global_ids.dtype != np.int64
        or anchor_episode_ids.dtype != np.int64
        or anchor_global_ids.ndim != 1
        or anchor_episode_ids.ndim != 1
        or anchor_global_ids.shape != anchor_episode_ids.shape
        or anchor_global_ids.size == 0
        or np.any(anchor_global_ids < 0)
        or np.any(anchor_episode_ids < 0)
    ):
        raise ConditioningArtifactError(
            "sparse anchor IDs must be nonempty nonnegative int64 [E] arrays"
        )
    anchor_keys = list(zip(anchor_global_ids.tolist(), anchor_episode_ids.tolist()))
    if anchor_keys != sorted(set(anchor_keys)):
        raise ConditioningArtifactError(
            "sparse anchors must be unique and lexicographically ordered by (global_env_id, episode_id)"
        )
    represented_episodes = sorted(
        {
            (
                int(global_ids[time_index, env_index]),
                int(episode_ids[time_index, env_index]),
            )
            for time_index in range(steps)
            for env_index in range(envs)
        }
    )
    if anchor_keys != represented_episodes:
        raise ConditioningArtifactError(
            "sparse anchor table must contain exactly one anchor for every represented episode"
        )
    episode_count = len(anchor_keys)

    robot_visible = np.zeros(image_count, dtype=np.bool_)
    for view in VIEWS:
        depth = _require_exact_array(
            canonical,
            f"depth_ctrl__{view}",
            dtype=np.uint8,
            shape=(
                image_count,
                int(canonical_contract["resolution"][0]),
                int(canonical_contract["resolution"][1]),
            ),
        )
        robot = _require_exact_array(
            canonical,
            f"rgb_robot__{view}",
            dtype=np.uint8,
            shape=(
                image_count,
                int(canonical_contract["resolution"][0]),
                int(canonical_contract["resolution"][1]),
                3,
            ),
        )
        anchors = _require_exact_array(
            canonical,
            f"rgb_anchor__{view}",
            dtype=np.uint8,
            shape=(
                episode_count,
                int(canonical_contract["resolution"][0]),
                int(canonical_contract["resolution"][1]),
                3,
            ),
        )
        _require_raster_content(depth, f"depth_ctrl__{view}", black_background=False)
        _require_raster_content(robot, f"rgb_robot__{view}", black_background=True)
        _require_raster_content(anchors, f"rgb_anchor__{view}", black_background=False)
        robot_visible |= np.any(robot.reshape(image_count, -1) != 0, axis=1)

    if bool(np.any(~robot_visible)):
        indices = np.flatnonzero(~robot_visible)[:8].tolist()
        raise ConditioningArtifactError(
            "rgb_robot is empty across all synchronized views for image rows: "
            f"{indices}"
        )

    return {
        "arrays": canonical,
        "num_steps": steps,
        "num_envs": envs,
        "transition_count": steps * envs,
        "episode_count": episode_count,
        "image_count": image_count,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "purpose": purpose,
        "contract": canonical_contract,
    }


def _control_contract() -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "views": list(VIEWS),
        "view_names": list(VIEWS),
        "resolution": [CONTROL_HEIGHT, CONTROL_WIDTH],
        "sensor_resolution": [SENSOR_HEIGHT, SENSOR_WIDTH],
        "control_resolution": [CONTROL_HEIGHT, CONTROL_WIDTH],
        "depth_mode": "h5_v4",
        "depth": {"dtype": "uint8", "encoding": DEPTH_ENCODING, "layout": "FHW"},
        "robot_rgb": {
            "dtype": "uint8",
            "encoding": "textured_robot_on_black",
            "layout": "FHWC",
        },
        "image_rows": {
            "layout": "F",
            "order": "transition_first_use",
            "sample_rule": "episode_step_mod_stride",
            "transition_mapping": "latest_same_episode_image_row",
        },
        "anchors": {
            "dtype": "uint8",
            "layout": "EHWC",
            "storage": "sparse_episode_table",
            "key": ["global_env_id", "episode_id"],
        },
        "raw_metric_depth_retained": False,
        "high_resolution_rasters_retained": False,
        "h5_v4_parameters": h5_v4_recipe_document(),
        "depth_h5v4_per_view": h5_v4_recipe_document(),
        "reference_policy": REFERENCE_POLICY,
        "reference_lag": REFERENCE_LAG,
        "reference_lag_unit": REFERENCE_LAG_UNIT,
        "control_hz": CONTROL_HZ,
        "render_hz": RENDER_HZ,
        "image_stride": IMAGE_STRIDE,
    }


def _canonical_json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConditioningArtifactError(f"{label} must be a JSON mapping")
    decoded = _plain_json_value(value, label)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise ConditioningArtifactError(f"{label} must be a JSON object")
    return decoded


def _validate_render_fields(
    *, prompt: Any, seed: Any, reference_lag: Any, metadata: Any
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip() or "\x00" in prompt:
        raise ConditioningArtifactError(
            "render prompt must be a nonempty string without NUL"
        )
    try:
        prompt_size = len(prompt.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ConditioningArtifactError("render prompt must be valid UTF-8") from exc
    if prompt_size > 16 * 1024:
        raise ConditioningArtifactError("render prompt exceeds 16 KiB")
    if type(seed) is not int or seed < 0 or seed > 2**63 - 1:
        raise ConditioningArtifactError("render seed must be an integer in [0, 2^63-1]")
    if type(reference_lag) is not int or reference_lag <= 0:
        raise ConditioningArtifactError(
            "render reference_lag must be a positive integer"
        )
    return {
        "mode": RENDER_MODE,
        "prompt": prompt,
        "seed": seed,
        "reference_lag": reference_lag,
        "reference_lag_unit": REFERENCE_LAG_UNIT,
        "control_hz": CONTROL_HZ,
        "render_hz": RENDER_HZ,
        "image_stride": IMAGE_STRIDE,
        "metadata": _canonical_json_mapping(metadata, "render metadata"),
    }


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _array_digest(array),
        }
        for key, array in sorted(arrays.items())
    }


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _check_path_chain(path: Path) -> None:
    path = _absolute(path)
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ConditioningArtifactError(f"symbolic links are forbidden: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ConditioningArtifactError(
                f"path ancestor is not a directory: {current}"
            )


def _ensure_parent(path: Path) -> None:
    _check_path_chain(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConditioningArtifactError(
            f"cannot create artifact directory: {exc}"
        ) from exc
    _check_path_chain(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_npz_temporary(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    _ensure_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            with zipfile.ZipFile(
                handle, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                for key in sorted(arrays):
                    info = zipfile.ZipInfo(
                        f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                    )
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
            os.fchmod(handle.fileno(), _READ_ONLY_MODE)
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    temporary = _write_npz_temporary(path, arrays)
    try:
        temporary_sha = sha256_file(temporary)
        try:
            os.link(temporary, path, follow_symlinks=False)
            _fsync_directory(path.parent)
        except FileExistsError:
            try:
                info = os.lstat(path)
                existing_sha = sha256_file(path)
            except (OSError, ArtifactError) as exc:
                raise ConditioningArtifactError(
                    f"cannot verify existing payload: {exc}"
                ) from exc
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_mode & 0o222
                or existing_sha != temporary_sha
            ):
                raise ConditioningArtifactError(
                    f"immutable conditioning payload conflict: {path}"
                )
        return temporary_sha
    except (OSError, ArtifactError) as exc:
        if isinstance(exc, ConditioningArtifactError):
            raise
        raise ConditioningArtifactError(
            f"cannot publish conditioning payload: {exc}"
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_output_paths(
    payload_path: str | os.PathLike[str], manifest_path: str | os.PathLike[str] | None
) -> tuple[Path, Path]:
    payload = _absolute(payload_path)
    manifest = (
        payload.with_name(f"{payload.stem}.manifest.json")
        if manifest_path is None
        else _absolute(manifest_path)
    )
    if payload.suffix != ".npz" or payload.name.startswith("."):
        raise ConditioningArtifactError(
            "conditioning payload must be a visible .npz file"
        )
    if manifest.suffix != ".json" or manifest.name.startswith("."):
        raise ConditioningArtifactError(
            "conditioning manifest must be a visible .json file"
        )
    if payload.parent != manifest.parent or payload == manifest:
        raise ConditioningArtifactError(
            "conditioning payload and manifest must be distinct siblings"
        )
    _ensure_parent(payload)
    _check_path_chain(manifest)
    return payload, manifest


def publish_conditioning_shard(
    payload_path: str | os.PathLike[str],
    arrays: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any] | None = None,
    purpose: str,
    prompt: str,
    seed: int,
    reference_lag: int,
    render_metadata: Mapping[str, Any],
    manifest_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Publish a deterministic NPZ followed by its immutable JSON seal."""

    payload, manifest_path_value = _validate_output_paths(payload_path, manifest_path)
    validated = validate_conditioning_arrays(arrays, contract, purpose=purpose)
    canonical_arrays = validated["arrays"]
    render = _validate_render_fields(
        prompt=prompt,
        seed=seed,
        reference_lag=reference_lag,
        metadata=render_metadata,
    )
    if render["reference_lag"] != validated["contract"]["reference_lag"]:
        raise ConditioningArtifactError(
            "render reference_lag differs from conditioning contract"
        )
    payload_sha = _publish_npz(payload, canonical_arrays)
    payload_size = os.stat(payload, follow_symlinks=False).st_size
    document = {
        "schema": SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "sealed": True,
        "learner_replay_ready": False,
        "replay_admission": REPLAY_ADMISSION,
        "purpose": purpose,
        "num_steps": validated["num_steps"],
        "num_envs": validated["num_envs"],
        "transition_count": validated["transition_count"],
        "episode_count": validated["episode_count"],
        "image_count": validated["image_count"],
        "action_dim": validated["action_dim"],
        "state_dim": validated["state_dim"],
        "control_contract": validated["contract"],
        "render": render,
        "arrays": _array_metadata(canonical_arrays),
        "payload": {
            "path": payload.name,
            "sha256": payload_sha,
            "size_bytes": payload_size,
        },
    }
    try:
        manifest_sha = publish_immutable_json(manifest_path_value, document)
    except ArtifactError as exc:
        raise ConditioningArtifactError(
            f"cannot publish conditioning manifest: {exc}"
        ) from exc
    for path, label in ((payload, "payload"), (manifest_path_value, "manifest")):
        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
            raise ConditioningArtifactError(
                f"published conditioning {label} is not read-only"
            )
    return {
        **document,
        "manifest_path": str(manifest_path_value),
        "manifest_sha256": manifest_sha,
        "payload_path": str(payload),
    }


def publish_raw_conditioning_artifact(
    payload_path: str | os.PathLike[str],
    arrays: Mapping[str, np.ndarray],
    *,
    prompt: str,
    seed: int,
    reference_lag: int,
    render_metadata: Mapping[str, Any],
    manifest_path: str | os.PathLike[str] | None = None,
    contract: Mapping[str, Any] | None = None,
    purpose: str = "train",
) -> dict[str, Any]:
    """Compatibility wrapper for :func:`publish_conditioning_shard`."""

    return publish_conditioning_shard(
        payload_path,
        arrays,
        contract=contract,
        purpose=purpose,
        prompt=prompt,
        seed=seed,
        reference_lag=reference_lag,
        render_metadata=render_metadata,
        manifest_path=manifest_path,
    )


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ConditioningArtifactError(
            f"{label} has missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )
    return dict(value)


def _validate_manifest(document: Any) -> dict[str, Any]:
    manifest = _require_exact_keys(document, _TOP_LEVEL_KEYS, "conditioning manifest")
    if (
        manifest["schema"] != SCHEMA
        or manifest["artifact_kind"] != ARTIFACT_KIND
        or manifest["sealed"] is not True
        or manifest["learner_replay_ready"] is not False
        or manifest["replay_admission"] != REPLAY_ADMISSION
    ):
        raise ConditioningArtifactError(
            "conditioning artifact identity/admission differs"
        )
    if manifest["purpose"] not in PURPOSES:
        raise ConditioningArtifactError("conditioning artifact purpose differs")
    for key in (
        "num_steps",
        "num_envs",
        "transition_count",
        "episode_count",
        "image_count",
        "action_dim",
        "state_dim",
    ):
        if type(manifest[key]) is not int or manifest[key] <= 0:
            raise ConditioningArtifactError(
                f"manifest {key} must be a positive integer"
            )
    if manifest["transition_count"] != manifest["num_steps"] * manifest["num_envs"]:
        raise ConditioningArtifactError("manifest transition count is inconsistent")
    validate_conditioning_contract(manifest["control_contract"])

    render = _require_exact_keys(manifest["render"], _RENDER_KEYS, "render contract")
    if render["mode"] != RENDER_MODE:
        raise ConditioningArtifactError("render mode differs")
    if (
        _validate_render_fields(
            prompt=render["prompt"],
            seed=render["seed"],
            reference_lag=render["reference_lag"],
            metadata=render["metadata"],
        )
        != render
    ):
        raise ConditioningArtifactError("render contract is not canonical")
    if render["reference_lag"] != manifest["control_contract"]["reference_lag"]:
        raise ConditioningArtifactError(
            "render reference_lag differs from control contract"
        )
    for key in ("reference_lag_unit", "control_hz", "render_hz", "image_stride"):
        if render[key] != manifest["control_contract"][key]:
            raise ConditioningArtifactError(f"render {key} differs from control contract")

    payload = _require_exact_keys(manifest["payload"], _PAYLOAD_KEYS, "payload")
    if (
        not isinstance(payload["path"], str)
        or Path(payload["path"]).name != payload["path"]
        or not payload["path"].endswith(".npz")
        or not isinstance(payload["sha256"], str)
        or _SHA256.fullmatch(payload["sha256"]) is None
        or type(payload["size_bytes"]) is not int
        or payload["size_bytes"] <= 0
    ):
        raise ConditioningArtifactError("conditioning payload descriptor is malformed")

    metadata = manifest["arrays"]
    if not isinstance(metadata, Mapping) or set(metadata) != REQUIRED_ARRAYS:
        raise ConditioningArtifactError("conditioning array inventory differs")
    for key, row in metadata.items():
        row = _require_exact_keys(row, _ARRAY_METADATA_KEYS, f"arrays.{key}")
        if (
            not isinstance(row["dtype"], str)
            or not isinstance(row["shape"], list)
            or not row["shape"]
            or any(type(size) is not int or size <= 0 for size in row["shape"])
            or not isinstance(row["sha256"], str)
            or _SHA256.fullmatch(row["sha256"]) is None
        ):
            raise ConditioningArtifactError(f"arrays.{key} metadata is malformed")
    return manifest


def _open_sealed_file(path: Path, label: str) -> BinaryIO:
    _check_path_chain(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
            raise ConditioningArtifactError(f"{label} must be a read-only regular file")
        return os.fdopen(descriptor, "rb")
    except Exception:
        if "descriptor" in locals():
            os.close(descriptor)
        raise


def _load_npz(
    path: Path, expected_sha: str, expected_size: int
) -> dict[str, np.ndarray]:
    try:
        with _open_sealed_file(path, "conditioning payload") as handle:
            info = os.fstat(handle.fileno())
            if info.st_size != expected_size:
                raise ConditioningArtifactError("conditioning payload size differs")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha:
                raise ConditioningArtifactError("conditioning payload SHA-256 differs")
            handle.seek(0)
            with zipfile.ZipFile(handle, mode="r") as archive:
                members = archive.infolist()
                names = [member.filename for member in members]
                expected_names = {f"{key}.npy" for key in REQUIRED_ARRAYS}
                if len(names) != len(set(names)) or set(names) != expected_names:
                    raise ConditioningArtifactError(
                        "conditioning NPZ member inventory differs"
                    )
                if any(
                    member.compress_type != zipfile.ZIP_STORED for member in members
                ):
                    raise ConditioningArtifactError(
                        "conditioning NPZ must use stored members"
                    )
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                return {
                    key: np.ascontiguousarray(archive[key]) for key in archive.files
                }
    except ConditioningArtifactError:
        raise
    except Exception as exc:
        raise ConditioningArtifactError(
            f"cannot load conditioning payload: {exc}"
        ) from exc


def load_raw_conditioning_artifact(
    manifest_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and recursively verify a sealed raw conditioning artifact."""

    manifest_file = _absolute(manifest_path)
    try:
        with _open_sealed_file(manifest_file, "conditioning manifest") as handle:
            raw_manifest = handle.read()
        document = load_strict_json(manifest_file)
    except (ArtifactError, OSError) as exc:
        raise ConditioningArtifactError(
            f"cannot load conditioning manifest: {exc}"
        ) from exc
    manifest = _validate_manifest(document)
    canonical_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if raw_manifest != canonical_bytes:
        raise ConditioningArtifactError("conditioning manifest is not canonical JSON")

    payload_path = manifest_file.parent / manifest["payload"]["path"]
    arrays = _load_npz(
        payload_path,
        manifest["payload"]["sha256"],
        manifest["payload"]["size_bytes"],
    )
    validated = validate_conditioning_arrays(
        arrays, manifest["control_contract"], purpose=manifest["purpose"]
    )
    for key in (
        "num_steps",
        "num_envs",
        "transition_count",
        "episode_count",
        "image_count",
        "action_dim",
        "state_dim",
    ):
        if validated[key] != manifest[key]:
            raise ConditioningArtifactError(f"payload {key} differs from its manifest")
    if _array_metadata(validated["arrays"]) != manifest["arrays"]:
        raise ConditioningArtifactError(
            "payload array metadata/digests differ from manifest"
        )
    return (
        {
            **manifest,
            "manifest_path": str(manifest_file),
            "manifest_sha256": sha256_file(manifest_file),
            "payload_path": str(payload_path),
        },
        validated["arrays"],
    )


def verify_raw_conditioning_artifact(
    manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify a raw artifact and return its manifest report without arrays."""

    report, _ = load_raw_conditioning_artifact(manifest_path)
    return report
