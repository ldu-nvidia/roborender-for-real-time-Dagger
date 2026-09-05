"""Frozen-actor rollout export helpers for round-based DAgger.

This module deliberately contains no Isaac Lab imports.  The native actor calls
``export_trainer_rollout`` after one frozen rollout; unit tests can exercise the
same conversion path with an in-memory storage double.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactError, publish_rollout_shard, verify_rollout_shard
from .actor_identity import actor_process_identity
from .native_actor_protocol import native_actor_generation, train_round_relative_root
from .policy_identity import (
    PolicyIdentityError,
    runtime_policy_identity_sha256,
    validate_homie_controller_sha256,
)


class ActorExportError(RuntimeError):
    """The actor rollout does not satisfy the immutable shard contract."""


_NATIVE_OBSERVATION_ENVIRONMENT = {
    "observation_source": "ROUND_DAGGER_OBSERVATION_SOURCE",
    "observation_runtime_sha256": "ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256",
    "isaac_runtime_sha256": "ROUND_DAGGER_ISAAC_RUNTIME_SHA256",
    "actor_python_freeze_sha256": "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256",
    "actor_source_inventory_sha256": (
        "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256"
    ),
    "gr00t_revision": "ROUND_DAGGER_GR00T_REVISION",
    "gr00t_source_tree_sha256": "ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256",
    "gpu_name": "ROUND_DAGGER_GPU_NAME",
    "compute_capability": "ROUND_DAGGER_COMPUTE_CAPABILITY",
    "gpu_count": "ROUND_DAGGER_GPU_COUNT",
    "physical_gpu_inventory": "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON",
    "physical_gpu_inventory_sha256": (
        "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256"
    ),
}

# BC/DAgger generation collection currently runs as one independent actor on
# admitted physical GPU 0.  The complete eight-H200 host inventory remains a
# separate runtime-attestation fact; it is not the actor's logical world size.
_NATIVE_GENERATION_ACTOR_WORLD_SIZE = 1
_CANONICAL_NONNEGATIVE_INTEGER = re.compile(r"0|[1-9][0-9]*\Z")
_NATIVE_GENERATION_NAMING_ENVIRONMENT = (
    "ROUND_DAGGER_CAMPAIGN_ROOT",
    "ROUND_DAGGER_GENERATION_ROOT",
    "ROUND_DAGGER_GENERATION_ID",
    "ROUND_DAGGER_MACRO_ROUND",
    "ROUND_DAGGER_ACTOR_ROLE",
    "ROUND_DAGGER_BETA",
    "ROUND_DAGGER_SATURATION_PHASE",
    "ROUND_DAGGER_SATURATION_WAVE_INDEX",
    "ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON",
)


def _native_generation_shard_ids_from_environment(
    environment: Mapping[str, str],
    *,
    rank: int,
    batch_size: int,
    environments_per_shard: int,
) -> tuple[str, ...]:
    """Derive exact generation-scoped shard IDs for one saturation rank."""

    # Macro-round/cohort variables are shared with historical Image3F actors.
    # The generation ID is the unambiguous discriminator for the native H200
    # generation naming contract.
    if not environment.get("ROUND_DAGGER_GENERATION_ID"):
        return ()
    missing = [
        name
        for name in _NATIVE_GENERATION_NAMING_ENVIRONMENT
        if not environment.get(name)
    ]
    if missing:
        raise ActorExportError(
            "native generation naming is incomplete: " + ", ".join(missing)
        )
    if environment.get("ROUND_DAGGER_OBSERVATION_SOURCE") != "native":
        raise ActorExportError("generation shard naming requires native observations")
    if environment.get("ROUND_DAGGER_PURPOSE") != "train":
        raise ActorExportError("native generation shard naming requires purpose=train")
    try:
        generation = native_actor_generation(
            environment["ROUND_DAGGER_GENERATION_ID"]
        )
    except ValueError as exc:
        raise ActorExportError(str(exc)) from exc
    if (
        environment["ROUND_DAGGER_MACRO_ROUND"] != str(generation.macro_round)
        or environment["ROUND_DAGGER_ACTOR_ROLE"] != generation.actor_role
        or environment["ROUND_DAGGER_BETA"] != generation.beta_text
    ):
        raise ActorExportError(
            "native generation macro-round/role/beta identity differs"
        )
    supplied_parent = environment.get("ROUND_DAGGER_PARENT_GENERATION_ID")
    if generation.parent_generation_id is None:
        if supplied_parent not in {None, ""}:
            raise ActorExportError("BC collection must not name a parent generation")
    elif supplied_parent != generation.parent_generation_id:
        raise ActorExportError(
            "DAgger collection must name the immediately preceding generation"
        )
    if environment["ROUND_DAGGER_SATURATION_PHASE"] != "collect":
        raise ActorExportError("generation shards may only be exported during collect")
    if (
        type(rank) is not int
        or rank < 0
        or rank >= _NATIVE_GENERATION_ACTOR_WORLD_SIZE
    ):
        raise ActorExportError("native generation actor rank must be exactly zero")
    if type(batch_size) is not int or batch_size < 1:
        raise ActorExportError("native generation batch size must be positive")
    if type(environments_per_shard) is not int or environments_per_shard < 1:
        raise ActorExportError("native generation environments_per_shard must be positive")

    identity = actor_process_identity(environment)
    if (
        identity is None
        or identity.world_size != _NATIVE_GENERATION_ACTOR_WORLD_SIZE
        or identity.rank != rank
    ):
        raise ActorExportError(
            "native generation shard naming requires the exact world-1/rank-0 actor identity"
        )

    wave_raw = environment["ROUND_DAGGER_SATURATION_WAVE_INDEX"]
    if _CANONICAL_NONNEGATIVE_INTEGER.fullmatch(wave_raw) is None:
        raise ActorExportError("saturation wave index must be a canonical non-negative integer")
    wave_index = int(wave_raw)
    if wave_index > 999:
        raise ActorExportError("saturation wave index exceeds the three-digit filename contract")

    campaign_root = Path(environment["ROUND_DAGGER_CAMPAIGN_ROOT"])
    generation_root = Path(environment["ROUND_DAGGER_GENERATION_ROOT"])
    cohort_root = Path(environment.get("ROUND_DAGGER_COHORT_DIR", ""))
    if not campaign_root.is_absolute():
        raise ActorExportError("native generation campaign root must be absolute")
    exact_generation_root = (
        campaign_root
        / train_round_relative_root(generation.generation_id)
        / "collection"
    )
    exact_cohort_root = (
        exact_generation_root / "conditioning" / "waves" / f"wave-{wave_index:03d}"
    )
    if generation_root != exact_generation_root or cohort_root != exact_cohort_root:
        raise ActorExportError(
            "native generation/cohort roots must match the exact campaign wave namespace"
        )

    part_count = (batch_size + environments_per_shard - 1) // environments_per_shard
    expected_names: list[str] = []
    rank_ids: list[str] = []
    for gpu in range(_NATIVE_GENERATION_ACTOR_WORLD_SIZE):
        for part_index in range(part_count):
            shard_index = (
                wave_index * _NATIVE_GENERATION_ACTOR_WORLD_SIZE * part_count
                + gpu * part_count
                + part_index
            )
            shard_id = (
                f"{generation.generation_id}__wave-{wave_index:03d}"
                f"__gpu-{gpu:02d}__shard-{shard_index:06d}"
            )
            expected_names.extend((f"{shard_id}.npz", f"{shard_id}.json"))
            if gpu == rank:
                rank_ids.append(shard_id)
    try:
        supplied_names = json.loads(
            environment["ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON"]
        )
    except json.JSONDecodeError as exc:
        raise ActorExportError("expected shard inventory is not valid JSON") from exc
    if supplied_names != expected_names:
        raise ActorExportError(
            "expected shard inventory differs from the exact wave/GPU/part naming contract"
        )
    return tuple(rank_ids)


def _validate_native_export_role(
    trainer: Any, environment: Mapping[str, str], cohort_dir: Path
) -> None:
    if getattr(trainer, "observation_source", None) != "native":
        return
    role = getattr(trainer, "native_role", None)
    expected_purpose = {
        "bc_collect": "train",
        "dagger_collect": "train",
        "validation_collect": "validation",
    }.get(role)
    if expected_purpose is None:
        raise ActorExportError("only native collection roles may export conditioning")
    if environment.get("ROUND_DAGGER_PURPOSE") != expected_purpose:
        raise ActorExportError("native collection role and artifact purpose disagree")
    if role == "validation_collect":
        raw_root = environment.get("ROUND_DAGGER_VALIDATION_CONDITIONING_ROOT")
        if not raw_root:
            raise ActorExportError(
                "validation_collect requires ROUND_DAGGER_VALIDATION_CONDITIONING_ROOT"
            )
        root = Path(raw_root)
        if not root.is_absolute() or not cohort_dir.is_absolute():
            raise ActorExportError("validation conditioning paths must be absolute")
        try:
            relative = cohort_dir.relative_to(root)
        except ValueError as exc:
            raise ActorExportError(
                "validation_collect cohort must be under the heldout validation root"
            ) from exc
        if not relative.parts or ".." in relative.parts:
            raise ActorExportError(
                "validation_collect cohort must be a child of the heldout validation root"
            )


def _native_rank_motion_ids(
    trainer: Any,
    environment: Mapping[str, str],
    *,
    rank: int,
    batch_size: int,
) -> tuple[str, ...] | None:
    """Resolve the fixed source motion for every local environment stream."""

    raw = environment.get("ROUND_DAGGER_GROUP_ROTATION_JSON")
    if raw is None:
        return None
    try:
        rotation = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ActorExportError("group rotation is not valid JSON") from exc
    if not isinstance(rotation, Mapping) or set(rotation) != {
        "start_group_offset",
        "rank_group_offsets",
        "counts",
    }:
        raise ActorExportError("group rotation fields differ")
    offsets = rotation["rank_group_offsets"]
    groups = tuple(getattr(trainer, "native_group_ids", ()) or ())
    identity = actor_process_identity(environment)
    if (
        not groups
        or len(set(groups)) != len(groups)
        or identity is None
        or identity.rank != rank
        or identity.world_size > 8
        or not isinstance(offsets, list)
        or len(offsets) != identity.world_size
        or any(type(value) is not int or value < 0 for value in offsets)
        or not 0 <= rank < identity.world_size
    ):
        raise ActorExportError("group rotation cannot resolve local motion IDs")
    offset = offsets[rank]
    return tuple(groups[(offset + index) % len(groups)] for index in range(batch_size))


def _observation_provenance_from_environment(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Return an exact artifact bundle while preserving historical Image3F."""

    source = environment.get("ROUND_DAGGER_OBSERVATION_SOURCE")
    native_names = tuple(_NATIVE_OBSERVATION_ENVIRONMENT.values())[1:]
    if source is None:
        ambient = [name for name in native_names if environment.get(name)]
        if ambient:
            raise ActorExportError(
                "observation provenance is missing ROUND_DAGGER_OBSERVATION_SOURCE"
            )
        return {}
    if source not in {"image3f", "native"}:
        raise ActorExportError(
            "ROUND_DAGGER_OBSERVATION_SOURCE must be exactly native or image3f"
        )
    if source == "image3f":
        forbidden = [name for name in native_names[1:] if environment.get(name)]
        if forbidden:
            raise ActorExportError(
                "Image3F actor cannot claim native runtime provenance: "
                + ", ".join(forbidden)
            )
        runtime = environment.get("ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256")
        # Explicit legacy launchers set the source discriminator but use the
        # pre-provenance artifact schema.  A new Image3F contract may opt into
        # the source-neutral source/runtime pair.
        if not runtime:
            return {}
        return {
            "observation_source": source,
            "observation_runtime_sha256": runtime,
        }
    missing = [
        name
        for name in _NATIVE_OBSERVATION_ENVIRONMENT.values()
        if not environment.get(name)
    ]
    if missing:
        raise ActorExportError(
            "native observation provenance is incomplete: " + ", ".join(missing)
        )
    result: dict[str, Any] = {
        field: environment[name]
        for field, name in _NATIVE_OBSERVATION_ENVIRONMENT.items()
    }
    try:
        result["gpu_count"] = int(result["gpu_count"])
    except (TypeError, ValueError) as exc:
        raise ActorExportError("ROUND_DAGGER_GPU_COUNT must be an integer") from exc
    try:
        result["physical_gpu_inventory"] = json.loads(
            result["physical_gpu_inventory"]
        )
    except json.JSONDecodeError as exc:
        raise ActorExportError(
            "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON is not valid JSON"
        ) from exc
    return result


def _as_numpy(value: Any) -> np.ndarray:
    """Detach a tensor-like value and return a C-contiguous CPU array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.ascontiguousarray(np.asarray(value))


def runtime_state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Content-address a policy state mapping independent of serialization.

    Tensor names, dtype, shape, and exact bytes are all included.  The result
    is stable across device placement and checkpoint container formats.
    """

    if not isinstance(state, Mapping) or not state:
        raise ActorExportError("policy state_dict must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        encoded_name = str(name).encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        if hasattr(value, "detach"):
            import torch

            value = value.detach().cpu().contiguous()
            dtype = str(value.dtype)
            shape = tuple(int(v) for v in value.shape)
            # Viewing as bytes also works for bfloat16, which NumPy cannot
            # represent directly on every supported PyTorch version.
            raw = value.reshape(-1).view(torch.uint8)
            payload = raw.numpy().tobytes(order="C")
        else:
            array = np.ascontiguousarray(np.asarray(value))
            dtype = array.dtype.str
            shape = tuple(int(v) for v in array.shape)
            payload = array.tobytes(order="C")
        encoded_dtype = dtype.encode("ascii")
        encoded_shape = ",".join(str(v) for v in shape).encode("ascii")
        digest.update(len(encoded_dtype).to_bytes(8, "big"))
        digest.update(encoded_dtype)
        digest.update(len(encoded_shape).to_bytes(8, "big"))
        digest.update(encoded_shape)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def runtime_policy_sha256(
    module: Any, *, homie_controller_sha256: Mapping[str, str]
) -> str:
    """Content-address every model that contributes to executed actor actions."""

    if hasattr(module, "module"):
        module = module.module
    if not hasattr(module, "state_dict"):
        raise ActorExportError("policy does not expose state_dict()")
    try:
        controllers = validate_homie_controller_sha256(homie_controller_sha256)
        return runtime_policy_identity_sha256(
            runtime_state_dict_sha256(module.state_dict()), controllers
        )
    except PolicyIdentityError as exc:
        raise ActorExportError(str(exc)) from exc


def denormalize_policy_rgb(
    vision: Any,
    *,
    image_mean: Sequence[float] | None,
    image_std: Sequence[float] | None,
) -> np.ndarray:
    """Convert stored normalized ``[T,B,H,W,3]`` vision to exact uint8 RGB."""

    rgb = _as_numpy(vision).astype(np.float32, copy=False)
    if rgb.ndim != 5 or rgb.shape[-1] != 3:
        raise ActorExportError(
            "vision_obs must have shape [T,B,H,W,3], got " f"{tuple(rgb.shape)}"
        )
    if (image_mean is None) != (image_std is None):
        raise ActorExportError("image_mean and image_std must be provided together")
    if image_mean is not None:
        mean = np.asarray(tuple(image_mean), dtype=np.float32)
        std = np.asarray(tuple(image_std), dtype=np.float32)
        if mean.shape != (3,) or std.shape != (3,) or not np.all(std > 0):
            raise ActorExportError("RGB normalization must contain three positive scales")
        rgb = rgb * std.reshape(1, 1, 1, 1, 3)
        rgb = rgb + mean.reshape(1, 1, 1, 1, 3)
    if not np.isfinite(rgb).all():
        raise ActorExportError("vision_obs contains non-finite values")
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


_CANONICAL_STORAGE_KEYS = (
    "vision_obs",
    "gt_actions",
    "student_actions",
    "actions",
    "expert_mask",
    "dones",
    "round_observation_is_raytraced",
    "round_global_env_ids",
    "round_episode_ids",
    "round_episode_steps",
    "round_reference_kinds",
    "round_reference_lags",
    "round_reference_source_steps",
    "round_reference_sha256",
    "round_generated_sha256",
    "round_conditioning_depth_nonzero_fraction",
    "round_conditioning_mask_nonzero_fraction",
)
_NATIVE_STORAGE_KEYS = (
    "gt_actions",
    "student_actions",
    "actions",
    "expert_mask",
    "dones",
    "round_global_env_ids",
    "round_episode_ids",
    "round_episode_steps",
)


def _storage_value(storage: Any, key: str) -> Any:
    try:
        return storage.query_key(key)
    except Exception as exc:  # pragma: no cover - trainer-specific exception
        raise ActorExportError(f"rollout storage is missing required key {key!r}") from exc


def _storage_array_slice(
    value: Any,
    *,
    key: str,
    steps: int,
    env_start: int,
    env_stop: int,
) -> np.ndarray:
    """Copy only one environment slice from rollout storage to host memory."""

    try:
        # This slice must happen before detach/cpu/numpy.  In production the
        # source is a CUDA tensor whose full [T,B,...] vision allocation is
        # several tens of GiB after conversion and RGB denormalization.
        sliced = value[:steps, env_start:env_stop]
    except Exception as exc:
        raise ActorExportError(
            f"storage key {key!r} cannot be sliced on [T,B]"
        ) from exc
    array = _as_numpy(sliced)
    expected_prefix = (steps, env_stop - env_start)
    if array.ndim < 2 or tuple(array.shape[:2]) != expected_prefix:
        raise ActorExportError(
            f"storage key {key!r} slice has prefix {tuple(array.shape[:2])}, "
            f"expected {expected_prefix}"
        )
    return array


def _rollout_storage(
    trainer: Any,
) -> tuple[
    int,
    int,
    Mapping[str, Any],
    Sequence[float] | None,
    Sequence[float] | None,
]:
    """Preflight one completed rollout without copying tensor payloads.

    Every required key and its [T,B] layout is checked before the first shard
    can be published.  This keeps fail-closed behavior while allowing the
    payload itself to be copied from CUDA in bounded environment slices.
    """

    storage = trainer.storage
    steps = int(storage.step)
    expected_steps = int(trainer.num_steps_per_env)
    if steps != expected_steps or steps <= 0:
        raise ActorExportError(
            f"rollout is incomplete: stored {steps}, expected {expected_steps} steps"
        )

    native = getattr(trainer, "observation_source", None) == "native"
    observation_keys = (
        ("actor_obs",)
        if native
        else tuple(
            key
            for key in sorted(getattr(trainer, "algo_obs_dim_dict", {}))
            if key != "vision_obs"
        )
    )
    if native and "actor_obs" not in getattr(trainer, "algo_obs_dim_dict", {}):
        raise ActorExportError("native student replay requires actor_obs")
    # A trainer may expose one of the canonical tensors as an additional
    # learner observation. Query duplicate source keys once; their canonical
    # and ``obs__`` outputs remain distinct.
    canonical_keys = _NATIVE_STORAGE_KEYS if native else _CANONICAL_STORAGE_KEYS
    keys = tuple(dict.fromkeys((*canonical_keys, *observation_keys)))

    values: dict[str, Any] = {}
    batch_size: int | None = None
    for key in keys:
        value = _storage_value(storage, key)
        try:
            shape = tuple(int(dimension) for dimension in value.shape)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ActorExportError(
                f"storage key {key!r} does not expose a valid shape"
            ) from exc
        if len(shape) < 2 or shape[0] < steps or shape[1] <= 0:
            raise ActorExportError(
                f"storage key {key!r} must contain at least [T,B]={steps},positive; "
                f"got {shape}"
            )
        if batch_size is None:
            batch_size = shape[1]
        elif shape[1] != batch_size:
            raise ActorExportError(
                "rollout storage keys disagree on environment dimension"
            )
        values[key] = value

    if batch_size is None:  # pragma: no cover - canonical keys make this unreachable
        raise ActorExportError("rollout storage is empty")
    cam_cfg = trainer.env.config.simulator.config.cameras
    return (
        steps,
        batch_size,
        values,
        getattr(cam_cfg, "image_mean", None),
        getattr(cam_cfg, "image_std", None),
    )


def _trainer_rollout_slice(
    trainer: Any,
    *,
    steps: int,
    values: Mapping[str, Any],
    image_mean: Sequence[float] | None,
    image_std: Sequence[float] | None,
    env_start: int,
    env_stop: int,
) -> dict[str, np.ndarray]:
    """Extract one contiguous environment slice into the actor shard schema."""

    def storage_array(key: str) -> np.ndarray:
        return _storage_array_slice(
            values[key],
            key=key,
            steps=steps,
            env_start=env_start,
            env_stop=env_stop,
        )

    native = getattr(trainer, "observation_source", None) == "native"
    arrays: dict[str, np.ndarray] = {
        "expert_actions": storage_array("gt_actions").astype(
            np.float32, copy=False
        ),
        "student_actions": storage_array("student_actions").astype(
            np.float32, copy=False
        ),
        "behavior_actions": storage_array("actions").astype(
            np.float32, copy=False
        ),
        "expert_mask": storage_array("expert_mask")[..., 0].astype(
            np.bool_, copy=False
        ),
        "dones": storage_array("dones")[..., 0].astype(np.bool_, copy=False),
        "global_env_ids": storage_array("round_global_env_ids")[..., 0].astype(
            np.int64, copy=False
        ),
        "episode_ids": storage_array("round_episode_ids")[..., 0].astype(
            np.int64, copy=False
        ),
        "episode_steps": storage_array("round_episode_steps")[..., 0].astype(
            np.int64, copy=False
        ),
    }
    if not native:
        arrays.update({
            "policy_rgb": denormalize_policy_rgb(
                storage_array("vision_obs"),
                image_mean=image_mean,
                image_std=image_std,
            ),
            "observation_is_raytraced": storage_array(
                "round_observation_is_raytraced"
            )[..., 0].astype(np.bool_, copy=False),
            "reference_kinds": storage_array("round_reference_kinds")[..., 0].astype(
                np.uint8, copy=False
            ),
            "reference_lags": storage_array("round_reference_lags")[..., 0].astype(
                np.int64, copy=False
            ),
            "reference_source_steps": storage_array(
                "round_reference_source_steps"
            )[..., 0].astype(np.int64, copy=False),
            "reference_sha256": storage_array("round_reference_sha256").astype(
                np.uint8, copy=False
            ),
            "generated_sha256": storage_array("round_generated_sha256").astype(
                np.uint8, copy=False
            ),
            "conditioning_depth_nonzero_fraction": storage_array(
                "round_conditioning_depth_nonzero_fraction"
            ).astype(np.float32, copy=False),
            "conditioning_mask_nonzero_fraction": storage_array(
                "round_conditioning_mask_nonzero_fraction"
            ).astype(np.float32, copy=False),
        })
        for key in sorted(getattr(trainer, "algo_obs_dim_dict", {})):
            if key == "vision_obs":
                continue
            arrays[f"obs__{key}"] = storage_array(key).astype(np.float32, copy=False)
    else:
        arrays["obs__actor_obs"] = storage_array("actor_obs").astype(
            np.float32, copy=False
        )
        try:
            controls = trainer.roborender_conditioning_slice(
                env_start=env_start,
                env_stop=env_stop,
                global_env_ids=arrays["global_env_ids"],
                episode_ids=arrays["episode_ids"],
                episode_steps=arrays["episode_steps"],
            )
        except Exception as exc:
            raise ActorExportError(
                f"cannot materialize RoboRender conditioning slice: {exc}"
            ) from exc
        overlap = sorted(set(arrays).intersection(controls))
        if overlap:
            raise ActorExportError(
                f"RoboRender conditioning keys overlap transition arrays: {overlap}"
            )
        arrays.update(controls)
    return arrays


def _trainer_rollout_parts(
    trainer: Any,
    *,
    environments_per_shard: int,
    existing_parts: Mapping[int, Mapping[str, Any]] | None = None,
) -> Iterator[tuple[int, dict[str, np.ndarray] | None]]:
    """Yield one host-bounded, canonical shard at a time."""

    if environments_per_shard <= 0:
        raise ActorExportError("environments_per_shard must be positive")
    steps, batch_size, values, mean, std = _rollout_storage(trainer)
    part_count = (batch_size + environments_per_shard - 1) // environments_per_shard
    committed = dict(existing_parts or {})
    if sorted(committed) != list(range(len(committed))) or len(committed) > part_count:
        raise ActorExportError(
            "existing rollout shards are not an exact contiguous part prefix"
        )
    for part_index, start in enumerate(range(0, batch_size, environments_per_shard)):
        stop = min(batch_size, start + environments_per_shard)
        if part_index in committed:
            report = committed[part_index]
            global_ids = _storage_array_slice(
                values["round_global_env_ids"],
                key="round_global_env_ids",
                steps=steps,
                env_start=start,
                env_stop=stop,
            )[..., 0].astype(np.int64, copy=False)
            if tuple(global_ids.shape) != (steps, stop - start) or not np.all(
                global_ids == global_ids[0:1]
            ):
                raise ActorExportError(
                    "existing shard resume cannot prove stable source global_env_ids"
                )
            expected_ids = sorted(int(value) for value in global_ids[0].tolist())
            if (
                report.get("global_env_ids") != expected_ids
                or report.get("transition_count") != steps * (stop - start)
            ):
                raise ActorExportError(
                    "existing verified shard does not match the current rollout part"
                )
            # The immutable payload and sidecar have already been fully read and
            # verified. Reuse them without allocating or rewriting RGB data.
            yield part_index, None
            continue
        yield part_index, _trainer_rollout_slice(
            trainer,
            steps=steps,
            values=values,
            image_mean=mean,
            image_std=std,
            env_start=start,
            env_stop=stop,
        )


def _verified_existing_rollout_parts(
    cohort_dir: Path,
    *,
    rank: int,
    rollout_id: int,
    expected_shard_ids: Sequence[str] | None = None,
) -> dict[int, dict[str, Any]]:
    """Return a fully verified committed prefix for an interrupted rank export."""

    shard_dir = cohort_dir / "shards"
    if shard_dir.is_symlink() or not shard_dir.is_dir():
        raise ActorExportError("actor cohort has no safe shards directory")
    expected_by_id = (
        {shard_id: part_index for part_index, shard_id in enumerate(expected_shard_ids)}
        if expected_shard_ids is not None
        else None
    )
    if expected_by_id is not None:
        if len(expected_by_id) != len(expected_shard_ids) or not expected_by_id:
            raise ActorExportError("native shard IDs must be non-empty and unique")
        prefix = expected_shard_ids[0].rsplit("__shard-", 1)[0] + "__shard-"
    else:
        prefix = f"rank-{rank:04d}_"
    reports: dict[int, dict[str, Any]] = {}
    try:
        for path in sorted(shard_dir.iterdir(), key=lambda item: item.name):
            if not path.name.startswith(prefix) or path.suffix != ".json":
                continue
            if path.is_symlink() or not path.is_file():
                raise ActorExportError(
                    f"unsafe existing rank shard sidecar: {path.name}"
                )
            report = verify_rollout_shard(path)
            if report["rank"] != rank or report["rollout_id"] != rollout_id:
                raise ActorExportError(
                    "existing rank shard has an unexpected rank or rollout identity"
                )
            part_index = int(report["part_index"])
            if expected_by_id is not None and (
                report["shard_id"] not in expected_by_id
                or expected_by_id[report["shard_id"]] != part_index
            ):
                raise ActorExportError(
                    "existing generation shard differs from its wave/GPU/part identity"
                )
            if part_index in reports:
                raise ActorExportError("existing rank shard repeats a part identity")
            reports[part_index] = report
    except ArtifactError as exc:
        raise ActorExportError(f"existing rank shard verification failed: {exc}") from exc
    if sorted(reports) != list(range(len(reports))):
        raise ActorExportError(
            "existing rollout shards are not an exact contiguous part prefix"
        )
    return reports


def trainer_rollout_arrays(trainer: Any) -> dict[str, np.ndarray]:
    """Extract a full rollout for tests/debugging into the actor schema.

    Production publication uses ``_trainer_rollout_parts`` instead so the
    complete vision batch is never materialized in host memory.
    """

    steps, batch_size, values, mean, std = _rollout_storage(trainer)
    return _trainer_rollout_slice(
        trainer,
        steps=steps,
        values=values,
        image_mean=mean,
        image_std=std,
        env_start=0,
        env_stop=batch_size,
    )


def _split_environment_axis(
    arrays: Mapping[str, np.ndarray], *, environments_per_shard: int
) -> list[dict[str, np.ndarray]]:
    if environments_per_shard <= 0:
        raise ActorExportError("environments_per_shard must be positive")
    values = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    if not values:
        raise ActorExportError("cannot split an empty rollout")
    batch_sizes = {value.shape[1] for value in values.values() if value.ndim >= 2}
    if len(batch_sizes) != 1:
        raise ActorExportError("rollout arrays disagree on environment dimension")
    batch_size = batch_sizes.pop()
    result = []
    for start in range(0, batch_size, environments_per_shard):
        stop = min(batch_size, start + environments_per_shard)
        result.append(
            {name: np.ascontiguousarray(value[:, start:stop]) for name, value in values.items()}
        )
    return result


def export_trainer_rollout(
    trainer: Any,
    *,
    policy_module: Any,
    initial_runtime_policy_sha256: str,
) -> list[dict[str, Any]]:
    """Publish all parts for one rank after proving the actor stayed frozen."""

    try:
        homie_controller_sha256 = validate_homie_controller_sha256(
            trainer.homie_controller_sha256
        )
    except (AttributeError, PolicyIdentityError) as exc:
        raise ActorExportError(
            "trainer lacks exact Homie walk/stand controller identities"
        ) from exc
    final_runtime_sha = runtime_policy_sha256(
        policy_module, homie_controller_sha256=homie_controller_sha256
    )
    if final_runtime_sha != initial_runtime_policy_sha256:
        raise ActorExportError(
            "frozen actor policy changed during collection: "
            f"{initial_runtime_policy_sha256} -> {final_runtime_sha}"
        )
    env = os.environ
    required = {
        "cohort_dir": "ROUND_DAGGER_COHORT_DIR",
        "campaign_id": "ROUND_DAGGER_CAMPAIGN_ID",
        "macro_round": "ROUND_DAGGER_MACRO_ROUND",
        "cohort_index": "ROUND_DAGGER_COHORT_INDEX",
        "policy_sha256": "ROUND_DAGGER_POLICY_SHA256",
        "teacher_sha256": "ROUND_DAGGER_TEACHER_SHA256",
        "beta_sha256": "ROUND_DAGGER_BETA_SHA256",
        "config_sha256": "ROUND_DAGGER_CONFIG_SHA256",
    }
    missing = [source for source in required.values() if not env.get(source)]
    if missing:
        raise ActorExportError("missing round actor environment: " + ", ".join(missing))
    observation_provenance = _observation_provenance_from_environment(env)
    renderer_sha256 = env.get("ROUND_DAGGER_RENDERER_SHA256")
    if observation_provenance.get("observation_source") != "native" and not renderer_sha256:
        raise ActorExportError("missing round actor environment: ROUND_DAGGER_RENDERER_SHA256")
    expected_runtime = env.get("ROUND_DAGGER_RUNTIME_POLICY_SHA256")
    if expected_runtime and expected_runtime != final_runtime_sha:
        raise ActorExportError("loaded policy does not match the round's runtime policy hash")
    actor_identity = actor_process_identity(env)
    rank = (
        actor_identity.rank
        if actor_identity is not None
        else int(getattr(trainer.accelerator, "process_index", 0))
    )
    rollout_id = int(env.get("ROUND_DAGGER_ROLLOUT_ID", "0"))
    per_shard = int(env.get("ROUND_DAGGER_ENVS_PER_SHARD", "4"))
    cohort_dir = Path(env[required["cohort_dir"]])
    _validate_native_export_role(trainer, env, cohort_dir)
    _, batch_size, _, _, _ = _rollout_storage(trainer)
    native_shard_ids = _native_generation_shard_ids_from_environment(
        env,
        rank=rank,
        batch_size=batch_size,
        environments_per_shard=per_shard,
    )
    native = observation_provenance.get("observation_source") == "native"
    native_motion_ids = (
        _native_rank_motion_ids(
            trainer,
            env,
            rank=rank,
            batch_size=batch_size,
        )
        if native
        else None
    )
    # Raw Isaac conditioning artifacts use a separate schema and are never
    # discovered by the replay verifier/learner.  An interrupted native wave
    # is simply recollected; create-only publication accepts only identical
    # bytes and otherwise fails closed.
    existing_parts = (
        {}
        if native
        else _verified_existing_rollout_parts(
            cohort_dir,
            rank=rank,
            rollout_id=rollout_id,
            expected_shard_ids=native_shard_ids or None,
        )
    )
    published: list[dict[str, Any]] = []
    for part_index, arrays in _trainer_rollout_parts(
        trainer,
        environments_per_shard=per_shard,
        existing_parts=existing_parts,
    ):
        if arrays is None:
            published.append(existing_parts[part_index])
            continue
        if native:
            from .conditioning_artifacts import publish_conditioning_shard

            shard_id = (
                native_shard_ids[part_index]
                if native_shard_ids
                else (
                    f"rank-{rank:04d}_rollout-{rollout_id:08d}"
                    f"_part-{part_index:04d}"
                )
            )
            shard_dir = cohort_dir / "shards"
            camera_specs = [
                {
                    "name": str(view["name"]),
                    "link": str(view["link"]),
                    "pos": [float(value) for value in view["pos"]],
                    "rot": [float(value) for value in view["rot"]],
                }
                for view in trainer.env.config.simulator.config.cameras.conditioning_views
            ]
            published.append(
                publish_conditioning_shard(
                    shard_dir / f"{shard_id}.npz",
                    arrays,
                    manifest_path=shard_dir / f"{shard_id}.json",
                    contract=trainer.roborender_conditioning_contract,
                    purpose=env.get("ROUND_DAGGER_PURPOSE", "train"),
                    prompt=trainer.roborender_prompt,
                    seed=trainer.roborender_seed,
                    reference_lag=trainer.roborender_reference_lag,
                    render_metadata={
                        "campaign_id": env[required["campaign_id"]],
                        "macro_round": int(env[required["macro_round"]]),
                        "cohort_index": int(env[required["cohort_index"]]),
                        "rank": rank,
                        "rollout_id": rollout_id,
                        "part_index": part_index,
                        "shard_id": shard_id,
                        "policy_sha256": env[required["policy_sha256"]],
                        "runtime_policy_sha256": final_runtime_sha,
                        "teacher_sha256": env[required["teacher_sha256"]],
                        "beta_sha256": env[required["beta_sha256"]],
                        "config_sha256": env[required["config_sha256"]],
                        "split_manifest_sha256": trainer.native_split_manifest_sha256,
                        "split_sha256": trainer.native_split_sha256,
                        "group_ids": list(trainer.native_group_ids),
                        "motion_ids": (
                            list(
                                native_motion_ids[
                                    part_index * per_shard :
                                    part_index * per_shard + arrays["global_env_ids"].shape[1]
                                ]
                            )
                            if native_motion_ids is not None
                            else []
                        ),
                        "panel_sha256": trainer.native_panel_sha256,
                        "camera_specs": camera_specs,
                        **observation_provenance,
                    },
                )
            )
        else:
            published.append(
                publish_rollout_shard(
                    cohort_dir,
                    rank=rank,
                    rollout_id=rollout_id,
                    part_index=part_index,
                    arrays=arrays,
                    campaign_id=env[required["campaign_id"]],
                    macro_round=int(env[required["macro_round"]]),
                    cohort_index=int(env[required["cohort_index"]]),
                    purpose=env.get("ROUND_DAGGER_PURPOSE", "train"),
                    policy_sha256=env[required["policy_sha256"]],
                    runtime_policy_sha256=final_runtime_sha,
                    teacher_sha256=env[required["teacher_sha256"]],
                    renderer_sha256=renderer_sha256,
                    beta_sha256=env[required["beta_sha256"]],
                    config_sha256=env[required["config_sha256"]],
                    native_shard_id=(
                        native_shard_ids[part_index] if native_shard_ids else None
                    ),
                    **observation_provenance,
                )
            )
        # Do not retain the previous host slice while the generator materializes
        # the next one. Publication reports contain metadata only.
        del arrays
    return published


__all__ = [
    "ActorExportError",
    "denormalize_policy_rgb",
    "export_trainer_rollout",
    "runtime_policy_sha256",
    "runtime_state_dict_sha256",
    "trainer_rollout_arrays",
]
