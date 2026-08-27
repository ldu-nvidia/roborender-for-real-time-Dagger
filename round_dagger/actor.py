"""Frozen-actor rollout export helpers for round-based DAgger.

This module deliberately contains no Isaac Lab imports.  The RTX trainer calls
``export_trainer_rollout`` after one frozen rollout; unit tests can exercise the
same conversion path with an in-memory storage double.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import publish_rollout_shard
from .actor_identity import actor_process_identity
from .policy_identity import (
    PolicyIdentityError,
    runtime_policy_identity_sha256,
    validate_homie_controller_sha256,
)


class ActorExportError(RuntimeError):
    """The actor rollout does not satisfy the immutable shard contract."""


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


def _storage_array(storage: Any, key: str, steps: int) -> np.ndarray:
    try:
        value = storage.query_key(key)
    except Exception as exc:  # pragma: no cover - trainer-specific exception
        raise ActorExportError(f"rollout storage is missing required key {key!r}") from exc
    value = _as_numpy(value[:steps])
    if value.shape[0] != steps:
        raise ActorExportError(f"storage key {key!r} contains too few steps")
    return value


def trainer_rollout_arrays(trainer: Any) -> dict[str, np.ndarray]:
    """Extract one completed trainer rollout into the canonical actor schema."""

    storage = trainer.storage
    steps = int(storage.step)
    expected_steps = int(trainer.num_steps_per_env)
    if steps != expected_steps or steps <= 0:
        raise ActorExportError(
            f"rollout is incomplete: stored {steps}, expected {expected_steps} steps"
        )
    cam_cfg = trainer.env.config.simulator.config.cameras
    mean = getattr(cam_cfg, "image_mean", None)
    std = getattr(cam_cfg, "image_std", None)
    arrays: dict[str, np.ndarray] = {
        "policy_rgb": denormalize_policy_rgb(
            _storage_array(storage, "vision_obs", steps),
            image_mean=mean,
            image_std=std,
        ),
        "expert_actions": _storage_array(storage, "gt_actions", steps).astype(
            np.float32, copy=False
        ),
        "student_actions": _storage_array(storage, "student_actions", steps).astype(
            np.float32, copy=False
        ),
        "behavior_actions": _storage_array(storage, "actions", steps).astype(
            np.float32, copy=False
        ),
        "expert_mask": _storage_array(storage, "expert_mask", steps)[..., 0].astype(
            np.bool_, copy=False
        ),
        "dones": _storage_array(storage, "dones", steps)[..., 0].astype(
            np.bool_, copy=False
        ),
        "observation_is_raytraced": _storage_array(
            storage, "round_observation_is_raytraced", steps
        )[..., 0].astype(np.bool_, copy=False),
        "global_env_ids": _storage_array(
            storage, "round_global_env_ids", steps
        )[..., 0].astype(np.int64, copy=False),
        "episode_ids": _storage_array(storage, "round_episode_ids", steps)[
            ..., 0
        ].astype(np.int64, copy=False),
        "episode_steps": _storage_array(storage, "round_episode_steps", steps)[
            ..., 0
        ].astype(np.int64, copy=False),
        "reference_kinds": _storage_array(
            storage, "round_reference_kinds", steps
        )[..., 0].astype(np.uint8, copy=False),
        "reference_lags": _storage_array(
            storage, "round_reference_lags", steps
        )[..., 0].astype(np.int64, copy=False),
        "reference_source_steps": _storage_array(
            storage, "round_reference_source_steps", steps
        )[..., 0].astype(np.int64, copy=False),
        "reference_sha256": _storage_array(
            storage, "round_reference_sha256", steps
        ).astype(np.uint8, copy=False),
        "generated_sha256": _storage_array(
            storage, "round_generated_sha256", steps
        ).astype(np.uint8, copy=False),
        "conditioning_depth_nonzero_fraction": _storage_array(
            storage, "round_conditioning_depth_nonzero_fraction", steps
        ).astype(np.float32, copy=False),
        "conditioning_mask_nonzero_fraction": _storage_array(
            storage, "round_conditioning_mask_nonzero_fraction", steps
        ).astype(np.float32, copy=False),
    }
    # Preserve every configured non-vision policy input so the camera-free
    # learner can reconstruct the exact forward call without Isaac Lab.
    for key in sorted(getattr(trainer, "algo_obs_dim_dict", {})):
        if key == "vision_obs":
            continue
        arrays[f"obs__{key}"] = _storage_array(storage, key, steps).astype(
            np.float32, copy=False
        )
    return arrays


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
        "renderer_sha256": "ROUND_DAGGER_RENDERER_SHA256",
        "beta_sha256": "ROUND_DAGGER_BETA_SHA256",
        "config_sha256": "ROUND_DAGGER_CONFIG_SHA256",
    }
    missing = [source for source in required.values() if not env.get(source)]
    if missing:
        raise ActorExportError("missing round actor environment: " + ", ".join(missing))
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
    parts = _split_environment_axis(
        trainer_rollout_arrays(trainer), environments_per_shard=per_shard
    )
    published: list[dict[str, Any]] = []
    for part_index, arrays in enumerate(parts):
        published.append(
            publish_rollout_shard(
                Path(env[required["cohort_dir"]]),
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
                renderer_sha256=env[required["renderer_sha256"]],
                beta_sha256=env[required["beta_sha256"]],
                config_sha256=env[required["config_sha256"]],
            )
        )
    return published


__all__ = [
    "ActorExportError",
    "denormalize_policy_rgb",
    "export_trainer_rollout",
    "runtime_policy_sha256",
    "runtime_state_dict_sha256",
    "trainer_rollout_arrays",
]
