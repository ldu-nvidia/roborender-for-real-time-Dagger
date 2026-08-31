"""Frozen-actor rollout export helpers for round-based DAgger.

This module deliberately contains no Isaac Lab imports.  The RTX trainer calls
``export_trainer_rollout`` after one frozen rollout; unit tests can exercise the
same conversion path with an in-memory storage double.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactError, publish_rollout_shard, verify_rollout_shard
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
    payload itself to be transferred from CUDA in bounded environment slices.
    """

    storage = trainer.storage
    steps = int(storage.step)
    expected_steps = int(trainer.num_steps_per_env)
    if steps != expected_steps or steps <= 0:
        raise ActorExportError(
            f"rollout is incomplete: stored {steps}, expected {expected_steps} steps"
        )

    observation_keys = tuple(
        key
        for key in sorted(getattr(trainer, "algo_obs_dim_dict", {}))
        if key != "vision_obs"
    )
    # A trainer may expose one of the canonical tensors as an additional
    # learner observation. Query duplicate source keys once; their canonical
    # and ``obs__`` outputs remain distinct.
    keys = tuple(dict.fromkeys((*_CANONICAL_STORAGE_KEYS, *observation_keys)))

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

    arrays: dict[str, np.ndarray] = {
        "policy_rgb": denormalize_policy_rgb(
            storage_array("vision_obs"),
            image_mean=image_mean,
            image_std=image_std,
        ),
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
        "observation_is_raytraced": storage_array(
            "round_observation_is_raytraced"
        )[..., 0].astype(np.bool_, copy=False),
        "global_env_ids": storage_array("round_global_env_ids")[..., 0].astype(
            np.int64, copy=False
        ),
        "episode_ids": storage_array("round_episode_ids")[..., 0].astype(
            np.int64, copy=False
        ),
        "episode_steps": storage_array("round_episode_steps")[..., 0].astype(
            np.int64, copy=False
        ),
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
    }
    for key in sorted(getattr(trainer, "algo_obs_dim_dict", {})):
        if key == "vision_obs":
            continue
        arrays[f"obs__{key}"] = storage_array(key).astype(np.float32, copy=False)
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
    cohort_dir: Path, *, rank: int, rollout_id: int
) -> dict[int, dict[str, Any]]:
    """Return a fully verified committed prefix for an interrupted rank export."""

    shard_dir = cohort_dir / "shards"
    if shard_dir.is_symlink() or not shard_dir.is_dir():
        raise ActorExportError("actor cohort has no safe shards directory")
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
    cohort_dir = Path(env[required["cohort_dir"]])
    existing_parts = _verified_existing_rollout_parts(
        cohort_dir, rank=rank, rollout_id=rollout_id
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
                renderer_sha256=env[required["renderer_sha256"]],
                beta_sha256=env[required["beta_sha256"]],
                config_sha256=env[required["config_sha256"]],
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
