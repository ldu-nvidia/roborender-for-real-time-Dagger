"""Origin-attested, vectorized composition with an initialized Raven trainer.

Raven/Isaac owns scientific transformations and the irreversible transition.
This module only validates, batches, and splits those native results.  In
particular it contains no h5_v4, raw-depth quantization, teacher, or Homie
formula.  One :class:`NativeRavenBatchComposition` corresponds to the physical
Isaac vector environment and is the only object allowed to call ``env.step``.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backend import BackendInvariantError, Capture, TransitionResult
from .contracts import (
    ACTION_DIM,
    HOMIE_ACTION_DIM,
    MODEL_VIEW_MAP,
    PINNED_RAVEN_COMMIT,
    PINNED_RAVEN_ROOT,
    RAVEN_VIEW_ORDER,
    RavenShapeProfile,
    StepKey,
    anchor_rgb_digest,
)
from .raven_adapter import RavenHooks


NATIVE_BINDING_VERSION = "roborender.dagger.native-raven.v2"

# Exact retained code paths.  Concrete callbacks are accepted only if every
# origin string matches and the immutable source manifest has been verified.
NATIVE_CALLBACK_ORIGINS = {
    "prepare_episode_slots": (
        "distill_trainer_videogen.py:_image_dagger_relabel/anchor-slot-initialization"
    ),
    "read_current_batch": (
        "ppo_trainer.py:TRLPPOTrainer._rollout_step/current-obs-dones-and-identities"
    ),
    "teacher_homie_labels": (
        "distill_trainer_obj_pred_homie_api.py:"
        "TRLDistillTrainerObjPredHomieAPI.policy_step/pre-student-prefix"
    ),
    "native_depth_archive": (
        "distill_trainer_videogen.py:"
        "TRLDistillTrainerVideoGen._save_collected_chunks/raw-depth-products"
    ),
    "build_action_state": (
        "ppo_trainer.py:TRLPPOTrainer._rollout_step/actions-plus-gt_actions"
    ),
    "decode_env_step": (
        "ppo_trainer.py:TRLPPOTrainer._rollout_step/env-step-result-processing"
    ),
}


@dataclass(frozen=True)
class NativeCurrentBatch:
    """One current, already-initialized Raven batch before ``env.step``."""

    keys: tuple[StepKey, ...]
    obs_dict: Mapping[str, Any]
    cur_dones: Any
    arrays: Mapping[str, np.ndarray]
    env_ids: np.ndarray
    seqs: np.ndarray


@dataclass(frozen=True)
class NativeActionLabels:
    """Exact native Raven label outputs for all physical environments."""

    teacher_action: np.ndarray  # [B,26]
    homie_action: np.ndarray  # [B,15]


@dataclass(frozen=True)
class NativeDepthArchive:
    """Exact native collection-format depth products for all environments."""

    depth_rawq: np.ndarray  # [B,H,W]
    depth_raw: np.ndarray  # [B,h,w]


@dataclass(frozen=True)
class NativeRavenEntrypoints:
    """Thin callbacks bound inside the certified Raven process.

    The concrete implementation in :mod:`dagger_rtx.native_entrypoints`
    executes the retained methods themselves.  It never duplicates their
    numerical implementations.
    """

    raven_root: str
    raven_commit: str
    origins: Mapping[str, str]
    prepare_episode_slots: Callable[[Any, int], None]
    read_current_batch: Callable[[Sequence[StepKey]], NativeCurrentBatch]
    teacher_homie_labels: Callable[[Any, NativeCurrentBatch], NativeActionLabels]
    native_depth_archive: Callable[
        [Any, Mapping[str, Mapping[str, Any]]], Mapping[str, NativeDepthArchive]
    ]
    build_action_state: Callable[[Any, np.ndarray, np.ndarray], Mapping[str, Any]]
    decode_env_step: Callable[[Any, Any], Sequence[TransitionResult]]


def _as_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    try:
        value = value.detach().to("cpu").numpy()
    except Exception as exc:  # pragma: no cover - real torch path
        raise BackendInvariantError(
            f"{name} is neither numpy nor a detachable native tensor"
        ) from exc
    return np.ascontiguousarray(value)


def _slot(container: Any, index: int, name: str) -> Any:
    if isinstance(container, Mapping):
        if index not in container:
            raise BackendInvariantError(f"native Raven {name} has no slot {index}")
        return container[index]
    if isinstance(container, (list, tuple)):
        if index >= len(container):
            raise BackendInvariantError(f"native Raven {name} has no slot {index}")
        return container[index]
    raise BackendInvariantError(f"native Raven {name} must be a mapping or sequence")


def _require_numpy(
    value: Any, name: str, shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise BackendInvariantError(f"{name} callback result must be a numpy array")
    if value.shape != shape or value.dtype != dtype:
        raise BackendInvariantError(
            f"{name} must be {shape} {dtype.name}; got {value.shape} {value.dtype}"
        )
    if np.issubdtype(dtype, np.floating) and not bool(np.isfinite(value).all()):
        raise BackendInvariantError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(value)


class NativeRavenBatchComposition:
    """Vectorized adapter for one physical Raven ``env``.

    All B rows are captured from one synchronized camera call and one native
    label call.  ``step_batch`` accepts all B selected actions and makes exactly
    one physical ``env.step({'actions': ..., 'gt_actions': ...})`` call.
    Durability around that call is provided by ``BatchIsaacTransitionOwner``.
    """

    def __init__(
        self,
        trainer: Any,
        entrypoints: NativeRavenEntrypoints,
        *,
        shape_profile: RavenShapeProfile | None = None,
    ):
        self.trainer = trainer
        self.entrypoints = entrypoints
        self.profile = shape_profile or RavenShapeProfile()
        self.num_envs = int(getattr(getattr(trainer, "env", None), "num_envs", 0))
        self._anchors: list[dict[str, np.ndarray] | None] = [None] * self.num_envs
        self._prompts: list[str | None] = [None] * self.num_envs
        self._raytraced: list[bool | None] = [None] * self.num_envs
        self._verify_surface()

    def _verify_surface(self) -> None:
        ep = self.entrypoints
        if ep.raven_root != PINNED_RAVEN_ROOT or ep.raven_commit != PINNED_RAVEN_COMMIT:
            raise BackendInvariantError("native callbacks do not attest the pinned Raven snapshot")
        if set(ep.origins) != set(NATIVE_CALLBACK_ORIGINS):
            raise BackendInvariantError(
                f"native callback origins must be exactly {sorted(NATIVE_CALLBACK_ORIGINS)}"
            )
        for name, expected in NATIVE_CALLBACK_ORIGINS.items():
            if ep.origins[name] != expected:
                raise BackendInvariantError(f"native callback {name} has the wrong retained origin")
            callback = getattr(ep, name)
            if not callable(callback) or inspect.iscoroutinefunction(callback):
                raise BackendInvariantError(f"native callback {name} must be synchronous")
        env = getattr(self.trainer, "env", None)
        simulator = getattr(env, "simulator", None)
        if env is None or simulator is None or self.num_envs < 1:
            raise BackendInvariantError("trainer must expose a nonempty initialized env.simulator")
        for owner, names in (
            (self.trainer, ("_id_sample_anchor", "_h5v4_control", "_resize_seq")),
            (simulator, ("get_conditioning_views", "get_policy_view_stack")),
            (env, ("step",)),
        ):
            for name in names:
                method = getattr(owner, name, None)
                if not callable(method) or inspect.iscoroutinefunction(method):
                    raise BackendInvariantError(f"missing synchronous native Raven method {name}")
        if getattr(self.trainer, "vg_depth_mode", None) != "h5_v4":
            raise BackendInvariantError("native Raven trainer is not configured for h5_v4")
        if list(getattr(self.trainer, "vg_control_res", ())) != [
            self.profile.control_h,
            self.profile.control_w,
        ]:
            raise BackendInvariantError("native Raven control resolution does not match wire profile")
        if tuple(getattr(self.trainer, "vg_view_names", ())) != RAVEN_VIEW_ORDER:
            raise BackendInvariantError("native Raven view_names are not in online policy order")
        if tuple(getattr(self.trainer, "_policy_views", ())) != RAVEN_VIEW_ORDER:
            raise BackendInvariantError("native Raven policy tile order is not canonical")

    def bind_episodes(
        self, bindings: Mapping[int, StepKey]
    ) -> Mapping[int, Mapping[str, Any]]:
        if not bindings:
            return {}
        for index, key in bindings.items():
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.num_envs:
                raise BackendInvariantError("episode binding has an invalid environment index")
            if key.episode_step_index != 0:
                raise BackendInvariantError("episode binding must begin at episode step zero")
        self.entrypoints.prepare_episode_slots(self.trainer, self.num_envs)
        result: dict[int, Mapping[str, Any]] = {}
        for index in sorted(bindings):
            self.trainer._id_sample_anchor(index)
            native_anchor = _slot(self.trainer._id_anchors, index, "_id_anchors")
            if not isinstance(native_anchor, Mapping) or set(native_anchor) != set(RAVEN_VIEW_ORDER):
                raise BackendInvariantError("_id_sample_anchor did not create canonical triplet")
            anchors: dict[str, np.ndarray] = {}
            for view in RAVEN_VIEW_ORDER:
                anchors[f"anchor_rgb__{view}"] = _require_numpy(
                    _as_numpy(native_anchor[view], f"anchor {view}"),
                    f"anchor_rgb__{view}",
                    (self.profile.control_h, self.profile.control_w, 3),
                    np.dtype("uint8"),
                ).copy()
            prompt = _slot(self.trainer._id_prompts, index, "_id_prompts")
            use_raytraced = _slot(
                self.trainer._id_use_raytraced, index, "_id_use_raytraced"
            )
            if not isinstance(prompt, str) or not prompt:
                raise BackendInvariantError("_id_sample_anchor returned an empty prompt")
            if not isinstance(use_raytraced, (bool, np.bool_)):
                raise BackendInvariantError("_id_sample_anchor returned non-boolean source")
            self._anchors[index] = anchors
            self._prompts[index] = prompt
            self._raytraced[index] = bool(use_raytraced)
            result[index] = {
                "use_raytraced": bool(use_raytraced),
                "anchor_sha256": anchor_rgb_digest(anchors),
                "raven_source": PINNED_RAVEN_ROOT,
                "raven_commit": PINNED_RAVEN_COMMIT,
            }
        return result

    def capture_batch(self, keys: Sequence[StepKey]) -> Sequence[Capture]:
        if len(keys) != self.num_envs:
            raise BackendInvariantError(
                f"capture_batch needs {self.num_envs} keys, got {len(keys)}"
            )
        if any(value is None for value in self._anchors + self._prompts + self._raytraced):
            raise BackendInvariantError("every native lane must bind an episode before capture")
        current = self.entrypoints.read_current_batch(keys)
        if not isinstance(current, NativeCurrentBatch) or current.keys != tuple(keys):
            raise BackendInvariantError("read_current_batch returned wrong native keys")
        b = self.num_envs
        _require_numpy(current.env_ids, "env_ids", (b,), np.dtype("int64"))
        _require_numpy(current.seqs, "seqs", (b,), np.dtype("int64"))

        # Exactly one call is the synchronized physical-camera boundary for B.
        views = self.trainer.env.simulator.get_conditioning_views()
        if not isinstance(views, Mapping) or not set(RAVEN_VIEW_ORDER).issubset(views):
            raise BackendInvariantError("get_conditioning_views omitted a canonical view")
        labels = self.entrypoints.teacher_homie_labels(self.trainer, current)
        if not isinstance(labels, NativeActionLabels):
            raise BackendInvariantError("teacher_homie_labels returned wrong result type")
        homie = _require_numpy(
            labels.homie_action, "homie_action", (b, HOMIE_ACTION_DIM), np.dtype("float32")
        )
        teacher = _require_numpy(
            labels.teacher_action, "teacher_action", (b, ACTION_DIM), np.dtype("float32")
        )
        if not np.array_equal(teacher[:, 11:], homie):
            raise BackendInvariantError("native teacher_action[:,11:] != homie_action")

        expected_state = {
            "actor_obs": (b, self.profile.actor_obs),
            "teacher_obs": (b, self.profile.teacher_obs),
            "critic_obs": (b, self.profile.critic_obs),
            "homie_obs": (b, self.profile.homie_obs),
            "gt_obj_pos": (b, 3),
        }
        if set(current.arrays) != set(expected_state):
            raise BackendInvariantError(f"current arrays must be exactly {sorted(expected_state)}")
        states = {
            name: _require_numpy(current.arrays[name], name, shape, np.dtype("float32"))
            for name, shape in expected_state.items()
        }

        # Exact retained archival block is executed once for the synchronized B
        # batch by the callback; this module does not contain its quantizer.
        archives = self.entrypoints.native_depth_archive(self.trainer, views)
        if not isinstance(archives, Mapping) or set(archives) != set(RAVEN_VIEW_ORDER):
            raise BackendInvariantError("native_depth_archive omitted a canonical view")

        shared: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        for view in RAVEN_VIEW_ORDER:
            item = views[view]
            if not isinstance(item, Mapping) or "depth" not in item or "robot_rgb" not in item:
                raise BackendInvariantError(f"conditioning view {view} lacks depth/robot_rgb")
            metric_batch = _as_numpy(item["depth"], f"metric depth {view}")
            if metric_batch.shape != (b, self.profile.depth_h, self.profile.depth_w, 1) or metric_batch.dtype != np.float32:
                raise BackendInvariantError(f"native depth {view} has wrong shape/dtype")
            metric = np.ascontiguousarray(metric_batch[..., 0])
            if not bool(np.isfinite(metric).all()):
                raise BackendInvariantError(f"native metric depth {view} is non-finite")
            archive = archives[view]
            rawq = _require_numpy(
                archive.depth_rawq, f"depth_rawq__{view}",
                (b, self.profile.depth_h, self.profile.depth_w), np.dtype("uint16"),
            )
            raw = _require_numpy(
                archive.depth_raw, f"depth_raw__{view}",
                (b, self.profile.control_h, self.profile.control_w), np.dtype("float16"),
            )
            control = self.trainer._h5v4_control(metric[None], view)
            control = self.trainer._resize_seq(
                control, [self.profile.control_h, self.profile.control_w]
            )
            control = _require_numpy(
                control, f"depth_ctrl_batch__{view}",
                (1, b, self.profile.control_h, self.profile.control_w), np.dtype("uint8"),
            )[0]
            robot = _as_numpy(item["robot_rgb"], f"robot RGB {view}")
            if robot.shape != (b, self.profile.depth_h, self.profile.depth_w, 3) or robot.dtype != np.uint8:
                raise BackendInvariantError(f"native robot RGB {view} has wrong shape/dtype")
            robot = self.trainer._resize_seq(
                robot[None], [self.profile.control_h, self.profile.control_w]
            )
            robot = _require_numpy(
                robot, f"rgb_robot_batch__{view}",
                (1, b, self.profile.control_h, self.profile.control_w, 3), np.dtype("uint8"),
            )[0]
            shared[view] = (metric, rawq, raw, control, robot)

        policy_stack = None
        if any(bool(value) for value in self._raytraced):
            policy_stack = _as_numpy(
                self.trainer.env.simulator.get_policy_view_stack(),
                "native raytraced policy stack",
            )
            expected = (
                b, len(RAVEN_VIEW_ORDER) * self.profile.policy_h,
                self.profile.policy_w, 3,
            )
            if policy_stack.shape != expected or policy_stack.dtype != np.float32:
                raise BackendInvariantError(f"native policy stack must be {expected} float32")
            if not bool(np.isfinite(policy_stack).all()) or bool((policy_stack < 0).any()) or bool((policy_stack > 1).any()):
                raise BackendInvariantError("native policy stack must be finite in [0,1]")

        captures: list[Capture] = []
        for index, key in enumerate(keys):
            arrays = {name: np.ascontiguousarray(value[index]) for name, value in states.items()}
            arrays["teacher_action"] = np.ascontiguousarray(teacher[index])
            arrays["homie_action"] = np.ascontiguousarray(homie[index])
            arrays.update({name: value.copy() for name, value in self._anchors[index].items()})
            for view in RAVEN_VIEW_ORDER:
                metric, rawq, raw, control, robot = shared[view]
                arrays[f"depth_metric__{view}"] = np.ascontiguousarray(metric[index])
                arrays[f"depth_rawq__{view}"] = np.ascontiguousarray(rawq[index])
                arrays[f"depth_raw__{view}"] = np.ascontiguousarray(raw[index])
                arrays[f"depth_ctrl__{view}"] = np.ascontiguousarray(control[index])
                arrays[f"rgb_robot__{view}"] = np.ascontiguousarray(robot[index])
            if self._raytraced[index]:
                assert policy_stack is not None
                for tile, view in enumerate(RAVEN_VIEW_ORDER):
                    start = tile * self.profile.policy_h
                    arrays[f"raytraced_rgb__{view}"] = np.ascontiguousarray(
                        policy_stack[index, start : start + self.profile.policy_h]
                    )
            metadata = {
                "view_order": list(RAVEN_VIEW_ORDER),
                "model_view_map": dict(MODEL_VIEW_MAP),
                "observation_source": "raytraced" if self._raytraced[index] else "generated_controls",
                "prompt": self._prompts[index],
                "env_id": int(current.env_ids[index]),
                "seq": int(current.seqs[index]),
                "depth_units": "meter",
                "depth_control_encoding": "raven_h5_v4",
                "robot_control_encoding": "textured_robot_rgb_on_black",
            }
            captures.append(Capture(key=key, metadata=metadata, arrays=arrays))
        return captures

    def step_batch(
        self, selected_actions: np.ndarray, teacher_actions: np.ndarray
    ) -> Sequence[TransitionResult]:
        selected = _require_numpy(
            selected_actions, "selected_actions", (self.num_envs, ACTION_DIM), np.dtype("float32")
        )
        teachers = _require_numpy(
            teacher_actions, "teacher_actions", (self.num_envs, ACTION_DIM), np.dtype("float32")
        )
        action_state = self.entrypoints.build_action_state(
            self.trainer, selected.copy(), teachers.copy()
        )
        if not isinstance(action_state, Mapping) or set(("actions", "gt_actions")) - set(action_state):
            raise BackendInvariantError("native action state omitted actions or gt_actions")
        embedded = _as_numpy(action_state["actions"], "action_state.actions")
        gt_embedded = _as_numpy(action_state["gt_actions"], "action_state.gt_actions")
        if not np.array_equal(embedded, selected) or not np.array_equal(gt_embedded, teachers):
            raise BackendInvariantError("native action-state callback changed selected/teacher actions")

        # Sole physical transition call.  No extra render_results() exists in
        # Raven's training rollout path.
        raw_result = self.trainer.env.step(action_state)
        results = tuple(self.entrypoints.decode_env_step(self.trainer, raw_result))
        if len(results) != self.num_envs:
            raise BackendInvariantError("decode_env_step returned wrong batch length")
        for result in results:
            if not isinstance(result, TransitionResult):
                raise BackendInvariantError("decode_env_step returned wrong result type")
            if not math.isfinite(result.reward):
                raise BackendInvariantError("native transition reward is non-finite")
            if type(result.done) is not bool or type(result.timeout) is not bool:
                raise BackendInvariantError("native done/timeout must be bool")
            if not isinstance(result.metadata, dict):
                raise BackendInvariantError("native transition metadata must be a dict")
        return results


def build_native_raven_batch(
    trainer: Any,
    entrypoints: NativeRavenEntrypoints,
    *,
    shape_profile: RavenShapeProfile | None = None,
) -> NativeRavenBatchComposition:
    """Validate and return the production vectorized native composition."""

    return NativeRavenBatchComposition(trainer, entrypoints, shape_profile=shape_profile)


def build_native_raven_hooks(
    trainer: Any,
    entrypoints: NativeRavenEntrypoints,
    *,
    shape_profile: RavenShapeProfile | None = None,
    env_index: int = 0,
) -> RavenHooks:
    """Compatibility adapter for a one-env validation smoke only.

    Production launchers use :func:`build_native_raven_batch`.  This adapter is
    intentionally unavailable when Raven is vectorized so it can never cause B
    separate calls to a physical vector ``env.step``.
    """

    batch = build_native_raven_batch(trainer, entrypoints, shape_profile=shape_profile)
    if batch.num_envs != 1 or env_index != 0:
        raise BackendInvariantError(
            "single-lane RavenHooks are validation-only; vector environments require the batch barrier"
        )
    latest_teacher: np.ndarray | None = None

    def bind(key: StepKey) -> Mapping[str, Any]:
        return batch.bind_episodes({0: key})[0]

    def capture(key: StepKey) -> Capture:
        nonlocal latest_teacher
        value = batch.capture_batch([key])[0]
        latest_teacher = value.arrays["teacher_action"].copy()
        return value

    def step(action: np.ndarray) -> TransitionResult:
        if latest_teacher is None:
            raise BackendInvariantError("step attempted before capture")
        return batch.step_batch(action[None], latest_teacher[None])[0]

    return RavenHooks(bind_episode=bind, capture=capture, step_once=step)
