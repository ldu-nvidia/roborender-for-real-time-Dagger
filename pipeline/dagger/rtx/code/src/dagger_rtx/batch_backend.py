from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .backend import Capture, TransitionResult
from .contracts import (
    ACTION_DIM,
    MODEL_VIEW_MAP,
    PINNED_RAVEN_COMMIT,
    PINNED_RAVEN_ROOT,
    RAVEN_CAPTURE_VIEWS,
    RAVEN_VIEW_ORDER,
    RavenShapeProfile,
    StepKey,
    anchor_rgb_digest,
)


class RavenBatchBackend(Protocol):
    """The vectorized Raven boundary owned by one initialized Isaac process.

    Lane-specific capture and H200 messages stay scalar.  Only ``step_batch``
    crosses the irreversible Isaac boundary, once for the complete action batch.
    """

    @property
    def num_envs(self) -> int: ...

    def bind_episodes(
        self, bindings: Mapping[int, StepKey]
    ) -> Mapping[int, Mapping[str, Any]]: ...

    def capture_batch(self, keys: Sequence[StepKey]) -> Sequence[Capture]: ...

    def step_batch(
        self,
        selected_actions: np.ndarray,
        teacher_actions: np.ndarray,
    ) -> Sequence[TransitionResult]: ...


class BatchBackendInvariantError(RuntimeError):
    pass


@dataclass
class DeterministicFakeBatchRaven:
    """A deterministic vectorized backend for barrier and crash tests only."""

    profile: RavenShapeProfile
    num_envs: int
    done_at_calls: Mapping[int, set[int]] = field(default_factory=dict)
    timeout_at_calls: Mapping[int, set[int]] = field(default_factory=dict)
    raise_at_batch_calls: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.batch_step_calls = 0
        self.capture_batch_calls = 0
        self.bind_batch_calls = 0
        self.bind_lane_calls = [0 for _ in range(self.num_envs)]
        self.active_episode: list[str | None] = [None for _ in range(self.num_envs)]
        self.active_episode_step = [0 for _ in range(self.num_envs)]
        self.use_raytraced = [False for _ in range(self.num_envs)]
        self.anchor_arrays: list[dict[str, np.ndarray]] = [
            {} for _ in range(self.num_envs)
        ]
        self.selected_batches: list[np.ndarray] = []
        self.teacher_batches: list[np.ndarray] = []

    def bind_episodes(
        self, bindings: Mapping[int, StepKey]
    ) -> Mapping[int, Mapping[str, Any]]:
        if not bindings:
            return {}
        self.bind_batch_calls += 1
        result: dict[int, Mapping[str, Any]] = {}
        for lane, key in sorted(bindings.items()):
            if isinstance(lane, bool) or not isinstance(lane, int) or not 0 <= lane < self.num_envs:
                raise BatchBackendInvariantError("binding lane is outside the vectorized batch")
            self.bind_lane_calls[lane] += 1
            self.active_episode[lane] = key.episode_id
            self.active_episode_step[lane] = 0
            self.use_raytraced[lane] = bool((lane + self.bind_lane_calls[lane]) % 2)
            anchors = {
                f"anchor_rgb__{view}": np.full(
                    (self.profile.control_h, self.profile.control_w, 3),
                    (lane * 17 + index + self.bind_lane_calls[lane]) % 256,
                    dtype=np.uint8,
                )
                for index, view in enumerate(RAVEN_VIEW_ORDER)
            }
            self.anchor_arrays[lane] = anchors
            result[lane] = {
                "use_raytraced": self.use_raytraced[lane],
                "anchor_sha256": anchor_rgb_digest(anchors),
                "raven_source": PINNED_RAVEN_ROOT,
                "raven_commit": PINNED_RAVEN_COMMIT,
            }
        return result

    def capture_batch(self, keys: Sequence[StepKey]) -> Sequence[Capture]:
        if len(keys) != self.num_envs:
            raise BatchBackendInvariantError("capture_batch must contain one key per environment")
        self.capture_batch_calls += 1
        return [self._capture_lane(lane, key) for lane, key in enumerate(keys)]

    def _capture_lane(self, lane: int, key: StepKey) -> Capture:
        if key.episode_id != self.active_episode[lane]:
            raise BatchBackendInvariantError(f"lane {lane} captured the wrong episode")
        if key.episode_step_index != self.active_episode_step[lane]:
            raise BatchBackendInvariantError(f"lane {lane} captured out of episode order")
        p = self.profile
        base = float(key.rollout_step_index + lane * 100)
        homie = np.arange(15, dtype=np.float32) + base
        arrays: dict[str, np.ndarray] = {
            "actor_obs": np.full((p.actor_obs,), base, dtype=np.float32),
            "teacher_obs": np.full((p.teacher_obs,), base + 1, dtype=np.float32),
            "critic_obs": np.full((p.critic_obs,), base + 2, dtype=np.float32),
            "homie_obs": np.full((p.homie_obs,), base + 3, dtype=np.float32),
            "gt_obj_pos": np.asarray((base, 1, 2), dtype=np.float32),
            "homie_action": homie,
            "teacher_action": np.concatenate(
                (np.arange(11, dtype=np.float32) - base, homie)
            ).astype(np.float32),
        }
        arrays.update({name: value.copy() for name, value in self.anchor_arrays[lane].items()})
        for index, view in enumerate(RAVEN_CAPTURE_VIEWS):
            arrays[f"depth_metric__{view}"] = np.full(
                (p.depth_h, p.depth_w), lane + index + 0.5, np.float32
            )
            arrays[f"depth_rawq__{view}"] = np.full(
                (p.depth_h, p.depth_w), lane + index, np.uint16
            )
            arrays[f"depth_raw__{view}"] = np.full(
                (p.control_h, p.control_w), lane + index + 0.5, np.float16
            )
            arrays[f"depth_ctrl__{view}"] = np.full(
                (p.control_h, p.control_w), lane + index, np.uint8
            )
            arrays[f"rgb_robot__{view}"] = np.full(
                (p.control_h, p.control_w, 3), lane + index, np.uint8
            )
            if self.use_raytraced[lane]:
                arrays[f"raytraced_rgb__{view}"] = np.full(
                    (p.policy_h, p.policy_w, 3),
                    (lane + index + 1) / float(self.num_envs + 4),
                    np.float32,
                )
        metadata = {
            "view_order": list(RAVEN_VIEW_ORDER),
            "model_view_map": dict(MODEL_VIEW_MAP),
            "observation_source": (
                "raytraced" if self.use_raytraced[lane] else "generated_controls"
            ),
            "prompt": f"fake Raven batch prompt lane {lane}",
            "env_id": lane,
            "seq": self.bind_lane_calls[lane],
            "depth_units": "meter",
            "depth_control_encoding": "raven_h5_v4",
            "robot_control_encoding": "textured_robot_rgb_on_black",
        }
        return Capture(key=key, metadata=metadata, arrays=arrays)

    def step_batch(
        self,
        selected_actions: np.ndarray,
        teacher_actions: np.ndarray,
    ) -> Sequence[TransitionResult]:
        expected = (self.num_envs, ACTION_DIM)
        if selected_actions.shape != expected or selected_actions.dtype != np.float32:
            raise BatchBackendInvariantError(
                f"selected action batch must be float32 {expected}"
            )
        if teacher_actions.shape != expected or teacher_actions.dtype != np.float32:
            raise BatchBackendInvariantError(
                f"teacher action batch must be float32 {expected}"
            )
        call = self.batch_step_calls
        self.batch_step_calls += 1
        self.selected_batches.append(selected_actions.copy())
        self.teacher_batches.append(teacher_actions.copy())
        if call in self.raise_at_batch_calls:
            raise RuntimeError("injected uncertain vectorized transition")
        results: list[TransitionResult] = []
        for lane in range(self.num_envs):
            self.active_episode_step[lane] += 1
            results.append(
                TransitionResult(
                    # Keep the fake reward exactly representable in float32 so
                    # strict remote seal checks exercise the barrier rather
                    # than an irrelevant JSON-vs-NPZ rounding difference.
                    reward=float(call * self.num_envs + lane),
                    done=call in set(self.done_at_calls.get(lane, set())),
                    timeout=call in set(self.timeout_at_calls.get(lane, set())),
                    metadata={"fake_batch_call_index": call, "lane_index": lane},
                )
            )
        return results
