from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .backend import Capture, TransitionResult
from .contracts import (
    MODEL_VIEW_MAP,
    PINNED_RAVEN_COMMIT,
    PINNED_RAVEN_ROOT,
    RAVEN_CAPTURE_VIEWS,
    RAVEN_VIEW_ORDER,
    RavenShapeProfile,
    StepKey,
    anchor_rgb_digest,
)


def fake_static_provenance() -> dict[str, str]:
    return {
        "raven_root": PINNED_RAVEN_ROOT,
        "raven_commit": PINNED_RAVEN_COMMIT,
        "raven_videogen_trainer_sha256": "1" * 64,
        "raven_teacher_homie_trainer_sha256": "2" * 64,
        "isaac_capture_sha256": "3" * 64,
        "image_dagger_config_sha256": "4" * 64,
        "teacher_checkpoint_sha256": "5" * 64,
        "homie_walk_sha256": "6" * 64,
        "homie_stand_sha256": "7" * 64,
    }


@dataclass
class DeterministicFakeRaven:
    profile: RavenShapeProfile
    done_at_calls: set[int] = field(default_factory=set)
    raise_at_calls: set[int] = field(default_factory=set)

    def __post_init__(self):
        self.step_calls = 0
        self.bind_calls = 0
        self.capture_calls = 0
        self.active_episode: str | None = None
        self.active_episode_step = 0
        self.actions: list[np.ndarray] = []
        self.use_raytraced = False
        self.anchor_arrays: dict[str, np.ndarray] = {}

    def bind_episode(self, *, key: StepKey) -> Mapping[str, Any]:
        self.bind_calls += 1
        self.active_episode = key.episode_id
        self.active_episode_step = 0
        self.use_raytraced = bool(self.bind_calls % 2)
        self.anchor_arrays = {
            f"anchor_rgb__{view}": np.full(
                (self.profile.control_h, self.profile.control_w, 3), index + self.bind_calls, np.uint8
            )
            for index, view in enumerate(RAVEN_VIEW_ORDER)
        }
        anchor = anchor_rgb_digest(self.anchor_arrays)
        return {
            "use_raytraced": self.use_raytraced,
            "anchor_sha256": anchor,
            "raven_source": PINNED_RAVEN_ROOT,
            "raven_commit": PINNED_RAVEN_COMMIT,
        }

    def capture_current(self, *, key: StepKey) -> Capture:
        if key.episode_id != self.active_episode or key.episode_step_index != self.active_episode_step:
            raise AssertionError("fake Raven capture out of episode order")
        self.capture_calls += 1
        p = self.profile
        base = float(key.rollout_step_index)
        homie = np.arange(15, dtype=np.float32) + base
        arrays: dict[str, np.ndarray] = {
            "actor_obs": np.full((p.actor_obs,), base, dtype=np.float32),
            "teacher_obs": np.full((p.teacher_obs,), base + 1, dtype=np.float32),
            "critic_obs": np.full((p.critic_obs,), base + 2, dtype=np.float32),
            "homie_obs": np.full((p.homie_obs,), base + 3, dtype=np.float32),
            "gt_obj_pos": np.asarray((base, 1, 2), dtype=np.float32),
            "homie_action": homie,
            "teacher_action": np.concatenate((np.arange(11, dtype=np.float32) - base, homie)),
        }
        arrays.update({name: value.copy() for name, value in self.anchor_arrays.items()})
        for index, view in enumerate(RAVEN_CAPTURE_VIEWS):
            arrays[f"depth_metric__{view}"] = np.full((p.depth_h, p.depth_w), index + 0.5, np.float32)
            arrays[f"depth_rawq__{view}"] = np.full((p.depth_h, p.depth_w), index, np.uint16)
            arrays[f"depth_raw__{view}"] = np.full((p.control_h, p.control_w), index + 0.5, np.float16)
            arrays[f"depth_ctrl__{view}"] = np.full((p.control_h, p.control_w), index, np.uint8)
            arrays[f"rgb_robot__{view}"] = np.full((p.control_h, p.control_w, 3), index, np.uint8)
            if self.use_raytraced:
                arrays[f"raytraced_rgb__{view}"] = np.full(
                    (p.policy_h, p.policy_w, 3), (index + 1) / 4.0, np.float32
                )
        metadata = {
            "view_order": list(RAVEN_VIEW_ORDER),
            "model_view_map": dict(MODEL_VIEW_MAP),
            "observation_source": "raytraced" if self.use_raytraced else "generated_controls",
            "prompt": "fake Raven prompt",
            "env_id": 7,
            "seq": self.bind_calls,
            "depth_units": "meter",
            "depth_control_encoding": "raven_h5_v4",
            "robot_control_encoding": "textured_robot_rgb_on_black",
        }
        return Capture(key=key, metadata=metadata, arrays=arrays)

    def step_once(self, action: np.ndarray) -> TransitionResult:
        call = self.step_calls
        self.step_calls += 1
        self.actions.append(action.copy())
        if call in self.raise_at_calls:
            raise RuntimeError("injected uncertain fake transition")
        done = call in self.done_at_calls
        self.active_episode_step += 1
        return TransitionResult(
            reward=float(call) / 10.0,
            done=done,
            timeout=False,
            metadata={"fake_call_index": call},
        )
