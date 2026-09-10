from __future__ import annotations

import inspect

import numpy as np
import pytest

import dagger_rtx.native_raven as native_module
from dagger_rtx.backend import BackendInvariantError
from dagger_rtx.contracts import (
    MODEL_VIEW_MAP,
    PINNED_RAVEN_COMMIT,
    PINNED_RAVEN_ROOT,
    RAVEN_VIEW_ORDER,
)
from dagger_rtx.native_raven import (
    NATIVE_CALLBACK_ORIGINS,
    NativeActionLabels,
    NativeCurrentBatch,
    NativeDepthArchive,
    NativeRavenEntrypoints,
    build_native_raven_batch,
    build_native_raven_hooks,
)

from conftest import make_key


class CaptureOnlySimulator:
    def __init__(self, calls: list[str], profile):
        self.calls = calls
        self.profile = profile

    def get_conditioning_views(self):
        self.calls.append("get_conditioning_views")
        p = self.profile
        return {
            view: {
                "depth": np.full(
                    (1, p.depth_h, p.depth_w, 1), index + 0.5, np.float32
                ),
                "robot_rgb": np.full(
                    (1, p.depth_h, p.depth_w, 3), index + 10, np.uint8
                ),
            }
            for index, view in enumerate(RAVEN_VIEW_ORDER)
        }

    def get_policy_view_stack(self):
        self.calls.append("get_policy_view_stack")
        p = self.profile
        return np.concatenate(
            [
                np.full((1, p.policy_h, p.policy_w, 3), index / 4, np.float32)
                for index, _view in enumerate(RAVEN_VIEW_ORDER)
            ],
            axis=1,
        )


class CaptureOnlyEnv:
    num_envs = 1

    def __init__(self, calls: list[str], profile):
        self.calls = calls
        self.simulator = CaptureOnlySimulator(calls, profile)
        self.step_calls = 0

    def step(self, _action_state):
        self.step_calls += 1
        raise AssertionError("capture-only test must never call env.step")


class CaptureOnlyTrainer:
    vg_depth_mode = "h5_v4"
    vg_view_names = list(RAVEN_VIEW_ORDER)
    _policy_views = list(RAVEN_VIEW_ORDER)

    def __init__(self, calls: list[str], profile, *, use_raytraced: bool = False):
        self.calls = calls
        self.profile = profile
        self.env = CaptureOnlyEnv(calls, profile)
        self.vg_control_res = [profile.control_h, profile.control_w]
        self._id_anchors = None
        self._id_prompts = None
        self._id_use_raytraced = None
        self.use_raytraced = use_raytraced

    def _id_sample_anchor(self, env_index: int):
        self.calls.append(f"_id_sample_anchor:{env_index}")
        self._id_anchors[env_index] = {
            view: np.full(
                (self.profile.control_h, self.profile.control_w, 3),
                index + 1,
                np.uint8,
            )
            for index, view in enumerate(RAVEN_VIEW_ORDER)
        }
        self._id_prompts[env_index] = "deterministic capture-only prompt"
        self._id_use_raytraced[env_index] = self.use_raytraced

    def _h5v4_control(self, depth, view):
        self.calls.append(f"_h5v4_control:{view}")
        return np.full(depth.shape, 77 + RAVEN_VIEW_ORDER.index(view), np.uint8)

    def _resize_seq(self, value, hw):
        channels = value.shape[-1] if value.ndim == 5 else 1
        self.calls.append(f"_resize_seq:{value.dtype.name}:{channels}")
        h, w = hw
        if value.ndim == 4:
            return np.full(
                (value.shape[0], value.shape[1], h, w), value.flat[0], value.dtype
            )
        return np.full(
            (value.shape[0], value.shape[1], h, w, 3), value.flat[0], value.dtype
        )


class VectorSimulator:
    """Deterministic two-lane stand-in for Raven's synchronized cameras."""

    def __init__(self, calls: list[str], profile):
        self.calls = calls
        self.profile = profile

    def get_conditioning_views(self):
        self.calls.append("get_conditioning_views")
        p = self.profile
        return {
            view: {
                "depth": np.stack(
                    [
                        np.full(
                            (p.depth_h, p.depth_w, 1),
                            10 * lane + view_index + 0.5,
                            np.float32,
                        )
                        for lane in range(2)
                    ]
                ),
                "robot_rgb": np.stack(
                    [
                        np.full(
                            (p.depth_h, p.depth_w, 3),
                            20 * lane + view_index,
                            np.uint8,
                        )
                        for lane in range(2)
                    ]
                ),
            }
            for view_index, view in enumerate(RAVEN_VIEW_ORDER)
        }

    def get_policy_view_stack(self):
        self.calls.append("get_policy_view_stack")
        p = self.profile
        return np.concatenate(
            [
                np.stack(
                    [
                        np.full(
                            (p.policy_h, p.policy_w, 3),
                            (10 * lane + view_index) / 20,
                            np.float32,
                        )
                        for lane in range(2)
                    ]
                )
                for view_index, _view in enumerate(RAVEN_VIEW_ORDER)
            ],
            axis=1,
        )


class VectorEnv:
    num_envs = 2

    def __init__(self, calls: list[str], profile):
        self.calls = calls
        self.simulator = VectorSimulator(calls, profile)
        self.step_calls = 0
        self.action_states = []

    def step(self, action_state):
        self.calls.append("env.step")
        self.step_calls += 1
        self.action_states.append(action_state)
        return {"native_batch_result": self.step_calls}


class VectorTrainer(CaptureOnlyTrainer):
    def __init__(self, calls: list[str], profile):
        super().__init__(calls, profile)
        self.env = VectorEnv(calls, profile)

    def _id_sample_anchor(self, env_index: int):
        self.calls.append(f"_id_sample_anchor:{env_index}")
        self._id_anchors[env_index] = {
            view: np.full(
                (self.profile.control_h, self.profile.control_w, 3),
                20 * env_index + view_index + 1,
                np.uint8,
            )
            for view_index, view in enumerate(RAVEN_VIEW_ORDER)
        }
        self._id_prompts[env_index] = f"deterministic vector prompt lane {env_index}"
        # Exercise the mixed-source split: lane 0 is ray-traced; lane 1 uses
        # generated controls and therefore must not carry ray-traced RGB.
        self._id_use_raytraced[env_index] = env_index == 0


def make_capture_only_entrypoints(calls, profile, *, bad_homie=False):
    def prepare_episode_slots(trainer, batch):
        calls.append(f"prepare_episode_slots:{batch}")
        assert batch == 1
        trainer._id_anchors = [None]
        trainer._id_prompts = [None]
        trainer._id_use_raytraced = {}

    def read_current_batch(keys):
        calls.append("read_current_batch")
        assert len(keys) == 1
        return NativeCurrentBatch(
            keys=tuple(keys),
            obs_dict={"native": "opaque"},
            cur_dones=np.array([False]),
            arrays={
                "actor_obs": np.zeros((1, profile.actor_obs), np.float32),
                "teacher_obs": np.ones((1, profile.teacher_obs), np.float32),
                "critic_obs": np.full((1, profile.critic_obs), 2, np.float32),
                "homie_obs": np.full((1, profile.homie_obs), 3, np.float32),
                "gt_obj_pos": np.arange(3, dtype=np.float32)[None],
            },
            env_ids=np.array([17], dtype=np.int64),
            seqs=np.array([4], dtype=np.int64),
        )

    def teacher_homie_labels(_trainer, _current):
        calls.append("teacher_homie_labels")
        homie = np.arange(15, dtype=np.float32)[None]
        teacher_homie = homie + 1 if bad_homie else homie
        return NativeActionLabels(
            teacher_action=np.concatenate(
                (np.arange(11, dtype=np.float32)[None], teacher_homie), axis=1
            ),
            homie_action=homie,
        )

    def native_depth_archive(_trainer, views):
        calls.append("native_depth_archive")
        assert tuple(views) == RAVEN_VIEW_ORDER
        return {
            view: NativeDepthArchive(
                depth_rawq=np.full(
                    (1, profile.depth_h, profile.depth_w),
                    100 + RAVEN_VIEW_ORDER.index(view),
                    np.uint16,
                ),
                depth_raw=np.full(
                    (1, profile.control_h, profile.control_w),
                    RAVEN_VIEW_ORDER.index(view) + 0.25,
                    np.float16,
                ),
            )
            for view in RAVEN_VIEW_ORDER
        }

    def forbidden_action_state(*_args):
        raise AssertionError("capture-only test crossed into action construction")

    def forbidden_decode(*_args):
        raise AssertionError("capture-only test crossed into transition decoding")

    return NativeRavenEntrypoints(
        raven_root=PINNED_RAVEN_ROOT,
        raven_commit=PINNED_RAVEN_COMMIT,
        origins=dict(NATIVE_CALLBACK_ORIGINS),
        prepare_episode_slots=prepare_episode_slots,
        read_current_batch=read_current_batch,
        teacher_homie_labels=teacher_homie_labels,
        native_depth_archive=native_depth_archive,
        build_action_state=forbidden_action_state,
        decode_env_step=forbidden_decode,
    )


def make_vector_entrypoints(calls, profile):
    def prepare_episode_slots(trainer, batch):
        calls.append(f"prepare_episode_slots:{batch}")
        assert batch == 2
        trainer._id_anchors = [None, None]
        trainer._id_prompts = [None, None]
        trainer._id_use_raytraced = [None, None]

    def read_current_batch(keys):
        calls.append("read_current_batch")
        assert len(keys) == 2
        return NativeCurrentBatch(
            keys=tuple(keys),
            obs_dict={"native": "opaque-vector-batch"},
            cur_dones=np.array([False, True]),
            arrays={
                "actor_obs": np.arange(2 * profile.actor_obs, dtype=np.float32).reshape(
                    2, profile.actor_obs
                ),
                "teacher_obs": np.arange(
                    2 * profile.teacher_obs, dtype=np.float32
                ).reshape(2, profile.teacher_obs),
                "critic_obs": np.arange(
                    2 * profile.critic_obs, dtype=np.float32
                ).reshape(2, profile.critic_obs),
                "homie_obs": np.arange(
                    2 * profile.homie_obs, dtype=np.float32
                ).reshape(2, profile.homie_obs),
                "gt_obj_pos": np.arange(6, dtype=np.float32).reshape(2, 3),
            },
            env_ids=np.array([101, 202], dtype=np.int64),
            seqs=np.array([7, 9], dtype=np.int64),
        )

    def teacher_homie_labels(_trainer, _current):
        calls.append("teacher_homie_labels")
        homie = np.arange(30, dtype=np.float32).reshape(2, 15)
        policy = np.stack(
            [
                np.arange(11, dtype=np.float32),
                np.arange(11, dtype=np.float32) + 100,
            ]
        )
        return NativeActionLabels(
            teacher_action=np.concatenate((policy, homie), axis=1),
            homie_action=homie,
        )

    def native_depth_archive(_trainer, views):
        calls.append("native_depth_archive")
        assert tuple(views) == RAVEN_VIEW_ORDER
        return {
            view: NativeDepthArchive(
                depth_rawq=np.stack(
                    [
                        np.full(
                            (profile.depth_h, profile.depth_w),
                            100 * lane + view_index,
                            np.uint16,
                        )
                        for lane in range(2)
                    ]
                ),
                depth_raw=np.stack(
                    [
                        np.full(
                            (profile.control_h, profile.control_w),
                            10 * lane + view_index + 0.25,
                            np.float16,
                        )
                        for lane in range(2)
                    ]
                ),
            )
            for view_index, view in enumerate(RAVEN_VIEW_ORDER)
        }

    def build_action_state(_trainer, selected, teachers):
        calls.append("build_action_state")
        return {"actions": selected.copy(), "gt_actions": teachers.copy()}

    def decode_env_step(_trainer, raw_result):
        calls.append("decode_env_step")
        assert raw_result == {"native_batch_result": 1}
        return (
            native_module.TransitionResult(
                reward=1.25,
                done=False,
                timeout=False,
                metadata={"native_env_index": 0},
            ),
            native_module.TransitionResult(
                reward=-2.5,
                done=True,
                timeout=False,
                metadata={"native_env_index": 1},
            ),
        )

    return NativeRavenEntrypoints(
        raven_root=PINNED_RAVEN_ROOT,
        raven_commit=PINNED_RAVEN_COMMIT,
        origins=dict(NATIVE_CALLBACK_ORIGINS),
        prepare_episode_slots=prepare_episode_slots,
        read_current_batch=read_current_batch,
        teacher_homie_labels=teacher_homie_labels,
        native_depth_archive=native_depth_archive,
        build_action_state=build_action_state,
        decode_env_step=decode_env_step,
    )


def test_b1_capture_uses_one_synchronized_native_batch_and_never_steps(tiny_profile):
    calls: list[str] = []
    trainer = CaptureOnlyTrainer(calls, tiny_profile)
    composition = build_native_raven_batch(
        trainer,
        make_capture_only_entrypoints(calls, tiny_profile),
        shape_profile=tiny_profile,
    )
    assert calls == []

    key = make_key()
    assignment = composition.bind_episodes({0: key})[0]
    assert assignment["use_raytraced"] is False
    assert calls == ["prepare_episode_slots:1", "_id_sample_anchor:0"]

    (capture,) = composition.capture_batch([key])
    assert capture.key == key
    assert capture.metadata == {
        "view_order": list(RAVEN_VIEW_ORDER),
        "model_view_map": dict(MODEL_VIEW_MAP),
        "observation_source": "generated_controls",
        "prompt": "deterministic capture-only prompt",
        "env_id": 17,
        "seq": 4,
        "depth_units": "meter",
        "depth_control_encoding": "raven_h5_v4",
        "robot_control_encoding": "textured_robot_rgb_on_black",
    }
    assert capture.arrays["teacher_action"].shape == (26,)
    assert np.array_equal(
        capture.arrays["teacher_action"][11:], capture.arrays["homie_action"]
    )
    for index, view in enumerate(RAVEN_VIEW_ORDER):
        assert capture.arrays[f"depth_metric__{view}"].shape == (
            tiny_profile.depth_h,
            tiny_profile.depth_w,
        )
        assert capture.arrays[f"depth_ctrl__{view}"].min() == 77 + index
        assert capture.arrays[f"depth_rawq__{view}"].min() == 100 + index
        assert capture.arrays[f"anchor_rgb__{view}"].min() == index + 1
    assert calls.count("get_conditioning_views") == 1
    assert calls.count("teacher_homie_labels") == 1
    assert calls.count("native_depth_archive") == 1
    assert "get_policy_view_stack" not in calls
    assert trainer.env.step_calls == 0


def test_b1_raytraced_capture_fetches_policy_stack_once_in_online_order(tiny_profile):
    calls: list[str] = []
    trainer = CaptureOnlyTrainer(calls, tiny_profile, use_raytraced=True)
    composition = build_native_raven_batch(
        trainer,
        make_capture_only_entrypoints(calls, tiny_profile),
        shape_profile=tiny_profile,
    )
    key = make_key()
    composition.bind_episodes({0: key})
    (capture,) = composition.capture_batch([key])

    assert capture.metadata["observation_source"] == "raytraced"
    assert calls.count("get_policy_view_stack") == 1
    assert capture.arrays["raytraced_rgb__head_color"].mean() == 0.0
    assert capture.arrays["raytraced_rgb__hand_left"].mean() == 0.25
    assert capture.arrays["raytraced_rgb__hand_right"].mean() == 0.5
    assert trainer.env.step_calls == 0


def test_b2_native_batch_is_synchronized_mixed_source_and_steps_once(tiny_profile):
    calls: list[str] = []
    trainer = VectorTrainer(calls, tiny_profile)
    composition = build_native_raven_batch(
        trainer,
        make_vector_entrypoints(calls, tiny_profile),
        shape_profile=tiny_profile,
    )
    keys = (
        make_key(episode_id="episode-lane-0"),
        make_key(episode_id="episode-lane-1"),
    )

    assignments = composition.bind_episodes({0: keys[0], 1: keys[1]})
    assert assignments[0]["use_raytraced"] is True
    assert assignments[1]["use_raytraced"] is False

    captures = tuple(composition.capture_batch(keys))
    assert tuple(capture.key for capture in captures) == keys
    assert calls.count("get_conditioning_views") == 1
    assert calls.count("teacher_homie_labels") == 1
    assert calls.count("native_depth_archive") == 1
    assert calls.count("get_policy_view_stack") == 1

    assert captures[0].metadata["observation_source"] == "raytraced"
    assert captures[1].metadata["observation_source"] == "generated_controls"
    for view_index, view in enumerate(RAVEN_VIEW_ORDER):
        ray_key = f"raytraced_rgb__{view}"
        assert ray_key in captures[0].arrays
        assert ray_key not in captures[1].arrays
        assert captures[0].arrays[ray_key].mean() == pytest.approx(view_index / 20)
    assert captures[0].metadata["env_id"] == 101
    assert captures[1].metadata["env_id"] == 202

    selected = np.arange(52, dtype=np.float32).reshape(2, 26) / 10
    teachers = np.stack(
        [capture.arrays["teacher_action"] for capture in captures]
    ).astype(np.float32)
    outcomes = tuple(composition.step_batch(selected, teachers))

    assert trainer.env.step_calls == 1
    assert calls.count("env.step") == 1
    assert calls.count("build_action_state") == 1
    assert calls.count("decode_env_step") == 1
    (action_state,) = trainer.env.action_states
    assert action_state["actions"].shape == (2, 26)
    assert action_state["actions"].dtype == np.float32
    assert action_state["gt_actions"].shape == (2, 26)
    assert action_state["gt_actions"].dtype == np.float32
    assert np.array_equal(action_state["actions"], selected)
    assert np.array_equal(action_state["gt_actions"], teachers)
    assert [outcome.reward for outcome in outcomes] == [1.25, -2.5]
    assert [outcome.done for outcome in outcomes] == [False, True]
    assert [outcome.timeout for outcome in outcomes] == [False, False]
    assert [outcome.metadata["native_env_index"] for outcome in outcomes] == [0, 1]


def test_teacher_homie_mismatch_fails_before_any_environment_step(tiny_profile):
    calls: list[str] = []
    trainer = CaptureOnlyTrainer(calls, tiny_profile)
    composition = build_native_raven_batch(
        trainer,
        make_capture_only_entrypoints(calls, tiny_profile, bad_homie=True),
        shape_profile=tiny_profile,
    )
    composition.bind_episodes({0: make_key()})
    with pytest.raises(BackendInvariantError, match=r"teacher_action\[:,11:\]"):
        composition.capture_batch([make_key()])
    assert trainer.env.step_calls == 0


def test_single_lane_hooks_are_validation_only_for_vectorized_raven(tiny_profile):
    calls: list[str] = []
    trainer = CaptureOnlyTrainer(calls, tiny_profile)
    trainer.env.num_envs = 2
    with pytest.raises(BackendInvariantError, match="vector environments require the batch barrier"):
        build_native_raven_hooks(
            trainer,
            make_capture_only_entrypoints(calls, tiny_profile),
            shape_profile=tiny_profile,
        )
    assert calls == []


def test_surface_verifier_rejects_unattested_callback_origin(tiny_profile):
    calls: list[str] = []
    trainer = CaptureOnlyTrainer(calls, tiny_profile)
    entrypoints = make_capture_only_entrypoints(calls, tiny_profile)
    entrypoints.origins["native_depth_archive"] = "unverified-copy"
    with pytest.raises(BackendInvariantError, match="wrong retained origin"):
        build_native_raven_batch(trainer, entrypoints, shape_profile=tiny_profile)
    assert calls == []


def test_composition_delegates_all_scientific_depth_math_to_raven():
    source = inspect.getsource(native_module)
    assert "._h5v4_control(" in source
    assert "native_depth_archive(" in source
    assert "np.percentile" not in source
    assert "np.log(" not in source
    assert "16383" not in source
