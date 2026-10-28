import torch
import pytest
from collections import deque
from dagger_rtx.phase_reset_state import ColdPhaseState


class Environment:
    def __init__(self):
        self.termination_level = torch.tensor(.5)
        self.reward_penalty_scale = torch.tensor(.2)
        self.current_rotation_noise_ranges = {"camera": [-.1, .1]}
        self.average_episode_length = 0.
        self.common_step_counter = 0
        self.config = {"unused": object()}
        self.log_dict = {"average_goal_reached": torch.tensor(0.)}
        self.history = torch.ones(6, 5)
    def update_camera_curriculum(self):
        self.current_rotation_noise_ranges["camera"][1] *= 2
        self.average_episode_length += 1
        if self.config:
            self.log_dict["x"] = torch.tensor(2.)


def test_round_reset_restores_curricula_without_touching_physics_history():
    env = Environment()
    saved = ColdPhaseState(env)
    identity = id(env.history)
    env.termination_level.add_(1)
    env.reward_penalty_scale.mul_(2)
    env.update_camera_curriculum()
    env.common_step_counter = 137
    names = saved.restore(env)
    assert env.termination_level == .5 and env.reward_penalty_scale == .2
    assert env.current_rotation_noise_ranges == {"camera": [-.1, .1]}
    assert env.average_episode_length == 0
    assert env.common_step_counter == 0
    assert set(env.log_dict) == {"average_goal_reached"}
    assert id(env.history) == identity and env.history.eq(1).all()
    assert "history" not in names


def test_warm_phase_restores_both_native_clocks_without_stepping_or_replacing_buffers():
    pair = [Environment(), Environment()]
    saved = [ColdPhaseState(env) for env in pair]
    for env, elapsed in zip(pair, (324, 8100)):
        env.common_step_counter = elapsed
    for env, snapshot in zip(pair, saved):
        snapshot.restore(env)
    assert [env.common_step_counter for env in pair] == [0, 0]
    assert all(env.history.eq(1).all() for env in pair)


def test_large_observations_cannot_be_cold_reset_state():
    env = Environment()
    env.observation_curriculum = torch.ones(65537)
    with pytest.raises(RuntimeError, match="large observation"):
        ColdPhaseState(env)


def test_previous_phase_task_outcomes_do_not_enter_next_curriculum():
    env = Environment()
    env.grasp_goal_reached_deque = deque(maxlen=500)
    env.average_grasp_goal_reached_rate = 0.
    env.current_last_stage_completed_task_buf = torch.zeros(6, dtype=torch.bool)
    env.skip_initial_goal_tracking = True
    saved = ColdPhaseState(env)
    env.grasp_goal_reached_deque.extend([True, True])
    env.average_grasp_goal_reached_rate = 1.
    env.current_last_stage_completed_task_buf.fill_(True)
    env.skip_initial_goal_tracking = False
    saved.restore(env)
    assert not env.grasp_goal_reached_deque and env.grasp_goal_reached_deque.maxlen == 500
    assert env.average_grasp_goal_reached_rate == 0.
    assert not env.current_last_stage_completed_task_buf.any()
    assert env.skip_initial_goal_tracking is True
