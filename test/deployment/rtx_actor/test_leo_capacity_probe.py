"""CPU-only bounds, reset, ordering and launcher tests; no Isaac/GPU imports."""
from test.deployment.source_paths import ACTOR_ROOT
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ACTOR = ACTOR_ROOT
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ACTOR))
sys.path.insert(0, str(ROOT / "pipeline/dagger/rtx/code/src"))

from leo_capacity_probe import run_teacher_capacity, run_teacher_camera_soak, step_acknowledgement
from qualify_native_isaac6 import command, IMAGE
from run_throughput_pair import validate_layout


class Batch:
    num_envs = 2

    def __init__(self, *, bad_tail=False, fail_step=False):
        self.bindings, self.captures, self.steps = [], [], []
        self.bad_tail, self.fail_step = bad_tail, fail_step
        self.arrays = {"teacher_action": np.arange(26, dtype=np.float32),
                       "homie_action": np.arange(11, 26, dtype=np.float32),
                       "actor_obs": np.zeros(113, dtype=np.float32)}
        for view in ("head_color", "hand_left", "hand_right"):
            self.arrays[f"depth_ctrl__{view}"] = np.zeros((240, 416), dtype=np.uint8)
            self.arrays[f"rgb_robot__{view}"] = np.zeros((240, 416, 3), dtype=np.uint8)
        if bad_tail:
            self.arrays["homie_action"][0] = -1

    def bind_episodes(self, keys):
        self.bindings.append(dict(keys))

    def capture_batch(self, keys):
        self.captures.append(list(keys))
        return [SimpleNamespace(key=k, arrays=self.arrays,
                                metadata={"observation_source": "raytraced"}) for k in keys]

    def step_batch(self, selected, teacher):
        assert np.array_equal(selected, teacher)
        assert selected.shape == (self.num_envs, 26)
        self.steps.append(selected.copy())
        if self.fail_step:
            raise RuntimeError("uncertain native result")
        return [SimpleNamespace(reward=1., done=len(self.steps) == 1 and lane == 0,
                                timeout=len(self.steps) == 2 and lane == 1)
                for lane in range(self.num_envs)]


def probe(batch, steps=3):
    events, calls = [], []
    lifecycle = SimpleNamespace(begin=lambda: calls.append("begin"), finish=lambda: calls.append("finish"))
    result = run_teacher_capacity(batch, lifecycle, steps=steps,
        acknowledgement=step_acknowledgement(steps), report=events.append)
    return result, events, calls


def test_exact_steps_teacher_actions_reset_rebinding_and_no_learning():
    batch = Batch()
    result, events, calls = probe(batch)
    assert calls == ["begin", "finish"]
    assert len(batch.steps) == 3
    assert result["environment_step_calls"] == 3
    assert result["physical_transition_rows"] == 6
    assert result["optimizer_steps"] == result["chunks_published"] == 0
    assert result["H200_renderer_tested"] is False
    assert result["production_qualified"] is False
    assert [set(row) for row in batch.bindings] == [{0, 1}, {0}, {1}]
    assert batch.captures[1][0].episode_step_index == 0
    assert batch.captures[1][0].episode_id != batch.captures[0][0].episode_id
    assert batch.captures[1][1].episode_step_index == 1
    assert batch.captures[2][1].episode_step_index == 0
    assert [e["phase"] for e in events] == ["capture_teacher_start", "env_step_start", "vector_step_complete"] * 3


@pytest.mark.parametrize('count', [16, 32, 64])
def test_wide_diagnostic_preserves_exact_lane_and_step_cardinality(count):
    batch=Batch();batch.num_envs=count
    result,events,calls=probe(batch)
    assert result['environment_step_calls']==3
    assert result['physical_transition_rows']==3*count
    assert all(actions.shape==(count,26) for actions in batch.steps)
    assert len(batch.captures[0])==count
    assert result['production_qualified'] is False


def test_capacity_bound_rejected_before_native_lifecycle():
    batch=Batch();batch.num_envs=65
    with pytest.raises(ValueError,match='bounded scope'):
        run_teacher_capacity(batch,None,steps=3,acknowledgement=step_acknowledgement(3),report=lambda row:None)
    assert not batch.steps


@pytest.mark.parametrize("steps", [0, 1, 2, 4, 81, True])
def test_unbounded_or_implicit_step_budget_rejected(steps):
    with pytest.raises(ValueError):
        step_acknowledgement(steps)


def test_wrong_ack_and_corrupt_teacher_fail_before_step():
    with pytest.raises(ValueError):
        run_teacher_capacity(Batch(), None, steps=3, acknowledgement="yes", report=lambda x: None)
    batch = Batch(bad_tail=True)
    with pytest.raises(RuntimeError, match="teacher/Homie"):
        probe(batch)
    assert batch.steps == []


def test_uncertain_step_is_not_retried_and_lifecycle_is_finished():
    batch = Batch(fail_step=True)
    calls, events = [], []
    lifecycle = SimpleNamespace(begin=lambda: calls.append("begin"), finish=lambda: calls.append("finish"))
    with pytest.raises(RuntimeError, match="uncertain native"):
        run_teacher_capacity(batch, lifecycle, steps=3,
            acknowledgement=step_acknowledgement(3), report=events.append)
    assert len(batch.steps) == 1 and calls == ["begin", "finish"]
    assert events[-1]["phase"] == "env_step_start"


def test_camera_soak_preserves_lifecycle_across_four_windows():
    batch = Batch(); batch.num_envs = 16
    calls, events = [], []
    lifecycle = SimpleNamespace(begin=lambda: calls.append("begin"), finish=lambda: calls.append("finish"))
    result = run_teacher_camera_soak(batch, lifecycle, steps=324,
        acknowledgement="I_ACKNOWLEDGE_324_NATIVE_CAMERA_SOAK_TEACHER_STEPS", report=events.append)
    assert calls == ["begin", "finish"]
    assert len(batch.steps) == 324
    assert result["schema"] == "roborender.leo.teacher-camera-soak.v1"
    assert result["mode"] == "teacher-camera-soak"
    assert result["production_qualified"] is False
    assert result["physical_transition_rows"] == 5184
    assert result["optimizer_steps"] == result["chunks_published"] == 0
    assert result["H200_renderer_tested"] is False
    assert [set(row) for row in batch.bindings] == [{*range(16)}, {0}, {1}]
    for step, keys in enumerate(batch.captures):
        assert all(key.rollout_step_index == step % 81 for key in keys)
        assert all(key.rollout_window_id == f"window-{step // 81:06d}" for key in keys)
        # Lane2 has no real done; a window boundary must not reset its episode.
        assert keys[2].episode_step_index == step
        assert keys[2].episode_id == batch.captures[0][2].episode_id


@pytest.mark.parametrize("steps,ack", [(10, "I_ACKNOWLEDGE_10_NATIVE_TEACHER_VECTOR_STEPS"),
    (324, "yes"), (True, "I_ACKNOWLEDGE_324_NATIVE_CAMERA_SOAK_TEACHER_STEPS")])
def test_camera_soak_budget_and_ack_fail_before_native(steps, ack):
    batch = Batch(); batch.num_envs = 16
    with pytest.raises(ValueError, match="camera soak"):
        run_teacher_camera_soak(batch, None, steps=steps, acknowledgement=ack, report=lambda x: None)
    assert not batch.steps


def test_camera_soak_launcher_cannot_connect_to_h200_or_expand_batch(tmp_path):
    (tmp_path / "port-manifest.json").write_text('{"files":[]}')
    options = dict(mode="teacher-camera-soak", deployment_host="leo-rtx-6k",
                   physical_gpu=0, capacity_steps=324)
    cmd = command("test", tmp_path, tmp_path, ROOT / "pipeline/dagger/rtx/code", 16, **options)
    assert "ROBORENDER_LEO_PROBE_MODE=teacher-camera-soak" in cmd
    assert "ROBORENDER_LEO_CAPACITY_ACK=I_ACKNOWLEDGE_324_NATIVE_CAMERA_SOAK_TEACHER_STEPS" in cmd
    assert "DAGGER_RTX_MODE=capture-only" in cmd
    assert cmd[cmd.index("--network") + 1] == "bridge"
    assert not any("H200_TOKEN" in arg or "DAGGER_H200_BASE" in arg for arg in cmd)
    for size in (2, 28, 32, 64):
        with pytest.raises(ValueError, match="camera soak"):
            command("test", tmp_path, tmp_path, ROOT / "pipeline/dagger/rtx/code", size, **options)
    with pytest.raises(ValueError, match="resource limits"):
        command("test", tmp_path, tmp_path, ROOT / "pipeline/dagger/rtx/code", 16,
                mode="teacher-capacity", deployment_host="leo-rtx-6k", physical_gpu=0, capacity_steps=324)


@pytest.mark.parametrize("gpu", [0, 1])
@pytest.mark.parametrize("num_envs", [8, 12, 16, 32, 64])
def test_qualified_command_is_scoped_no_h200_no_learner(tmp_path, gpu, num_envs):
    (tmp_path / "port-manifest.json").write_text('{"files":[]}')
    cmd = command("rtx-leo-migration-capacity-test", tmp_path, tmp_path,
        ROOT / "pipeline/dagger/rtx/code", num_envs, mode="teacher-capacity",
        physical_gpu=gpu, deployment_host="leo-rtx-6k", memory_gib=24,
        run_id="rtx-leo-migration-capacity-test", benchmark_barrier=tmp_path,
        capacity_steps=3, compact_host_heap=True)
    # Preserve native asset resolution, but do not expose host/H200 services.
    assert cmd[cmd.index("--network")+1] == "bridge"
    assert cmd[cmd.index("--memory")+1] == "24g"
    assert cmd[cmd.index("--memory-swap")+1] == "24g"
    assert "trainer._target_=isaac6_capacity_trainer.Isaac6TeacherCapacityTrainer" in cmd
    assert "DAGGER_RTX_MODE=capture-only" in cmd
    assert "ROBORENDER_LEO_CAPACITY_STEPS=3" in cmd
    assert "ROBORENDER_LEO_CAPACITY_ACK="+step_acknowledgement(3) in cmd
    assert "DAGGER_BENCHMARK_BARRIER_DIR=/benchmark-barrier" in cmd
    assert "WANDB_MODE=disabled" in cmd
    assert not any("H200_TOKEN" in arg or "DAGGER_H200_BASE" in arg for arg in cmd)
    assert IMAGE in cmd


def test_wrong_host_display_gpu_and_unsafe_pair_budget_rejected(tmp_path):
    (tmp_path / "port-manifest.json").write_text('{"files":[]}')
    with pytest.raises(ValueError, match="Leo only"):
        command("test", tmp_path, tmp_path, ROOT / "pipeline/dagger/rtx/code", 8, mode="teacher-capacity")
    with pytest.raises(ValueError, match="display GPU2"):
        command("test", tmp_path, tmp_path, ROOT / "pipeline/dagger/rtx/code", 8,
                mode="teacher-capacity", deployment_host="leo-rtx-6k", physical_gpu=2)
    assert validate_layout([0, 1], 24, 4, "leo-rtx-6k") == 54
    with pytest.raises(ValueError):
        validate_layout([0, 1], 32, 4, "leo-rtx-6k")
