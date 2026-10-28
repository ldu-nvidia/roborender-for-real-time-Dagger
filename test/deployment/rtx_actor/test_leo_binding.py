import json
from pathlib import Path

import pytest

from leo_binding import GPU_UUIDS, HOSTNAME, checked_gpu
from qualify_native_isaac6 import command, GPU_UUID


def plan(tmp_path, **kwargs):
    (tmp_path / "port-manifest.json").write_text(json.dumps({"files": []}))
    return command("leo-test", tmp_path / "out", tmp_path, tmp_path / "code", 2, **kwargs)


@pytest.mark.parametrize("gpu", [0, 1])
def test_leo_has_explicit_uuid_and_separate_cache(tmp_path, gpu):
    cmd = plan(tmp_path, physical_gpu=gpu, deployment_host=HOSTNAME)
    assert cmd[cmd.index("--gpus") + 1] == "device=" + GPU_UUIDS[gpu]
    assert f"CUDA_VISIBLE_DEVICES={GPU_UUIDS[gpu]}" in cmd
    assert any(f"runtime/isaac6-cache-leo-gpu{gpu}" in item for item in cmd)
    assert "algo.config.num_learning_epochs=0" in cmd
    assert "algo.config.init_at_random_ep_len=true" in cmd
    assert "simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]" in cmd


def test_beta_default_unchanged_and_gpu0_remains_forbidden(tmp_path):
    cmd = plan(tmp_path)
    assert cmd[cmd.index("--gpus") + 1] == "device=" + GPU_UUID
    with pytest.raises(ValueError):
        plan(tmp_path, physical_gpu=0)


@pytest.mark.parametrize("gpu,mode", [(2, "capture-only"), (True, "capture-only"), (0, "beta-production"), (1, "multi-window-check")])
def test_leo_rejects_display_gpu_and_unqualified_production(gpu, mode):
    with pytest.raises(ValueError):
        checked_gpu(gpu, mode)


def test_leo_bounded_crosshost_uses_real_native_owner(tmp_path):
    cmd = plan(tmp_path, physical_gpu=0, deployment_host=HOSTNAME,
               mode="isolated-one-step", run_id="rtx-leo-migration-one-step",
               token_file=Path("/private/token"), h200_pid=123)
    assert "DAGGER_PHYSICAL_GPU=0" in cmd
    assert "DAGGER_NATIVE_STEP_ACK=I_ACKNOWLEDGE_ONE_NATIVE_VECTOR_STEP" in cmd
    assert "trainer._target_=dagger_rtx.native_service_trainer.NativeDaggerRTXServiceTrainer" in cmd
    with pytest.raises(ValueError):
        plan(tmp_path, physical_gpu=0, deployment_host=HOSTNAME,
             mode="isolated-one-step", run_id="rtx-beta-migration-wrong-host",
             token_file=Path("/private/token"), h200_pid=123)


def test_unknown_host_not_allowed(tmp_path):
    with pytest.raises(ValueError):
        plan(tmp_path, deployment_host="other-host")
