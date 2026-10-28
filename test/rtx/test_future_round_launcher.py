from __future__ import annotations
from test.deployment.source_paths import RTX_ROOT

import hashlib
import json
from pathlib import Path
import subprocess

import pytest


PROJECT = RTX_ROOT
LAUNCHER = PROJECT / "scripts" / "launch_native_raven.sh"
COORDINATOR_RUN_ID = "dagger-r1-coordinator"
WANDB_RUN_ID = "dagger-r2-teacher50-seed911223831-b28-50iter"


def _token(tmp_path: Path) -> Path:
    path = tmp_path / "token"
    path.write_text("test-token\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _config(*, fidelity: bool = True) -> dict:
    pids = list(range(1, 9))
    value = {
        "schema": "roborender.cross_gpu_dagger_round.v1",
        "run_id": COORDINATOR_RUN_ID,
        "existing_run_config_sha256": "b" * 64,
        "round_id": 2,
        "teacher_beta": 0.5,
        "seed": 911223831,
        "rtx_actor": {
            "simulator_seed": 911223831,
            "videogen_seed": 42,
            "runtime_root": f"/mnt/data2/ldu/roborender-runtime/{WANDB_RUN_ID}",
        },
        "h200": {
            "boundary_contract": "roborender.dagger.rtx.v4",
            "gateway_ports": list(range(18720, 18728)),
            "expected_gateway_pids": pids,
            "expected_gateway_instance_ids": [f"{index:064x}" for index in pids],
        },
        "student": {"initial_checkpoint_sha256": "a" * 64},
        "execution": {
            "window_count": 50,
            "stop_after_actor_barriers": False,
            "wait_for_checkpoint_advance": True,
            "learner_submission_authorized": True,
            "validation_authorized": True,
        },
        "validation": {"interval_iterations": 50},
        "logging": {
            "rtx_collection_wandb": False,
            "wandb_project": "dagger",
            "wandb_run_id": WANDB_RUN_ID,
            "wandb_run_name": WANDB_RUN_ID,
        },
    }
    if fidelity:
        value["raven_fidelity"] = {
            "raytraced_native_augmentation": True,
            "full_batch_sync_batchnorm": True,
            "random_initial_episode_lengths": True,
        }
    return value


def _invoke(tmp_path: Path, config: dict, *, ack: str):
    config_path = PROJECT / "config" / f".pytest-round2-{tmp_path.name}.json"
    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    config_path.write_bytes(payload)
    try:
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "production",
                "28",
                COORDINATOR_RUN_ID,
                str(config["round_id"]),
                f"{COORDINATOR_RUN_ID}-g1",
                "0",
                str(_token(tmp_path)),
                ack,
                "--gpu",
                "1",
                "--wire-protocol",
                "v4",
                "--expected-h200-pid",
                "1",
                "--expected-h200-pids",
                ",".join(str(pid) for pid in range(1, 9)),
                "--production-config",
                str(config_path),
                "--production-config-sha256",
                hashlib.sha256(payload).hexdigest(),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        config_path.unlink(missing_ok=True)


def test_round2_fifty_window_ack_and_distinct_identity_reach_fidelity_gate(tmp_path):
    result = _invoke(
        tmp_path,
        _config(fidelity=False),
        ack="I_ACKNOWLEDGE_FIFTY_NATIVE_81_STEP_WINDOWS_TO_VALIDATION",
    )
    assert result.returncode == 2
    assert "complete Raven-fidelity contract" in result.stderr
    assert "requires round1" not in result.stderr
    assert "W&B" not in result.stderr


@pytest.mark.parametrize(
    ("round_id", "beta", "teacher_percent"),
    [(2, 0.5, 50), (3, 0.25, 25), (4, 0.0, 0)],
)
def test_each_scheduled_future_round_reaches_common_fifty_window_gate(
    tmp_path, round_id, beta, teacher_percent
):
    config = _config(fidelity=False)
    identity = f"dagger-r{round_id}-teacher{teacher_percent}-b28-50iter"
    config["round_id"] = round_id
    config["teacher_beta"] = beta
    config["logging"]["wandb_run_id"] = identity
    config["logging"]["wandb_run_name"] = identity
    config["rtx_actor"]["runtime_root"] = (
        f"/mnt/data2/ldu/roborender-runtime/{identity}"
    )
    result = _invoke(
        tmp_path,
        config,
        ack="I_ACKNOWLEDGE_FIFTY_NATIVE_81_STEP_WINDOWS_TO_VALIDATION",
    )
    assert result.returncode == 2
    assert "complete Raven-fidelity contract" in result.stderr
    assert "requires round1" not in result.stderr


def test_round2_rejects_legacy_round1_ack_before_any_host_preflight(tmp_path):
    result = _invoke(
        tmp_path,
        _config(),
        ack="I_ACKNOWLEDGE_FORTY_NATIVE_81_STEP_WINDOWS",
    )
    assert result.returncode == 2
    assert "I_ACKNOWLEDGE_FIFTY_NATIVE_81_STEP_WINDOWS_TO_VALIDATION" in result.stderr


def test_round2_rejects_coordinator_identity_reused_for_wandb(tmp_path):
    config = _config()
    config["logging"]["wandb_run_id"] = COORDINATOR_RUN_ID
    config["logging"]["wandb_run_name"] = COORDINATOR_RUN_ID
    result = _invoke(
        tmp_path,
        config,
        ack="I_ACKNOWLEDGE_FIFTY_NATIVE_81_STEP_WINDOWS_TO_VALIDATION",
    )
    assert result.returncode == 2
    assert "distinct immutable W&B identity" in result.stderr


def test_round2_runtime_root_is_derived_from_distinct_wandb_identity(tmp_path):
    config = _config()
    config["rtx_actor"]["runtime_root"] = (
        f"/mnt/data2/ldu/roborender-runtime/{COORDINATOR_RUN_ID}"
    )
    result = _invoke(
        tmp_path,
        config,
        ack="I_ACKNOWLEDGE_FIFTY_NATIVE_81_STEP_WINDOWS_TO_VALIDATION",
    )
    assert result.returncode == 2
    assert (
        f"production runtime_root must equal /mnt/data2/ldu/roborender-runtime/"
        f"{WANDB_RUN_ID}" in result.stderr
    )


def test_trainer_uses_round_specific_fresh_segment_ack_and_wandb_identity():
    source = (
        PROJECT / "code/src/dagger_rtx/native_service_trainer.py"
    ).read_text(encoding="utf-8")
    assert 'if window_index == 0 and round_id == 1' in source
    assert '"I_ACKNOWLEDGE_FIFTY_NATIVE_81_STEP_WINDOWS_TO_VALIDATION"' in source
    assert "production_wandb_identity(" in source
    assert "wandb_run_id=wandb_run_id" in source
