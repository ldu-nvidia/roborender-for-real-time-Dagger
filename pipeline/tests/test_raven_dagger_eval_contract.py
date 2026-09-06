from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DAGGER = ROOT / "pipeline/dagger"
GR00T = ROOT / "pipeline/GR00T-VisualSim2Real"


def _load_validator():
    path = DAGGER / "validate_raven_image3f_contract.py"
    spec = importlib.util.spec_from_file_location("raven_contract_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_raven_image3f_contract_is_complete_and_static():
    result = _load_validator().validate(full_assets=False)
    assert result["status"] == "raven_image3f_dagger_contract_valid"
    assert result["raven_commit"] == "94d7985"
    assert result["scientific_parameters_mutable"] == ["num_envs"]
    assert result["anchors"] == 2048


def test_only_environment_count_is_exposed_as_experiment_scale_input():
    launcher = (DAGGER / "run_raven_dagger_round_h200.sh").read_text()
    assert "NUM_ENVS=${3:-16}" in launcher
    for fixed in (
        "algo.config.num_steps_per_env=81",
        "algo.config.num_mini_batches=1",
        "algo.config.value_loss_coef=0.0",
        "algo.config.entropy_coef=0.0",
        "algo.config.videogen.rollout_action_source=teacher",
        "algo.config.videogen.raytraced_mix_prob=0.5",
        "algo.config.videogen.seed=42",
        "algo.config.videogen.anchor_index=0",
    ):
        assert fixed in launcher
    for forbidden in ("MIX=${", "BETA=${", "STEPS=${", "CFG=${", "SEED=${"):
        assert forbidden not in launcher

    four_rounds = (DAGGER / "run_raven_four_rounds_h200.sh").read_text()
    assert "for round in 1 2 3 4" in four_rounds
    assert 'NUM_ENVS=${2:-16}' in four_rounds
    assert 'CHECKPOINT=${candidates[0]#* }' in four_rounds


def test_raven_checkpoint_and_additional_round_semantics_are_exact():
    assert (
        __import__("subprocess").check_output(
            ["git", "-C", str(GR00T), "rev-parse", "HEAD"], text=True
        ).strip()
        == "94d79851d9c471b1210be81c75116d9bea835a4d"
    )
    ppo = (GR00T / "gr00t/rl/trl/trainer/ppo_trainer.py").read_text()
    for restored in (
        "model.policy.load_state_dict",
        "self.optimizer.load_state_dict",
        "self.lr_scheduler.load_state_dict",
        "self.env.load_env_state_dict",
        'checkpoint["state"].__dict__.items()',
    ):
        assert restored in ppo
    assert "for batch_idx in range(1, args.num_total_batches + 1):" in ppo
    launcher = (DAGGER / "run_raven_dagger_round_h200.sh").read_text()
    assert "raven_exact_image_dagger" not in launcher
    assert "resume=false" not in launcher


def test_online_server_preserves_raven_generation_contract():
    server = (DAGGER / "image3f_server.py").read_text()
    assert 'VIEW_MAP = [("ext1", "head_color"), ("ext2", "hand_left"), ("wrist", "hand_right")]' in server
    assert 'Image3FSession(memory_mode="auto", vae_mode="untiled"' in server
    assert 'seed=int(req.get("seed", 42))' in server

    contract = json.loads(
        (DAGGER / "raven_image3f_dagger_contract.v1.json").read_text()
    )
    assert contract["image3f"] == {
        "steps": 5,
        "cfg_scale": 1.0,
        "timestep_schedule": [
            1.0,
            0.7539682388305664,
            0.6122448446350098,
            0.44871795177459717,
            0.24193546175956726,
            0.0,
        ],
        "tea_cache_threshold": 0.3,
        "tea_cache_model_id": "Image3FJoint-5step",
        "vae_mode": "untiled",
        "frames_per_request": 1,
        "views_per_request": 3,
        "adapter_sha256": "c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588",
        "engine_architecture": "SM90",
        "engine_sha256": "9a11777c62e9d8ceb8bc380e4933353de2bd472e8b94716d29a4d1c39c3a70a5",
    }


def test_evaluation_is_raven_raytraced_300_episode_contract():
    launcher = (
        ROOT / "pipeline/evaluation/run_raven_raytraced_300_eval_h200.sh"
    ).read_text()
    assert 'cp -- "$BASELINE_CONFIG" "$STAGE/config.yaml"' in launcher
    assert 'cp -- "$CHECKPOINT" "$STAGE/last.pt"' in launcher
    assert "algo.config.eval.num_eval_episodes=300" in launcher
    assert "unset IMAGE_DAGGER_EVAL_RELABEL" in launcher
    assert "OBS=${" not in launcher
    assert "EPISODES=${" not in launcher
