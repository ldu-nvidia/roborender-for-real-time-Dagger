from __future__ import annotations
from test.deployment.source_paths import REPO_ROOT

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = REPO_ROOT
DAGGER = ROOT / "pipeline/dagger"
GR00T = ROOT / "pipeline/GR00T-VisualSim2Real"
RUNTIME_DEPLOYMENT = ROOT / "renderer/online_image3f/runtime/config/deployment.json"
requires_raven = pytest.mark.skipif(
    not (GR00T / ".git").exists(),
    reason="requires the ignored, pinned Raven checkout",
)
requires_full_static_contract = pytest.mark.skipif(
    not (GR00T / ".git").exists() or not RUNTIME_DEPLOYMENT.is_file(),
    reason="requires ignored Raven and renderer deployment materializations",
)


def _load_validator():
    path = DAGGER / "validate_raven_image3f_contract.py"
    spec = importlib.util.spec_from_file_location("raven_contract_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_full_static_contract
def test_raven_image3f_contract_is_complete_and_static():
    result = _load_validator().validate(full_assets=False)
    assert result["status"] == "raven_image3f_dagger_contract_valid"
    assert result["raven_commit"] == "94d7985"
    assert result["scientific_parameters_mutable"] == ["num_envs"]
    assert result["anchors"] == 2048


def test_canonical_round_schedule_is_fixed_and_legacy_override_is_disabled():
    launcher = (DAGGER / "run_raven_dagger_round_h200.sh").read_text()
    assert "NUM_ENVS=${3:-16}" in launcher
    for fixed in (
        "algo.config.num_steps_per_env=81",
        "algo.config.num_mini_batches=1",
        "algo.config.value_loss_coef=0.0",
        "algo.config.entropy_coef=0.0",
        "algo.config.rollout_with_teacher_num_steps=0",
        "algo.config.enforce_teacher_rollout=false",
        "algo.config.ratio_teacher_rollout=0.0",
        "algo.config.videogen.rollout_action_source=beta",
        "algo.config.videogen.beta_schedule=constant",
        'algo.config.videogen.beta_start="$BETA"',
        'algo.config.videogen.beta_end="$BETA"',
        "algo.config.videogen.raytraced_mix_prob=0.5",
        "algo.config.videogen.seed=42",
        "algo.config.videogen.anchor_index=0",
    ):
        assert fixed in launcher
    for forbidden in ("MIX=${", "STEPS=${", "CFG=${", "SEED=${"):
        assert forbidden not in launcher

    schedule = json.loads((DAGGER / "canonical_round_schedule.v1.json").read_text())
    assert schedule["selection"] == "per_decision_per_environment_bernoulli"
    assert schedule["teacher_labels_every_visited_state"] is True
    assert schedule["legacy_teacher_override"] is False
    assert [row["teacher_probability"] for row in schedule["rounds"]] == [
        0.75,
        0.5,
        0.25,
        0.0,
    ]
    assert [row["student_probability"] for row in schedule["rounds"]] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    for shell_row in (
        "1) BETA=0.75",
        "2) BETA=0.50",
        "3) BETA=0.25",
        "4) BETA=0.00",
    ):
        assert shell_row in launcher

    four_rounds = (DAGGER / "run_raven_four_rounds_h200.sh").read_text()
    assert "for round in 1 2 3 4" in four_rounds
    assert 'NUM_ENVS=${2:-16}' in four_rounds
    assert 'CHECKPOINT=${candidates[0]#* }' in four_rounds


@requires_raven
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


@requires_raven
def test_live_action_order_relabels_then_selects_then_steps_isaac():
    trainer = (
        GR00T / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
    ).read_text()
    relabel = 'obs_dict = self._image_dagger_relabel(obs_dict, cur_dones)'
    policy = 'policy_state_dict = super().policy_step(policy_model, obs_dict, cur_dones=cur_dones)'
    beta = 'beta = self._dagger_beta()'
    selection = 'policy_state_dict["actions"] = mixed'
    assert trainer.index(relabel, trainer.index("def policy_step")) < trainer.index(policy)
    assert trainer.index(policy) < trainer.index(beta) < trainer.index(selection)

    ppo = (GR00T / "gr00t/rl/trl/trainer/ppo_trainer_homie_api.py").read_text()
    package = 'actor_state = {"actions": policy_state_dict["actions"]}'
    step = "obs_dict, rewards, dones, infos = self.env.step(actor_state)"
    assert ppo.index(package) < ppo.index(step, ppo.index(package))


def test_online_server_preserves_raven_generation_contract():
    server = (DAGGER / "image3f_server.py").read_text()
    assert 'VIEW_MAP = [("ext1", "head_color"), ("ext2", "hand_left"), ("wrist", "hand_right")]' in server
    assert 'Image3FSession(memory_mode="auto", vae_mode="untiled"' in server
    assert 'seed=int(req.get("seed", 42))' in server
    assert "MAX_TCP_REQUEST_BYTES = 64 * 1024 * 1024" in server
    assert "if not 0 < n <= MAX_TCP_REQUEST_BYTES:" in server

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


def test_historical_raytraced_evaluation_cli_is_retired():
    launcher = (
        ROOT / "pipeline/evaluation/run_raven_raytraced_300_eval_h200.sh"
    ).read_text()
    assert 'Retired native-RGB student evaluation' in launcher
    assert 'pipeline.deployment.bc_success_dispatch' in launcher
    assert launcher.splitlines()[-1] == 'exit 2'
    assert 'eval_agent_trl.py' not in launcher
    assert not any(line.lstrip().startswith('source ') for line in launcher.splitlines())
