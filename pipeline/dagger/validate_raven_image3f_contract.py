#!/usr/bin/env python3
"""Fail-closed static audit of the Raven-equivalent Image3F DAgger stage."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipeline"
GR00T = PIPELINE / "GR00T-VisualSim2Real"
CONTRACT = Path(__file__).with_name("raven_image3f_dagger_contract.v1.json")
RAVEN_CONFIG = (
    GR00T
    / "gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger_3view.yaml"
)
RAVEN_TRAINER = (
    GR00T / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
)
RAVEN_CLIENT = GR00T / "gr00t/rl/utils/image3f_frame_client.py"
SERVER = Path(__file__).with_name("image3f_server.py")
DEPLOYMENT = ROOT / "renderer/online_image3f/runtime/config/deployment.json"
SOURCE_DEPLOYMENT = ROOT / "renderer/online_image3f/source/config/deployment.json"
ANCHORS = (
    ROOT
    / "renderer/anchors/roborender_online_anchor_pool_teacher50_2048"
)
ADAPTER = (
    ROOT
    / "renderer/online_image3f/checkpoints/"
    "roborender_online_dagger_image3f_5step_lora_step05000.safetensors"
)
ENGINE = (
    ROOT
    / "renderer/online_image3f/engines/sm90/"
    "roborender_online_dagger_image3f_sm90_bf16_3view.engine"
)
SCHEDULE = Path(__file__).with_name("canonical_round_schedule.v1.json")
LAUNCHER = Path(__file__).with_name("run_raven_dagger_round_h200.sh")
ONLINE_TRAINER_METHODS = (
    "policy_step",
    "_dagger_beta",
    "_advance_dagger_beta",
    "_id_sample_anchor",
    "_image_dagger_relabel",
    "_rollout_step",
    "_depth_to_control_video",
    "_h5v4_control",
    "_resize_seq",
    "_normalize_like_env",
)


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _git_blob(relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(GR00T), "show", f"HEAD:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout


def _function_ast(source: str, names: tuple[str, ...]) -> dict[str, str]:
    tree = ast.parse(source)
    found = {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    require(set(found) == set(names), "online trainer method inventory differs")
    return found


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ContractError(f"{path}: missing literal assignment {name}")


def validate(*, full_assets: bool = False) -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    reference = contract["reference"]
    head = subprocess.run(
        ["git", "-C", str(GR00T), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(
        head.startswith(reference["gr00t_commit"]),
        "active GR00T checkout is not Raven 94d7985",
    )
    drift = subprocess.run(
        ["git", "-C", str(GR00T), "status", "--porcelain", "--", "gr00t"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    allowed_bc_drift = {
        " M gr00t/rl/config/exp/loco_manip/wsdpt_videogen_offline_bc_3view.yaml",
        " M gr00t/rl/trl/trainer/distill_trainer_videogen.py",
    }
    require(
        set(drift).issubset(allowed_bc_drift),
        "active Raven GR00T tree has drift outside the pinned offline-BC extension",
    )
    exact_sources = {
        "client_sha256": RAVEN_CLIENT,
        "config_sha256": RAVEN_CONFIG,
    }
    for key, path in exact_sources.items():
        require(path.is_file(), f"missing Raven executable source: {path}")
        require(
            sha256(path) == reference[key],
            f"Raven source drift: {path}",
        )

    reference_trainer = _git_blob(
        "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
    )
    require(
        hashlib.sha256(reference_trainer).hexdigest()
        == reference["trainer_sha256"],
        "Raven trainer Git blob differs from the pinned reference",
    )
    trainer_digest = sha256(RAVEN_TRAINER)
    if trainer_digest != reference["trainer_sha256"]:
        require(
            _function_ast(RAVEN_TRAINER.read_text(encoding="utf-8"), ONLINE_TRAINER_METHODS)
            == _function_ast(reference_trainer.decode("utf-8"), ONLINE_TRAINER_METHODS),
            "online Image3F trainer methods differ from Raven 94d7985",
        )

    config = yaml.safe_load(RAVEN_CONFIG.read_text(encoding="utf-8"))
    vg = config["algo"]["config"]["videogen"]
    cameras = config["simulator"]["config"]["cameras"]
    rollout = contract["rollout"]
    conditioning = contract["conditioning"]
    require(config["policy_view_count"] == 3, "policy_view_count must be 3")
    require(
        cameras["policy_views"] == contract["policy"]["views"],
        "policy view order drift",
    )
    require(
        cameras["camera_resolutions"]
        == contract["policy"]["per_view_resolution"],
        "policy resolution drift",
    )
    require(
        cameras["depth_hi_resolution"] == conditioning["capture_resolution"],
        "conditioning capture resolution drift",
    )
    require(vg["mode"] == "image_dagger", "DAgger mode must be image_dagger")
    require(
        config["algo"]["config"]["num_steps_per_env"]
        == rollout["num_steps_per_env"],
        "81-step update cadence drift",
    )
    require(
        config["algo"]["config"]["num_mini_batches"]
        == rollout["num_mini_batches"],
        "minibatch count drift",
    )
    for key in ("value_loss_coef", "entropy_coef"):
        require(
            config["algo"]["config"][key] == rollout[key],
            f"{key} drift",
        )
    expected_vg = {
        "rollout_action_source": rollout["action_source"],
        "raytraced_mix_prob": rollout["raytraced_mix_prob"],
        "seed": rollout["seed"],
        "anchor_index": conditioning["anchor"]["frame_index_within_anchor_file"],
        "control_resolution": conditioning["control_resolution"],
        "depth_mode": conditioning["depth_mode"],
        "view_names": conditioning["view_order"],
        "primary_view": "head_color",
    }
    for key, expected in expected_vg.items():
        require(vg[key] == expected, f"videogen.{key} drift")
    require(
        vg["depth_h5v4_per_view"] == contract["depth_h5_v4"],
        "h5_v4 depth recipe drift",
    )

    ppo = (GR00T / "gr00t/rl/trl/trainer/ppo_trainer.py").read_text(encoding="utf-8")
    for fragment in (
        "for batch_idx in range(1, args.num_total_batches + 1):",
        'self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])',
        'self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])',
        'self.env.load_env_state_dict(checkpoint["env_state_dict"])',
        'for key, value in checkpoint["state"].__dict__.items()',
    ):
        require(fragment in ppo, f"Raven checkpoint/update semantics missing: {fragment}")
    homie_ppo = (
        GR00T / "gr00t/rl/trl/trainer/ppo_trainer_homie_api.py"
    ).read_text(encoding="utf-8")
    behavior_package = 'actor_state = {"actions": policy_state_dict["actions"]}'
    isaac_step = "obs_dict, rewards, dones, infos = self.env.step(actor_state)"
    require(behavior_package in homie_ppo, "selected behavior action is not packaged")
    require(isaac_step in homie_ppo, "Isaac environment step is missing")
    require(
        homie_ppo.index(behavior_package) < homie_ppo.index(isaac_step),
        "Isaac is stepped before the selected behavior action is packaged",
    )

    view_map = _literal_assignment(SERVER, "VIEW_MAP")
    require(
        dict(view_map) == conditioning["image3f_slots"],
        "online Image3F slot mapping drift",
    )
    server_text = SERVER.read_text(encoding="utf-8")
    require(
        'Image3FSession(memory_mode="auto", vae_mode="untiled"' in server_text,
        "online server must use Raven's untiled VAE",
    )

    deployment = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    image3f = contract["image3f"]
    generation = deployment["generation_contract"]
    require(
        generation["source_view_mapping"] == conditioning["image3f_slots"],
        "deployment Image3F slot mapping drift",
    )
    configured_adapter = (DEPLOYMENT.parent.parent / deployment["adapter"]["path"]).resolve()
    require(
        configured_adapter == ADAPTER.resolve(),
        "deployment adapter path is not the canonical online checkpoint",
    )
    require(generation["steps"] == image3f["steps"], "Image3F step count drift")
    require(generation["cfg_scale"] == image3f["cfg_scale"], "Image3F CFG drift")
    require(
        generation["timestep_schedule"] == image3f["timestep_schedule"],
        "Image3F five-step timestep schedule drift",
    )
    source_generation = json.loads(
        SOURCE_DEPLOYMENT.read_text(encoding="utf-8")
    )["generation_contract"]
    require(
        generation["timestep_schedule"] == source_generation["timestep_schedule"],
        "runtime timestep schedule differs from the supplied Image3F source contract",
    )
    require(
        deployment["acceleration"]["tea_cache"]["threshold"]
        == image3f["tea_cache_threshold"],
        "TeaCache threshold drift",
    )
    require(
        deployment["acceleration"]["tea_cache"]["model_id"]
        == image3f["tea_cache_model_id"],
        "TeaCache model ID drift",
    )

    anchors = sorted(ANCHORS.glob("anchor_*.npz"))
    require(len(anchors) == 2048, f"expected 2048 anchors, found {len(anchors)}")
    for path in (ADAPTER, ENGINE):
        require(path.is_file(), f"missing Image3F asset: {path}")
    if full_assets:
        require(sha256(ADAPTER) == image3f["adapter_sha256"], "adapter SHA drift")
        require(sha256(ENGINE) == image3f["engine_sha256"], "SM90 engine SHA drift")

    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    require(
        schedule.get("schema") == "roborender.canonical_dagger_round_schedule.v1",
        "canonical DAgger schedule schema differs",
    )
    require(
        [row["teacher_probability"] for row in schedule["rounds"]]
        == [0.75, 0.5, 0.25, 0.0],
        "canonical DAgger teacher schedule differs",
    )
    require(
        [row["student_probability"] for row in schedule["rounds"]]
        == [0.25, 0.5, 0.75, 1.0],
        "canonical DAgger student schedule differs",
    )
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "algo.config.rollout_with_teacher_num_steps=0",
        "algo.config.enforce_teacher_rollout=false",
        "algo.config.ratio_teacher_rollout=0.0",
        "algo.config.videogen.rollout_action_source=beta",
        "algo.config.videogen.beta_schedule=constant",
        'algo.config.videogen.beta_start="$BETA"',
        'algo.config.videogen.beta_end="$BETA"',
    ):
        require(fragment in launcher, f"active action-selection override missing: {fragment}")

    return {
        "status": "raven_image3f_dagger_contract_valid",
        "raven_commit": reference["gr00t_commit"],
        "scientific_parameters_mutable": ["num_envs"],
        "allowed_runtime_overrides": contract["allowed_differences"],
        "anchors": len(anchors),
        "full_asset_hashes_verified": full_assets,
        "online_trainer_matches_raven_ast": True,
        "canonical_teacher_betas": [0.75, 0.5, 0.25, 0.0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-assets", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(full_assets=args.full_assets), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
