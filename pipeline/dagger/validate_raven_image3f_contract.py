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
    ).stdout
    require(not drift, "active Raven GR00T source tree has local drift")
    exact_sources = {
        "trainer_sha256": RAVEN_TRAINER,
        "client_sha256": RAVEN_CLIENT,
        "config_sha256": RAVEN_CONFIG,
    }
    for key, path in exact_sources.items():
        require(path.is_file(), f"missing Raven executable source: {path}")
        require(
            sha256(path) == reference[key],
            f"Raven source drift: {path}",
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

    return {
        "status": "raven_image3f_dagger_contract_valid",
        "raven_commit": reference["gr00t_commit"],
        "scientific_parameters_mutable": ["num_envs"],
        "allowed_runtime_overrides": contract["allowed_differences"],
        "anchors": len(anchors),
        "full_asset_hashes_verified": full_assets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-assets", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(full_assets=args.full_assets), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
