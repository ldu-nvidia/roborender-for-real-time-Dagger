"""Small structurally valid production policy bundles for control-plane tests."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Callable

from round_dagger import artifacts
from round_dagger.gr00t_policy import (
    BASE_MODULE_TARGET,
    POLICY_TARGET,
    build_portable_policy_spec,
)
from round_dagger.policy_identity import runtime_policy_identity_sha256


HOMIE_CONTROLLER_SHA256 = {
    "walk": "4" * 64,
    "stand": "5" * 64,
}


_FakePolicy = type("VisionActorObjPred", (), {})
_FakePolicy.__module__ = "gr00t.rl.trl.modules.vision_actor_critic_modules_obj_pred"


def complete_runtime_policy_sha256(policy_state_sha256: str) -> str:
    return runtime_policy_identity_sha256(
        policy_state_sha256, HOMIE_CONTROLLER_SHA256
    )


def portable_policy_spec(checkpoint_sha256: str, policy_state_sha256: str) -> dict:
    policy = _FakePolicy()
    policy.num_actions = 11
    policy.obj_pred_mlp = types.SimpleNamespace(output_dim=3)
    actor_config = {
        "_target_": POLICY_TARGET,
        "running_mean_std": False,
        "backbone": {
            "vision_module": {
                "_target_": BASE_MODULE_TARGET,
                "process_output_dim": True,
                "module_config_dict": {
                    "input_dim": ["vision_obs"],
                    "output_dim": ["vision_feature_dim"],
                    "layer_config": {
                        "type": "ResNet",
                        "resnet_type": "resnet18",
                        "pretrained": True,
                        "trainable": True,
                    },
                },
            },
            "mlp_module": {
                "_target_": BASE_MODULE_TARGET,
                "process_output_dim": True,
                "module_config_dict": {
                    "input_dim": ["actor_obs", "vision_feature_dim"],
                    "output_dim": [-1],
                    "layer_config": {
                        "type": "MLP",
                        "hidden_dims": [512, 256, 128],
                        "activation": "SiLU",
                    },
                },
            },
            "obj_pred_mlp": {
                "_target_": BASE_MODULE_TARGET,
                "process_output_dim": True,
                "module_config_dict": {
                    "input_dim": ["vision_feature_dim"],
                    "output_dim": [3],
                    "layer_config": {
                        "type": "MLP",
                        "hidden_dims": [256, 128, 64],
                        "activation": "SiLU",
                    },
                },
            },
        },
    }
    return build_portable_policy_spec(
        policy=policy,
        actor_config=actor_config,
        algo_config={
            "init_noise_std": 0.3,
            "freeze_noise_std": True,
            "clamp_noise_std": False,
        },
        module_dim_dict={"vision_feature_dim": 128},
        env_config=types.SimpleNamespace(
            robot=types.SimpleNamespace(
                algo_obs_dim_dict={
                    "actor_obs": 113,
                    "vision_obs": 324 * 192 * 3,
                },
                actions_dim=31,
            )
        ),
        policy_views=["head_color", "hand_left", "hand_right"],
        policy_view_layout="vertical",
        per_view_resolution=[108, 192],
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.229, 0.224, 0.225],
        homie_action_dim=15,
        replay_action_dim=26,
        checkpoint_sha256=checkpoint_sha256,
        policy_state_sha256=policy_state_sha256,
        homie_controller_sha256=HOMIE_CONTROLLER_SHA256,
    )


def publish_policy_bundle(
    checkpoint: Path,
    policy_state_sha256: str,
    *,
    mutate_spec: Callable[[dict], None] | None = None,
) -> tuple[Path, Path]:
    checkpoint_sha256 = artifacts.sha256_file(checkpoint)
    spec = portable_policy_spec(checkpoint_sha256, policy_state_sha256)
    if mutate_spec is not None:
        mutate_spec(spec)
    spec_path = Path(f"{checkpoint}.policy_spec.json")
    manifest_path = Path(f"{checkpoint}.manifest.json")
    artifacts.publish_immutable_json(spec_path, spec)
    artifacts.publish_immutable_json(
        manifest_path,
        {
            "schema": "fast_roborender.round_dagger.bootstrap_policy.v1",
            "checkpoint_path": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "runtime_policy_sha256": spec["runtime_policy_sha256"],
            "policy_spec_path": spec_path.name,
            "policy_spec_sha256": artifacts.sha256_file(spec_path),
        },
    )
    return spec_path, manifest_path
