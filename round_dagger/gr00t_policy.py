#!/usr/bin/env python3
"""Portable, camera-free construction contract for the GR00T DAgger student.

The RTX collector exports this small resolved specification beside the bootstrap
checkpoint.  The H100 learner reconstructs the exact policy architecture from
the specification, initializes the ResNet without pretrained weights (and
therefore without any network access), then strictly loads the checkpoint.

Heavy runtime dependencies are intentionally imported only inside construction
and checkpoint functions.  Specification validation and export remain usable
in the lightweight control-plane Python environment.
"""

from __future__ import annotations

import copy
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from round_dagger.artifacts import canonical_json_sha256, load_strict_json
from round_dagger.checkpoint_compat import (
    CheckpointCompatibilityError,
    load_hash_pinned_policy_state_dict,
)
from round_dagger.policy_identity import (
    PolicyIdentityError,
    runtime_policy_identity_sha256,
    validate_homie_controller_sha256,
)


POLICY_SPEC_SCHEMA = "fast_roborender.round_dagger.gr00t_policy_spec.v3"
POLICY_TARGET = (
    "gr00t.rl.trl.modules.vision_actor_critic_modules_obj_pred.VisionActorObjPred"
)
BASE_MODULE_TARGET = "gr00t.rl.agents.modules.modules.BaseModule"
ADAPTER_ID = "gr00t.vision_actor_obj_pred.forward.v1"
VISION_SHAPE = (324, 192, 3)
POLICY_ACTION_DIM = 11
HOMIE_ACTION_DIM = 15
REPLAY_ACTION_DIM = 26
ENV_COMMAND_DIM = 31
OBJECT_PREDICTION_DIM = 3
TRAIN_ACTION_SLICE = (0, POLICY_ACTION_DIM)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class GR00TPolicyError(RuntimeError):
    """A portable policy specification or checkpoint is incompatible."""


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GR00TPolicyError(f"{label} must be a lowercase SHA-256")
    return value


def _true_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GR00TPolicyError(f"{label} must be an integer >= {minimum}")
    return value


def _float_vector(value: Any, label: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise GR00TPolicyError(f"{label} must contain exactly three numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise GR00TPolicyError(f"{label}[{index}] must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise GR00TPolicyError(f"{label}[{index}] must be numeric") from exc
        if not math.isfinite(number):
            raise GR00TPolicyError(f"{label}[{index}] must be finite")
        result.append(number)
    return result


def _plain(value: Any) -> Any:
    """Convert DictConfig/SimpleNamespace structures into canonical JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    # OmegaConf's ``ListConfig`` implements the sequence protocol but is not a
    # builtin list/tuple.  Falling through to ``__dict__`` silently serialized
    # those values as ``{}``, producing an artifact that was file-valid but
    # impossible to reconstruct on the camera-free learner.  Treat every
    # non-string sequence as a JSON array so the portable spec really is the
    # resolved runtime architecture.
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    raise GR00TPolicyError(
        f"cannot serialize policy configuration value {type(value)!r}"
    )


def _member(value: Any, key: str, label: str) -> Any:
    if isinstance(value, Mapping):
        if key not in value:
            raise GR00TPolicyError(f"{label} lacks {key}")
        return value[key]
    if not hasattr(value, key):
        raise GR00TPolicyError(f"{label} lacks {key}")
    return getattr(value, key)


def _force_resnet_offline(
    actor_config: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    actor = copy.deepcopy(dict(actor_config))
    try:
        layer = actor["backbone"]["vision_module"]["module_config_dict"]["layer_config"]
    except (KeyError, TypeError) as exc:
        raise GR00TPolicyError(
            "actor configuration lacks the ResNet layer contract"
        ) from exc
    if not isinstance(layer, Mapping) or layer.get("type") != "ResNet":
        raise GR00TPolicyError("production GR00T policy must use a ResNet vision layer")
    source_pretrained = bool(layer.get("pretrained", False))
    layer["pretrained"] = False
    return actor, source_pretrained


def _runtime_class_path(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def build_portable_policy_spec(
    *,
    policy: Any,
    actor_config: Any,
    algo_config: Any,
    module_dim_dict: Any,
    env_config: Any,
    policy_views: Sequence[str],
    policy_view_layout: str,
    per_view_resolution: Sequence[int],
    rgb_mean: Sequence[float],
    rgb_std: Sequence[float],
    homie_action_dim: int,
    replay_action_dim: int,
    checkpoint_sha256: str,
    policy_state_sha256: str,
    homie_controller_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Capture the exact initialized student architecture without simulator state."""

    if _runtime_class_path(policy) != POLICY_TARGET:
        raise GR00TPolicyError(
            f"bootstrap policy class is {_runtime_class_path(policy)!r}, expected {POLICY_TARGET!r}"
        )
    actor_plain = _plain(actor_config)
    if (
        not isinstance(actor_plain, Mapping)
        or actor_plain.get("_target_") != POLICY_TARGET
    ):
        raise GR00TPolicyError(
            "resolved actor target does not match VisionActorObjPred"
        )
    actor_offline, source_pretrained = _force_resnet_offline(actor_plain)
    module_dims_raw = _plain(module_dim_dict)
    if not isinstance(module_dims_raw, Mapping):
        raise GR00TPolicyError("module_dim_dict must be a mapping")
    module_dims = {
        "vision_feature_dim": _true_int(
            module_dims_raw.get("vision_feature_dim"),
            "module_dim_dict.vision_feature_dim",
            minimum=1,
        )
    }
    robot = _member(env_config, "robot", "env_config")
    obs_dims_raw = _plain(_member(robot, "algo_obs_dim_dict", "env_config.robot"))
    if not isinstance(obs_dims_raw, Mapping):
        raise GR00TPolicyError("algo_obs_dim_dict must be a mapping")
    obs_dims = {
        key: _true_int(obs_dims_raw.get(key), f"algo_obs_dim_dict.{key}", minimum=1)
        for key in ("actor_obs", "vision_obs")
    }
    actor_obs_dim = obs_dims["actor_obs"]
    env_command_dim = _true_int(
        _member(robot, "actions_dim", "env_config.robot"), "actions_dim", minimum=1
    )
    if env_command_dim != ENV_COMMAND_DIM:
        raise GR00TPolicyError(
            f"simulator command dimension is {env_command_dim}, expected {ENV_COMMAND_DIM}"
        )
    if not isinstance(policy_views, Sequence) or isinstance(policy_views, (str, bytes)):
        raise GR00TPolicyError("policy_views must be a sequence")
    views = [str(item) for item in policy_views]
    if views != ["head_color", "hand_left", "hand_right"]:
        raise GR00TPolicyError(
            "policy view order must be head_color, hand_left, hand_right"
        )
    if policy_view_layout != "vertical":
        raise GR00TPolicyError("policy view layout must be vertical")
    if (
        not isinstance(per_view_resolution, Sequence)
        or isinstance(per_view_resolution, (str, bytes))
        or len(per_view_resolution) != 2
    ):
        raise GR00TPolicyError("per_view_resolution must be [height, width]")
    height = _true_int(per_view_resolution[0], "per_view_resolution[0]", minimum=1)
    width = _true_int(per_view_resolution[1], "per_view_resolution[1]", minimum=1)
    vision_shape = [height * len(views), width, 3]
    if tuple(vision_shape) != VISION_SHAPE:
        raise GR00TPolicyError(
            f"three-view policy image is {vision_shape}, expected {list(VISION_SHAPE)}"
        )
    if (
        obs_dims.get("vision_obs")
        != vision_shape[0] * vision_shape[1] * vision_shape[2]
    ):
        raise GR00TPolicyError(
            "vision_obs dimension does not match the 3-view image stack"
        )
    predicted_actions = _true_int(
        getattr(policy, "num_actions", None), "policy.num_actions"
    )
    if predicted_actions != POLICY_ACTION_DIM:
        raise GR00TPolicyError(
            f"student action dimension is {predicted_actions}, expected {POLICY_ACTION_DIM}"
        )
    homie_actions = _true_int(homie_action_dim, "homie_action_dim", minimum=1)
    replay_actions = _true_int(replay_action_dim, "replay_action_dim", minimum=1)
    if homie_actions != HOMIE_ACTION_DIM:
        raise GR00TPolicyError(
            f"Homie action dimension is {homie_actions}, expected {HOMIE_ACTION_DIM}"
        )
    if replay_actions != predicted_actions + homie_actions:
        raise GR00TPolicyError(
            "replay action dimension must equal policy plus Homie action dimensions"
        )
    if replay_actions != REPLAY_ACTION_DIM:
        raise GR00TPolicyError(
            f"replay action dimension is {replay_actions}, expected {REPLAY_ACTION_DIM}"
        )
    # The live Hydra configuration may still contain the source-only ``-1``
    # placeholder even though the instantiated student head is resolved to 11.
    # A portable learner spec must never depend on that trainer-side mutation.
    try:
        actor_offline["backbone"]["mlp_module"]["module_config_dict"]["output_dim"] = [
            predicted_actions
        ]
    except (KeyError, TypeError) as exc:
        raise GR00TPolicyError(
            "actor configuration lacks the policy action-head output contract"
        ) from exc
    obj_head = getattr(policy, "obj_pred_mlp", None)
    obj_dim = _true_int(
        getattr(obj_head, "output_dim", None), "obj_pred_mlp.output_dim"
    )
    if obj_dim != OBJECT_PREDICTION_DIM:
        raise GR00TPolicyError("object prediction head must output three coordinates")
    algo_plain = _plain(algo_config)
    if not isinstance(algo_plain, Mapping):
        raise GR00TPolicyError("algo_config must be a mapping")
    freeze_noise_std = algo_plain.get("freeze_noise_std", False)
    clamp_noise_std = algo_plain.get("clamp_noise_std", False)
    if not isinstance(freeze_noise_std, bool) or not isinstance(clamp_noise_std, bool):
        raise GR00TPolicyError("noise flags must be boolean")
    minimal_algo = {
        "init_noise_std": _finite_number(
            algo_plain.get("init_noise_std", 1.0), "algo_config.init_noise_std"
        ),
        "freeze_noise_std": freeze_noise_std,
        "clamp_noise_std": clamp_noise_std,
        "max_noise_std": _finite_number(
            algo_plain.get("max_noise_std", 1.0), "algo_config.max_noise_std"
        ),
    }
    try:
        controller_sha256 = validate_homie_controller_sha256(
            homie_controller_sha256
        )
        policy_state_sha256 = _sha256(
            policy_state_sha256, "policy_state_sha256"
        )
        runtime_policy_sha256 = runtime_policy_identity_sha256(
            policy_state_sha256, controller_sha256
        )
    except PolicyIdentityError as exc:
        raise GR00TPolicyError(str(exc)) from exc

    document = {
        "schema": POLICY_SPEC_SCHEMA,
        "policy_target": POLICY_TARGET,
        "adapter_id": ADAPTER_ID,
        "actor_config": actor_offline,
        "algo_config": minimal_algo,
        "module_dim_dict": module_dims,
        "env_config": {
            "robot": {
                "algo_obs_dim_dict": obs_dims,
                "actions_dim": env_command_dim,
            }
        },
        "input_contract": {
            "actor_obs_dim": actor_obs_dim,
            "vision_shape": vision_shape,
            "vision_layout": "NHWC",
            "view_layout": policy_view_layout,
            "views": views,
            "rgb_mean": _float_vector(rgb_mean, "rgb_mean"),
            "rgb_std": _float_vector(rgb_std, "rgb_std"),
        },
        "output_contract": {
            "policy_action_dim": predicted_actions,
            "homie_action_dim": homie_actions,
            "replay_action_dim": replay_actions,
            "env_command_dim": env_command_dim,
            "train_action_slice": [0, predicted_actions],
            "object_prediction_dim": obj_dim,
        },
        "initialization_contract": {
            "pretrained_for_constructor": False,
            "source_actor_requested_pretrained": source_pretrained,
            "strict_checkpoint_load": True,
            "frozen_parameter_groups": ["obj_pred_mlp", "std"],
        },
        "checkpoint_sha256": _sha256(checkpoint_sha256, "checkpoint_sha256"),
        "policy_state_sha256": policy_state_sha256,
        "homie_controller_sha256": controller_sha256,
        "runtime_policy_sha256": runtime_policy_sha256,
    }
    return validate_policy_spec(document)


def _exact_mapping(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        missing = (
            sorted(expected - set(value))
            if isinstance(value, Mapping)
            else sorted(expected)
        )
        extra = sorted(set(value) - expected) if isinstance(value, Mapping) else []
        raise GR00TPolicyError(f"{label} has missing={missing} extra={extra}")
    return dict(value)


def _mapping_member(value: Any, key: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GR00TPolicyError(f"{label} must be a mapping")
    member = value.get(key)
    if not isinstance(member, Mapping):
        raise GR00TPolicyError(f"{label}.{key} must be a mapping")
    return dict(member)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GR00TPolicyError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GR00TPolicyError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise GR00TPolicyError(f"{label} must be finite")
    return result


def validate_policy_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate one portable GR00T policy specification."""

    document = _exact_mapping(
        raw,
        {
            "schema",
            "policy_target",
            "adapter_id",
            "actor_config",
            "algo_config",
            "module_dim_dict",
            "env_config",
            "input_contract",
            "output_contract",
            "initialization_contract",
            "checkpoint_sha256",
            "policy_state_sha256",
            "homie_controller_sha256",
            "runtime_policy_sha256",
        },
        "policy spec",
    )
    if document["schema"] != POLICY_SPEC_SCHEMA:
        raise GR00TPolicyError("policy spec schema mismatch")
    if document["policy_target"] != POLICY_TARGET:
        raise GR00TPolicyError("policy target mismatch")
    if document["adapter_id"] != ADAPTER_ID:
        raise GR00TPolicyError("policy adapter mismatch")
    actor = document["actor_config"]
    if not isinstance(actor, Mapping) or actor.get("_target_") != POLICY_TARGET:
        raise GR00TPolicyError("actor_config target mismatch")
    backbone = actor.get("backbone")
    if not isinstance(backbone, Mapping):
        raise GR00TPolicyError("actor_config lacks backbone")
    for name in ("vision_module", "mlp_module", "obj_pred_mlp"):
        module = backbone.get(name)
        if (
            not isinstance(module, Mapping)
            or module.get("_target_") != BASE_MODULE_TARGET
        ):
            raise GR00TPolicyError(f"actor backbone {name} target mismatch")
    vision_module = _mapping_member(backbone, "vision_module", "actor_config.backbone")
    vision_module_config = _mapping_member(
        vision_module, "module_config_dict", "actor_config.backbone.vision_module"
    )
    layer = _mapping_member(
        vision_module_config,
        "layer_config",
        "actor_config.backbone.vision_module.module_config_dict",
    )
    if layer.get("type") != "ResNet" or layer.get("pretrained") is not False:
        raise GR00TPolicyError("ResNet constructor must have pretrained=false")
    if layer.get("resnet_type") not in {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    }:
        raise GR00TPolicyError("unsupported ResNet architecture")
    module_dims = _exact_mapping(
        document["module_dim_dict"], {"vision_feature_dim"}, "module_dim_dict"
    )
    _true_int(
        module_dims["vision_feature_dim"],
        "module_dim_dict.vision_feature_dim",
        minimum=1,
    )
    algo = _exact_mapping(
        document["algo_config"],
        {"init_noise_std", "freeze_noise_std", "clamp_noise_std", "max_noise_std"},
        "algo_config",
    )
    _finite_number(algo["init_noise_std"], "algo_config.init_noise_std")
    _finite_number(algo["max_noise_std"], "algo_config.max_noise_std")
    if not isinstance(algo["freeze_noise_std"], bool) or not isinstance(
        algo["clamp_noise_std"], bool
    ):
        raise GR00TPolicyError("noise flags must be boolean")
    if algo["freeze_noise_std"] is not True:
        raise GR00TPolicyError("action-only learner requires frozen action noise")
    env = _exact_mapping(document["env_config"], {"robot"}, "env_config")
    robot = _exact_mapping(
        env["robot"], {"algo_obs_dim_dict", "actions_dim"}, "env_config.robot"
    )
    if robot["actions_dim"] != ENV_COMMAND_DIM:
        raise GR00TPolicyError("simulator command dimension must be 31")
    dims = _exact_mapping(
        robot["algo_obs_dim_dict"],
        {"actor_obs", "vision_obs"},
        "env_config.robot.algo_obs_dim_dict",
    )
    actor_obs_dim = _true_int(
        dims["actor_obs"], "env_config.robot.algo_obs_dim_dict.actor_obs", minimum=1
    )
    vision_obs_dim = _true_int(
        dims["vision_obs"],
        "env_config.robot.algo_obs_dim_dict.vision_obs",
        minimum=1,
    )
    if vision_obs_dim != math.prod(VISION_SHAPE):
        raise GR00TPolicyError("vision observation dimension must match 324x192x3")
    inputs = _exact_mapping(
        document["input_contract"],
        {
            "actor_obs_dim",
            "vision_shape",
            "vision_layout",
            "view_layout",
            "views",
            "rgb_mean",
            "rgb_std",
        },
        "input_contract",
    )
    if inputs["actor_obs_dim"] != actor_obs_dim:
        raise GR00TPolicyError("actor_obs dimension disagrees with env config")
    if inputs["vision_shape"] != list(VISION_SHAPE):
        raise GR00TPolicyError("policy vision shape must be 324x192x3")
    if inputs["vision_layout"] != "NHWC" or inputs["view_layout"] != "vertical":
        raise GR00TPolicyError("policy vision layout mismatch")
    if inputs["views"] != ["head_color", "hand_left", "hand_right"]:
        raise GR00TPolicyError("policy view order mismatch")
    _float_vector(inputs["rgb_mean"], "rgb_mean")
    std = _float_vector(inputs["rgb_std"], "rgb_std")
    if any(item <= 0 for item in std):
        raise GR00TPolicyError("rgb_std must be positive")
    outputs = _exact_mapping(
        document["output_contract"],
        {
            "policy_action_dim",
            "homie_action_dim",
            "replay_action_dim",
            "env_command_dim",
            "train_action_slice",
            "object_prediction_dim",
        },
        "output_contract",
    )
    policy_action_dim = _true_int(
        outputs["policy_action_dim"], "output_contract.policy_action_dim", minimum=1
    )
    homie_action_dim = _true_int(
        outputs["homie_action_dim"], "output_contract.homie_action_dim", minimum=1
    )
    replay_action_dim = _true_int(
        outputs["replay_action_dim"], "output_contract.replay_action_dim", minimum=1
    )
    env_command_dim = _true_int(
        outputs["env_command_dim"], "output_contract.env_command_dim", minimum=1
    )
    if policy_action_dim != POLICY_ACTION_DIM:
        raise GR00TPolicyError("production policy action dimension must be 11")
    if homie_action_dim != HOMIE_ACTION_DIM:
        raise GR00TPolicyError("production Homie action dimension must be 15")
    if replay_action_dim != policy_action_dim + homie_action_dim:
        raise GR00TPolicyError(
            "replay action dimension must equal policy plus Homie action dimensions"
        )
    if replay_action_dim != REPLAY_ACTION_DIM:
        raise GR00TPolicyError("production replay action dimension must be 26")
    if env_command_dim != robot["actions_dim"] or env_command_dim != ENV_COMMAND_DIM:
        raise GR00TPolicyError("simulator command dimension must be 31")
    if outputs["train_action_slice"] != [0, policy_action_dim]:
        raise GR00TPolicyError(
            "train action slice must be the policy-width replay-action prefix"
        )
    if outputs["object_prediction_dim"] != OBJECT_PREDICTION_DIM:
        raise GR00TPolicyError("object prediction dimension must be 3")
    mlp_module = _mapping_member(backbone, "mlp_module", "actor_config.backbone")
    mlp_config = _mapping_member(
        mlp_module, "module_config_dict", "actor_config.backbone.mlp_module"
    )
    if mlp_config.get("output_dim") != [policy_action_dim]:
        raise GR00TPolicyError(
            "portable actor action head must resolve to policy_action_dim"
        )
    initialization = _exact_mapping(
        document["initialization_contract"],
        {
            "pretrained_for_constructor",
            "source_actor_requested_pretrained",
            "strict_checkpoint_load",
            "frozen_parameter_groups",
        },
        "initialization_contract",
    )
    if initialization["pretrained_for_constructor"] is not False:
        raise GR00TPolicyError("constructor pretrained flag must be false")
    if initialization["strict_checkpoint_load"] is not True:
        raise GR00TPolicyError("checkpoint loading must be strict")
    if initialization["frozen_parameter_groups"] != ["obj_pred_mlp", "std"]:
        raise GR00TPolicyError("action-only learner parameter-freeze contract mismatch")
    if not isinstance(initialization["source_actor_requested_pretrained"], bool):
        raise GR00TPolicyError("source pretrained provenance must be boolean")
    _sha256(document["checkpoint_sha256"], "checkpoint_sha256")
    policy_state_sha256 = _sha256(
        document["policy_state_sha256"], "policy_state_sha256"
    )
    try:
        controller_sha256 = validate_homie_controller_sha256(
            document["homie_controller_sha256"]
        )
        expected_runtime_sha256 = runtime_policy_identity_sha256(
            policy_state_sha256, controller_sha256
        )
    except PolicyIdentityError as exc:
        raise GR00TPolicyError(str(exc)) from exc
    if document["runtime_policy_sha256"] != expected_runtime_sha256:
        raise GR00TPolicyError(
            "runtime_policy_sha256 does not bind policy state and Homie controllers"
        )
    # Ensure the document itself contains only canonical JSON values.
    canonical_json_sha256(document)
    return copy.deepcopy(document)


def load_policy_spec(path: str | os.PathLike[str]) -> dict[str, Any]:
    return validate_policy_spec(load_strict_json(path))


def _extract_spec(bundle_or_spec: Mapping[str, Any]) -> dict[str, Any]:
    if bundle_or_spec.get("schema") == POLICY_SPEC_SCHEMA:
        return validate_policy_spec(bundle_or_spec)
    model_config = bundle_or_spec.get("model_config")
    if not isinstance(model_config, Mapping):
        raise GR00TPolicyError("learner bundle lacks inline model_config policy spec")
    spec = validate_policy_spec(model_config)
    if bundle_or_spec.get("rgb_mean") != spec["input_contract"]["rgb_mean"]:
        raise GR00TPolicyError("learner RGB mean differs from policy spec")
    if bundle_or_spec.get("rgb_std") != spec["input_contract"]["rgb_std"]:
        raise GR00TPolicyError("learner RGB std differs from policy spec")
    if bundle_or_spec.get("rgb_layout", "NHWC") != "NHWC":
        raise GR00TPolicyError("VisionActorObjPred requires NHWC learner images")
    if (
        bundle_or_spec.get("train_action_slice")
        != spec["output_contract"]["train_action_slice"]
    ):
        raise GR00TPolicyError("learner action slice differs from policy spec")
    if bundle_or_spec.get("initial_checkpoint_sha256") != spec["checkpoint_sha256"]:
        raise GR00TPolicyError("learner checkpoint digest differs from policy spec")
    return spec


def create_vision_actor_obj_pred(bundle: Mapping[str, Any]) -> Any:
    """Hydra-instantiate the exact GR00T student without simulator imports."""

    spec = _extract_spec(bundle)
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise GR00TPolicyError(
            "GR00T policy construction requires hydra-core and OmegaConf"
        ) from exc
    actor_config = OmegaConf.create(copy.deepcopy(spec["actor_config"]))
    env_config = OmegaConf.create(copy.deepcopy(spec["env_config"]))
    algo_config = OmegaConf.create(copy.deepcopy(spec["algo_config"]))
    try:
        policy = instantiate(
            actor_config,
            env_config=env_config,
            algo_config=algo_config,
            module_dim_dict=copy.deepcopy(spec["module_dim_dict"]),
            _recursive_=False,
        )
    except Exception as exc:
        raise GR00TPolicyError(f"cannot instantiate VisionActorObjPred: {exc}") from exc
    if _runtime_class_path(policy) != POLICY_TARGET:
        raise GR00TPolicyError("Hydra instantiated an unexpected policy class")
    if int(policy.num_actions) != int(spec["output_contract"]["policy_action_dim"]):
        raise GR00TPolicyError(
            "instantiated policy action dimension differs from portable policy spec"
        )
    if int(policy.obj_pred_mlp.output_dim) != OBJECT_PREDICTION_DIM:
        raise GR00TPolicyError("instantiated object prediction dimension is not 3")
    # The replay supervises only the spec-derived action prefix.  Exclude the auxiliary
    # object head and action-distribution sigma from DDP/AdamW entirely: leaving
    # them trainable would either trigger unfinished-reduction failures or
    # silently move them through weight decay despite having no targets.
    policy.obj_pred_mlp.requires_grad_(False)
    policy.std.requires_grad_(False)
    return policy


def load_checkpoint_strict(
    policy: Any,
    checkpoint_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load one exact GR00T policy state and reject missing/unexpected keys.

    Auxiliary RTX trainer objects embedded beside the policy are deserialized
    only as inert, discarded compatibility values.  The returned document is
    intentionally sanitized to contain no trainer state.
    """

    try:
        state = load_hash_pinned_policy_state_dict(
            checkpoint_path,
            expected_sha256=_sha256(
                expected_sha256, "expected checkpoint SHA-256"
            ),
            map_location=map_location,
        )
    except CheckpointCompatibilityError as exc:
        raise GR00TPolicyError(f"cannot load policy checkpoint: {exc}") from exc
    try:
        policy.load_state_dict(state, strict=True)
    except Exception as exc:
        raise GR00TPolicyError(f"strict policy checkpoint load failed: {exc}") from exc
    return {"policy_state_dict": state}


@dataclass(frozen=True)
class VisionActorObjPredAdapter:
    """Learner adapter for VisionActorObjPred.forward(obs_dict)."""

    actor_obs_dim: int
    vision_shape: tuple[int, int, int] = VISION_SHAPE
    policy_action_dim: int = POLICY_ACTION_DIM
    adapter_id: str = ADAPTER_ID

    def actions(self, model: Any, observations: Mapping[str, Any]) -> Any:
        if set(("actor_obs", "vision_obs")) - set(observations):
            raise GR00TPolicyError("policy batch lacks actor_obs or vision_obs")
        actor_obs = observations["actor_obs"]
        vision_obs = observations["vision_obs"]
        if (actor_obs.ndim, vision_obs.ndim) not in ((2, 4), (3, 5)):
            raise GR00TPolicyError(
                "policy batch must be [B,D]/[B,H,W,C] or explicit [B,S,...]"
            )
        if tuple(actor_obs.shape[:-1]) != tuple(vision_obs.shape[:-3]):
            raise GR00TPolicyError("actor and vision batch dimensions differ")
        if int(actor_obs.shape[-1]) != self.actor_obs_dim:
            raise GR00TPolicyError("actor_obs width differs from portable policy spec")
        if tuple(int(item) for item in vision_obs.shape[-3:]) != self.vision_shape:
            raise GR00TPolicyError("vision_obs shape differs from portable policy spec")
        # The upstream actor's nominal rollout path receives an explicit
        # singleton sequence dimension.  Replay batches intentionally store
        # only [B, ...], so bridge [B,D]/[B,H,W,C] to
        # [B,1,D]/[B,1,H,W,C] here and remove that dimension from the action
        # result before computing the supervised loss.  Keeping this quirk in
        # the adapter avoids changing either immutable replay data or GR00T.
        inserted_sequence_dim = actor_obs.ndim == 2 and vision_obs.ndim == 4
        model_observations = dict(observations)
        if inserted_sequence_dim:
            model_observations["actor_obs"] = actor_obs.unsqueeze(1)
            model_observations["vision_obs"] = vision_obs.unsqueeze(1)
        output = model(model_observations)
        if not isinstance(output, Mapping) or set(("actions", "obj_pred")) - set(
            output
        ):
            raise GR00TPolicyError(
                "VisionActorObjPred forward result lacks actions or obj_pred"
            )
        actions = output["actions"]
        if int(actions.shape[-1]) != self.policy_action_dim:
            raise GR00TPolicyError("VisionActorObjPred returned the wrong action width")
        obj_pred = output["obj_pred"]
        if (
            tuple(obj_pred.shape[:-1]) != tuple(actions.shape[:-1])
            or int(obj_pred.shape[-1]) != OBJECT_PREDICTION_DIM
        ):
            raise GR00TPolicyError(
                "VisionActorObjPred returned the wrong object-prediction shape"
            )
        if inserted_sequence_dim:
            if actions.ndim != 3 or int(actions.shape[1]) != 1:
                raise GR00TPolicyError(
                    "VisionActorObjPred did not preserve the singleton replay sequence"
                )
            actions = actions.squeeze(1)
        return actions


def vision_actor_obj_pred_adapter_factory(
    bundle: Mapping[str, Any],
) -> VisionActorObjPredAdapter:
    spec = _extract_spec(bundle)
    return VisionActorObjPredAdapter(
        actor_obs_dim=int(spec["input_contract"]["actor_obs_dim"]),
        policy_action_dim=int(spec["output_contract"]["policy_action_dim"]),
    )


__all__ = [
    "POLICY_SPEC_SCHEMA",
    "POLICY_TARGET",
    "BASE_MODULE_TARGET",
    "ADAPTER_ID",
    "VISION_SHAPE",
    "POLICY_ACTION_DIM",
    "HOMIE_ACTION_DIM",
    "REPLAY_ACTION_DIM",
    "ENV_COMMAND_DIM",
    "TRAIN_ACTION_SLICE",
    "GR00TPolicyError",
    "build_portable_policy_spec",
    "validate_policy_spec",
    "load_policy_spec",
    "create_vision_actor_obj_pred",
    "load_checkpoint_strict",
    "VisionActorObjPredAdapter",
    "vision_actor_obj_pred_adapter_factory",
]
