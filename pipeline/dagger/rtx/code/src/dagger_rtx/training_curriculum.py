"""Raven's small training checkpoint state, scoped to one warm native actor.

This is deliberately not a simulator/RNG checkpoint. The population tag binds
either the legacy per-actor curriculum or the explicit paired B32 protocol.
Raven exports log_dict and loads only keys that already exist on the environment.
The allowlist below is that curriculum intersection for the pinned task. Lane
completion/history/physics buffers are not exported by Raven and stay cold.
"""
from __future__ import annotations

import hashlib
import math
import re
from contextlib import contextmanager

from .contracts import canonical_json, require_identifier, require_sha256

SCHEMA = "roborender.training-curriculum-export.v1"
REQUEST_SCHEMA = "roborender.training-curriculum-request.v1"
POPULATION = "independent_per_actor_native_population_v1"
SHARED_POPULATION = "shared_paired_native_population_v1"
ALLOWLIST = frozenset({"termination_level", "average_episode_length", "reward_penalty_scale",
    "soft_dof_pos_curriculum_value", "soft_dof_vel_curriculum_value", "soft_torque_curriculum_value",
    "current_noise_curriculum_value", "reset_from_teleop_curri_ratio"})
_BINDING = {"run_id", "round_id", "actor_id", "physical_gpu", "gpu_uuid", "native_pid",
    "worker_instance_id", "source_sha256", "command_id", "num_envs", "completed_updates",
    "boundary", "final_checkpoint_sha256", "motion_assignment_sha256", "config_sha256",
    "diagnostic_scope_sha256"}


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _get(value, path, default=None):
    for name in path.split("."):
        try:
            value = value[name] if isinstance(value, dict) else getattr(value, name)
        except (KeyError, AttributeError):
            return default
    return value


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("curriculum values and bounds must be finite numbers")
    return float(value)


def field_bounds(env):
    """Only fields the pinned getter emits and loader can actually restore."""
    cfg = env.config
    result = {"average_episode_length": (0.0, _number(float(env.max_episode_length)) + 1.0)}
    termination = _get(cfg, "termination_curriculum")
    if termination is not None:
        result["termination_level"] = (_get(termination, "min_level"), _get(termination, "max_level"))
    options = (
        ("use_reward_penalty_curriculum", "reward_penalty_scale", "rewards.reward_min_penalty_scale", "rewards.reward_max_penalty_scale"),
        ("use_reward_limits_dof_pos_curriculum", "soft_dof_pos_curriculum_value", "rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit", "rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit"),
        ("use_reward_limits_dof_vel_curriculum", "soft_dof_vel_curriculum_value", "rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit", "rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit"),
        ("use_reward_limits_torque_curriculum", "soft_torque_curriculum_value", "rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit", "rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit"),
        ("add_noise_currculum", "current_noise_curriculum_value", "obs.noise_value_min", "obs.noise_value_max"),
    )
    for flag, name, low, high in options:
        if bool(getattr(env, flag, False)):
            result[name] = (_get(cfg, low), _get(cfg, high))
    if bool(_get(cfg, "reset_from_dataset.reset_from_teleop_curriculum", False)):
        result["reset_from_teleop_curri_ratio"] = (0.0, 1.0)
    for name, limits in result.items():
        low, high = map(_number, limits)
        if low > high or name not in vars(env):
            raise ValueError("curriculum bounds or native attribute are incompatible: " + name)
        result[name] = [low, high]
    return result


def _encode(value):
    if hasattr(value, "detach"):
        if value.numel() != 1 or str(value.dtype) not in {"torch.float32", "torch.float64"}:
            raise ValueError("curriculum tensors must contain one finite floating value")
        return dict(kind="tensor", dtype=str(value.dtype).removeprefix("torch."),
                    shape=list(value.shape), value=_number(value.detach().cpu().item()))
    if type(value) is not float:
        raise ValueError("unsupported original curriculum state type")
    return dict(kind="float", dtype="float64", shape=[], value=_number(value))


def _validate_field(name, row, bounds):
    if not isinstance(row, dict) or set(row) != {"kind", "dtype", "shape", "value"}:
        raise ValueError("invalid curriculum scalar encoding")
    expected_shape = [1] if name == "reset_from_teleop_curri_ratio" and row["kind"] == "tensor" else []
    if (row["kind"] not in {"float", "tensor"} or row["dtype"] not in {"float32", "float64"}
            or row["shape"] != expected_shape
            or (row["kind"] == "float" and row["dtype"] != "float64")):
        raise ValueError("curriculum field type or shape changed: " + name)
    number = _number(row["value"])
    low, high = bounds
    # float32 config endpoints such as 0.1 are not exactly representable.
    tolerance = 2e-7 * max(1.0, abs(low), abs(high)) if row["dtype"] == "float32" else 0.0
    if not low - tolerance <= number <= high + tolerance:
        raise ValueError("curriculum value is outside configured bounds: " + name)


def validate_binding(value):
    if not isinstance(value, dict) or set(value) != _BINDING:
        raise ValueError("curriculum requires exact training/owner/checkpoint binding")
    for name in ("run_id", "actor_id", "command_id"):
        require_identifier(value[name], name)
    for name in ("source_sha256", "final_checkpoint_sha256", "motion_assignment_sha256", "config_sha256"):
        require_sha256(value[name], name)
    if (type(value["round_id"]) is not int or not 1 <= value["round_id"] <= 4
            or type(value["physical_gpu"]) is not int or value["physical_gpu"] not in (0, 1)
            or value["actor_id"] != f"rtx-g{value['physical_gpu']}"
            or type(value["num_envs"]) is not int or value["num_envs"] != 16
            or type(value["native_pid"]) is not int or value["native_pid"] <= 0
            or not isinstance(value["worker_instance_id"], str)
            or not re.fullmatch("[a-f0-9]{32}", value["worker_instance_id"])
            or not isinstance(value["gpu_uuid"], str) or not value["gpu_uuid"].startswith("GPU-")):
        raise ValueError("curriculum native actor identity is invalid")
    count = value["completed_updates"]
    if value["boundary"] == "production_update100":
        valid = type(count) is int and count == 100 and value["diagnostic_scope_sha256"] is None
    elif value["boundary"] == "diagnostic_update4":
        require_sha256(value["diagnostic_scope_sha256"], "diagnostic scope")
        valid = type(count) is int and count == 4
    else:
        valid = False
    if not valid:
        raise ValueError("curriculum export requires an exact completed training boundary")
    return value


def validate_export(value, *, expected_binding=None):
    if (not isinstance(value, dict) or set(value) != {"schema", "phase", "population", "binding", "bounds", "fields"}
            or value["schema"] != SCHEMA or value["phase"] != "training"
            or value["population"] not in (POPULATION, SHARED_POPULATION)):
        raise ValueError("training-only per-actor curriculum export required")
    validate_binding(value["binding"])
    if expected_binding is not None and value["binding"] != expected_binding:
        raise ValueError("curriculum predecessor identity changed")
    fields, bounds = value["fields"], value["bounds"]
    if (not isinstance(fields, dict) or not isinstance(bounds, dict) or set(fields) != set(bounds)
            or not {"average_episode_length"} <= set(fields) <= ALLOWLIST):
        raise ValueError("unsupported or missing original curriculum fields")
    for name, row in fields.items():
        limits = bounds[name]
        if not isinstance(limits, list) or len(limits) != 2 or _number(limits[0]) > _number(limits[1]):
            raise ValueError("invalid curriculum field bounds")
        _validate_field(name, row, limits)
    canonical_json(value)
    return value


def export_training_curriculum(env, binding, *, phase, population=POPULATION):
    if phase != "training" or bool(getattr(env, "is_evaluating", False)):
        raise ValueError("evaluation cannot export training curriculum")
    validate_binding(binding)
    if int(env.num_envs) != binding["num_envs"]:
        raise ValueError("curriculum native population changed")
    original = env.get_env_state_dict()
    bounds = field_bounds(env)
    restorable = {name for name in original if name in vars(env)}
    if restorable != set(bounds):
        raise ValueError("original getter/loader curriculum intersection is unsupported or incomplete")
    value = dict(schema=SCHEMA, phase="training", population=population, binding=dict(binding),
                 bounds=bounds, fields={name: _encode(original[name]) for name in sorted(restorable)})
    return validate_export(value)


def restore_training_curriculum(env, value, *, expected_binding, phase, population=POPULATION):
    """Call after cold-phase preparation, before reset_all; never calls physics."""
    if phase != "training" or bool(getattr(env, "is_evaluating", False)):
        raise ValueError("evaluation cannot restore training curriculum")
    validate_export(value, expected_binding=expected_binding)
    if value['population'] != population:
        raise ValueError('cannot switch independent/shared curriculum on a resumed round')
    if int(env.num_envs) != expected_binding["num_envs"] or field_bounds(env) != value["bounds"]:
        raise ValueError("curriculum population/configuration changed")
    import torch
    decoded = {}
    for name, row in value["fields"].items():
        target = getattr(env, name)
        if hasattr(target, "detach") and (list(target.shape) != row["shape"] or
                str(target.dtype).removeprefix("torch.") != row["dtype"]):
            raise ValueError("curriculum target tensor shape/dtype changed: " + name)
        decoded[name] = (torch.tensor(row["value"], dtype=getattr(torch, row["dtype"])).reshape(row["shape"])
                         if row["kind"] == "tensor" else row["value"])
    # Use the original loader: in particular a float becomes a native tensor,
    # and tensor values preserve their saved dtype. No log or lane buffers enter.
    env.load_env_state_dict(decoded)
    return dict(mode="restore_predecessor", export_sha256=digest(value),
                fields=sorted(decoded), population=population,
                restored_before_initial_reset=True, physical_or_rng_state_restored=False,
                before_initial_reset=current_values(env))


def current_values(env):
    return {name: _encode(getattr(env, name)) for name in sorted(field_bounds(env))}


def bootstrap_training_curriculum(env, *, phase):
    if phase != 'training' or bool(getattr(env, 'is_evaluating', False)):
        raise ValueError('only training can bootstrap curriculum')
    paths = {
        'termination_level': 'termination_curriculum.initial_level',
        'reward_penalty_scale': 'rewards.reward_initial_penalty_scale',
        'soft_dof_pos_curriculum_value': 'rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit',
        'soft_dof_vel_curriculum_value': 'rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit',
        'soft_torque_curriculum_value': 'rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit',
        'current_noise_curriculum_value': 'obs.noise_initial_value',
    }
    values = current_values(env)
    for name, row in values.items():
        expected = (_get(env.config, paths[name], 1.0 if name == 'termination_level' else None)
                    if name in paths else 0.0)
        expected = _number(expected)
        tolerance = 2e-7 * max(1.0, abs(expected)) if row['dtype'] == 'float32' else 0.0
        if abs(row['value'] - expected) > tolerance:
            raise ValueError('bootstrap state differs from cold configuration: ' + name)
    return dict(mode='bootstrap_from_config', population=POPULATION, fields=sorted(values),
                restored_before_initial_reset=True, physical_or_rng_state_restored=False,
                before_initial_reset=values)


@contextmanager
def observe_initial_reset(env, receipt):
    """Observe the existing reset call; never add/reset physical work ourselves."""
    if receipt is None:
        yield
        return
    previous = env.reset_all
    owned = "reset_all" in vars(env)
    calls = 0

    def observed(*args, **kwargs):
        nonlocal calls
        result = previous(*args, **kwargs)
        calls += 1
        if calls != 1:
            raise RuntimeError("curriculum phase must have exactly one initial reset_all")
        receipt["after_initial_reset"] = current_values(env)
        receipt["initial_reset_calls"] = calls
        return result

    env.reset_all = observed
    try:
        yield
        if calls != 1:
            raise RuntimeError("curriculum phase never observed its initial reset")
    finally:
        if owned:
            env.reset_all = previous
        else:
            del env.reset_all


def bootstrap_request():
    return dict(schema=REQUEST_SCHEMA, mode="bootstrap_from_config", predecessor=None)


def validate_request(value):
    if not isinstance(value, dict) or set(value) != {"schema", "mode", "predecessor"} or value["schema"] != REQUEST_SCHEMA:
        raise ValueError("explicit versioned training curriculum request required")
    if value["mode"] == "bootstrap_from_config":
        if value["predecessor"] is not None:
            raise ValueError("configuration bootstrap cannot inherit state")
    elif value["mode"] == "restore_predecessor":
        prior = value["predecessor"]
        if not isinstance(prior, dict) or set(prior) != {"binding", "export_sha256", "native_terminal_receipt_sha256"}:
            raise ValueError("exact predecessor terminal/export binding required")
        validate_binding(prior["binding"])
        require_sha256(prior["export_sha256"], "curriculum export")
        require_sha256(prior["native_terminal_receipt_sha256"], "native terminal receipt")
    else:
        raise ValueError("unknown training curriculum mode")
    return value


def build_restore_request(wrapper):
    value = validate_export(wrapper["training_curriculum"])
    if wrapper.get("training_curriculum_sha256") != digest(value):
        raise ValueError("terminal wrapper curriculum digest changed")
    result = dict(schema=REQUEST_SCHEMA, mode="restore_predecessor", predecessor=dict(
        binding=value["binding"], export_sha256=digest(value),
        native_terminal_receipt_sha256=wrapper["native_terminal_receipt_sha256"]))
    return validate_request(result)


def resolve_predecessor(request, owner_state, *, parent_checkpoint_sha256, physical_gpu, source_sha256):
    """Resolve only the exact completed training command on this warm owner.

    The external wrapper binds the terminal receipt, which binds the export;
    the export does not contain the terminal hash (there is no digest cycle).
    """
    validate_request(request)
    if request["mode"] == "bootstrap_from_config":
        return None
    prior = request["predecessor"]
    binding = prior["binding"]
    if (binding["physical_gpu"] != physical_gpu or binding["source_sha256"] != source_sha256
            or binding["final_checkpoint_sha256"] != parent_checkpoint_sha256
            or any(owner_state.get(name) != binding[name] for name in ("native_pid", "worker_instance_id", "gpu_uuid", "source_sha256"))):
        raise ValueError("curriculum predecessor is not this actor/source/parent checkpoint")
    row = owner_state.get("commands", {}).get(binding["command_id"], {})
    receipt = row.get("receipt", {})
    if (row.get("status") != "COMPLETE" or receipt.get("complete") is not True
            or row.get("command", {}).get("operation") not in {"collect_production_segment", "probe_continuous_training"}
            or digest(receipt) != prior["native_terminal_receipt_sha256"]
            or receipt.get("training_curriculum_sha256") != prior["export_sha256"]):
        raise ValueError("curriculum predecessor lacks its exact completed native training receipt")
    value = validate_export(receipt.get("training_curriculum"), expected_binding=binding)
    if digest(value) != prior["export_sha256"]:
        raise ValueError("curriculum predecessor export changed")
    return value
