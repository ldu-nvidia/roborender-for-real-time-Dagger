"""Explicit original-PPO teacher control, separate from Raven distillation.

Importing the mode/receipt helpers is stdlib-only. Tensor operations are imported
only inside native/CPU numerical helpers; success criteria and reset logic do
not live here. The default remains the original distillation call convention.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

DISTILL = "distill"
RAVEN_PPO = "raven_ppo"
PPO_SOURCE_SHA256 = "0c09d2be2da9918942e4f8aa59390d93381c05b2e2d54c352942c07a38c85637"
TEACHER_SHA256 = "943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38"
WALK_SHA256 = "2efa9d39c6ace247d6d68b43452f591cd5b22fb2f5e58dcfd3b6c5d1b30df77e"
STAND_SHA256 = "ceb976d2745bdaa99e51ba7141e441dc7a140638ff8e2b0efaf50efd20290054"


def validate_mode(mode=DISTILL, arm="teacher"):
    if mode not in (DISTILL, RAVEN_PPO) or (mode == RAVEN_PPO and arm != "teacher"):
        raise ValueError("raven_ppo diagnostic mode requires the privileged teacher arm")
    return mode


def validate_trace_steps(value):
    if type(value) is not int or not 0 <= value <= 24:
        raise ValueError("numeric diagnostic trace must be bounded to 0..24 vector steps")
    return value


def control_contract(mode=DISTILL):
    validate_mode(mode)
    if mode == DISTILL:
        return {"schema": "roborender.teacher-action-control.v1", "mode": DISTILL,
                "convention": "original_raven_distillation_teacher_gt"}
    return {"schema": "roborender.teacher-action-control.v2", "mode": RAVEN_PPO,
        "convention": "original_raven_ppo_deterministic_teacher_plus_homie",
        "original_ppo_source_sha256": PPO_SOURCE_SHA256,
        "teacher_checkpoint_sha256": TEACHER_SHA256,
        "homie_walk_sha256": WALK_SHA256, "homie_stand_sha256": STAND_SHA256,
        "teacher_action": "rollout_then_action_mean_detach",
        "rollout_sample_rng_consumed": True,
        "homie_action": "action_mean", "homie_switch_threshold": 0.5,
        "homie_one_step_observations": 86, "teacher_observations": 226,
        "teacher_delta_actions_slice": [44, 55], "teacher_velocity_feedback_slice": [44, 47],
        "env_step_keys": ["actions"], "gt_actions_key_present": False,
        "pre_step_teacher_velocity_correction": False, "student_forward_calls": 0,
        "original_clamp_noise_std": True, "original_max_noise_std": 1.0,
        "std_state_contract": "checkpoint_raw_or_exact_original_max_clamp"}


def validate_control_contract(value, mode=DISTILL):
    if (not isinstance(value, dict)
            or json.dumps(value, sort_keys=True, allow_nan=False) !=
               json.dumps(control_contract(mode), sort_keys=True, allow_nan=False)):
        raise ValueError("teacher action control contract differs")
    return deepcopy(value)


def action_mode(arm, mode, student_mode):
    validate_mode(mode, arm)
    if arm != "teacher":
        return student_mode
    return ("raven_ppo_privileged_teacher_plus_homie_raw" if mode == RAVEN_PPO
            else "privileged_teacher_plus_homie")


def prepare_ppo_teacher(trainer, *, original_ppo=False):
    """Once per actual model, attest exact teacher parameters including RMS.

    This never loads environment state or initializes a device. The original
    teacher load is strict; this extra diagnostic gate checks that the live
    frozen model actually still equals the pinned checkpoint, rather than
    labeling a receipt from a pathname alone.
    """
    import torch
    if type(original_ppo) is not bool:
        raise ValueError("original PPO attestation selector must be boolean")
    teacher = trainer.ref_model
    std_policy = None
    if original_ppo:
        # A teacher instantiated by the distillation trainer inherits the
        # STUDENT .001 noise cap. Original PPO uses max_noise_std=1.0. This
        # affects only its discarded sampled action, not the deterministic
        # mean; still reproduce the original distribution/RNG path exactly.
        # Do not mutate trainer.config, teacher.algo_config or any parameter.
        std_policy = getattr(teacher, "_original_ppo_std_policy", None)
        if std_policy is None:
            inherited_max = getattr(teacher, "max_noise_std", None)
            if (getattr(teacher, "clamp_noise_std", None) is not True
                    or type(inherited_max) not in (int, float)
                    or not math.isfinite(inherited_max) or inherited_max <= 0
                    or teacher.algo_config.get("use_clampped_std", False)):
                raise RuntimeError("unsupported inherited original PPO noise control")
            std_policy = {"inherited_clamp_noise_std": True,
                "inherited_max_noise_std": float(inherited_max),
                "original_clamp_noise_std": True, "original_max_noise_std": 1.0,
                "binding_scope": "teacher_instance_only_before_first_ppo_forward",
                "trainer_and_student_config_modified": False}
            teacher.max_noise_std = 1.0
            teacher._original_ppo_std_policy = deepcopy(std_policy)
        if (teacher.clamp_noise_std is not True or teacher.max_noise_std != 1.0
                or teacher.algo_config.get("use_clampped_std", False)):
            raise RuntimeError("original PPO noise control changed after binding")
    actual = teacher.state_dict()
    homie_specs = (("walk", trainer.unwrapped_model.homie_walk_model,
                    trainer.config.homie_walk_model_path, WALK_SHA256),
                   ("stand", trainer.unwrapped_model.homie_stand_model,
                    trainer.config.homie_stand_model_path, STAND_SHA256))
    versions = [("teacher", key, value.data_ptr(), value._version) for key, value in actual.items()]
    for role, model, _path, _sha in homie_specs:
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("original PPO Homie must remain frozen/eval: " + role)
        versions.extend((role, key, value.data_ptr(), value._version)
                        for key, value in model.state_dict().items())
    versions = (original_ppo, tuple(versions), json.dumps(std_policy, sort_keys=True))
    cached = getattr(trainer, "_ppo_teacher_parameter_attestation", None)
    if (getattr(teacher, "is_recurrent", None) is not False
            or teacher.max_rollout_history != 1 or teacher.training
            or teacher.input_key not in {"teacher_obs", "actor_obs"}):
        raise RuntimeError("original PPO teacher architecture/mode differs")
    rms = teacher.running_mean_std
    if (rms is None or rms.training or rms.norm_only or not rms.per_channel
            or rms.epsilon != 1e-5 or tuple(rms.running_mean.shape) != (226,)
            or tuple(rms.running_var.shape) != (226,)):
        raise RuntimeError("original PPO teacher RMS contract differs")
    if cached is not None and cached[0] is teacher and cached[1] == versions:
        return deepcopy(cached[2])
    path = Path(trainer.config.network_load_dict.teacher_actor.path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != TEACHER_SHA256:
        raise RuntimeError("original PPO teacher checkpoint identity differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("actor_model_state_dict", checkpoint.get("policy_state_dict"))
    if not isinstance(state, dict) or set(state) != set(actual):
        raise RuntimeError("live original PPO teacher/RMS differs from pinned checkpoint")
    mismatches = [key for key in state if not torch.equal(actual[key].detach().cpu(), state[key])]
    runtime_std = actual.get("std")
    expected_std = state.get("std")
    if original_ppo:
        if (runtime_std is None or expected_std is None or tuple(runtime_std.shape) != (11,)
                or not torch.isfinite(runtime_std).all() or not torch.isfinite(expected_std).all()
                or (expected_std <= 0).any()):
            raise RuntimeError("original PPO checkpoint/runtime std invalid")
        runtime_std = runtime_std.detach().cpu()
        legally_clamped = expected_std.clamp(max=1.0)
        if not (torch.equal(runtime_std, expected_std) or torch.equal(runtime_std, legally_clamped)):
            raise RuntimeError("live original PPO std differs from exact raw/original clamp")
        mismatches = [key for key in mismatches if key != "std"]
    if mismatches:
        raise RuntimeError("live original PPO teacher/RMS differs from pinned checkpoint: "
                           + ",".join(mismatches))
    homie_proofs = {}
    for role, model, checkpoint_path, expected_sha in homie_specs:
        homie_path = Path(checkpoint_path)
        if hashlib.sha256(homie_path.read_bytes()).hexdigest() != expected_sha:
            raise RuntimeError("original PPO Homie checkpoint identity differs: " + role)
        homie_state = torch.load(homie_path, map_location="cpu", weights_only=False)["model_state_dict"]
        live = model.state_dict()
        # HomieActorModule deliberately retains actor/estimator/std only, not
        # the unused critic from HIMActorCritic's original full checkpoint.
        if (not live or not set(live) <= set(homie_state)
                or any(not torch.equal(value.detach().cpu(), homie_state[key]) for key, value in live.items())):
            raise RuntimeError("live original PPO Homie differs from pinned checkpoint: " + role)
        homie_proofs[role] = {"checkpoint_sha256": expected_sha,
            "strict_parameter_equality": True, "updates_enabled": False}
    proof = {"checkpoint_sha256": TEACHER_SHA256, "strict_parameter_and_rms_equality": True,
             "teacher_input_key": teacher.input_key, "rms_updates_enabled": False,
             "teacher_history_steps": 1, "teacher_is_recurrent": False,
             "homie_parameter_attestations": homie_proofs}
    if original_ppo:
        proof.update(schema="roborender.teacher-parameter-attestation.v2",
            strict_parameter_and_rms_equality=torch.equal(runtime_std, expected_std),
            strict_mean_parameters_and_rms_equality=True,
            std_attestation={**std_policy, "checkpoint_std": expected_std.tolist(),
                "runtime_std": runtime_std.tolist(),
                "runtime_state": "checkpoint_raw" if torch.equal(runtime_std, expected_std)
                                 else "original_ppo_max_clamp",
                "allowed_runtime_states": ["checkpoint_raw", "original_ppo_max_clamp"],
                "optimizer_or_rms_updates_allowed": False})
    trainer._ppo_teacher_parameter_attestation = (teacher, versions, proof)
    return deepcopy(proof)


def validate_parameter_attestation(proof):
    """Validate versioned PPO evidence without importing Torch or model files."""
    common = {"schema": "roborender.teacher-parameter-attestation.v2",
        "checkpoint_sha256": TEACHER_SHA256, "strict_mean_parameters_and_rms_equality": True,
        "rms_updates_enabled": False, "teacher_history_steps": 1, "teacher_is_recurrent": False,
        "homie_parameter_attestations": {
            role: {"checkpoint_sha256": value, "strict_parameter_equality": True, "updates_enabled": False}
            for role, value in (("walk", WALK_SHA256), ("stand", STAND_SHA256))}}
    if (not isinstance(proof, dict)
            or set(proof) != set(common) | {"teacher_input_key", "strict_parameter_and_rms_equality", "std_attestation"}
            or json.dumps({key: proof.get(key) for key in common}, sort_keys=True, allow_nan=False)
                != json.dumps(common, sort_keys=True, allow_nan=False)
            or proof["teacher_input_key"] not in {"teacher_obs", "actor_obs"}):
        raise ValueError("original PPO teacher parameter/RMS attestation differs")
    std = proof["std_attestation"]
    fixed = {"inherited_clamp_noise_std": True, "original_clamp_noise_std": True,
        "original_max_noise_std": 1.0, "binding_scope": "teacher_instance_only_before_first_ppo_forward",
        "trainer_and_student_config_modified": False,
        "allowed_runtime_states": ["checkpoint_raw", "original_ppo_max_clamp"],
        "optimizer_or_rms_updates_allowed": False}
    if (not isinstance(std, dict)
            or set(std) != set(fixed) | {"inherited_max_noise_std", "checkpoint_std", "runtime_std", "runtime_state"}
            or any(std.get(key) != value for key, value in fixed.items())
            or type(std["inherited_max_noise_std"]) not in (int, float)
            or not math.isfinite(std["inherited_max_noise_std"]) or std["inherited_max_noise_std"] <= 0):
        raise ValueError("original PPO std attestation control differs")
    for key in ("checkpoint_std", "runtime_std"):
        values = std[key]
        if (not isinstance(values, list) or len(values) != 11
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values)):
            raise ValueError("original PPO std attestation values invalid")
    raw = std["runtime_std"] == std["checkpoint_std"]
    if (std["runtime_std"] not in [std["checkpoint_std"], [min(v, 1.0) for v in std["checkpoint_std"]]]
            or std["runtime_state"] != ("checkpoint_raw" if raw else "original_ppo_max_clamp")
            or type(proof["strict_parameter_and_rms_equality"]) is not bool
            or proof["strict_parameter_and_rms_equality"] is not raw):
        raise ValueError("original PPO std attestation is not exact raw/original clamp")
    return deepcopy(proof)


def raven_ppo_step(trainer, obs):
    """Original PPO eval loop's forward/call order, with only an input-key alias.

    Actor.rollout samples a discarded Normal action before the deterministic
    mean is selected. Do not replace it with act_inference: that changes native
    RNG consumption. Both Homie models receive the unchanged original history.
    """
    import torch
    with torch.no_grad():
        teacher = trainer.ref_model
        teacher_obs = obs["teacher_obs"]
        if teacher_obs.ndim != 2 or teacher_obs.shape[-1] != 226:
            raise RuntimeError("original PPO teacher needs sorted 226-D privileged observations")
        inputs = obs
        if teacher.input_key == "actor_obs":
            inputs = dict(obs)
            inputs["actor_obs"] = teacher_obs
        elif teacher.input_key != "teacher_obs":
            raise RuntimeError("original PPO teacher input key differs")
        teacher.rollout(obs_dict=inputs)
        raw_teacher = teacher.action_mean.detach()
        homie_obs = obs["homie_obs"]
        if tuple(homie_obs.shape) != (teacher_obs.shape[0], 516):
            raise RuntimeError("original PPO Homie needs unchanged 6x86 history")
        models = trainer.unwrapped_model
        if models.homie_switch_threshold != 0.5:
            raise RuntimeError("original PPO Homie switch threshold differs")
        walk_out = models.homie_walk_model(homie_obs)
        stand_out = models.homie_stand_model(homie_obs)
        commands = homie_obs[..., -86:-83]
        mask = torch.norm(commands, dim=-1, keepdim=True) > models.homie_switch_threshold
        while mask.dim() < walk_out["actions"].dim():
            mask = mask.unsqueeze(-1)
        homie = torch.where(mask, walk_out["action_mean"], stand_out["action_mean"])
        chosen = torch.cat([raw_teacher, homie], dim=-1)
        if tuple(chosen.shape) != (teacher_obs.shape[0], 26) or not torch.isfinite(chosen).all():
            raise RuntimeError("original PPO raw teacher-plus-Homie action differs")
        return {"actions": chosen}, raw_teacher


def select_policy_actions(trainer, obs, dones, *, arm, mode=DISTILL, capture_raw=False):
    """Keep the old branch intact; opt-in PPO never calls student/policy_step."""
    import torch
    validate_mode(mode, arm)
    if mode == RAVEN_PPO:
        return raven_ppo_step(trainer, obs)
    raw = []
    original = None
    if capture_raw:
        original = trainer.ref_model.act_inference
        def observed(*args, **kwargs):
            value = original(*args, **kwargs)
            raw.append(value.detach().clone())
            return value
        trainer.ref_model.act_inference = observed
    try:
        actions = trainer.policy_step(trainer.policy_model, obs, cur_dones=dones)
    finally:
        if original is not None:
            trainer.ref_model.act_inference = original
    teacher = actions["gt_actions"]
    student = actions["action_mean"]
    count = obs["teacher_obs"].shape[0]
    if (tuple(teacher.shape) != (count, 26) or tuple(student.shape) != (count, 26)
            or not torch.equal(teacher[:, 11:], student[:, 11:])):
        raise RuntimeError("11D policy/15D Homie action contract mismatch")
    chosen = teacher if arm == "teacher" else student
    if not torch.isfinite(chosen).all():
        raise RuntimeError("non-finite action")
    actions["actions"] = chosen
    actions["action_mean"] = chosen
    if capture_raw and len(raw) != 1:
        raise RuntimeError("numeric teacher trace requires exactly one original teacher forward")
    return actions, raw[0] if raw else None


# Compatibility name for historical diagnostic callers; no second algorithm.
diagnostic_step = select_policy_actions


def numeric_state(env):
    """Small native-control/physics trace only; no image or simulation calls."""
    import torch
    sim = env.simulator
    targets = sim._robot.data.joint_pos_target
    if not isinstance(targets, torch.Tensor) and hasattr(targets, "torch"):
        targets = targets.torch
    targets = targets[:, sim.dof_ids]
    values = {
        "last_delta_actions": env._last_delta_actions,
        "cumulative_delta_actions": env._delta_actions,
        "right_arm_qvel": sim.dof_vel[:, env.right_arm_dof_indices],
        "right_palm_position": sim._rigid_body_pos[:, env.right_palm_body_idx, :],
        "hold_hand_distance": env.get_hold_object_hand_distance(),
        "hold_tray_distance": env.get_hold_object_tray_distance(),
        "right_finger_cumulative_command": env._delta_actions[:, 10:11],
        "right_finger_joint_targets": targets[:, env._right_hand_dof_idx],
        "right_finger_joint_positions": sim.dof_pos[:, env._right_hand_dof_idx],
        "right_finger_joint_velocities": sim.dof_vel[:, env._right_hand_dof_idx],
        "applied_whole_body_actions": env.actions_after_delay,
    }
    for name in ("front_object", "hold_object", "tray_object"):
        values[name + "_root_state"] = sim.get_task_root_state(getattr(env.config.task, name))
    for name in ("hold_object_to_hand_contact_forces", "front_object_to_hand_contact_forces",
                 "front_object_table_contact_forces", "table_to_hand_contact_forces",
                 "table_to_body_contact_forces"):
        if hasattr(sim, name):
            values[name] = getattr(sim, name)
    result = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or value.numel() > 16384 or not torch.isfinite(value).all():
            raise RuntimeError("numeric diagnostic state is not bounded/finite: " + name)
        result[name] = value.detach().cpu().tolist()
    # Optional getter-only diagnostics must not invalidate a native trajectory
    # if a particular backend does not expose an effective-drive readback.
    # Never confuse this evidence with the user target buffer above.
    if getattr(env, "num_envs", None) in (1, 2):
        from native_physics_readback import drive_readback
        try:
            result["effective_drive_readback"] = drive_readback(env)
        except Exception as exc:
            result["effective_drive_readback"] = {
                "status": "unavailable", "error_type": type(exc).__name__, "read_only": True}
    return result
