"""Versioned initial robot-refresh corrections and historical receipt parsing.

No simulator, policy, allocation or file APIs. Clearing is permitted only AFTER
the original independent panel reset has applied robot state and built its
observations. Ordinary done-driven resets are never routed through this helper.
"""
from copy import deepcopy
import json

ORIGINAL = "raven_original"
CLEAR_ONCE = "clear_already_applied_once"
STUDENT_CLEAR_ONCE = "student_clear_already_applied_once_v1"
SMOKE_STUDENT_CLEAR_ONCE = "smoke_student_clear_already_applied_once_v1"
SCALED_STUDENT_CLEAR_ONCE = "scaled_student_clear_already_applied_once_v1"
SCALED_TEACHER_CLEAR_ONCE = "scaled_teacher_clear_already_applied_once_v1"
FIELD = "diagnostic_initial_robot_refresh"


def require_corrected_execution(value, *, arm, teacher_action_mode, num_envs=None, has_resume=False):
    """Live execution never defaults to the obsolete double-refresh reset.

    ORIGINAL remains a reader label for immutable historical evidence only.
    Missing/legacy selectors fail before reset, model loading or native steps.
    """
    if value in (None, ORIGINAL):
        raise ValueError('pre-fix success reset is retired; explicit corrected production/smoke protocol required')
    return validate_mode(value, arm=arm, teacher_action_mode=teacher_action_mode,
                         num_envs=num_envs, has_resume=has_resume)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_mode(value=ORIGINAL, *, arm="teacher", teacher_action_mode="raven_ppo",
                  num_envs=None, has_resume=False):
    if value not in (ORIGINAL, CLEAR_ONCE, STUDENT_CLEAR_ONCE, SMOKE_STUDENT_CLEAR_ONCE, SCALED_STUDENT_CLEAR_ONCE, SCALED_TEACHER_CLEAR_ONCE):
        raise ValueError("unknown diagnostic initial robot refresh mode")
    if value == CLEAR_ONCE and (arm != "teacher" or teacher_action_mode != "raven_ppo"
            or has_resume or (num_envs is not None and (type(num_envs) is not int or num_envs not in (1, 2)))):
        raise ValueError("clear-once initial refresh requires fresh non-resume teacher PPO B1/B2 diagnostic")
    if value in (STUDENT_CLEAR_ONCE, SMOKE_STUDENT_CLEAR_ONCE) and (arm not in ("bc", "dagger") or teacher_action_mode != "distill"
            or has_resume or (num_envs is not None and (type(num_envs) is not int or num_envs not in (1, 2)))):
        raise ValueError("student clear-once initial refresh requires fresh non-resume BC/DAgger student B1/B2 evaluation")
    if value == SCALED_STUDENT_CLEAR_ONCE and (arm not in ('bc', 'dagger') or teacher_action_mode != 'distill'
            or has_resume or (num_envs is not None and (type(num_envs) is not int or not 1 <= num_envs <= 16))):
        raise ValueError('scaled clear-once initial refresh requires fresh BC student B1..16 evaluation')
    if value == SCALED_TEACHER_CLEAR_ONCE and (arm != 'teacher' or teacher_action_mode != 'raven_ppo'
            or has_resume or (num_envs is not None and (type(num_envs) is not int or not 1 <= num_envs <= 16))):
        raise ValueError('scaled teacher reset requires fresh privileged PPO teacher B1..16')
    return value


def control_contract(value=CLEAR_ONCE):
    if value not in (CLEAR_ONCE, STUDENT_CLEAR_ONCE, SMOKE_STUDENT_CLEAR_ONCE, SCALED_STUDENT_CLEAR_ONCE, SCALED_TEACHER_CLEAR_ONCE):
        raise ValueError("only the opt-in reset intervention has an extra control contract")
    student = value in (STUDENT_CLEAR_ONCE, SMOKE_STUDENT_CLEAR_ONCE)
    schema = ('roborender.initial-robot-refresh-control.v4' if value == SCALED_TEACHER_CLEAR_ONCE else
              'roborender.initial-robot-refresh-control.v3' if value == SCALED_STUDENT_CLEAR_ONCE else
              'roborender.initial-robot-refresh-control.v2' if student else 'roborender.initial-robot-refresh-control.v1')
    return {"schema": schema,
        "mode": value,
        "scope": "independent_panel_initial_reset_only",
        "boundary": "after_original_reset_writes_forward_and_observations_before_first_policy_step",
        "operation": "clear_only_already_applied_pending_robot_refresh_flags_once",
        "ordinary_done_resets_modified": False, "robot_or_object_targets_modified": False,
        "policy_or_rng_modified": False, "extra_physics_steps": 0,
        "success_criteria_modified": False, "production_activation": value in (STUDENT_CLEAR_ONCE, SCALED_STUDENT_CLEAR_ONCE)}


def _attestation_schema(mode):
    if mode == SCALED_TEACHER_CLEAR_ONCE:
        return 'roborender.initial-robot-refresh-attestation.v4'
    if mode == SCALED_STUDENT_CLEAR_ONCE:
        return 'roborender.initial-robot-refresh-attestation.v3'
    return ('roborender.initial-robot-refresh-attestation.v2' if mode in (STUDENT_CLEAR_ONCE, SMOKE_STUDENT_CLEAR_ONCE)
            else 'roborender.initial-robot-refresh-attestation.v1')


def validate_attestation(value, *, num_envs=None, mode=CLEAR_ONCE):
    if not isinstance(value, dict):
        raise ValueError("initial robot refresh attestation missing")
    count = value.get("num_envs")
    allowed = range(1, 17) if mode in (SCALED_STUDENT_CLEAR_ONCE, SCALED_TEACHER_CLEAR_ONCE) else (1, 2)
    if type(count) is not int or count not in allowed or (num_envs is not None and count != num_envs):
        raise ValueError("initial robot refresh attestation cardinality differs")
    expected = {"schema": _attestation_schema(mode),
        "control": control_contract(mode), "num_envs": count,
        "pending_before": [True] * count, "pending_after": [False] * count,
        "cleared_lanes": count, "clear_calls": 1, "native_steps_added": 0}
    if canonical(value) != canonical(expected):
        raise ValueError("initial robot refresh attestation differs")
    return deepcopy(value)


def clear_already_applied_initial_refresh(env, *, mode=CLEAR_ONCE, arm="teacher", teacher_action_mode="raven_ppo"):
    """One bool-buffer operation, no native API/RNG/target operations."""
    import torch
    validate_mode(mode, arm=arm, teacher_action_mode=teacher_action_mode, num_envs=env.num_envs)
    control = control_contract(mode)
    if getattr(env, "_diagnostic_initial_refresh_clear_calls", 0) != 0:
        raise RuntimeError("initial robot refresh intervention already applied")
    pending = env.need_to_refresh_envs
    if hasattr(env, "_rr_initial_reset_generation"):
        # New shared overlay already consumed the flag at reset_all completion.
        # Preserve the frozen success receipt without applying the fix twice.
        from initial_reset_contract import require_completed_initial_reset
        receipt = require_completed_initial_reset(env)
        generation = receipt["generation"]
        if getattr(env, "_diagnostic_initial_refresh_generation", 0) >= generation:
            raise RuntimeError("initial robot refresh intervention already applied for this reset")
        env._diagnostic_initial_refresh_generation = generation
        before = [True] * env.num_envs
    elif (not isinstance(pending, torch.Tensor) or pending.dtype != torch.bool
            or tuple(pending.shape) != (env.num_envs,) or not pending.all().item()):
        raise RuntimeError("initial robot refresh flags are not all already-applied pending lanes")
    else:
        # Compatibility for the previously qualified explicit panel correction.
        before = pending.detach().cpu().tolist()
        pending.zero_()
    env._diagnostic_initial_refresh_clear_calls = 1
    evidence = {"schema": _attestation_schema(mode),
        "control": control, "num_envs": env.num_envs,
        "pending_before": before, "pending_after": pending.detach().cpu().tolist(),
        "cleared_lanes": env.num_envs, "clear_calls": 1, "native_steps_added": 0}
    return validate_attestation(evidence, num_envs=env.num_envs, mode=mode)


def result_mode(result, *, expected=None, num_envs=None):
    value = validate_mode(result.get(FIELD, ORIGINAL), arm=result.get("arm"),
        teacher_action_mode=result.get("teacher_action_mode", "distill"), num_envs=num_envs)
    if expected is not None and value != expected:
        raise ValueError("initial robot refresh result differs from requested diagnostic")
    extras = {"initial_robot_refresh_control", "initial_robot_refresh_attestation", "initial_robot_refresh_attestations"}
    if value == ORIGINAL:
        if extras & set(result):
            raise ValueError("original refresh result cannot carry intervention evidence")
        return value
    if canonical(result.get("initial_robot_refresh_control")) != canonical(control_contract(value)):
        raise ValueError("initial robot refresh control contract differs")
    single = "initial_robot_refresh_attestation" in result
    multiple = "initial_robot_refresh_attestations" in result
    if single == multiple:
        raise ValueError("initial robot refresh evidence must be one cohort or an aggregate")
    if single:
        validate_attestation(result["initial_robot_refresh_attestation"], num_envs=num_envs, mode=value)
    else:
        values = result["initial_robot_refresh_attestations"]
        if not isinstance(values, list) or not values:
            raise ValueError("initial robot refresh aggregate lacks cohort evidence")
        for row in values:
            validate_attestation(row, mode=value)
    return value


def validate_panel_mode(panel, value=ORIGINAL, *, arm, teacher_action_mode, has_resume=False):
    validate_mode(value, arm=arm, teacher_action_mode=teacher_action_mode, has_resume=has_resume)
    if value != ORIGINAL:
        for row in panel["cohorts"]:
            validate_mode(value, arm=arm, teacher_action_mode=teacher_action_mode,
                          num_envs=len(row["assignment"]["lanes"]))
    return value
