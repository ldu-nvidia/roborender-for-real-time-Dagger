"""Versioned same-motion-support success panels, separate from legacy B5.

Each case is the first completed native episode for a declared cohort/lane.
Seeds are reproducible reset instructions, not claims of fixed reset indices.
Actual native reset samples/states are attested by the runtime executor.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

try:
    from .initial_robot_refresh_diagnostic import (
        ORIGINAL as ORIGINAL_INITIAL_REFRESH, CLEAR_ONCE, FIELD as INITIAL_REFRESH_FIELD,
        control_contract as initial_refresh_contract, result_mode as initial_refresh_result_mode,
        validate_panel_mode as validate_initial_refresh_panel, validate_attestation as validate_initial_refresh_attestation,
    )
except ImportError:
    from initial_robot_refresh_diagnostic import (
        ORIGINAL as ORIGINAL_INITIAL_REFRESH, CLEAR_ONCE, FIELD as INITIAL_REFRESH_FIELD,
        control_contract as initial_refresh_contract, result_mode as initial_refresh_result_mode,
        validate_panel_mode as validate_initial_refresh_panel, validate_attestation as validate_initial_refresh_attestation,
    )

try:
    from .native_motion_assignment import validate_motion_assignment
    from .teacher_action_diagnostic import (
        DISTILL, RAVEN_PPO, action_mode, control_contract, validate_control_contract, validate_mode,
        validate_parameter_attestation,
    )
except ImportError:
    from native_motion_assignment import validate_motion_assignment
    from teacher_action_diagnostic import (
        DISTILL, RAVEN_PPO, action_mode, control_contract, validate_control_contract, validate_mode,
        validate_parameter_attestation,
    )

SCHEMA = "roborender.explicit-motion-success-panel.v1"
RESULT_SCHEMA = "roborender.explicit-motion-success-result.v1"
GOAL_ACCOUNTING_VERSION = "roborender.raven.pre_reset_goal.v1"
COHORT_RECEIPT_SCHEMA = "roborender.explicit-motion-completed-cohort.v1"
PARTITION_SCHEMA = "roborender.explicit-motion-panel-partition.v1"
PARTITION_RESULT_SCHEMA = "roborender.explicit-motion-panel-partition-result.v1"


def _execution_identity(checkpoint_sha256, arm, teacher_action_mode=DISTILL):
    if (not isinstance(checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256)
            or arm not in {"teacher", "bc", "dagger"}):
        raise ValueError("success resume requires the exact checkpoint SHA and behavior arm")
    validate_mode(teacher_action_mode, arm)


def _result_mode(result):
    mode = validate_mode(result.get("teacher_action_mode", DISTILL), result.get("arm"))
    if "teacher_control" in result or mode == RAVEN_PPO:
        validate_control_contract(result.get("teacher_control"), mode)
    if mode == RAVEN_PPO:
        validate_parameter_attestation(result.get("teacher_parameter_attestation"))
    initial_refresh_result_mode(result)
    return mode


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def panel_sha256(value):
    return hashlib.sha256(canonical(validate_panel(value))).hexdigest()


def _rendered_contract(panel):
    # Preserve the already admitted, frozen paired200 diagnostic separately.
    if str(panel.get('panel_id', '')).startswith('scaled-rendered295-'):
        try:
            from . import scaled_success_contract as contract
        except ImportError:
            import scaled_success_contract as contract
    elif panel.get('panel_id') == 'bc116-training-support-image3f-200-20261023':
        try:
            from . import rendered_training_support_contract as contract
        except ImportError:
            import rendered_training_support_contract as contract
    else:
        try:
            from . import rendered_student_success_contract as contract
        except ImportError:
            import rendered_student_success_contract as contract
    return contract


def validate_panel(value):
    if isinstance(value, dict) and value.get("observation_mode") == "image3f_3view":
        # Structural parser only: execution contracts pin full panel identities.
        validate_panel(_rendered_contract(value).raytraced_shape(value))
        return deepcopy(value)
    keys = {"schema", "panel_id", "purpose", "motion_inventory_sha256", "observation_mode",
            "action_mode", "metric", "max_vector_steps_per_cohort", "episode_count", "cohorts"}
    if (not isinstance(value, dict) or set(value) != keys or value["schema"] != SCHEMA
            or not isinstance(value["panel_id"], str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", value["panel_id"])
            or value["purpose"] not in {"validation", "test"}
            or value["observation_mode"] != "raytraced_3view"
            or value["action_mode"] != "deterministic_action_mean_11_plus_homie_15"
            or value["metric"] != "native_goal_or_until_done"
            or type(value["max_vector_steps_per_cohort"]) is not int
            or value["max_vector_steps_per_cohort"] != 450
            or type(value["episode_count"]) is not int or not 1 <= value["episode_count"] <= 12000
            or not isinstance(value["cohorts"], list) or not 1 <= len(value["cohorts"]) <= 300):
        raise ValueError("invalid explicit-motion success panel")
    count, cases, cohorts, seeds = 0, set(), set(), set()
    for row in value["cohorts"]:
        if not isinstance(row, dict) or set(row) != {"assignment", "case_ids"}:
            raise ValueError("explicit panel cohort fields differ")
        assignment = validate_motion_assignment(row["assignment"])
        if (assignment["run_id"] != value["panel_id"]
                or assignment["purpose"] != value["purpose"]
                or assignment["motion_inventory_sha256"] != value["motion_inventory_sha256"]
                or assignment["cohort_id"] in cohorts or assignment["simulator_seed"] in seeds
                or not isinstance(row["case_ids"], list)
                or len(row["case_ids"]) != len(assignment["lanes"])):
            raise ValueError("panel native assignment/count/seed identity differs")
        for case in row["case_ids"]:
            if (not isinstance(case, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", case)
                    or case in cases):
                raise ValueError("explicit panel case identities must be unique")
            cases.add(case)
        count += len(row["case_ids"])
        cohorts.add(assignment["cohort_id"])
        seeds.add(assignment["simulator_seed"])
    if count != value["episode_count"]:
        raise ValueError("panel episode count differs from declared lanes")
    return deepcopy(value)


def summarize(panel, results):
    panel = validate_panel(panel)
    if len(results) != len(panel["cohorts"]):
        raise ValueError("success panel is missing a cohort")
    outcomes, checkpoints, accounting, arms, modes = [], set(), set(), set(), set()
    initial_refresh_modes, initial_refresh_attestations = set(), []
    for spec, result in zip(panel["cohorts"], results):
        assignment = spec["assignment"]
        n = len(assignment["lanes"])
        if (result.get("complete") is not True or result.get("completed") != n
                or result.get("expected") != n or result.get("optimizer_steps") != 0
                or result.get("policy_unchanged") is not True or result.get("accounting_disagreements") != 0
                or result.get("seed") != assignment["simulator_seed"]
                or result.get("observation_mode") != panel["observation_mode"]
                or result.get("arm") not in {"teacher", "bc", "dagger"}
                or result.get("goal_accounting_version") != GOAL_ACCOUNTING_VERSION
                or any(not isinstance(result.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", result[key])
                       for key in ("checkpoint_sha256", "runtime_physics_sha256"))
                or result.get("motion_assignment_sha256") != hashlib.sha256(canonical(assignment)).hexdigest()
                or len(result.get("outcomes", [])) != n
                or {r.get("lane") for r in result["outcomes"]} != set(range(n))):
            raise ValueError("native explicit-motion cohort is incomplete or inconsistent")
        if panel["observation_mode"] == "image3f_3view":
            _rendered_contract(panel).validate_cohort_observations(result, count=n)
        elif "rendered_observation_receipt" in result or "paired_initial_state_attestation" in result:
            raise ValueError("ray-traced cohort cannot carry rendered-observation evidence")
        checkpoints.add(result["checkpoint_sha256"])
        accounting.add(result["goal_accounting_version"])
        arms.add(result["arm"])
        modes.add(_result_mode(result))
        initial_refresh = initial_refresh_result_mode(result, num_envs=n)
        initial_refresh_modes.add(initial_refresh)
        if initial_refresh != ORIGINAL_INITIAL_REFRESH:
            if "initial_robot_refresh_attestation" not in result:
                raise ValueError("native cohort requires one initial refresh attestation")
            initial_refresh_attestations.append(deepcopy(result["initial_robot_refresh_attestation"]))
        for outcome in sorted(result["outcomes"], key=lambda row: row["lane"]):
            lane = outcome["lane"]
            if (type(outcome.get("native_success")) is not bool
                    or type(outcome.get("exported_success")) is not bool
                    or outcome["native_success"] != outcome["exported_success"]
                    or type(outcome.get("steps")) is not int or not 1 <= outcome["steps"] <= 450):
                raise ValueError("native outcome is not a complete truthful episode receipt")
            motion = assignment["lanes"][lane]
            outcomes.append({**outcome, "case_id": spec["case_ids"][lane],
                "cohort_id": assignment["cohort_id"], "simulator_seed": assignment["simulator_seed"],
                "motion_id": motion["motion_id"], "motion_sha256": motion["sha256"]})
        native_successes = sum(row["native_success"] for row in result["outcomes"])
        if (result.get("original_raven_metric_completed") != n
                or result.get("original_raven_metric_successes") != native_successes
                or result.get("native_successes") != native_successes
                or result.get("exported_successes") != native_successes):
            raise ValueError("native Raven metric disagrees with completed outcome receipts")
    if (len(checkpoints) != 1 or len(accounting) != 1 or len(arms) != 1 or len(modes) != 1
            or len(initial_refresh_modes) != 1):
        raise ValueError("checkpoint/goal accounting/behavior arm/teacher mode changed within a panel")
    successes = sum(row["exported_success"] for row in outcomes)
    arm = arms.pop()
    mode = modes.pop()
    summary = {"schema": RESULT_SCHEMA, "complete": True, "panel_sha256": panel_sha256(panel),
        "panel_id": panel["panel_id"], "purpose": panel["purpose"],
        "checkpoint_sha256": checkpoints.pop(), "goal_accounting_version": accounting.pop(),
        "arm": arm, "metric": panel["metric"], "action_mode": action_mode(arm, mode, panel["action_mode"]),
        "augmentation_applied": False,
        "runtime_physics_sha256_per_cohort": [r["runtime_physics_sha256"] for r in results],
        "completed_episodes": len(outcomes), "expected_episodes": panel["episode_count"],
        "successes": successes, "success_rate": successes / len(outcomes),
        "optimizer_steps": 0, "policy_unchanged": True, "observation_mode": panel["observation_mode"],
        "motion_inventory_sha256": panel["motion_inventory_sha256"], "outcomes": outcomes,
        "legacy_five_motion_benchmark": False,
        "case_semantics": "first completed native episode per explicit seed/lane; actual resets attested"}
    if panel["observation_mode"] == "image3f_3view":
        summary["rendered_observation_receipts"] = [deepcopy(r["rendered_observation_receipt"]) for r in results]
        if panel['panel_id'] == 'bc116-training-support-image3f-200-20261023':
            summary["paired_initial_state_attestations"] = [deepcopy(r["paired_initial_state_attestation"]) for r in results]
    if mode != DISTILL:
        summary.update(teacher_action_mode=mode, teacher_control=control_contract(mode),
            teacher_parameter_attestation=deepcopy(results[0]["teacher_parameter_attestation"]))
    initial_refresh = initial_refresh_modes.pop()
    if initial_refresh != ORIGINAL_INITIAL_REFRESH:
        summary.update({INITIAL_REFRESH_FIELD: initial_refresh,
            "initial_robot_refresh_control": initial_refresh_contract(initial_refresh),
            "initial_robot_refresh_attestations": initial_refresh_attestations})
    return summary


def validate_result(panel, result, checkpoint_sha256, *, allowed_arms=("bc", "dagger"),
                    teacher_action_mode=DISTILL, diagnostic_initial_robot_refresh=ORIGINAL_INITIAL_REFRESH):
    """Validate a small host-import receipt; no tensor/image/checkpoint rereads."""
    panel = validate_panel(panel)
    validate_mode(teacher_action_mode, result.get("arm") if isinstance(result, dict) else None)
    validate_initial_refresh_panel(panel, diagnostic_initial_robot_refresh,
        arm=result.get("arm") if isinstance(result, dict) else None, teacher_action_mode=teacher_action_mode)
    if (not isinstance(checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256)
            or not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA
            or result.get("complete") is not True or result.get("panel_sha256") != panel_sha256(panel)
            or result.get("panel_id") != panel["panel_id"] or result.get("purpose") != panel["purpose"]
            or result.get("checkpoint_sha256") != checkpoint_sha256
            or result.get("goal_accounting_version") != GOAL_ACCOUNTING_VERSION
            or result.get("arm") not in allowed_arms
            or result.get("metric") != panel["metric"]
            or _result_mode(result) != teacher_action_mode
            or result.get("action_mode") != action_mode(result.get("arm"), teacher_action_mode, panel["action_mode"])
            or result.get("augmentation_applied") is not False
            or result.get("optimizer_steps") != 0 or result.get("policy_unchanged") is not True
            or result.get("observation_mode") != panel["observation_mode"]
            or result.get("motion_inventory_sha256") != panel["motion_inventory_sha256"]
            or result.get("legacy_five_motion_benchmark") is not False
            or result.get("expected_episodes") != panel["episode_count"]
            or result.get("completed_episodes") != panel["episode_count"]
            or not isinstance(result.get("outcomes"), list)
            or len(result["outcomes"]) != panel["episode_count"]):
        raise ValueError("explicit success result identity/completion/behavior differs")
    physics = result.get("runtime_physics_sha256_per_cohort")
    if panel["observation_mode"] == "image3f_3view":
        rendered_contract = _rendered_contract(panel)
        paired_diagnostic = panel['panel_id'] == 'bc116-training-support-image3f-200-20261023'
        rendered = result.get("rendered_observation_receipts")
        pairs = result.get("paired_initial_state_attestations") if paired_diagnostic else [None] * len(panel['cohorts'])
        if (not isinstance(rendered, list) or not isinstance(pairs, list)
                or len(rendered) != len(panel["cohorts"]) or len(pairs) != len(rendered)
                or not isinstance(physics, list) or len(physics) != len(rendered)
                or (not paired_diagnostic and 'paired_initial_state_attestations' in result)):
            raise ValueError("rendered success aggregate lacks actual per-cohort evidence")
        for spec, receipt, paired, physical in zip(panel["cohorts"], rendered, pairs, physics):
            cohort = {"rendered_observation_receipt": receipt, "runtime_physics_sha256": physical,
                "vector_steps": receipt.get("vector_steps_rendered") if isinstance(receipt, dict) else None,
                'checkpoint_sha256': checkpoint_sha256, 'motion_assignment_sha256': hashlib.sha256(canonical(spec['assignment'])).hexdigest()}
            if paired_diagnostic:
                cohort['paired_initial_state_attestation'] = paired
            rendered_contract.validate_cohort_observations(cohort, count=len(spec["case_ids"]))
    elif {"rendered_observation_receipts", "paired_initial_state_attestations", "rendered_training_support_protocol", 'rendered_student_success_protocol'} & set(result):
        raise ValueError("ray-traced result cannot silently adopt rendered evidence")
    initial_refresh_result_mode(result, expected=diagnostic_initial_robot_refresh)
    if diagnostic_initial_robot_refresh != ORIGINAL_INITIAL_REFRESH:
        attestations = result.get("initial_robot_refresh_attestations")
        if not isinstance(attestations, list) or len(attestations) != len(panel["cohorts"]):
            raise ValueError("success aggregate lacks per-cohort initial refresh attestation")
        for evidence, cohort in zip(attestations, panel["cohorts"]):
            validate_initial_refresh_attestation(evidence, num_envs=len(cohort["case_ids"]),
                                                mode=diagnostic_initial_robot_refresh)
    if (not isinstance(physics, list) or len(physics) != len(panel["cohorts"])
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in physics)):
        raise ValueError("explicit success result lacks per-cohort runtime physics evidence")
    expected = {}
    for cohort in panel["cohorts"]:
        assignment = cohort["assignment"]
        for lane, case_id in enumerate(cohort["case_ids"]):
            motion = assignment["lanes"][lane]
            expected[case_id] = {"lane": lane, "cohort_id": assignment["cohort_id"],
                "simulator_seed": assignment["simulator_seed"], "motion_id": motion["motion_id"],
                "motion_sha256": motion["sha256"]}
    seen, successes = set(), 0
    for row in result["outcomes"]:
        case = row.get("case_id") if isinstance(row, dict) else None
        if (not isinstance(case, str) or case not in expected or case in seen
                or any(row.get(key) != value for key, value in expected[case].items())
                or type(row.get("steps")) is not int or not 1 <= row["steps"] <= 450
                or type(row.get("native_success")) is not bool
                or type(row.get("exported_success")) is not bool
                or row["native_success"] != row["exported_success"]
                or type(row.get("timeout")) is not bool):
            raise ValueError("explicit success case motion/seed/native outcome differs")
        seen.add(case)
        successes += row["exported_success"]
    if (type(result.get("successes")) is not int or result["successes"] != successes
            or type(result.get("success_rate")) not in (int, float)
            or result["success_rate"] != successes / panel["episode_count"]):
        raise ValueError("explicit success aggregate does not equal completed cases")
    return deepcopy(result)


def make_cohort_receipt(panel, index, result, *, checkpoint_sha256, arm="bc",
                        teacher_action_mode=DISTILL, diagnostic_initial_robot_refresh=ORIGINAL_INITIAL_REFRESH):
    """Bind one *complete* native cohort to its original sealed full panel.

    This is a small CPU-only receipt, not a simulator snapshot. A failed or
    partially completed cohort cannot be resumed by this mechanism; its scene
    must start freshly. Complete earlier cohorts need not be executed again.
    """
    panel = validate_panel(panel)
    _execution_identity(checkpoint_sha256, arm, teacher_action_mode)
    if (type(index) is not int or not 0 <= index < len(panel["cohorts"])
            or not isinstance(result, dict)):
        raise ValueError("invalid completed cohort index or arm")
    spec = panel["cohorts"][index]
    one = deepcopy(panel)
    one["cohorts"] = [spec]
    one["episode_count"] = len(spec["case_ids"])
    # The temporary singleton is used only for existing native receipt checks;
    # its synthetic SHA is never published as the tested panel's identity.
    checked = summarize(one, [result])
    validate_result(one, checked, checkpoint_sha256, allowed_arms=(arm,), teacher_action_mode=teacher_action_mode,
                    diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
    receipt = {"schema": COHORT_RECEIPT_SCHEMA, "panel_sha256": panel_sha256(panel),
        "panel_id": panel["panel_id"], "purpose": panel["purpose"],
        "checkpoint_sha256": checkpoint_sha256, "arm": arm,
        "cohort_index": index, "cohort_id": spec["assignment"]["cohort_id"],
        "cohort_spec_sha256": hashlib.sha256(canonical(spec)).hexdigest(),
        "case_ids": deepcopy(spec["case_ids"]),
        "result_sha256": hashlib.sha256(canonical(result)).hexdigest(),
        "result": deepcopy(result)}
    if teacher_action_mode != DISTILL:
        receipt["teacher_action_mode"] = teacher_action_mode
    if diagnostic_initial_robot_refresh != ORIGINAL_INITIAL_REFRESH:
        receipt[INITIAL_REFRESH_FIELD] = diagnostic_initial_robot_refresh
    return receipt


def validate_cohort_receipt(panel, receipt, *, checkpoint_sha256, arm="bc",
                            teacher_action_mode=DISTILL, diagnostic_initial_robot_refresh=ORIGINAL_INITIAL_REFRESH):
    if not isinstance(receipt, dict) or type(receipt.get("cohort_index")) is not int:
        raise ValueError("invalid completed cohort receipt")
    expected = make_cohort_receipt(panel, receipt["cohort_index"], receipt.get("result"),
                                  checkpoint_sha256=checkpoint_sha256, arm=arm,
                                  teacher_action_mode=teacher_action_mode,
                                  diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
    if canonical(receipt) != canonical(expected):
        raise ValueError("completed cohort receipt panel/checkpoint/result binding differs")
    return expected


def _completed_prefix(panel, indices, receipts, *, checkpoint_sha256, arm, teacher_action_mode=DISTILL,
                      diagnostic_initial_robot_refresh=ORIGINAL_INITIAL_REFRESH):
    if not isinstance(receipts, (list, tuple)) or len(receipts) > len(indices):
        raise ValueError("completed receipts are not a bounded cohort prefix")
    checked = []
    for index, receipt in zip(indices, receipts):
        value = validate_cohort_receipt(panel, receipt, checkpoint_sha256=checkpoint_sha256, arm=arm,
                                       teacher_action_mode=teacher_action_mode,
                                       diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
        if value["cohort_index"] != index:
            raise ValueError("completed receipts contain a hole, duplicate, or reordered cohort")
        checked.append(value)
    return checked


def completed_progress(panel, receipts, *, checkpoint_sha256, arm="bc", teacher_action_mode=DISTILL,
                       diagnostic_initial_robot_refresh=ORIGINAL_INITIAL_REFRESH):
    """Validate an exact completed prefix without pretending it is a final score."""
    panel = validate_panel(panel)
    _execution_identity(checkpoint_sha256, arm, teacher_action_mode)
    validate_initial_refresh_panel(panel, diagnostic_initial_robot_refresh, arm=arm,
                                  teacher_action_mode=teacher_action_mode)
    checked = _completed_prefix(panel, list(range(len(panel["cohorts"]))), receipts,
                                checkpoint_sha256=checkpoint_sha256, arm=arm,
                                teacher_action_mode=teacher_action_mode,
                                diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
    progress = {"phase": "explicit_success_resume_progress", "complete": False,
        "panel_sha256": panel_sha256(panel), "checkpoint_sha256": checkpoint_sha256,
        "cohorts_completed": len(checked), "expected_cohorts": len(panel["cohorts"]),
        "completed_episodes": sum(len(r["case_ids"]) for r in checked),
        "expected_episodes": panel["episode_count"],
        "successes": sum(r["result"]["native_successes"] for r in checked),
        "optimizer_steps": 0}
    if diagnostic_initial_robot_refresh != ORIGINAL_INITIAL_REFRESH:
        progress[INITIAL_REFRESH_FIELD] = diagnostic_initial_robot_refresh
    return progress


def make_partition(panel, cohort_indices):
    """Describe a bounded contiguous execution slice, never a new test panel."""
    panel = validate_panel(panel)
    if (not isinstance(cohort_indices, (list, tuple)) or not cohort_indices
            or any(type(i) is not int or not 0 <= i < len(panel["cohorts"]) for i in cohort_indices)
            or list(cohort_indices) != list(range(cohort_indices[0], cohort_indices[-1] + 1))):
        raise ValueError("partition must be a nonempty contiguous original cohort range")
    cases = [case for i in cohort_indices for case in panel["cohorts"][i]["case_ids"]]
    return {"schema": PARTITION_SCHEMA, "panel_sha256": panel_sha256(panel),
        "panel_id": panel["panel_id"], "purpose": panel["purpose"],
        "cohort_indices": list(cohort_indices), "case_ids": cases,
        "expected_episodes": len(cases), "full_expected_episodes": panel["episode_count"]}


def partition_panel(panel, *, max_cohorts):
    panel = validate_panel(panel)
    if type(max_cohorts) is not int or max_cohorts < 1:
        raise ValueError("max_cohorts must be a positive integer")
    return [make_partition(panel, list(range(start, min(start + max_cohorts, len(panel["cohorts"])))))
            for start in range(0, len(panel["cohorts"]), max_cohorts)]


def _validate_partition(panel, partition):
    if not isinstance(partition, dict):
        raise ValueError("invalid original-panel partition")
    expected = make_partition(panel, partition.get("cohort_indices"))
    if canonical(partition) != canonical(expected):
        raise ValueError("partition differs from original sealed full panel")
    return expected


def _partition_result(panel, partition, receipts, *, checkpoint_sha256, arm, teacher_action_mode=DISTILL,
                      diagnostic_initial_robot_refresh=ORIGINAL_INITIAL_REFRESH):
    partition = _validate_partition(panel, partition)
    checked = _completed_prefix(panel, partition["cohort_indices"], receipts,
                                checkpoint_sha256=checkpoint_sha256, arm=arm,
                                teacher_action_mode=teacher_action_mode,
                                diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
    if len(checked) != len(partition["cohort_indices"]):
        raise ValueError("partition still has incomplete cohorts")
    result = {"schema": PARTITION_RESULT_SCHEMA, "panel_sha256": panel_sha256(panel),
        "panel_id": panel["panel_id"], "checkpoint_sha256": checkpoint_sha256, "arm": arm,
        "partition": partition, "partition_sha256": hashlib.sha256(canonical(partition)).hexdigest(),
        "complete": True, "cohort_receipts": checked,
        "completed_episodes": partition["expected_episodes"],
        "expected_episodes": partition["expected_episodes"],
        "full_expected_episodes": panel["episode_count"],
        "successes": sum(r["result"]["native_successes"] for r in checked),
        "optimizer_steps": 0, "policy_unchanged": True}
    if teacher_action_mode != DISTILL:
        result["teacher_action_mode"] = teacher_action_mode
    if diagnostic_initial_robot_refresh != ORIGINAL_INITIAL_REFRESH:
        result[INITIAL_REFRESH_FIELD] = diagnostic_initial_robot_refresh
    return result


def run_partition(panel, partition, *, execute_cohort, checkpoint_sha256, arm="bc",
                  completed_receipts=(), save_receipt=lambda receipt: None,
                  report=lambda value: None, teacher_action_mode=DISTILL,
                  diagnostic_initial_robot_refresh=ORIGINAL_INITIAL_REFRESH):
    """Run a complete-cohort slice under one bounded native-owner lifecycle.

    execute_cohort always receives the original full-panel cohort index. The
    caller owns native scene teardown/recreation and durable atomic storage of
    save_receipt. Partial outcomes never count, and no subset success rate is
    exposed as the full-panel test metric.
    """
    panel = validate_panel(panel)
    _execution_identity(checkpoint_sha256, arm, teacher_action_mode)
    validate_initial_refresh_panel(panel, diagnostic_initial_robot_refresh, arm=arm,
        teacher_action_mode=teacher_action_mode, has_resume=bool(completed_receipts))
    partition = _validate_partition(panel, partition)
    indices = partition["cohort_indices"]
    receipts = _completed_prefix(panel, indices, completed_receipts,
                                 checkpoint_sha256=checkpoint_sha256, arm=arm,
                                 teacher_action_mode=teacher_action_mode,
                                 diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
    event = {"phase": "explicit_success_partition_resume", "complete": False,
            "panel_sha256": panel_sha256(panel), "checkpoint_sha256": checkpoint_sha256,
            "cohort_indices": indices, "cohorts_completed": len(receipts),
            "completed_episodes": sum(len(r["case_ids"]) for r in receipts),
            "successes": sum(r["result"]["native_successes"] for r in receipts),
            "expected_episodes": panel["episode_count"], "optimizer_steps": 0}
    if diagnostic_initial_robot_refresh != ORIGINAL_INITIAL_REFRESH:
        event[INITIAL_REFRESH_FIELD] = diagnostic_initial_robot_refresh
    report(event)
    for index in indices[len(receipts):]:
        cohort = panel["cohorts"][index]
        report({"phase": "explicit_success_cohort_start", "panel_sha256": panel_sha256(panel),
                "cohort_index": index, "cohort_id": cohort["assignment"]["cohort_id"],
                "case_ids": cohort["case_ids"]})
        receipt = make_cohort_receipt(panel, index, execute_cohort(cohort, index),
                                     checkpoint_sha256=checkpoint_sha256, arm=arm,
                                     teacher_action_mode=teacher_action_mode,
                                     diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
        save_receipt(deepcopy(receipt))
        receipts.append(receipt)
        event = {"phase": "explicit_success_partition_progress", "complete": False,
                "panel_sha256": panel_sha256(panel), "cohorts_completed": len(receipts),
                "completed_episodes": sum(len(r["case_ids"]) for r in receipts),
                "successes": sum(r["result"]["native_successes"] for r in receipts),
                "expected_episodes": panel["episode_count"], "optimizer_steps": 0}
        if diagnostic_initial_robot_refresh != ORIGINAL_INITIAL_REFRESH:
            event[INITIAL_REFRESH_FIELD] = diagnostic_initial_robot_refresh
        report(event)
    return _partition_result(panel, partition, receipts, checkpoint_sha256=checkpoint_sha256, arm=arm,
                             teacher_action_mode=teacher_action_mode,
                             diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)


def aggregate_partitions(panel, results, *, checkpoint_sha256, arm="bc", teacher_action_mode=DISTILL,
                         diagnostic_initial_robot_refresh=ORIGINAL_INITIAL_REFRESH):
    """Emit the original full-panel metric only after exact once-only coverage."""
    panel = validate_panel(panel)
    _execution_identity(checkpoint_sha256, arm, teacher_action_mode)
    validate_initial_refresh_panel(panel, diagnostic_initial_robot_refresh, arm=arm,
                                  teacher_action_mode=teacher_action_mode)
    if not isinstance(results, (list, tuple)) or not results:
        raise ValueError("success panel is missing partitions")
    by_index = {}
    for value in results:
        if not isinstance(value, dict):
            raise ValueError("invalid success partition result")
        expected = _partition_result(panel, value.get("partition"), value.get("cohort_receipts"),
                                     checkpoint_sha256=checkpoint_sha256, arm=arm,
                                     teacher_action_mode=teacher_action_mode,
                                     diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
        if canonical(value) != canonical(expected):
            raise ValueError("partition result identity/accounting differs")
        for receipt in expected["cohort_receipts"]:
            index = receipt["cohort_index"]
            if index in by_index:
                raise ValueError("success partitions contain overlapping or duplicate cohorts")
            by_index[index] = receipt["result"]
    if set(by_index) != set(range(len(panel["cohorts"]))):
        raise ValueError("success panel is missing a cohort partition")
    summary = summarize(panel, [by_index[i] for i in range(len(panel["cohorts"]))])
    return validate_result(panel, summary, checkpoint_sha256, allowed_arms=(arm,),
                           teacher_action_mode=teacher_action_mode,
                           diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)


def run_panel(panel, *, execute_cohort, report=lambda value: None,
              completed_receipts=(), expected_checkpoint_sha256=None, expected_arm="bc",
              save_receipt=lambda receipt: None, teacher_action_mode=DISTILL,
              diagnostic_initial_robot_refresh=ORIGINAL_INITIAL_REFRESH):
    """CPU-testable fail-closed orchestration; never omit failed/incomplete lanes.

    The default behavior is unchanged. Opt-in resume accepts only an ordered,
    verified prefix of whole completed cohorts and pins the exact checkpoint.
    This resumes final-test accounting, not simulator or partial-episode state.
    """
    panel = validate_panel(panel)
    validate_mode(teacher_action_mode, expected_arm)
    validate_initial_refresh_panel(panel, diagnostic_initial_robot_refresh, arm=expected_arm,
        teacher_action_mode=teacher_action_mode, has_resume=bool(completed_receipts))
    if completed_receipts or expected_checkpoint_sha256 is not None:
        partition = make_partition(panel, list(range(len(panel["cohorts"]))))
        value = run_partition(panel, partition, execute_cohort=execute_cohort,
            checkpoint_sha256=expected_checkpoint_sha256, arm=expected_arm,
            completed_receipts=completed_receipts, save_receipt=save_receipt, report=report,
            teacher_action_mode=teacher_action_mode,
            diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
        return aggregate_partitions(panel, [value], checkpoint_sha256=expected_checkpoint_sha256,
                                    arm=expected_arm, teacher_action_mode=teacher_action_mode,
                                    diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
    results = []
    for index, cohort in enumerate(panel["cohorts"]):
        report({"phase": "explicit_success_cohort_start", "cohort_index": index,
                "cohort_id": cohort["assignment"]["cohort_id"], "case_ids": cohort["case_ids"]})
        results.append(execute_cohort(cohort, index))
        # Reuse the full validation on a bounded prefix before advancing.
        prefix = deepcopy(panel)
        prefix["cohorts"] = prefix["cohorts"][:len(results)]
        prefix["episode_count"] = sum(len(row["case_ids"]) for row in prefix["cohorts"])
        checked = summarize(prefix, results)
        if checked.get("teacher_action_mode", DISTILL) != teacher_action_mode:
            raise ValueError("native teacher mode differs from requested diagnostic")
        initial_refresh_result_mode(checked, expected=diagnostic_initial_robot_refresh)
        report({"phase": "explicit_success_progress", "cohorts_completed": len(results),
                "completed_episodes": checked["completed_episodes"], "successes": checked["successes"],
                "expected_episodes": panel["episode_count"], "optimizer_steps": 0})
    return summarize(panel, results)
