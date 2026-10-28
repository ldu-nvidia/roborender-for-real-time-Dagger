"""Explicit BC116 incident continuation, never a simulator or legacy resume.

The immutable 146-case prefix and one weaker, durable first-episode outcome
remain distinct. A new owner may restart only the unfinished case in its exact
original B2 cohort, after a before-action comparison against saved initial
evidence. This module contains no native or GPU operations.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

try:
    from .explicit_motion_panel import canonical, panel_sha256, completed_progress, validate_cohort_receipt
    from .rendered_student_success_contract import validate_config, validate_cohort_observations
    from .initial_robot_refresh_diagnostic import STUDENT_CLEAR_ONCE, validate_attestation
    from .success_diagnostic_core import FirstEpisodeAccounting
except ImportError:
    from explicit_motion_panel import canonical, panel_sha256, completed_progress, validate_cohort_receipt
    from rendered_student_success_contract import validate_config, validate_cohort_observations
    from initial_robot_refresh_diagnostic import STUDENT_CLEAR_ONCE, validate_attestation
    from success_diagnostic_core import FirstEpisodeAccounting

EVIDENCE_SCHEMA = "roborender.rendered-success-continuation-evidence.v1"
ADMISSION_SCHEMA = "roborender.rendered-success-continuation-admission.v1"
RESULT_SCHEMA = "roborender.rendered-success-continuation-result.v1"
GATE_SCHEMA = "roborender.rendered-success-continuation-initial-state.v1"
OPERATION = "evaluate_rendered_continuation"
AUDIT_SHA256 = "a803703d1583ad0d5328cfadcd0de1ddc7425399882fface1ce712cead0513a4"
USER_REQUEST = "please recover the connection and continue success rate calculation from where it dropped"
TRANSPORT = {"schema": "roborender.image3f.cached-reply.v1", "recovery_timeout_s": 75.0}
MISSING_CHECKS = ["terminal policy_unchanged equality", "terminal native goal metric equality",
    "complete rendered observation receipt with anchors and input/output bindings",
    "complete original cohort result/receipt"]
REQUIRED_FILES = {"panel.json", "observation-config.json", "resident-state.json",
    "cohort-073/native-motion-reset.json", "cohort-073/initial-state.json",
    "cohort-073/runtime-physics.json", "cohort-073/steps.jsonl", "render-timings/cohort-073.jsonl"} | {
    f"cohort-receipts/cohort-{index:03d}.json" for index in range(73)}


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def raw_digest(value):
    return hashlib.sha256(value).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_pinned(path, expected_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError("continuation document must be a bounded regular file")
    raw = path.read_bytes()
    if len(raw) > 16 << 20 or not _sha(expected_sha256) or raw_digest(raw) != expected_sha256:
        raise ValueError("continuation document raw bytes differ from admission")
    value = json.loads(raw)
    canonical(value)
    return value


def validate_evidence(document, panel, config):
    """Recheck original byte-bound artifacts and recompute partial accounting."""
    config = validate_config(config, panel)
    if (not isinstance(document, dict) or set(document) != {"schema", "audit_raw", "files"}
            or document["schema"] != EVIDENCE_SCHEMA or not isinstance(document["audit_raw"], str)
            or raw_digest(document["audit_raw"].encode()) != AUDIT_SHA256
            or not isinstance(document["files"], dict) or set(document["files"]) != REQUIRED_FILES):
        raise ValueError("continuation requires the exact original audited evidence bundle")
    audit = json.loads(document["audit_raw"])
    files = document["files"]
    for name, content in files.items():
        expected = audit["evidence_files"][name]
        if (not isinstance(content, str) or len(content.encode()) != expected["bytes"]
                or raw_digest(content.encode()) != expected["sha256"]):
            raise ValueError("original evidence file changed: " + name)
    original_panel = json.loads(files["panel.json"])
    original_config = json.loads(files["observation-config.json"])
    if (original_panel != panel or original_config != config
            or audit["checkpoint_sha256"] != config["checkpoint_sha256"]
            or panel["episode_count"] != 295 or len(panel["cohorts"]) != 148
            or len(panel["cohorts"][73]["case_ids"]) != 2
            or len(panel["cohorts"][-1]["case_ids"]) != 1):
        raise ValueError("continuation panel/checkpoint/rendering configuration differs")
    receipts = [json.loads(files[f"cohort-receipts/cohort-{index:03d}.json"]) for index in range(73)]
    progress = completed_progress(panel, receipts, checkpoint_sha256=config["checkpoint_sha256"], arm="bc",
        teacher_action_mode="distill", diagnostic_initial_robot_refresh=STUDENT_CLEAR_ONCE)
    if (progress != audit["validated_progress"] or progress["completed_episodes"] != 146
            or progress["successes"] != 49 or progress["cohorts_completed"] != 73):
        raise ValueError("immutable historical accepted prefix differs")
    for index, receipt in enumerate(receipts):
        validate_cohort_observations(receipt["result"], count=2, config=config, panel=panel, cohort_index=index)
    trace = [json.loads(line) for line in files["cohort-073/steps.jsonl"].splitlines()]
    timings = [json.loads(line) for line in files["render-timings/cohort-073.jsonl"].splitlines()]
    if len(trace) != 203 or len(timings) != 203:
        raise ValueError("original partial trace boundary differs")
    account = FirstEpisodeAccounting(2)
    events = []
    for step, (row, timing) in enumerate(zip(trace, timings)):
        if row["step"] != step or timing["step"] != step or timing["lanes_rendered"] != 2 or row["arm"] != "bc":
            raise ValueError("original partial native/rendered step sequence differs")
        lanes = row["lanes"]
        outcomes = account.update(step, [x["current_last_stage_completed_task_buf"] for x in lanes],
            row["exported_goals"], [x["reset_buf"] for x in lanes], [x["time_out_buf"] for x in lanes], lanes)
        if outcomes != row["completed_outcomes"]:
            raise ValueError("original first-episode accounting differs")
        events.extend(dict(trace_step=step, case_id=panel["cohorts"][73]["case_ids"][x["lane"]], outcome=x)
                      for x in outcomes)
    if (events != audit["partial_completion_events"] or len(events) != 1
            or events[0]["trace_step"] != 146 or events[0]["outcome"]["lane"] != 0
            or events[0]["outcome"]["native_success"] is not True
            or events[0]["outcome"]["exported_success"] is not True
            or audit["missing_checks"] != MISSING_CHECKS):
        raise ValueError("durable partial result or its weaker evidence class differs")
    reset = json.loads(files["cohort-073/native-motion-reset.json"])
    if (reset["assignment"] != panel["cohorts"][73]["assignment"]
            or reset["assignment_sha256"] != digest(reset["assignment"])
            or reset["sample_indices_forced"] is not False or reset["seed_reapplied_per_episode"] is not False):
        raise ValueError("original partial reset assignment differs")
    refresh = deepcopy(audit["partial_initial_refresh_attestation"])
    refresh.pop("phase", None)
    validate_attestation(refresh, num_envs=2, mode=STUDENT_CLEAR_ONCE)
    state = json.loads(files["resident-state.json"])
    command_id = "bc116-image3f-fixed295-20261024"
    command = state["commands"][command_id]["command"]
    if (command["operation"] != "evaluate_explicit_panel" or command["worker_instance_id"] != state["worker_instance_id"]
            or command["payload"]["checkpoint_sha256"] != config["checkpoint_sha256"]
            or state["source_sha256"] != audit["source_sha256"]):
        raise ValueError("historical owner/command identity differs")
    return dict(original_owner=dict(worker_instance_id=state["worker_instance_id"], command_id=command_id,
            source_sha256=state["source_sha256"]), panel_sha256=panel_sha256(panel),
        checkpoint_sha256=config["checkpoint_sha256"], observation_config_sha256=digest(config),
        audit_sha256=AUDIT_SHA256, historical_receipts=receipts,
        historical_receipt_raw_sha256=[audit["evidence_files"][f"cohort-receipts/cohort-{i:03d}.json"]["sha256"] for i in range(73)],
        partial=dict(cohort_index=73, case_id=events[0]["case_id"], outcome=events[0]["outcome"],
            completion_step=146, trace_steps=203, missing_checks=deepcopy(MISSING_CHECKS),
            trace_sha256=audit["evidence_files"]["cohort-073/steps.jsonl"]["sha256"],
            render_timings_sha256=audit["evidence_files"]["render-timings/cohort-073.jsonl"]["sha256"],
            native_reset=reset, initial_state=json.loads(files["cohort-073/initial-state.json"]),
            runtime_physics=json.loads(files["cohort-073/runtime-physics.json"]), initial_refresh_attestation=refresh))


def read_evidence(path, raw_sha256, panel, config):
    result = validate_evidence(read_pinned(path, raw_sha256), panel, config)
    result["evidence_sha256"] = raw_sha256
    return result


def validate_admission(admission, evidence, *, source_sha256, native_state, output_name):
    keys = {"schema", "evidence_sha256", "original_worker_instance_id", "original_command_id", "original_source_sha256",
        "fresh_worker_instance_id", "fresh_command_id", "fresh_source_sha256", "output_name", "user_request",
        "old_owner_reconciliation_sha256", "old_owner_process_absence_verified",
        "restart_unfinished_case_from_original_initial_conditions", "preserve_partial_case_as_weaker_evidence",
        "retain_all_h200_compute", "simulator_snapshot_restore", "transport_recovery"}
    if not isinstance(admission, dict) or set(admission) != keys or admission["schema"] != ADMISSION_SCHEMA:
        raise ValueError("separate explicit continuation admission required")
    expected = dict(evidence_sha256=evidence["evidence_sha256"], fresh_source_sha256=source_sha256,
        fresh_worker_instance_id=native_state["worker_instance_id"], output_name=output_name,
        user_request=USER_REQUEST, old_owner_process_absence_verified=True,
        restart_unfinished_case_from_original_initial_conditions=True,
        preserve_partial_case_as_weaker_evidence=True, retain_all_h200_compute=True,
        simulator_snapshot_restore=False, transport_recovery=TRANSPORT)
    expected.update({"original_" + name: value for name, value in evidence["original_owner"].items()})
    if (any(admission[name] != value for name, value in expected.items())
            or not _sha(admission["old_owner_reconciliation_sha256"])
            or not _sha(source_sha256) or native_state["source_sha256"] != source_sha256
            or source_sha256 == evidence["original_owner"]["source_sha256"]
            or admission["fresh_worker_instance_id"] == admission["original_worker_instance_id"]
            or not isinstance(admission["fresh_command_id"], str)
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", admission["fresh_command_id"]) is None
            or admission["fresh_command_id"] == admission["original_command_id"]):
        raise ValueError("continuation admission owner/source/evidence/authorization differs")
    if native_state["phase"] not in {"IDLE", "RUNNING"}:
        raise ValueError("continuation owner is busy or uncertain")
    if ((native_state["phase"] == "IDLE" and native_state.get("active_command_id") is not None)
            or (native_state["phase"] == "RUNNING" and (
                native_state.get("active_command_id") != admission["fresh_command_id"]
                or admission["fresh_command_id"] not in native_state["commands"]))):
        raise ValueError("continuation active command differs from exact admission")
    for command_id, row in native_state["commands"].items():
        command = row["command"]
        if command_id == admission["fresh_command_id"]:
            if (row["status"] != "RUNNING" or native_state["active_command_id"] != command_id
                    or command["operation"] != OPERATION or command["payload"]["output_name"] != output_name):
                raise ValueError("continuation command is not the exact active admitted command")
        elif (command["operation"] != "capture_initial_b2" or row["status"] != "COMPLETE"
                or row.get("receipt", {}).get("environment_step_calls") != 0):
            raise ValueError("continuation requires a fresh owner with no previous stepping command")
    return deepcopy(admission)


def compare_initial_boundary(evidence, *, native_reset, initial_state, runtime_physics,
                             initial_refresh_attestation, checkpoint_sha256):
    """Called after native reset/checkpoint load, before any policy action."""
    partial = evidence["partial"]
    actual = dict(native_reset=native_reset, initial_state=initial_state, runtime_physics=runtime_physics,
                  initial_refresh_attestation=initial_refresh_attestation)
    for field, value in actual.items():
        if canonical(value) != canonical(partial[field]):
            raise ValueError("continuation before-action original boundary mismatch: " + field)
    if checkpoint_sha256 != evidence["checkpoint_sha256"]:
        raise ValueError("continuation before-action checkpoint mismatch")
    validate_attestation(initial_refresh_attestation, num_envs=2, mode=STUDENT_CLEAR_ONCE)
    return dict(schema=GATE_SCHEMA, passed=True, original_cohort_index=73,
        checkpoint_sha256=checkpoint_sha256, compared_sha256={name: digest(value) for name, value in actual.items()},
        physical_lanes=2, scored_lanes=[1], unscored_companion_lanes=[0],
        environment_step_calls_before_gate=0, simulator_snapshot_restored=False)


def _case(panel, index, outcome, evidence_class, reference):
    spec = panel["cohorts"][index]
    lane = outcome["lane"]
    motion = spec["assignment"]["lanes"][lane]
    return dict(deepcopy(outcome), case_id=spec["case_ids"][lane], cohort_index=index,
        cohort_id=spec["assignment"]["cohort_id"], simulator_seed=spec["assignment"]["simulator_seed"],
        motion_id=motion["motion_id"], motion_sha256=motion["sha256"],
        evidence_class=evidence_class, evidence_reference=reference)


def merge_results(panel, config, evidence, admission, new_receipts, *, require_complete=True):
    """Score original IDs exactly once; a companion result never replaces 0146."""
    checkpoint = config["checkpoint_sha256"]
    cases = []
    for index, receipt in enumerate(evidence["historical_receipts"]):
        cases.extend(_case(panel, index, row, "historical_complete_rendered_cohort",
            evidence["historical_receipt_raw_sha256"][index]) for row in receipt["result"]["outcomes"])
    partial = evidence["partial"]
    cases.append(_case(panel, 73, partial["outcome"], "historical_partial_trace_missing_terminal_checks",
                       partial["trace_sha256"]))
    if not isinstance(new_receipts, list) or len(new_receipts) > 75:
        raise ValueError("new continuation receipts must be an ordered original-cohort prefix")
    for index, receipt in enumerate(new_receipts, 73):
        if receipt.get("cohort_index") != index:
            raise ValueError("new continuation receipts contain a gap, duplicate or changed original index")
        validate_cohort_receipt(panel, receipt, checkpoint_sha256=checkpoint, arm="bc",
            teacher_action_mode="distill", diagnostic_initial_robot_refresh=STUDENT_CLEAR_ONCE)
        validate_cohort_observations(receipt["result"], count=len(panel["cohorts"][index]["case_ids"]),
            config=config, panel=panel, cohort_index=index)
        if index == 73:
            expected_gate = compare_initial_boundary(evidence, checkpoint_sha256=checkpoint,
                **{name: partial[name] for name in ("native_reset", "initial_state", "runtime_physics", "initial_refresh_attestation")})
            if receipt["result"].get("continuation_initial_state_gate") != expected_gate:
                raise ValueError("restarted unfinished case lacks its before-action original-state gate")
        for row in receipt["result"]["outcomes"]:
            if index == 73 and row["lane"] == 0:
                continue
            kind = "fresh_unfinished_case_from_verified_original_initial_state" if index == 73 else "fresh_complete_rendered_cohort"
            cases.append(_case(panel, index, row, kind, digest(receipt)))
    expected_ids = [case for spec in panel["cohorts"] for case in spec["case_ids"]]
    cases.sort(key=lambda row: expected_ids.index(row["case_id"]))
    actual_ids = [row["case_id"] for row in cases]
    if actual_ids != expected_ids[:len(cases)] or len(set(actual_ids)) != len(cases):
        raise ValueError("continuation score must contain each original case exactly once")
    complete = len(cases) == 295
    if require_complete and not complete:
        raise ValueError("continuation still lacks original cases")
    strong = [row for row in cases if row["evidence_class"] != "historical_partial_trace_missing_terminal_checks"]
    successes = sum(row["native_success"] for row in cases)
    strong_successes = sum(row["native_success"] for row in strong)
    return dict(schema=RESULT_SCHEMA, complete=complete, historical_full295_contract_complete=False,
        panel_id=panel["panel_id"], panel_sha256=panel_sha256(panel), checkpoint_sha256=checkpoint,
        observation_config_sha256=digest(config), observation_mode="image3f_3view", arm="bc", purpose="test",
        expected_episodes=295, completed_episodes=len(cases), successes=successes, success_rate=successes / len(cases),
        strict_accepted_episodes=len(strong), strict_accepted_successes=strong_successes,
        strict_accepted_success_rate=strong_successes / len(strong), weaker_evidence_episodes=1,
        weaker_evidence_successes=1, weaker_evidence_missing_checks=deepcopy(MISSING_CHECKS),
        original_accepted_episodes=146, original_accepted_successes=49, new_scored_episodes=len(cases)-147,
        optimizer_steps=0, simulator_snapshot_restored=False, partial_companion_outcome_scored=False,
        original_owner=deepcopy(evidence["original_owner"]), fresh_worker_instance_id=admission["fresh_worker_instance_id"],
        fresh_command_id=admission["fresh_command_id"], fresh_source_sha256=admission["fresh_source_sha256"],
        evidence_sha256=evidence["evidence_sha256"], admission_sha256=digest(admission), transport_recovery=deepcopy(TRANSPORT),
        outcomes=cases, new_cohort_receipts=deepcopy(new_receipts))
