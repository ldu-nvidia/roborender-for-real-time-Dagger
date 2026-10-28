"""CPU-only tests: these synthetic receipts do not qualify a native owner."""
from test.deployment.source_paths import ACTOR_ROOT
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = ACTOR_ROOT
sys.path.insert(0, str(HERE))
import rendered_success_continuation as continuation
import rendered_success_observations as observations
import initial_robot_refresh_diagnostic as refresh
from explicit_motion_panel import canonical, make_cohort_receipt, completed_progress
from success_diagnostic_core import FirstEpisodeAccounting
from test.deployment.rtx_actor.test_rendered_student_success_contract import accepted, rendering_config, cohort_result
from test.deployment.rtx_actor.test_rendered_student_native_routing import evaluator_function

_BUNDLE_CACHE = []


@pytest.fixture
def bundle(accepted, monkeypatch):
    if _BUNDLE_CACHE:
        value = deepcopy(_BUNDLE_CACHE[0])
        monkeypatch.setattr(continuation, "AUDIT_SHA256", continuation.raw_digest(value.document["audit_raw"].encode()))
        return value
    panel = accepted
    config = rendering_config(panel)
    receipts = []
    for index in range(73):
        result = cohort_result(panel, index, config=config, successes=[index * 2 < 49, index * 2 + 1 < 49])
        receipts.append(make_cohort_receipt(panel, index, result, checkpoint_sha256=config["checkpoint_sha256"],
            diagnostic_initial_robot_refresh=refresh.STUDENT_CLEAR_ONCE))
    state = dict(worker_instance_id="original-owner", source_sha256="a" * 64, commands={
        "bc116-image3f-fixed295-20261024": dict(command=dict(operation="evaluate_explicit_panel",
            worker_instance_id="original-owner", payload=dict(checkpoint_sha256=config["checkpoint_sha256"])))})
    assignment = panel["cohorts"][73]["assignment"]
    reset = dict(schema="roborender.explicit-motion-success-reset.v1",
        assignment=assignment, assignment_sha256=continuation.digest(assignment),
        sample_indices_forced=False, seed_reapplied_per_episode=False, lanes=[
            dict(assignment["lanes"][i], sample_idx=i+5, sample_idx_in_seq=i+4, reset_rng_sha256="e" * 64)
            for i in range(2)])
    initial = dict(actor_obs=dict(dtype="float32", shape=[2, 113], sha256="f" * 64))
    physics = dict(physics_dt=.005, physics_unchanged=True)
    initial_refresh = receipts[0]["result"]["initial_robot_refresh_attestation"]
    account, traces, timings, events = FirstEpisodeAccounting(2), [], [], []
    for step in range(203):
        lanes = [dict(current_last_stage_completed_task_buf=(step == 146 and lane == 0),
            reset_buf=int(step == 146 and lane == 0), time_out_buf=False) for lane in range(2)]
        goals = [step == 146, False]
        outcomes = account.update(step, [row["current_last_stage_completed_task_buf"] for row in lanes],
            goals, [row["reset_buf"] for row in lanes], [False, False], lanes)
        traces.append(dict(step=step, arm="bc", lanes=lanes, exported_goals=goals, completed_outcomes=outcomes))
        timings.append(dict(step=step, lanes_rendered=2))
        events.extend(dict(trace_step=step, case_id=panel["cohorts"][73]["case_ids"][x["lane"]], outcome=x)
                      for x in outcomes)
    documents = {"panel.json": panel, "observation-config.json": config, "resident-state.json": state,
        "cohort-073/native-motion-reset.json": reset, "cohort-073/initial-state.json": initial,
        "cohort-073/runtime-physics.json": physics}
    documents.update({f"cohort-receipts/cohort-{i:03d}.json": r for i, r in enumerate(receipts)})
    files = {name: canonical(value).decode() + "\n" for name, value in documents.items()}
    files["cohort-073/steps.jsonl"] = "".join(canonical(row).decode() + "\n" for row in traces)
    files["render-timings/cohort-073.jsonl"] = "".join(canonical(row).decode() + "\n" for row in timings)
    audit = dict(evidence_files={name: dict(bytes=len(value.encode()), sha256=continuation.raw_digest(value.encode()))
                               for name, value in files.items()}, checkpoint_sha256=config["checkpoint_sha256"],
        source_sha256=state["source_sha256"], validated_progress=completed_progress(panel, receipts,
            checkpoint_sha256=config["checkpoint_sha256"], diagnostic_initial_robot_refresh=refresh.STUDENT_CLEAR_ONCE),
        partial_completion_events=events, missing_checks=continuation.MISSING_CHECKS,
        partial_initial_refresh_attestation=initial_refresh)
    audit_raw = canonical(audit).decode() + "\n"
    monkeypatch.setattr(continuation, "AUDIT_SHA256", continuation.raw_digest(audit_raw.encode()))
    document = dict(schema=continuation.EVIDENCE_SCHEMA, audit_raw=audit_raw, files=files)
    evidence = continuation.validate_evidence(document, panel, config)
    evidence["evidence_sha256"] = continuation.digest(document)
    fresh_state = dict(worker_instance_id="fresh-owner", source_sha256="c" * 64, phase="IDLE", commands={}, active_command_id=None)
    admission = dict(schema=continuation.ADMISSION_SCHEMA, evidence_sha256=evidence["evidence_sha256"],
        original_worker_instance_id="original-owner", original_source_sha256="a" * 64,
        original_command_id="bc116-image3f-fixed295-20261024", fresh_worker_instance_id="fresh-owner",
        fresh_source_sha256="c" * 64, fresh_command_id="fresh-continuation", output_name="fresh-continuation",
        user_request=continuation.USER_REQUEST, old_owner_reconciliation_sha256="d" * 64,
        old_owner_process_absence_verified=True, restart_unfinished_case_from_original_initial_conditions=True,
        preserve_partial_case_as_weaker_evidence=True, retain_all_h200_compute=True,
        simulator_snapshot_restore=False, transport_recovery=continuation.TRANSPORT)
    value = SimpleNamespace(panel=panel, config=config, document=document, evidence=evidence,
                            admission=admission, state=fresh_state)
    _BUNDLE_CACHE.append(deepcopy(value))
    return value


def gate(bundle):
    return continuation.compare_initial_boundary(bundle.evidence, checkpoint_sha256=bundle.config["checkpoint_sha256"],
        **{key: bundle.evidence["partial"][key] for key in (
            "native_reset", "initial_state", "runtime_physics", "initial_refresh_attestation")})


def new_receipt(bundle, index, companion_success=False):
    successes = [companion_success, False] if index == 73 else [False] * len(bundle.panel["cohorts"][index]["case_ids"])
    row = cohort_result(bundle.panel, index, config=bundle.config, successes=successes)
    if index == 73:
        row["continuation_initial_state_gate"] = gate(bundle)
    return make_cohort_receipt(bundle.panel, index, row, checkpoint_sha256=bundle.config["checkpoint_sha256"],
                              diagnostic_initial_robot_refresh=refresh.STUDENT_CLEAR_ONCE)


@pytest.mark.parametrize("file", ["cohort-receipts/cohort-002.json", "cohort-073/steps.jsonl", "cohort-073/initial-state.json"])
def test_historical_raw_bytes_cannot_be_changed(bundle, file):
    value = deepcopy(bundle.document)
    value["files"][file] += " "
    with pytest.raises(ValueError, match="original evidence file changed"):
        continuation.validate_evidence(value, bundle.panel, bundle.config)


@pytest.mark.parametrize("fault", ["owner", "source", "authorization", "snapshot", "transport", "old_owner", "prior_step", "active_command"])
def test_admission_rejects_stale_or_unapproved_owner(bundle, fault):
    admission, state = deepcopy(bundle.admission), deepcopy(bundle.state)
    if fault == "owner": admission["fresh_worker_instance_id"] = "other-owner"
    if fault == "source": admission["fresh_source_sha256"] = "e" * 64
    if fault == "authorization": admission["user_request"] = "watch only"
    if fault == "snapshot": admission["simulator_snapshot_restore"] = True
    if fault == "transport": admission["transport_recovery"] = None
    if fault == "old_owner": admission["old_owner_process_absence_verified"] = False
    if fault == "prior_step":
        state["commands"]["earlier"] = dict(status="COMPLETE", command=dict(operation="capture_initial_b2"),
                                               receipt=dict(environment_step_calls=1))
    if fault == "active_command":
        state.update(phase="RUNNING", active_command_id="wrong-command")
        state["commands"][admission["fresh_command_id"]] = dict(status="RUNNING", command=dict(
            operation=continuation.OPERATION, payload=dict(output_name=admission["output_name"])))
    with pytest.raises(ValueError):
        continuation.validate_admission(admission, bundle.evidence, source_sha256=state["source_sha256"],
                                        native_state=state, output_name=admission["output_name"])


def test_admission_allows_only_exact_zero_step_bootstrap_and_current_command(bundle):
    state = deepcopy(bundle.state)
    state["commands"]["bootstrap"] = dict(status="COMPLETE", command=dict(operation="capture_initial_b2"),
                                            receipt=dict(environment_step_calls=0))
    kwargs = dict(source_sha256=state["source_sha256"], native_state=state, output_name=bundle.admission["output_name"])
    assert continuation.validate_admission(bundle.admission, bundle.evidence, **kwargs) == bundle.admission
    state.update(phase="RUNNING", active_command_id=bundle.admission["fresh_command_id"])
    state["commands"][state["active_command_id"]] = dict(status="RUNNING", command=dict(
        operation=continuation.OPERATION, payload=dict(output_name=bundle.admission["output_name"])))
    assert continuation.validate_admission(bundle.admission, bundle.evidence, **kwargs) == bundle.admission


@pytest.mark.parametrize("field", ["native_reset", "initial_state", "runtime_physics", "initial_refresh_attestation", "checkpoint"])
def test_original_initial_gate_rejects_every_changed_component(bundle, field):
    actual = {key: deepcopy(bundle.evidence["partial"][key]) for key in (
        "native_reset", "initial_state", "runtime_physics", "initial_refresh_attestation")}
    checkpoint = bundle.config["checkpoint_sha256"]
    if field == "checkpoint": checkpoint = "f" * 64
    else: actual[field]["changed"] = True
    with pytest.raises(ValueError, match="before-action"):
        continuation.compare_initial_boundary(bundle.evidence, checkpoint_sha256=checkpoint, **actual)


def test_exact295_disjoint_scoring_keeps_historical_partial_and_all_failures(bundle):
    receipts = [new_receipt(bundle, index) for index in range(73, 148)]
    result = continuation.merge_results(bundle.panel, bundle.config, bundle.evidence, bundle.admission, receipts)
    assert result["completed_episodes"] == 295 and result["strict_accepted_episodes"] == 294
    assert result["successes"] == 50 and result["strict_accepted_successes"] == 49
    assert result["weaker_evidence_episodes"] == 1 and not result["historical_full295_contract_complete"]
    assert not result["partial_companion_outcome_scored"] and result["new_scored_episodes"] == 148
    assert result["outcomes"][146]["native_success"] is True
    assert result["outcomes"][147]["native_success"] is False
    assert len({row["case_id"] for row in result["outcomes"]}) == 295
    receipts[0] = new_receipt(bundle, 73, companion_success=True)
    changed = continuation.merge_results(bundle.panel, bundle.config, bundle.evidence, bundle.admission, receipts)
    assert changed["successes"] == result["successes"]
    with pytest.raises(ValueError, match="gap, duplicate"):
        continuation.merge_results(bundle.panel, bundle.config, bundle.evidence, bundle.admission, receipts[1:])


def test_continuation_native_route_preserves_original_indices_and_cached_transport(bundle, tmp_path, monkeypatch):
    prepared, actions, adapters = [], [], []
    trainer = SimpleNamespace(env=object(), _continuation_initial_evidence={"stale": True})
    class Adapter:
        def __init__(self, owner, **kwargs):
            assert owner is trainer
            self.binding = kwargs["evaluation_binding"]
            self.index = self.binding["cohort_index"]
            assert kwargs["anchor_seed"] == (bundle.config["anchors"]["seed"] + self.index) % (2**32)
            kwargs["client_factory"](0)
            self.closed = False
            adapters.append(self)
        def close(self): self.closed = True
    class Evaluator:
        def evaluate_cohort(self, assignment, **kwargs):
            index = self._rendered_success_evaluation_binding["cohort_index"]
            assert assignment == bundle.panel["cohorts"][index]["assignment"]
            assert hasattr(self, "_continuation_initial_evidence") == (index == 73)
            actions.append(index)
            return new_receipt(bundle, index)["result"]
    def client(host, port, **kwargs):
        assert kwargs == dict(transport_recovery=continuation.TRANSPORT["schema"], recovery_timeout_s=75.0)
        return object()
    monkeypatch.setattr(observations, "GeneratedSuccessObservations", Adapter)
    monkeypatch.setitem(sys.modules, "pipeline.dagger.services.render_student", SimpleNamespace(_Image3FLoopbackClient=client))
    execute = evaluator_function(tmp_path, Evaluator)
    result = execute(trainer, bundle.panel, root=tmp_path / "new-attempt", checkpoint_path="/readonly/selected.pt",
        checkpoint_sha256=bundle.config["checkpoint_sha256"], prepare_cohort=prepared.append,
        report=lambda row: None, rendered_observation_config=bundle.config, arm="bc", teacher_action_mode="distill",
        diagnostic_initial_robot_refresh=refresh.STUDENT_CLEAR_ONCE, _continuation=(bundle.evidence, bundle.admission))
    assert actions == list(range(73, 148)) and all(adapter.closed for adapter in adapters)
    assert len(prepared[0]["lanes"]) == 2 and len(prepared[-1]["lanes"]) == 1
    assert result["completed_episodes"] == 295 and result["strict_accepted_episodes"] == 294


def test_native_before_action_hook_and_historical_resume_guard_remain_explicit():
    tree = ast.parse((HERE / "isaac6_motion_evaluator.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExplicitMotionSuccessEvaluator")
    hook = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_diagnostic_after_reset")
    text = ast.unparse(hook)
    assert "compare_initial_boundary" in text and "initial_state(self.env, obs)" in text
    assert "runtime_physics_evidence(self.env)" in text and "DIAGNOSTIC_CHECKPOINT_SHA256" in text
    with pytest.raises(ValueError):
        refresh.validate_mode(refresh.STUDENT_CLEAR_ONCE, arm="bc", teacher_action_mode="distill", has_resume=True)


@pytest.mark.parametrize("fault", [None, "reset_rng", "initial_state", "physics", "checkpoint"])
def test_actual_native_reset_hook_rejects_mismatch_before_policy_action(bundle, tmp_path, monkeypatch, fault):
    partial = deepcopy(bundle.evidence["partial"])
    assignment = bundle.panel["cohorts"][73]["assignment"]
    actual_initial, actual_physics = deepcopy(partial["initial_state"]), deepcopy(partial["runtime_physics"])
    resets = {i: {name: partial["native_reset"]["lanes"][i][name] for name in (
        "sample_idx", "sample_idx_in_seq", "reset_rng_sha256")} for i in range(2)}
    checkpoint = bundle.config["checkpoint_sha256"]
    if fault == "reset_rng": resets[1]["reset_rng_sha256"] = "0" * 64
    if fault == "initial_state": actual_initial["actor_obs"]["sha256"] = "0" * 64
    if fault == "physics": actual_physics["physics_dt"] = .01
    if fault == "checkpoint": checkpoint = "0" * 64
    monkeypatch.setenv("DIAGNOSTIC_CHECKPOINT_SHA256", checkpoint)
    monkeypatch.setitem(sys.modules, "isaac6_success_diagnostic", SimpleNamespace(
        initial_state=lambda env, obs: actual_initial, digest=lambda value: value,
        runtime_physics_evidence=lambda env: actual_physics))
    owner = SimpleNamespace(motion_assignment=assignment, _first_native_resets=resets,
        _continuation_initial_evidence=bundle.evidence, env=SimpleNamespace(
            config=SimpleNamespace(domain_rand=SimpleNamespace(image_augmentation=SimpleNamespace(enabled=False))),
            simulator=SimpleNamespace(native_motion_assignment_sha256=continuation.digest(assignment)),
            reset_dataset=SimpleNamespace(motion_files=[row["path"] for row in assignment["lanes"]],
                sample_idx_based_on_demo_clip_number=[(0, 100), (0, 100)]),
            _diagnostic_initial_refresh_attestation=partial["initial_refresh_attestation"]))
    tree = ast.parse((HERE / "isaac6_motion_evaluator.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExplicitMotionSuccessEvaluator")
    hook = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_diagnostic_after_reset")
    published, policy_actions = [], []
    scope = dict(assert_motion_environment=lambda config, motion: motion, hashlib=__import__("hashlib"),
        canonical=canonical, torch=SimpleNamespace(initial_seed=lambda: assignment["simulator_seed"]), os=os,
        publish_json=lambda path, value: published.append((path.name, deepcopy(value))))
    exec(compile(ast.Module(body=[hook], type_ignores=[]), "actual-reset-hook", "exec"), scope)
    def attempt():
        scope["_diagnostic_after_reset"](owner, {}, tmp_path)
        policy_actions.append("first-policy-action")
    if fault:
        with pytest.raises(ValueError, match="before-action"): attempt()
        assert not policy_actions and not hasattr(owner, "_continuation_initial_state_gate")
    else:
        attempt()
        assert owner._continuation_initial_state_gate == gate(bundle)
        assert published[-1][0] == "continuation-initial-state-gate.json"
        assert len(policy_actions) == 1


def test_pinned_documents_reject_changed_bytes_and_symlinks(tmp_path):
    path = tmp_path / "document.json"
    path.write_bytes(b"{}\n")
    raw_sha = continuation.raw_digest(path.read_bytes())
    assert continuation.read_pinned(path, raw_sha) == {}
    with pytest.raises(ValueError): continuation.read_pinned(path, "a" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular file"): continuation.read_pinned(link, raw_sha)
