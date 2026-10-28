"""Synthetic policy-switch tests; no live workload or capacity certification."""
from copy import deepcopy
import json

import pytest

from pipeline.success_rate import dispatch as policies
from pipeline.success_rate import scaled_teacher_benchmark as teacher
from pipeline.success_rate.__main__ import main
from pipeline.success_rate.dispatch import TeacherTransport, transport_for
from test.pipeline.test_scaled_teacher_benchmark import material, native_result
from test.deployment.rtx_actor.test_production_success_contract import accepted
from test.pipeline.test_scaled_success_dispatch import Transport, posts
from test.pipeline.test_scaled_success_qualification import evidence


def make_plan(material, **kwargs):
    panel, owners = material
    values = dict(policy="teacher", protocol=policies.FIXED_PROTOCOL,
        command_prefix="teacher-switch-test", shadow_checkpoint_sha256="b" * 64)
    values.update(kwargs)
    return policies.build_plan(panel, owners, **values)


def result_for(material, plan, i):
    return teacher.wrap_result(material[0], plan["commands"][i]["artifacts"][0]["value"],
                               native_result(material, i))


def receipt_for(material, plan, i):
    entry = plan["commands"][i]
    owner = plan["owners"][entry["owner_index"]]
    return dict(status="COMPLETE", payload_sha256=entry["payload_sha256"], receipt=dict(
        complete=True, purpose="test", optimizer_steps=0, source_sha256=owner["source_sha256"],
        native_pid=owner["native_pid"], native_initializations=1, profile="explicit_evaluation",
        same_kit=True, physics_pause_requested=True, kit_identity=123,
        command_id=entry["body"]["command_id"], worker_instance_id=owner["worker_instance_id"],
        resource_guard={"synthetic_cpu_only": True}, result=result_for(material, plan, i)))


def test_exact_teacher_native_payload_preserves_physical_cases(material):
    panel, owners = material
    before = deepcopy(panel)
    plan = make_plan(material)
    assert policies.validate_plan(plan) == plan and panel == before
    assert len(plan["commands"]) == 19
    assert plan["observation_mode"] == "privileged_state"
    for i, command in enumerate(plan["commands"]):
        body = command["body"]
        assert body["operation"] == "evaluate_scaled_teacher_panel"
        payload = body["payload"]
        assert set(payload) == {"panel_file", "panel_sha256", "benchmark_file", "benchmark_sha256",
            "checkpoint_sha256", "arm", "teacher_action_mode", "diagnostic_initial_robot_refresh", "output_name"}
        assert payload["teacher_action_mode"] == "raven_ppo" and payload["arm"] == "teacher"
        spec = command["artifacts"][0]["value"]
        assert spec["num_envs"] == (7 if i == 18 else 16)
        assert spec["owner"] == owners[i % 2]
        assert spec["renderer_calls"] == spec["student_forward_calls"] == spec["optimizer_steps"] == 0


@pytest.mark.parametrize("mutation", [dict(policy="unknown"), dict(protocol=policies.POOLED_PROTOCOL),
    dict(protocol="scaled-fixed295-b32-v1"), dict(config={}), dict(arm="bc"),
    dict(qualification_only=True), dict(qualification_evidence={}), dict(shadow_checkpoint_sha256=None)])
def test_no_silent_policy_protocol_or_capacity_fallback(material, mutation):
    with pytest.raises(ValueError):
        make_plan(material, **mutation)


def test_low_teacher_score_is_preserved_and_advisory_only(material):
    plan = make_plan(material)
    rows = [result_for(material, plan, i) for i in range(19)]
    result = policies.aggregate(plan, rows)
    assert result["successes"] == 19 and result["completed"] == 295
    assert result["success_rate"] == 19 / 295
    assert result["complete"] is True  # Low performance is not an incomplete run.
    assert result["sanity_check"] == dict(expected_success_rate_gt=.8, expectation_met=False,
        advisory_only=True, low_rate_requires_investigation=True, certifies_student_renderer=False)
    with pytest.raises(ValueError, match="partial"):
        policies.aggregate(plan, rows[:-1])
    rows[0]["benchmark"]["command_id"] = "other-run"
    with pytest.raises(ValueError, match="different"):
        policies.aggregate(plan, rows)


def test_student_teacher_switch_keeps_same_physical_benchmark(evidence, material):
    _, owners = material
    owners = deepcopy(owners)
    for owner in owners:
        owner["source_sha256"] = evidence["proof"]["source_sha256"]
    panel, config = evidence["panel"], evidence["config"]
    student = policies.build_plan(panel, owners, policy="student", protocol=policies.FIXED_PROTOCOL,
        config=config, arm="bc", qualification_evidence=evidence["proof"], command_prefix="student-matched")
    baseline = policies.build_plan(panel, owners, policy="teacher", protocol=policies.FIXED_PROTOCOL,
        shadow_checkpoint_sha256=config["checkpoint_sha256"], command_prefix="teacher-matched")
    assert student["panel"] == baseline["panel"] and student["owners"] == baseline["owners"]
    assert policies.validate_plan(student) == student
    for s, t in zip(student["commands"], baseline["commands"]):
        assert s["cohort_index"] == t["cohort_index"] and s["owner_index"] == t["owner_index"]
        assert s["body"]["payload"]["panel_sha256"] == t["body"]["payload"]["panel_sha256"]
        assert s["body"]["operation"] == "evaluate_scaled_rendered_panel"
        assert t["body"]["operation"] == "evaluate_scaled_teacher_panel"


@pytest.mark.parametrize("successes,met", [(236, False), (237, True)])
def test_expected_above_80_percent_is_strict_and_does_not_filter_cases(material, successes, met):
    plan = make_plan(material)
    remaining = successes
    rows = []
    for i, entry in enumerate(plan["commands"]):
        row = native_result(material, i)
        count = min(remaining, row["completed"])
        remaining -= count
        for j, outcome in enumerate(row["outcomes"]):
            outcome.update(native_success=j < count, exported_success=j < count)
        row.update(native_successes=count, exported_successes=count, original_raven_metric_successes=count)
        rows.append(teacher.wrap_result(material[0], entry["artifacts"][0]["value"], row))
    value = policies.aggregate(plan, rows)
    assert value["successes"] == successes and value["completed"] == 295
    assert value["sanity_check"]["expectation_met"] is met


class TeacherFakeTransport(Transport):
    def stage_artifacts(self, owner, artifacts):
        assert len(artifacts) == 2
        self.calls.append(("stage", owner["worker_instance_id"]))


def test_lost_acceptance_reconciles_without_duplicate_teacher_submission(material, tmp_path):
    plan = make_plan(material)
    transport = TeacherFakeTransport(plan, tmp_path)
    transport.lose_after = True
    runner = policies.coordinator(plan, tmp_path, transport, resource_validator=lambda *_: None)
    first = runner.tick()
    assert first["workers"][0]["status"] == "TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER"
    assert len(posts(transport)) == 1
    runner = policies.coordinator(plan, tmp_path, transport, resource_validator=lambda *_: None)
    second = runner.tick()
    assert second["workers"][0]["status"] == "IN_FLIGHT"
    assert len([p for p in posts(transport) if p[2].endswith("c000")]) == 1


def test_teacher_complete_receipt_requires_same_warm_owner_and_guard(material, tmp_path):
    plan = make_plan(material)
    guards = []
    runner = policies.coordinator(plan, tmp_path, TeacherFakeTransport(plan, tmp_path),
        resource_validator=lambda owner, receipt: guards.append((owner, receipt)))
    entry = plan["commands"][0]
    row = receipt_for(material, plan, 0)
    assert runner._validate_complete(entry, row) == row["receipt"]["result"]
    assert len(guards) == 1
    for key, value in (("same_kit", False), ("native_initializations", 2), ("native_pid", 0),
                       ("optimizer_steps", 1), ("command_id", "wrong")):
        bad = deepcopy(row)
        bad["receipt"][key] = value
        with pytest.raises(ValueError):
            runner._validate_complete(entry, bad)
    assert len(guards) == 1


def test_teacher_transport_never_requires_or_contacts_renderer(material):
    plan = make_plan(material)
    entry = plan["commands"][0]
    owner = plan["owners"][0]
    transport = object.__new__(TeacherTransport)
    calls = []
    def invoke(owner, action, **kwargs):
        assert action == "stage"
        calls.append(kwargs)
        return dict(staged=True, raw_sha256=kwargs["artifact"]["raw_sha256"])
    transport.invoke = invoke
    artifacts = plan["shared_artifacts"] + entry["artifacts"]
    transport.stage_artifacts(owner, artifacts)
    assert len(calls) == 2
    changed = deepcopy(artifacts)
    changed[1]["raw_sha256"] = "a" * 64
    with pytest.raises(ValueError):
        transport.stage_artifacts(owner, changed)
    with pytest.raises(ValueError):
        transport_for("unknown")


def test_cli_teacher_selection_and_summary(material, tmp_path, capsys):
    panel, owners = material
    for name, value in (("panel", panel), ("owners", owners)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    path = tmp_path / "plan.json"
    main(["plan", "--policy", "teacher", "--protocol", policies.FIXED_PROTOCOL,
          "--panel", str(tmp_path / "panel.json"), "--owners", str(tmp_path / "owners.json"),
          "--shadow-checkpoint-sha256", "b" * 64, "--command-prefix", "teacher-switch-test",
          "--output", str(path)])
    plan = json.loads(path.read_text())
    assert plan == make_plan(material)
    main(["validate-plan", "--plan", str(path)])
    paths = []
    for i in range(19):
        result = tmp_path / f"cohort-{i:03d}.json"
        result.write_text(json.dumps(result_for(material, plan, i)))
        paths.append(str(result))
    output = tmp_path / "summary.json"
    main(["aggregate", "--plan", str(path), "--results", *paths, "--output", str(output)])
    assert json.loads(output.read_text())["sanity_check"]["expectation_met"] is False
    capsys.readouterr()
