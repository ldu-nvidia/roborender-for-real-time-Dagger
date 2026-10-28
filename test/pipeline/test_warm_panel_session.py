from test.deployment.source_paths import PIPELINE_ROOT
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.evaluation.beta_small_batch_panel import canonical, make_panel, summarize
from pipeline.evaluation.warm_panel_session import CHECKPOINT_PATHS, SCHEMA, read_request, run_pair, validate_request


def inputs():
    panel = make_panel("a" * 64, smoke=True)
    request = {"schema": SCHEMA, "panel_sha256": hashlib.sha256(canonical(panel)).hexdigest(),
               "cohort_limit": 2, "checkpoint_sha256": {"bc": "b" * 64, "dagger": "c" * 64}}
    return panel, request


def result(panel, arm, checkpoint):
    cohorts = [{"seed": cohort["seed"], "complete": True, "completed": 5,
                "optimizer_steps": 0, "policy_unchanged": True, "accounting_disagreements": 0,
                "outcomes": [{"lane": i, "exported_success": arm == "dagger" and i == 2,
                              "steps": 10, "timeout": False} for i in range(5)]}
               for cohort in panel["cohorts"]]
    return {**summarize(panel, cohorts), "checkpoint_sha256": checkpoint,
            "policy_unchanged": True, "paired_initial_states": arm == "dagger",
            "runtime_physics_sha256": "d" * 64, "goal_accounting_version": panel["goal_accounting_version"]}


def test_two_frozen_policies_reuse_one_owner_with_reference_state_binding(tmp_path):
    panel, request = inputs()
    owner, calls = object(), []

    def evaluate(**kwargs):
        calls.append((owner, kwargs))
        kwargs["output"].mkdir()
        if kwargs["arm"] == "dagger":
            assert kwargs["reference"] == calls[0][1]["output"]
            assert kwargs["reference"].exists()
        return result(panel, kwargs["arm"], kwargs["checkpoint_sha256"])

    rows, receipt = run_pair(request, panel, tmp_path, evaluate)
    assert len(calls) == 2 and calls[0][0] is calls[1][0]
    assert [(row[1]["arm"], row[1]["checkpoint"]) for row in calls] == list(CHECKPOINT_PATHS.items())
    assert calls[0][1]["reference"] is None
    assert receipt["complete"] and receipt["optimizer_steps"] == 0
    assert rows["bc"]["successes"] == 0 and rows["dagger"]["successes"] == 2
    assert receipt["receipt_audit"]["paired_outcomes"]["candidate_only"] == 2


@pytest.mark.parametrize("change", ["field", "hash", "panel", "partial", "boolean_count"])
def test_request_rejects_changed_identity_or_partial_panel(change):
    panel, request = inputs()
    if change == "field": request["command"] = "anything"
    if change == "hash": request["checkpoint_sha256"]["dagger"] = "invalid"
    if change == "panel": request["panel_sha256"] = "e" * 64
    if change == "partial": request["cohort_limit"] = 1
    if change == "boolean_count": request["cohort_limit"] = True
    with pytest.raises(ValueError):
        validate_request(request, panel)


def test_canonical_request_and_raw_digest_must_match(tmp_path):
    panel, request = inputs()
    path = tmp_path / "request.json"
    path.write_bytes(canonical(request))
    assert read_request(path, hashlib.sha256(path.read_bytes()).hexdigest(), panel) == request
    with pytest.raises(ValueError, match="identity"):
        read_request(path, "a" * 64, panel)
    path.write_text(json.dumps(request, indent=2))
    with pytest.raises(ValueError, match="canonical"):
        read_request(path, hashlib.sha256(path.read_bytes()).hexdigest(), panel)


def test_failure_or_existing_outputs_never_replay_a_panel(tmp_path):
    panel, request = inputs()
    calls = []

    def fail(**kwargs):
        calls.append(kwargs["arm"])
        kwargs["output"].mkdir()
        raise RuntimeError("native failure after a physical step")

    with pytest.raises(RuntimeError, match="native failure"):
        run_pair(request, panel, tmp_path, fail)
    assert calls == ["bc"]
    with pytest.raises(RuntimeError, match="never replay"):
        run_pair(request, panel, tmp_path, fail)
    assert calls == ["bc"]


@pytest.mark.parametrize("change", ["wrong_checkpoint", "changed_weights", "trained", "reset_mismatch", "physics"])
def test_warm_comparison_rejects_policy_and_reset_errors(tmp_path, change):
    panel, request = inputs()

    def evaluate(**kwargs):
        kwargs["output"].mkdir()
        row = result(panel, kwargs["arm"], kwargs["checkpoint_sha256"])
        if kwargs["arm"] == "dagger":
            key, value = {"wrong_checkpoint": ("checkpoint_sha256", "e" * 64),
                          "changed_weights": ("policy_unchanged", False),
                          "trained": ("optimizer_steps", 1),
                          "reset_mismatch": ("paired_initial_states", False),
                          "physics": ("runtime_physics_sha256", "e" * 64)}[change]
            row[key] = value
        return row

    with pytest.raises((RuntimeError, ValueError)):
        run_pair(request, panel, tmp_path, evaluate)


def test_native_callback_hot_loads_correct_policy_and_restores_environment(monkeypatch, tmp_path):
    """Exercise the actual native adapter with a CPU stand-in for Isaac."""
    panel, request = inputs()
    source = PIPELINE_ROOT / "deployment/rtx_actor/isaac6_panel_evaluator.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WarmPanelEvaluator")
    calls = []

    class Base:
        def evaluate_panel(self, output, reference):
            arm = os.environ["DIAGNOSTIC_ARM"]
            assert os.environ["DIAGNOSTIC_CHECKPOINT_PATH"] == CHECKPOINT_PATHS[arm]
            digest = os.environ["DIAGNOSTIC_CHECKPOINT_SHA256"]
            calls.append((id(self.env), arm, reference))
            output.mkdir()
            return result(panel, arm, digest)

    def native_path(name):
        if str(name) == "/dagger-runtime": return tmp_path
        if str(name) == "/panel.json":
            return SimpleNamespace(read_bytes=lambda: canonical(panel))
        return Path(name)

    namespace = {"BetaPanelEvaluator": Base, "Path": native_path, "os": os, "json": json,
                 "validate_panel": lambda value: value,
                 "read_request": lambda *args: copy.deepcopy(request), "run_pair": run_pair,
                 "canonical": canonical, "time": __import__("time"), "emit": lambda row: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    worker = namespace["WarmPanelEvaluator"]()
    worker.env = SimpleNamespace(simulator=SimpleNamespace(sim=object()))
    monkeypatch.setenv("DIAGNOSTIC_ARM", "previous")
    monkeypatch.delenv("DIAGNOSTIC_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("DIAGNOSTIC_CHECKPOINT_SHA256", "previous-hash")
    monkeypatch.setenv("WARM_PANEL_REQUEST_SHA256", "unused-by-stand-in")
    worker.train()
    receipt = json.loads((tmp_path / "warm-session.json").read_text())
    assert [row[1] for row in calls] == ["bc", "dagger"] and calls[0][0] == calls[1][0]
    assert calls[1][2] == str(tmp_path / "bc")
    assert receipt["native_initializations"] == receipt["checkpoint_switches"] == 1
    assert os.environ["DIAGNOSTIC_ARM"] == "previous"
    assert "DIAGNOSTIC_CHECKPOINT_PATH" not in os.environ
    assert os.environ["DIAGNOSTIC_CHECKPOINT_SHA256"] == "previous-hash"
