"""CPU-only diagnostic admission/routing tests; never load Isaac or real data."""
from test.deployment.source_paths import ACTOR_ROOT
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = ACTOR_ROOT
sys.path.insert(0, str(HERE))
import training_support_success_contract as contract
import initial_robot_refresh_diagnostic as refresh
from explicit_motion_panel import canonical, run_panel, summarize, validate_panel
from native_artifact_io import publish_json
from teacher_action_diagnostic import DISTILL, RAVEN_PPO, validate_mode
from test.deployment.rtx_actor.test_explicit_motion_panel import result as base_result


@pytest.fixture
def accepted(monkeypatch):
    # Synthetic selection bytes: real frozen constants are never rewritten.
    selection = {"schema": "roborender.bc-train-chunk-source-selection.v1", "selected_epoch": 116,
        "checkpoint": {"sha256": contract.CHECKPOINT_SHA256},
        "bindings": {"train_index": {"raw_sha256": contract.TRAIN_INDEX_RAW_SHA256}},
        "sampling": {"unit": "admitted_training_chunk", "sample_size": 200, "population_size": 2560,
            "seed": 20261023, "without_replacement": True, "deduplication_applied": False,
            "sample_order": "random_draw_order", "selected_index_ordinals": list(range(200))},
        "samples": []}
    for index in range(200):
        chunk = f"bc_chunk_{index:06d}"
        motion = f"motion-{index % 10}"
        selection["samples"].append({"draw_index": index, "admitted_train_index": index,
            "train_index_item": {"chunk_id": chunk}, "source_record": {"chunk_id": chunk,
                "split": "train", "selected_for_split": True, "overcollection": False,
                "teacher_only": True, "frames": 81, "motion_id": motion,
                "motion_sha256": hashlib.sha256(motion.encode()).hexdigest()}})
    monkeypatch.setattr(contract, "SOURCE_SELECTION_RAW_SHA256", hashlib.sha256(canonical(selection) + b"\n").hexdigest())
    panel = contract.build_panel(selection)
    monkeypatch.setattr(contract, "PANEL_SHA256", hashlib.sha256(canonical(panel)).hexdigest())
    monkeypatch.setattr(contract, "PANEL_RAW_SHA256", hashlib.sha256(canonical(panel) + b"\n").hexdigest())
    return selection, panel


def cohort_result(cohort):
    value = base_result(cohort)
    value["checkpoint_sha256"] = contract.CHECKPOINT_SHA256
    evidence = {"schema": "roborender.initial-robot-refresh-attestation.v2",
        "control": refresh.control_contract(refresh.STUDENT_CLEAR_ONCE), "num_envs": 2,
        "pending_before": [True, True], "pending_after": [False, False],
        "cleared_lanes": 2, "clear_calls": 1, "native_steps_added": 0}
    value.update(diagnostic_initial_robot_refresh=refresh.STUDENT_CLEAR_ONCE,
        initial_robot_refresh_control=refresh.control_contract(refresh.STUDENT_CLEAR_ONCE),
        initial_robot_refresh_attestation=evidence)
    return value


def complete_result(panel):
    value = summarize(panel, [cohort_result(row) for row in panel["cohorts"]])
    value.update(training_support_success_protocol=contract.protocol_identity(panel),
                 diagnostic_only=True, exact_historical_replay=False)
    return value


def test_panel_preserves_every_draw_duplicate_motion_and_case_mapping(accepted):
    selection, panel = accepted
    assert contract.require_panel(panel, raw_sha256=contract.PANEL_RAW_SHA256) == panel
    rows = [lane for cohort in panel["cohorts"] for lane in cohort["assignment"]["lanes"]]
    assert [r["motion_id"] for r in rows] == [r["source_record"]["motion_id"] for r in selection["samples"]]
    assert Counter(r["motion_id"] for r in rows) == {f"motion-{i}": 20 for i in range(10)}
    assert len(panel["cohorts"]) == len({r["assignment"]["simulator_seed"] for r in panel["cohorts"]}) == 100
    assert all(len(r["case_ids"]) == 2 for r in panel["cohorts"])
    assert contract.build_panel(selection) == panel
    assert contract.evaluation_options(panel) == {"teacher_action_mode": "distill",
        "diagnostic_initial_robot_refresh": refresh.STUDENT_CLEAR_ONCE}


@pytest.mark.parametrize("fault", ["draw", "duplicate_chunk", "deduplication", "motion", "checkpoint", "split"])
def test_changed_source_selection_is_never_admitted(accepted, fault):
    selection, _ = deepcopy(accepted)
    if fault == "draw": selection["samples"].reverse()
    elif fault == "duplicate_chunk": selection["samples"][1] = deepcopy(selection["samples"][0])
    elif fault == "deduplication": selection["sampling"]["deduplication_applied"] = True
    elif fault == "motion": selection["samples"][0]["source_record"]["motion_sha256"] = "f" * 64
    elif fault == "checkpoint": selection["checkpoint"]["sha256"] = "f" * 64
    else: selection["samples"][0]["source_record"]["split"] = "validation"
    with pytest.raises(ValueError): contract.build_panel(selection)


def test_source_reader_rejects_changed_bytes_raw_identity_and_symlink(accepted, tmp_path):
    selection, _ = accepted
    source = tmp_path / "source.json"
    source.write_bytes(canonical(selection) + b"\n")
    assert contract.read_source_selection(source, contract.SOURCE_SELECTION_RAW_SHA256) == selection
    with pytest.raises(ValueError): contract.read_source_selection(source, "f" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError): contract.read_source_selection(link, contract.SOURCE_SELECTION_RAW_SHA256)
    source.write_bytes(canonical(selection))
    with pytest.raises(ValueError): contract.read_source_selection(source, contract.SOURCE_SELECTION_RAW_SHA256)


@pytest.mark.parametrize("fault", ["case", "order", "seed", "motion", "path", "denominator", "panel_id", "raw"])
def test_changed_panel_fails_exact_diagnostic_gate(accepted, fault):
    _, panel = deepcopy(accepted)
    raw = contract.PANEL_RAW_SHA256
    if fault == "case": panel["cohorts"][0]["case_ids"][0] = "replacement"
    elif fault == "order": panel["cohorts"].reverse()
    elif fault == "seed": panel["cohorts"][0]["assignment"]["simulator_seed"] += 1
    elif fault == "motion": panel["cohorts"][0]["assignment"]["lanes"][0]["sha256"] = "f" * 64
    elif fault == "path": panel["cohorts"][0]["assignment"]["lanes"][0]["path"] = "/different/motion.npz"
    elif fault == "denominator": panel["episode_count"] = 199
    elif fault == "panel_id": panel["panel_id"] = "production-test"
    else: raw = "f" * 64
    with pytest.raises(ValueError): contract.evaluation_options(panel, raw_sha256=raw)


def test_diagnostic_result_is_complete_student_only_and_never_benchmark(accepted):
    _, panel = accepted
    result = complete_result(panel)
    assert contract.validate_training_support_result(panel, result, contract.CHECKPOINT_SHA256) == result
    assert result["successes"] == 100 and result["completed_episodes"] == 200
    protocol = result["training_support_success_protocol"]
    assert protocol["optimizer_steps"] == protocol["teacher_action_probability"] == 0
    assert protocol["max_episode_seconds"] == 8 and protocol["diagnostic_only"]
    assert not protocol["fixed295_benchmark"] and not protocol["production_promotion_allowed"]
    from production_success_contract import require_panel as production_panel
    from smoke_success_contract import require_panel_smoke
    for gate in (production_panel, require_panel_smoke):
        with pytest.raises(ValueError): gate(panel)


@pytest.mark.parametrize("fault", ["checkpoint", "teacher", "dagger", "optimizer", "partial", "success_count",
    "reset", "protocol", "replay", "production", "smoke", "case", "allowed_arms"])
def test_result_cannot_change_behavior_accounting_or_scope(accepted, fault):
    _, panel = accepted
    result = complete_result(panel)
    checkpoint, arms = contract.CHECKPOINT_SHA256, ("bc",)
    if fault == "checkpoint": checkpoint = "f" * 64
    elif fault in {"teacher", "dagger"}: result["arm"] = fault
    elif fault == "optimizer": result["optimizer_steps"] = 1
    elif fault == "partial": result["completed_episodes"] = 199
    elif fault == "success_count": result["successes"] = 200
    elif fault == "reset": result["diagnostic_initial_robot_refresh"] = "raven_original"
    elif fault == "protocol": result.pop("training_support_success_protocol")
    elif fault == "replay": result["exact_historical_replay"] = True
    elif fault == "production": result["production_success_protocol"] = {}
    elif fault == "smoke": result["smoke_success_protocol"] = {}
    elif fault == "case": result["outcomes"][0]["case_id"] = "replacement"
    else: arms = ("bc", "dagger")
    with pytest.raises(ValueError): contract.validate_training_support_result(panel, result, checkpoint, allowed_arms=arms)


def resident_admission(accepted, tmp_path, monkeypatch):
    selection, panel = accepted
    paths = {"panel.json": tmp_path / "panel.json", "selection.json": tmp_path / "selection.json"}
    paths["panel.json"].write_bytes(canonical(panel) + b"\n")
    paths["selection.json"].write_bytes(canonical(selection) + b"\n")
    tree = ast.parse((HERE / "resident_isaac_trainer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ResidentRavenBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
    checked = []
    def identity(path, expected, maximum=None):
        checked.append(Path(path))
        if Path(path).parent == Path("/diagnostic-assets"): return expected
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        return expected
    scope = {"os": os, "json": json, "Path": Path, "file_identity": identity}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "resident-diagnostic-admission", "exec"), scope)
    monkeypatch.setenv("RESIDENT_PRODUCTION_FULL295", "1")
    monkeypatch.setenv("RESIDENT_PIPELINE_MODE", "production")
    owner = SimpleNamespace(runtime_root=tmp_path, _consumed_outputs=set(), artifact=lambda name: paths[name],
                            motion_inventory_sha256=contract.MOTION_INVENTORY_SHA256)
    payload = {"panel_file": "panel.json", "panel_sha256": contract.PANEL_RAW_SHA256,
        "checkpoint_sha256": contract.CHECKPOINT_SHA256, "arm": "bc", "output_name": "training-support-diagnostic",
        "diagnostic_protocol": contract.SCHEMA, "training_support_selection_file": "selection.json",
        "training_support_selection_sha256": contract.SOURCE_SELECTION_RAW_SHA256,
        **contract.evaluation_options(panel)}
    return lambda: scope["validate"](owner, "evaluate_explicit_panel", payload), payload, checked, paths, owner


def test_historical_raytraced_student_panel_is_rejected_before_loading_checkpoint(accepted, tmp_path, monkeypatch):
    call, _, checked, paths, _ = resident_admission(accepted, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="requires rendered observations"):
        call()
    assert checked == [paths["panel.json"]]


def test_historical_raytraced_student_execution_cannot_publish_a_new_receipt(accepted, tmp_path, monkeypatch):
    selection, panel = accepted
    admission, payload, _, _, owner = resident_admission(accepted, tmp_path, monkeypatch)
    tree = ast.parse((HERE / "resident_isaac_trainer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ResidentRavenBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_run_explicit_test")
    calls, published = [], []
    def evaluate(trainer, supplied_panel, **kwargs):
        assert supplied_panel == panel and kwargs["training_support_selection"] == selection
        assert kwargs["checkpoint_sha256"] == contract.CHECKPOINT_SHA256
        assert kwargs["diagnostic_initial_robot_refresh"] == refresh.STUDENT_CLEAR_ONCE
        assert not kwargs["completed_receipts"] and not kwargs["reference_root"]
        calls.append(kwargs)
        return complete_result(panel)
    monkeypatch.setitem(sys.modules, "isaac6_motion_evaluator", SimpleNamespace(evaluate_explicit_panel=evaluate))
    owner.validate = lambda operation, supplied: admission()
    owner.source_sha256 = "e" * 64
    owner.trainer = SimpleNamespace(env=SimpleNamespace(simulator=SimpleNamespace(sim=SimpleNamespace(pause=lambda: None))))
    owner.scene_initializations, owner.phase_resets, owner.profile = 1, 0, "explicit_evaluation"
    owner._kit_unchanged = lambda: None
    scope = {"Path": Path, "json": json, "os": os,
        "atomic_write_json": lambda path, value, **kwargs: published.append((path, deepcopy(value)))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "resident-diagnostic-receipt", "exec"), scope)
    with pytest.raises(ValueError, match="requires rendered observations"):
        scope["_run_explicit_test"](owner, payload, lambda *a, **k: None)
    assert not calls and not published and not owner._consumed_outputs


@pytest.mark.parametrize("fault", ["protocol", "missing_source", "missing_protocol", "source_hash", "source_bytes",
    "checkpoint", "teacher", "dagger", "old_reset", "smoke_owner", "missing_owner_flag", "resume", "reference",
    "numeric_trace", "inventory", "missing_explicit_mode", "all_diagnostic_fields_missing", "used_output"])
def test_native_rejections_precede_checkpoint_or_scene_work(accepted, tmp_path, monkeypatch, fault):
    call, payload, checked, paths, owner = resident_admission(accepted, tmp_path, monkeypatch)
    if fault == "protocol": payload["diagnostic_protocol"] = "other"
    elif fault == "missing_source": payload.pop("training_support_selection_file")
    elif fault == "missing_protocol": payload.pop("diagnostic_protocol")
    elif fault == "source_hash": payload["training_support_selection_sha256"] = "f" * 64
    elif fault == "source_bytes": paths["selection.json"].write_bytes(b"{}\n")
    elif fault == "checkpoint": payload["checkpoint_sha256"] = "f" * 64
    elif fault in {"teacher", "dagger"}: payload["arm"] = fault
    elif fault == "old_reset": payload["diagnostic_initial_robot_refresh"] = "raven_original"
    elif fault == "smoke_owner": monkeypatch.setenv("RESIDENT_PIPELINE_MODE", "smoke")
    elif fault == "missing_owner_flag": monkeypatch.delenv("RESIDENT_PRODUCTION_FULL295")
    elif fault == "resume": payload.update(completed_receipts_file="prior", completed_receipts_sha256="f" * 64)
    elif fault == "reference": payload["reference_output_name"] = "teacher-reference"
    elif fault == "numeric_trace": payload["numeric_trace_steps"] = 0
    elif fault == "inventory": owner.motion_inventory_sha256 = "f" * 64
    elif fault == "missing_explicit_mode": payload.pop("teacher_action_mode")
    elif fault == "used_output": owner._consumed_outputs.add(payload["output_name"])
    else:
        for key in ("diagnostic_protocol", "training_support_selection_file", "training_support_selection_sha256"):
            payload.pop(key)
    with pytest.raises(ValueError): call()
    assert not any(p.parent == Path("/diagnostic-assets") for p in checked)


def test_future_evaluator_rejects_historical_raytraced_student_before_scene_setup(accepted, tmp_path):
    selection, panel = accepted
    tree = ast.parse((HERE / "isaac6_motion_evaluator.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_explicit_panel")
    prepared = []
    class OwnedPath(type(Path())):
        def is_relative_to(self, *others):
            return super().is_relative_to(tmp_path if others == ("/dagger-runtime",) else others[0])
    class FakeEvaluator:
        def evaluate_cohort(self, assignment, **kwargs):
            pytest.fail("historical raytraced student execution cannot reach a native cohort")
    scope = {"validate_panel": validate_panel, "validate_mode": validate_mode,
        "require_corrected_execution": refresh.require_corrected_execution,
        "validate_initial_refresh_panel": refresh.validate_panel_mode,
        "STUDENT_CLEAR_ONCE": refresh.STUDENT_CLEAR_ONCE, "SMOKE_STUDENT_CLEAR_ONCE": refresh.SMOKE_STUDENT_CLEAR_ONCE,
        "DISTILL": DISTILL, "RAVEN_PPO": RAVEN_PPO, "Path": OwnedPath,
        "ExplicitMotionSuccessEvaluator": FakeEvaluator, "publish_json": publish_json, "run_panel": run_panel}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "native-diagnostic-evaluator", "exec"), scope)
    output = tmp_path / "fresh-diagnostic"
    kwargs = dict(root=output, checkpoint_path="/diagnostic-assets/frozen.pt", checkpoint_sha256=contract.CHECKPOINT_SHA256,
        prepare_cohort=prepared.append, report=lambda row: None, arm="bc", training_support_selection=selection,
        **contract.evaluation_options(panel))
    with pytest.raises(ValueError, match="requires rendered observation configuration"):
        scope["evaluate_explicit_panel"](SimpleNamespace(), panel, **kwargs)
    assert not prepared and not output.exists()
    # Historical evidence remains readable; it is not new execution authority.
    value = complete_result(panel)
    assert contract.validate_training_support_result(panel, value, contract.CHECKPOINT_SHA256) == value
    # Direct evaluator calls cannot bypass the resident preflight either.
    for change in ({"checkpoint_sha256": "f" * 64}, {"arm": "teacher"},
                   {"training_support_selection": None}, {"reference_root": "/teacher"},
                   {"numeric_trace_steps": 0}, {"completed_receipts": [{}]},
                   {"diagnostic_initial_robot_refresh": "raven_original"},
                   {"teacher_action_mode": "raven_ppo"}):
        altered = {**kwargs, "root": tmp_path / "must-not-start", **change}
        with pytest.raises(ValueError):
            scope["evaluate_explicit_panel"](SimpleNamespace(), panel, **altered)
        assert not (tmp_path / "must-not-start").exists()
        assert not prepared
