"""No GPU imports: scene budgets, input identity, failure/no-retry semantics."""
from test.deployment.source_paths import ACTOR_ROOT, REPO_ROOT
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(ACTOR_ROOT))
from warm_camera_diagnostic import SCHEMA, canonical, read_plan, run_stages, validate_plan


def plan():
    rows = [{"motion_id": f"m{i}", "filename": f"m{i}.npz", "sha256": f"{i:064x}", "bytes": 1}
            for i in range(295)]
    inventory = {"schema": "roborender.motion-inventory.v1", "motion_count": 295,
                 "motions": rows, "motion_records_sha256": hashlib.sha256(canonical(rows)).hexdigest()}
    assignment = {"schema": "roborender.native-motion-cohort.v1", "run_id": "reference-only",
                  "cohort_id": "c1", "purpose": "training", "simulator_seed": 1,
                  "motion_inventory_sha256": hashlib.sha256(canonical(inventory)).hexdigest(),
                  "lanes": [{"lane": i, "motion_id": row["motion_id"],
                             "path": "/motions/" + row["filename"], "sha256": row["sha256"]}
                            for i, row in enumerate(rows[:16])]}
    final = {"assignment": assignment, "steps": 324}
    preludes = [{"assignment": {**deepcopy(assignment), "simulator_seed": i+20}, "steps": 3}
                for i in range(11)]
    return {"schema": SCHEMA, "inventory": inventory,
            "arms": {"0": [final], "1": preludes + [deepcopy(final)]}}


def test_identical_final_inputs_and_bounded_warm_lifecycles():
    value = validate_plan(plan())
    assert sum(s["steps"] for s in value["arms"]["0"]) == 324
    assert sum(s["steps"] for s in value["arms"]["1"]) == 357


@pytest.mark.parametrize("mutation", [
    lambda p: p["arms"]["1"][-1]["assignment"].update(simulator_seed=99),
    lambda p: p["arms"]["1"][0].update(steps=81),
    lambda p: p["arms"]["1"].pop(),
    lambda p: p["arms"]["1"][0]["assignment"].update(purpose="test"),
    lambda p: p["arms"]["1"][0]["assignment"]["lanes"].pop(),
    lambda p: p["arms"]["1"][0]["assignment"].update(motion_inventory_sha256="f"*64),
    lambda p: p["arms"]["1"][0]["assignment"]["lanes"][0].update(sha256="e"*64),
    lambda p: p["inventory"]["motions"][0].update(bytes=2),
])
def test_reject_changed_inputs_or_unbounded_budget(mutation):
    value = plan()
    mutation(value)
    with pytest.raises(ValueError):
        validate_plan(value)


def test_plan_digest_and_regular_file(tmp_path):
    path = tmp_path / "plan.json"
    path.write_bytes(canonical(plan()))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_plan(path, sha)["schema"] == SCHEMA
    with pytest.raises(ValueError, match="digest"):
        read_plan(path, "f"*64)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular"):
        read_plan(link, sha)


def test_exact_scene_order_and_no_retry_after_irreversible_failure():
    events, switches, calls = [], [], []
    def execute(steps, report):
        calls.append(steps)
        report({"phase": "capture_teacher_start", "completed_vector_steps": 0})
        if len(calls) == 2:
            raise RuntimeError("native capture stuck")
        return {"result": "PASS", "environment_step_calls": steps,
                "physical_transition_rows": 16*steps, "optimizer_steps": 0, "chunks_published": 0}
    with pytest.raises(RuntimeError, match="stuck"):
        run_stages(plan()["arms"]["1"], switch=switches.append, execute=execute, report=events.append)
    assert len(switches) == len(calls) == 2
    assert events[-1]["total_completed_vector_steps"] == 3


def test_exact_completed_accounting():
    events = []
    def execute(steps, report):
        return {"result": "PASS", "environment_step_calls": steps,
                "physical_transition_rows": 16*steps, "optimizer_steps": 0, "chunks_published": 0}
    results = run_stages(plan()["arms"]["1"], switch=lambda assignment: None,
                        execute=execute, report=events.append)
    assert len(results) == 12
    assert events[-1]["completed_vector_steps"] == 357


def test_partial_receipt_cannot_advance_scene():
    switches = []
    with pytest.raises(RuntimeError, match="exact teacher"):
        run_stages(plan()["arms"]["1"], switch=switches.append,
                   execute=lambda steps, report: {"result": "PASS"}, report=lambda row: None)
    assert len(switches) == 1


def test_native_diagnostic_receipts_are_host_readable():
    import ast
    tree = ast.parse((ACTOR_ROOT / "warm_camera_diagnostic.py").read_text())
    writes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"]
    assert len(writes) == 2
    assert all(any(keyword.arg == "mode" and keyword.value.value == 0o644
                   for keyword in node.keywords) for node in writes)


def test_launcher_rejects_warm_plan_outside_isolated_soak(tmp_path):
    from qualify_native_isaac6 import command
    with pytest.raises(ValueError, match="paired isolated"):
        command("unused", tmp_path, tmp_path, tmp_path, 16, mode="teacher-capacity",
                physical_gpu=0, deployment_host="leo-rtx-6k", warm_camera_plan=tmp_path / "missing")


def test_launcher_pins_read_only_plan_and_has_no_h200_credentials(tmp_path):
    from qualify_native_isaac6 import command
    path = tmp_path / "plan.json"
    path.write_bytes(canonical(plan()))
    (tmp_path / "port-manifest.json").write_text('{"files":[]}')
    root = REPO_ROOT
    cmd = command("rtx-leo-migration-warm-test", tmp_path, tmp_path,
                  root / "pipeline/dagger/rtx/code", 16, mode="teacher-camera-soak",
                  physical_gpu=1, deployment_host="leo-rtx-6k", capacity_steps=324,
                  run_id="rtx-leo-migration-warm-test", benchmark_barrier=tmp_path,
                  warm_camera_plan=path, warm_camera_source_sha256="a"*64)
    assert f"type=bind,src={path},dst=/run/warm-camera-plan.json,readonly" in cmd
    assert f"type=bind,src={root / 'pipeline/evaluation'},dst=/evaluation-source,readonly" in cmd
    assert any(arg.startswith("PYTHONPATH=/evaluation-source:") for arg in cmd)
    assert "ROBORENDER_WARM_CAMERA_GPU=1" in cmd
    assert "ROBORENDER_WARM_CAMERA_PLAN_SHA256=" + hashlib.sha256(path.read_bytes()).hexdigest() in cmd
    assert cmd[cmd.index("--network")+1] == "bridge"
    assert not any("H200_TOKEN" in arg or "DAGGER_H200_BASE" in arg for arg in cmd)


def test_missing_transitive_resident_helper_rejected_before_docker(tmp_path):
    from qualify_native_isaac6 import command
    path = tmp_path / "plan.json"
    path.write_bytes(canonical(plan()))
    code_root = tmp_path / "source/pipeline/dagger/rtx/code"
    code_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="evaluation helper"):
        command("rtx-leo-migration-test", tmp_path, tmp_path, code_root, 16,
                mode="teacher-camera-soak", physical_gpu=0, deployment_host="leo-rtx-6k",
                capacity_steps=324, benchmark_barrier=tmp_path, warm_camera_plan=path,
                warm_camera_source_sha256="a"*64)


def test_pair_cli_checks_plan_before_host_without_local_import_shadowing(tmp_path, monkeypatch):
    import run_throughput_pair
    path = tmp_path / "plan.json"
    path.write_bytes(canonical(plan()))
    monkeypatch.setattr(sys, "argv", ["pair", "--run-id", "rtx-leo-migration-test",
        "--deployment-host", "leo-rtx-6k", "--code-root", str(tmp_path),
        "--num-envs", "16", "--rtx-gpus", "0", "1", "--mode", "teacher-camera-soak",
        "--capacity-steps", "324", "--warm-camera-plan", str(path),
        "--conservative-capacity-guard"])
    monkeypatch.setattr(run_throughput_pair.socket, "gethostname", lambda: "not-leo")
    monkeypatch.setattr(run_throughput_pair.signal, "signal", lambda *args: None)
    with pytest.raises(RuntimeError, match="wrong host/run scope"):
        run_throughput_pair.main()
