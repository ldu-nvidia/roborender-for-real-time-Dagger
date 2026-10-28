"""Synthetic CPU admission tests; no native/source-transition qualification."""
from test.deployment.source_paths import REPO_ROOT

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.bc_training.handoff_common import bound, publish
from pipeline.dagger.coverage_production import digest, make_first_config, make_plan
from pipeline.deployment import cohort_source_transition as transition
from pipeline.deployment.source_snapshot import file_record
from test.pipeline.test_dagger_coverage_production import production,rendering_dependencies


def manifest(root):
    rows = [file_record(root, path.relative_to(root).as_posix())
            for path in sorted(root.rglob("*")) if path.is_file()]
    return {"schema": "roborender.source-snapshot.v1", "source_sha256": digest(rows),
            "file_count": len(rows), "files": rows}


@pytest.fixture
def inputs(tmp_path, monkeypatch, production):
    old_root, new_root = tmp_path / "historical", tmp_path / "execution"
    old_root.mkdir(); new_root.mkdir()
    pairs = {}
    for index, name in enumerate({**transition.REVIEWED_TRANSPORT_PAIRS, **transition.REVIEWED_CONTROL_PAIRS}):
        old = f"# historical transport fixture {index}\n"
        new = f"# execution transport fixture {index}\n"
        for root, raw in ((old_root, old), (new_root, new)):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(raw)
        pairs[name] = (hashlib.sha256(old.encode()).hexdigest(), hashlib.sha256(new.encode()).hexdigest())
    for root in (old_root, new_root):
        path = root / "pipeline/dagger/services/raven_learner_worker.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unchanged scientific fixture\n")
    old_manifest, new_manifest = manifest(old_root), manifest(new_root)
    old_sha, new_sha = old_manifest["source_sha256"], new_manifest["source_sha256"]
    monkeypatch.setattr(transition, "HISTORICAL_SOURCE_SHA256", old_sha)
    monkeypatch.setattr(transition, "REVIEWED_TRANSPORT_PAIRS",
                        {name: pairs[name] for name in transition.REVIEWED_TRANSPORT_PAIRS})
    monkeypatch.setattr(transition, "REVIEWED_CONTROL_PAIRS",
                        {name: pairs[name] for name in transition.REVIEWED_CONTROL_PAIRS})

    _, base = production
    old_plan = make_plan(
        run_id="dagger-leo-source-transition-test", inventory=base["inventory"], panel=base["panel"],
        excluded_process_seeds=base["excluded_process_seeds"], seed=base["seed"],
        motion_root=base["motion_root"], source_sha256=old_sha,
        teacher_sha256=base["teacher_sha256"], policy_spec_sha256=base["policy_spec_sha256"],
        environments_per_actor=16,
    )
    new_plan = deepcopy(old_plan); new_plan["source_sha256"] = new_sha
    old_plan_ref = publish(tmp_path / "historical-plan.json", old_plan)
    new_plan_ref = publish(tmp_path / "execution-plan.json", new_plan)
    old = make_first_config(old_plan, coverage_plan=old_plan_ref,
                           checkpoint=tmp_path / "bootstrap.pt", checkpoint_sha256="d" * 64,
                           gateway_pids=[101, 102, 103, 104],
                           rendered_observation_config=rendering_dependencies(tmp_path,old_plan))
    new = deepcopy(old)
    new["source_sha256"] = new_sha
    new["coverage_plan"] = new_plan_ref
    new["h200"]["gateway_pids"] = [201, 202, 203, 204]
    receipt = {
        "schema": transition.RECEIPT_SCHEMA, "import_complete": True,
        "run_id": old["run_id"], "round_id": 1, "source_sha256": old_sha,
        "completed_updates": 36, "completed_cohorts": 18, "next_iteration": 37, "next_cohort_index": 18,
        "source_database_unchanged": True, "old_namespace_preserved": True,
        "fresh_native_runtime_required": True, "allocation_retained": True,
        "native_replay_allowed": False, "native_work_started": False, "compute_release_requested": False,
        "parent_checkpoint_sha256": "e" * 64, "parent_optimizer_global_step": 18596,
        "parent_learning_rate": .0002, "mse_history_iterations": [0, 20],
        "checkpoint_materialization": {"checkpoint_count": 37, "all_hashes_verified": True,
                                       "all_paths_rewritten": True},
        "fixed_heldout_manifest": publish(tmp_path / "heldout.json", {"fixture": "unchanged"}),
        "fixed_heldout_admission": publish(tmp_path / "heldout-admission.json", {"fixture": "unchanged"}),
    }
    return {
        "historical_root": old_root, "execution_root": new_root,
        "historical_manifest": publish(tmp_path / "historical-manifest.json", old_manifest),
        "execution_manifest": publish(tmp_path / "execution-manifest.json", new_manifest),
        "historical_config": publish(tmp_path / "historical-config.json", old),
        "execution_config": publish(tmp_path / "execution-config.json", new),
        "import_receipt": publish(tmp_path / "import.json", receipt),
    }


def replace_ref(inputs, key, value, tmp_path):
    inputs[key] = publish(tmp_path / f"changed-{key}.json", value)


def test_admission_preserves_history_and_distinguishes_actual_execution_source(inputs):
    files = [Path(inputs[name]["path"]) for name in (
        "historical_manifest", "historical_config", "import_receipt")]
    before = {path: path.read_bytes() for path in files}
    result = transition.build_admission(**inputs)
    assert result["schema"] == transition.SCHEMA
    assert result["historical_source_sha256"] != result["execution_source_sha256"]
    assert result["completed_updates"] == 36 and result["completed_cohorts"] == 18
    assert result["next_iteration"] == 37
    assert result["next_coverage_epoch"] == 2 and result["next_cohort_index"] == 8
    assert result["parent_optimizer_global_step"] == 18596
    assert result["mse_history_iterations"] == [0, 20]
    assert result["historical_artifacts_preserved"] is True
    assert result["fresh_native_runtime_required"] is True
    assert result["fresh_native_qualification_required"] is True
    assert result["allocation_retained"] is True
    for key in ("execution_authorized", "native_work_started", "native_replay_allowed",
                "ledger_mutation_performed", "compute_release_requested"):
        assert result[key] is False
    assert {path: path.read_bytes() for path in files} == before


def test_historical_reviewed_pairs_reject_drifted_current_source():
    """v1 pins a historical transition, not every future mutable checkout.

    These metadata rows exercise the exact-pair gate only; they are not a
    historical source manifest or native admission. Never refresh its trusted
    hashes merely to make new runtime/scientific code pass this old transition.
    """
    root = REPO_ROOT
    pairs = {**transition.REVIEWED_TRANSPORT_PAIRS, **transition.REVIEWED_CONTROL_PAIRS}
    actual = {name: file_record(root, name) for name in pairs}
    historical = {name: {**actual[name], 'sha256': old} for name, (old, _) in pairs.items()}
    reviewed = {name: {**actual[name], 'sha256': new} for name, (_, new) in pairs.items()}
    assert transition._source_delta(historical, reviewed) == sorted(pairs)
    assert any(actual[name]['sha256'] != new for name, (_, new) in pairs.items())
    with pytest.raises(ValueError, match='outside the exact reviewed hash pair'):
        transition._source_delta(historical, actual)


@pytest.mark.parametrize("name", list(transition.REVIEWED_CONTROL_PAIRS))
def test_control_hook_cannot_be_arbitrarily_changed(inputs, tmp_path, name):
    (inputs["execution_root"] / name).write_text("# unreviewed orchestration edit\n")
    replace_ref(inputs, "execution_manifest", manifest(inputs["execution_root"]), tmp_path)
    with pytest.raises(ValueError, match="exact reviewed hash pair"):
        transition.build_admission(**inputs)


@pytest.mark.parametrize("fault", ["learner", "transport", "renderer", "reset", "model",
                                   "removed", "extra", "symlink", "mode"])
def test_source_admission_rejects_unreviewed_or_unbound_source(inputs, tmp_path, fault):
    root = inputs["execution_root"]
    if fault == "learner":
        (root / "pipeline/dagger/services/raven_learner_worker.py").write_text("# changed learner\n")
    elif fault == "transport":
        (root / next(iter(transition.REVIEWED_TRANSPORT_PAIRS))).write_text("# another transport policy\n")
    elif fault in {"renderer", "reset", "model"}:
        name = {
            "renderer": "pipeline/dagger/services/render_student.py",
            "reset": "pipeline/deployment/rtx_actor/resident_isaac_trainer.py",
            "model": "pipeline/dagger/services/student_model.py",
        }[fault]
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unreviewed scientific addition\n")
    elif fault == "removed":
        (root / "pipeline/dagger/services/raven_learner_worker.py").unlink()
    elif fault == "extra":
        (root / "untracked.py").write_text("# undeclared runtime code\n")
    elif fault == "symlink":
        (root / "shadow.py").symlink_to(root / "pipeline/dagger/services/raven_learner_worker.py")
    else:
        (root / next(iter(transition.REVIEWED_TRANSPORT_PAIRS))).chmod(0o755)
    if fault not in {"extra", "symlink"}:
        replace_ref(inputs, "execution_manifest", manifest(root), tmp_path)
    with pytest.raises(ValueError):
        transition.build_admission(**inputs)


def test_manifest_content_is_verified_against_disk(inputs):
    (inputs["execution_root"] / next(iter(transition.REVIEWED_TRANSPORT_PAIRS))).write_text("# tamper\n")
    with pytest.raises(ValueError, match="bytes/mode"):
        transition.build_admission(**inputs)


@pytest.mark.parametrize("fault", ["duplicate", "bad_digest", "missing_transport"])
def test_manifest_and_exact_pair_structure_are_checked(inputs, tmp_path, fault):
    value = bound(inputs["execution_manifest"])
    if fault == "duplicate":
        value["files"].append(deepcopy(value["files"][-1]))
        value["file_count"] += 1
        value["source_sha256"] = digest(value["files"])
    elif fault == "bad_digest":
        value["source_sha256"] = "f" * 64
    else:
        old = bound(inputs["historical_manifest"])
        path = next(iter(transition.REVIEWED_TRANSPORT_PAIRS))
        (inputs["execution_root"] / path).write_bytes((inputs["historical_root"] / path).read_bytes())
        value["files"] = [next(row for row in old["files"] if row["path"] == path)
                          if row["path"] == path else row for row in value["files"]]
        value["source_sha256"] = digest(value["files"])
    replace_ref(inputs, "execution_manifest", value, tmp_path)
    with pytest.raises(ValueError):
        transition.build_admission(**inputs)


@pytest.mark.parametrize("fault", ["run", "wandb", "source_relabel", "initial_parent", "batch", "beta", "panel"])
def test_round_config_rejects_scientific_identity_changes(inputs, tmp_path, fault):
    value = bound(inputs["execution_config"])
    if fault == "run":
        value["run_id"] += "-other"
    elif fault == "wandb":
        value["logging"]["wandb_run_id"] += "-other"
    elif fault == "source_relabel":
        value["source_sha256"] = bound(inputs["historical_config"])["source_sha256"]
    elif fault == "initial_parent":
        value["student"]["initial_checkpoint_sha256"] = "f" * 64
    elif fault == "beta":
        value["teacher_beta"] = .5
        value["student_action_probability"] = .5
    elif fault == "panel":
        value["validation"]["scenario_manifest_sha256"] = "f" * 64
    else:
        for actor in value["rtx_actors"]:
            actor["concurrent_environments"] = 2
    replace_ref(inputs, "execution_config", value, tmp_path)
    with pytest.raises(ValueError):
        transition.build_admission(**inputs)


@pytest.mark.parametrize("field,value", [("teacher_sha256", "f" * 64), ("seed", 77),
                                         ("learning_rate", .01)])
def test_coverage_plan_cannot_change_science(inputs, tmp_path, field, value):
    config = bound(inputs["execution_config"])
    plan = bound(config["coverage_plan"])
    plan[field] = value
    config["coverage_plan"] = publish(tmp_path / "changed-plan.json", plan)
    replace_ref(inputs, "execution_config", config, tmp_path)
    with pytest.raises(ValueError):
        transition.build_admission(**inputs)


@pytest.mark.parametrize("field,value", [
    ("completed_updates", 34), ("completed_cohorts", 17), ("next_iteration", 38),
    ("source_sha256", "f" * 64), ("fresh_native_runtime_required", False),
    ("native_replay_allowed", True), ("compute_release_requested", True),
    ("allocation_retained", False), ("mse_history_iterations", [0]),
])
def test_import_boundary_and_retention_cannot_be_relaxed(inputs, tmp_path, field, value):
    receipt = bound(inputs["import_receipt"])
    receipt[field] = value
    replace_ref(inputs, "import_receipt", receipt, tmp_path)
    with pytest.raises(ValueError):
        transition.build_admission(**inputs)


def test_source_roots_must_be_distinct(inputs):
    inputs["execution_root"] = inputs["historical_root"]
    with pytest.raises(ValueError, match="separate"):
        transition.build_admission(**inputs)
