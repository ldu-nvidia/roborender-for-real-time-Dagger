"""CPU-only tests for the post-import, pre-request gateway PID rebind."""

import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

from pipeline.bc_training.handoff_common import bound, publish
from pipeline.dagger.coverage_production import digest
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import actor_ids, validate_config
from pipeline.dagger.services.store import StateStore
from pipeline.deployment import cohort_boundary_runtime_rebind as rebind
from pipeline.deployment.cohort_boundary_recovery import RECEIPT_SCHEMA
from pipeline.deployment import cohort_source_transition as transition
from test.deployment.test_cohort_source_transition import inputs as transition_inputs
from test.pipeline.test_dagger_coverage_production import production


RUN = "dagger-leo-runtime-rebind"
SOURCE = "a" * 64


def fixture(tmp_path, monkeypatch, *, completed_updates=26, environments_per_actor=2):
    root = tmp_path / "ledger"
    store = StateStore(root)
    old = {"run_id": RUN, "source_sha256": SOURCE, "h200": {"gateway_pids": [101, 102, 103, 104]},
           "rtx_actors": [{"physical_gpu_index": gpu, "concurrent_environments": environments_per_actor}
                          for gpu in (0, 1)],
           "scientific": {"beta": .75}}
    new = json.loads(json.dumps(old)); new["h200"]["gateway_pids"] = [201, 202, 203, 204]
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir()
    checkpoints = []
    for index in range(completed_updates + 1):
        step = 2560 + index
        parameter = torch.tensor([float(index + 1)], dtype=torch.float32)
        temporary = checkpoint_root / f"checkpoint-{index:02d}.pt"
        torch.save({
            "schema": "roborender.bc_student_checkpoint.v3",
            "global_step": step,
            "policy_spec": {"fixture": True},
            "policy_state_dict": {"vision_module.fixture.weight": parameter},
            "optimizer_state_dict": {
                "param_groups": [{
                    "params": [0], "weight_decay": 0.0, "amsgrad": False,
                    "betas": (0.9, 0.999), "eps": 1e-8, "maximize": False,
                    "lr": 0.0002,
                }],
                "state": {0: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }},
            },
            "lr_scheduler_state_dict": None,
            "fixture_index": index,
        }, temporary)
        sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        final = checkpoint_root / f"{sha}.pt"
        temporary.rename(final); final.chmod(0o444)
        checkpoints.append((sha, final.stat().st_size, final))
    with store.transaction() as db:
        for sha, size, path in checkpoints:
            db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)",
                       (sha, size, "{}", str(path), 1.0))
        db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (RUN, json.dumps(actor_ids(old)), 81, digest(old), checkpoints[0][0], checkpoints[-1][0],
                    1, .75, "collecting", 148, 1., 2.))
        for iteration in range(1, completed_updates + 2):
            db.execute("INSERT INTO round_iterations VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (RUN, 1, iteration, checkpoints[iteration - 1][0],
                        checkpoints[iteration][0] if iteration <= completed_updates else None,
                        None, None, "advanced" if iteration <= completed_updates else "collecting", 1., 2.))
        for iteration in range(1, completed_updates + 1):
            job = f"job-{iteration}"
            db.execute("INSERT INTO learner_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (job, RUN, 1, "b" * 64, "[]", "succeeded", "{}", "{}", "/log", None, 1., 2.))
            db.execute("INSERT INTO learner_job_iterations VALUES (?,?,?,?)", (job, RUN, 1, iteration))
    active = rebind._validate_training_checkpoint(
        checkpoints[-1][2], checkpoints[-1][0], 2560 + completed_updates, 0.0002
    )
    import_ref = publish(root / "cohort-prefix-import.json", {
        "schema": RECEIPT_SCHEMA, "import_complete": True, "destination_root": str(root.resolve()),
        "run_id": RUN, "source_sha256": SOURCE, "completed_cohorts": completed_updates // 2,
        "completed_updates": completed_updates, "next_iteration": completed_updates + 1,
        "native_work_started": False, "compute_release_requested": False,
        "parent_checkpoint_sha256": checkpoints[-1][0], "parent_optimizer_global_step": 2560 + completed_updates,
        "parent_learning_rate": 0.0002,
        "checkpoint_materialization": {
            "schema": "roborender.dagger-cohort-checkpoint-materialization.v1",
            "checkpoint_count": completed_updates + 1,
            "total_size_bytes": sum(row[1] for row in checkpoints),
            "hardlinked_count": 0, "reflinked_count": 0, "copied_count": completed_updates + 1,
            "content_addressed_directory": str(checkpoint_root),
            "all_hashes_verified": True, "all_paths_rewritten": True,
            "source_paths_preserved": True, "active_checkpoint": active,
        },
    })
    old_ref = publish(tmp_path / "old.json", old); new_ref = publish(tmp_path / "new.json", new)
    services = [{"pid": pid, "start_ticks": str(pid * 10), "cmdline_sha256": f"{pid:064x}"}
                for pid in new["h200"]["gateway_pids"]]
    service_ref = publish(tmp_path / "services.json", {
        "schema": rebind.SERVICE_SCHEMA, "run_id": RUN, "source_sha256": SOURCE,
        "ledger_root": str(root.resolve()), "services": services, "fresh_services": True,
        "requests_submitted": 0, "allocation_retained": True, "compute_release_requested": False,
    })
    monkeypatch.setattr(rebind, "validate_config", lambda value, runtime=False: value)
    identities = {row["pid"]: row for row in services}
    monkeypatch.setattr(rebind, "_proc_identity", lambda pid: identities[pid])
    return root, import_ref, old_ref, new_ref, service_ref, old, new


def test_exact_runtime_only_rebind_is_idempotent_and_inert(tmp_path, monkeypatch):
    root, imported, old_ref, new_ref, services, old, new = fixture(tmp_path, monkeypatch)
    result = rebind.execute_rebind(root, imported, old_ref, new_ref, services)
    assert result["old_config_sha256"] == digest(old)
    assert result["new_config_sha256"] == digest(new)
    assert result["requests_submitted"] == 0 and result["native_work_started"] is False
    with StateStore(root).connection() as db:
        assert db.execute("SELECT config_sha256 FROM runs WHERE run_id=?", (RUN,)).fetchone()[0] == digest(new)
        assert db.execute("SELECT COUNT(*) FROM iteration_barriers").fetchone()[0] == 0
    assert rebind.execute_rebind(root, imported, old_ref, new_ref, services) == result


def test_nonruntime_config_delta_is_rejected(tmp_path, monkeypatch):
    root, imported, old_ref, _, services, _, new = fixture(tmp_path, monkeypatch)
    new["scientific"]["beta"] = .5
    changed = publish(tmp_path / "changed.json", new)
    with pytest.raises(ValueError, match="only h200.gateway_pids"):
        rebind.inspect_rebind(root, imported, old_ref, changed, services)


def test_existing_iteration27_effect_is_rejected(tmp_path, monkeypatch):
    root, imported, old_ref, new_ref, services, _, _ = fixture(tmp_path, monkeypatch)
    with StateStore(root).transaction() as db:
        db.execute("INSERT INTO iteration_barriers VALUES (?,?,?,?,?,?,?)",
                   (RUN, 1, 27, "actor", "c" * 64, "[]", 1.))
    with pytest.raises(ValueError, match="iteration27"):
        rebind.execute_rebind(root, imported, old_ref, new_ref, services)


def test_b16_iteration37_runtime_rebind_preserves_36_update_prefix(tmp_path, monkeypatch):
    root, imported, old_ref, new_ref, services, old, new = fixture(
        tmp_path, monkeypatch, completed_updates=36, environments_per_actor=16,
    )
    preview = rebind.inspect_rebind(root, imported, old_ref, new_ref, services)
    assert preview["completed_updates"] == 36
    assert preview["next_iteration"] == 37
    with StateStore(root).connection() as db:
        assert db.execute("SELECT config_sha256 FROM runs").fetchone()[0] == digest(old)
        before = [tuple(row) for row in db.execute("SELECT * FROM round_iterations ORDER BY iteration_id")]
    result = rebind.execute_rebind(root, imported, old_ref, new_ref, services)
    assert result["requests_submitted"] == 0
    assert result["native_work_started"] is False
    assert result["compute_release_requested"] is False
    with StateStore(root).connection() as db:
        assert db.execute("SELECT config_sha256 FROM runs").fetchone()[0] == digest(new)
        assert len(json.loads(db.execute("SELECT actor_ids_json FROM runs").fetchone()[0])) == 32
        assert db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 37
        assert db.execute("SELECT COUNT(*) FROM learner_jobs").fetchone()[0] == 36
        assert [tuple(row) for row in db.execute("SELECT * FROM round_iterations ORDER BY iteration_id")] == before
    assert rebind.execute_rebind(root, imported, old_ref, new_ref, services) == result


@pytest.mark.parametrize("fault,message", [
    ("tail_barrier", "iteration37"),
    ("tail_prepare", "iteration37"),
    ("tail_candidate", "post-update36"),
    ("actor_population", "pinned population"),
    ("checkpoint_lineage", "post-update36"),
    ("failed_prefix_job", "prefix lineage changed"),
])
def test_b16_rebind_rejects_changed_boundary(tmp_path, monkeypatch, fault, message):
    root, imported, old_ref, new_ref, services, _, _ = fixture(
        tmp_path, monkeypatch, completed_updates=36, environments_per_actor=16,
    )
    with StateStore(root).transaction() as db:
        if fault == "tail_barrier":
            db.execute("INSERT INTO iteration_barriers VALUES (?,?,?,?,?,?,?)",
                       (RUN, 1, 37, actor_ids(json.loads(Path(old_ref["path"]).read_bytes()))[0],
                        "c" * 64, "[]", 1.))
        elif fault == "tail_prepare":
            db.execute("INSERT INTO step_prepares VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       ("tail", "c" * 64, RUN, 1, "actor", "window-000036", 0, "episode", 0,
                        "d" * 64, "receipt", "{}", "{}", "/artifact", "e" * 64, "prepared", 1., 2.))
        elif fault == "tail_candidate":
            db.execute("UPDATE round_iterations SET candidate_checkpoint_sha256=? WHERE iteration_id=37",
                       ("c" * 64,))
        elif fault == "actor_population":
            db.execute("UPDATE runs SET actor_ids_json='[]'")
        elif fault == "checkpoint_lineage":
            db.execute("UPDATE round_iterations SET input_checkpoint_sha256=? WHERE iteration_id=36",
                       ("c" * 64,))
        else:
            db.execute("UPDATE learner_jobs SET status='failed' WHERE job_id='job-36'")
    with pytest.raises(ValueError, match=message):
        rebind.execute_rebind(root, imported, old_ref, new_ref, services)
    assert not (root / "runtime-rebind.json").exists()


@pytest.mark.parametrize("field,value", [
    ("completed_cohorts", 13), ("completed_updates", 35), ("next_iteration", 27),
])
def test_b16_rebind_rejects_inconsistent_import_counts(tmp_path, monkeypatch, field, value):
    root, imported, old_ref, new_ref, services, _, _ = fixture(
        tmp_path, monkeypatch, completed_updates=36, environments_per_actor=16,
    )
    receipt = json.loads(Path(imported["path"]).read_bytes())
    receipt[field] = value
    changed = publish(tmp_path / "changed-import.json", receipt)
    with pytest.raises(ValueError, match="prefix-import receipt"):
        rebind.inspect_rebind(root, changed, old_ref, new_ref, services)


@pytest.fixture
def source_rebind(tmp_path, monkeypatch, transition_inputs):
    """Use the real source/config/plan validator with a temporary owned ledger."""
    old = bound(transition_inputs["historical_config"])
    new = bound(transition_inputs["execution_config"])
    monkeypatch.setattr(sys.modules[__name__], "RUN", old["run_id"])
    monkeypatch.setattr(sys.modules[__name__], "SOURCE", old["source_sha256"])
    root, imported, _, _, services, _, _ = fixture(
        tmp_path, monkeypatch, completed_updates=36, environments_per_actor=16,
    )
    monkeypatch.setattr(rebind, "validate_config", validate_config)
    with StateStore(root).transaction() as db:
        initial = db.execute("SELECT initial_checkpoint_sha256 FROM runs").fetchone()[0]
        old["student"]["initial_checkpoint_sha256"] = initial
        new["student"]["initial_checkpoint_sha256"] = initial
        db.execute("UPDATE runs SET config_sha256=?", (digest(old),))
    old_ref = publish(tmp_path / "source-old-config.json", old)
    new_ref = publish(tmp_path / "source-new-config.json", new)
    # Complete the source-transition fixture with the actual checkpoint tree
    # and active policy/AdamW load proof produced by the ledger fixture.
    imported_value = {**bound(transition_inputs["import_receipt"]), **bound(imported)}
    imported = publish(tmp_path / "source-import.json", imported_value)
    service_value = bound(services)
    service_value["source_sha256"] = new["source_sha256"]
    services = publish(tmp_path / "source-services.json", service_value)
    inputs = {**transition_inputs, "historical_config": old_ref,
              "execution_config": new_ref, "import_receipt": imported}
    admission = publish(tmp_path / "source-transition.json", transition.build_admission(**inputs))
    return {"args": (root, imported, old_ref, new_ref, services),
            "source_transition_ref": admission, "old": old, "new": new, "transition_inputs": inputs}


def test_source_rebind_is_atomic_idempotent_and_preserves_historical_artifacts(source_rebind):
    case = source_rebind
    root, imported, old_ref, new_ref, services = case["args"]
    refs = [imported, old_ref, case["old"]["coverage_plan"],
            case["transition_inputs"]["historical_manifest"]]
    historical_bytes = {ref["path"]: Path(ref["path"]).read_bytes() for ref in refs}
    with StateStore(root).connection() as db:
        before_run = dict(db.execute("SELECT * FROM runs").fetchone())
        before_iterations = [tuple(row) for row in db.execute("SELECT * FROM round_iterations ORDER BY iteration_id")]
        before_checkpoints = [tuple(row) for row in db.execute("SELECT * FROM checkpoints ORDER BY sha256")]
    preview = rebind.inspect_rebind(*case["args"], source_transition_ref=case["source_transition_ref"])
    assert preview["schema"] == rebind.SOURCE_TRANSITION_SCHEMA
    assert preview["source_sha256"] == preview["execution_source_sha256"] == case["new"]["source_sha256"]
    assert preview["historical_source_sha256"] == case["old"]["source_sha256"]
    assert preview["historical_coverage_plan"] == case["old"]["coverage_plan"]
    assert preview["execution_coverage_plan"] == case["new"]["coverage_plan"]
    with StateStore(root).connection() as db:
        assert dict(db.execute("SELECT * FROM runs").fetchone()) == before_run
    result = rebind.execute_rebind(*case["args"], source_transition_ref=case["source_transition_ref"])
    assert result["database_update_complete"] is True
    assert result["execution_authorized"] is False
    assert result["native_work_started"] is False and result["native_replay_allowed"] is False
    assert result["compute_release_requested"] is False and result["allocation_retained"] is True
    with StateStore(root).connection() as db:
        assert dict(db.execute("SELECT * FROM runs").fetchone()) == {
            **before_run, "config_sha256": digest(case["new"]),
        }
        assert [tuple(row) for row in db.execute("SELECT * FROM round_iterations ORDER BY iteration_id")] == before_iterations
        assert [tuple(row) for row in db.execute("SELECT * FROM checkpoints ORDER BY sha256")] == before_checkpoints
    assert {path: Path(path).read_bytes() for path in historical_bytes} == historical_bytes
    assert bound({"path": str(root / "runtime-rebind-intent.json"),
                  "raw_sha256": hashlib.sha256((root / "runtime-rebind-intent.json").read_bytes()).hexdigest()})[
        "source_transition"] == case["source_transition_ref"]
    assert rebind.execute_rebind(*case["args"], source_transition_ref=case["source_transition_ref"]) == result


def test_source_change_without_explicit_transition_is_rejected(source_rebind):
    with pytest.raises(ValueError, match="only h200.gateway_pids"):
        rebind.execute_rebind(*source_rebind["args"])


@pytest.mark.parametrize("fault", ["different_import", "different_config_ref", "claimed_hash", "claimed_work"])
def test_source_rebind_rejects_forged_or_misbound_admission(source_rebind, tmp_path, fault):
    case = source_rebind
    value = bound(case["source_transition_ref"])
    if fault == "different_import":
        value["import_receipt"] = publish(tmp_path / "other-import.json", bound(case["args"][1]))
    elif fault == "different_config_ref":
        value["execution_config"] = publish(tmp_path / "other-config.json", case["new"])
    elif fault == "claimed_hash":
        value["execution_source_sha256"] = "f" * 64
    else:
        value["execution_authorized"] = True
    changed = publish(tmp_path / "forged-transition.json", value)
    with pytest.raises(ValueError, match="source.transition"):
        rebind.execute_rebind(*case["args"], source_transition_ref=changed)
    with StateStore(case["args"][0]).connection() as db:
        assert db.execute("SELECT config_sha256 FROM runs").fetchone()[0] == digest(case["old"])


@pytest.mark.parametrize("fault", ["config_science", "plan_science", "old_service_source", "tail_prepare",
                                   "success_evaluation", "promotion"])
def test_source_rebind_rechecks_science_services_and_clean_boundary(source_rebind, tmp_path, fault):
    case = source_rebind
    args = list(case["args"])
    transition_ref = case["source_transition_ref"]
    if fault in {"config_science", "plan_science"}:
        config = bound(args[3])
        if fault == "config_science":
            config["logging"]["wandb_run_id"] += "-another"
        else:
            plan = bound(config["coverage_plan"])
            plan["teacher_sha256"] = "f" * 64
            config["coverage_plan"] = publish(tmp_path / "bad-science-plan.json", plan)
        args[3] = publish(tmp_path / "bad-science-config.json", config)
        value = bound(transition_ref)
        value["execution_config"] = args[3]
        value["execution_config_sha256"] = digest(config)
        value["execution_coverage_plan"] = config["coverage_plan"]
        transition_ref = publish(tmp_path / "bad-science-transition.json", value)
    elif fault == "old_service_source":
        service = bound(args[4]); service["source_sha256"] = case["old"]["source_sha256"]
        args[4] = publish(tmp_path / "old-source-services.json", service)
    elif fault == "success_evaluation":
        with StateStore(args[0]).transaction() as db:
            db.execute("INSERT INTO success_evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       ("unexpected-test", RUN, 1, "test", "c" * 64, "d" * 64, "[]",
                        "raytraced", "raven_goal_reached_rate", "running", 1., 2.))
    elif fault == "promotion":
        with StateStore(args[0]).transaction() as db:
            db.execute("INSERT INTO promotions VALUES (?,?,?,?)", (RUN, 1, "c" * 64, 1.))
    else:
        with StateStore(args[0]).transaction() as db:
            db.execute("INSERT INTO step_prepares VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       ("tail", "c" * 64, RUN, 1, "actor", "window-000036", 0, "episode", 0,
                        "d" * 64, "receipt", "{}", "{}", "/artifact", "e" * 64, "prepared", 1., 2.))
    with pytest.raises(ValueError):
        rebind.execute_rebind(*args, source_transition_ref=transition_ref)
    with StateStore(args[0]).connection() as db:
        assert db.execute("SELECT config_sha256 FROM runs").fetchone()[0] == digest(case["old"])
    assert not (args[0] / "runtime-rebind.json").exists()


def test_source_rebind_recovers_result_publication_failure_without_repeating_mutation(source_rebind, monkeypatch):
    case = source_rebind
    original = rebind.publish
    def fail_result(path, value):
        if Path(path).name == "runtime-rebind.json":
            raise OSError("simulated result publication failure")
        return original(path, value)
    monkeypatch.setattr(rebind, "publish", fail_result)
    with pytest.raises(OSError, match="publication failure"):
        rebind.execute_rebind(*case["args"], source_transition_ref=case["source_transition_ref"])
    root = case["args"][0]
    assert (root / "runtime-rebind-intent.json").is_file()
    assert not (root / "runtime-rebind.json").exists()
    with StateStore(root).connection() as db:
        assert db.execute("SELECT config_sha256 FROM runs").fetchone()[0] == digest(case["new"])
    monkeypatch.setattr(rebind, "publish", original)
    result = rebind.execute_rebind(*case["args"], source_transition_ref=case["source_transition_ref"])
    assert result["database_update_complete"] is True
    assert result["source_transition"] == case["source_transition_ref"]
