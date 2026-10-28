"""CPU-only continuation guards; fixtures are not native qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.bc_training.handoff_common import binding, bound, publish
from pipeline.deployment import cohort_continuation as continuation
from pipeline.deployment import resident_production_collection as collector
from pipeline.dagger.coverage_production import digest, load_plan
from test.deployment.test_cohort_source_transition import inputs
from test.pipeline.test_dagger_coverage_production import production


def source_admission(inputs, tmp_path):
    from pipeline.deployment.cohort_source_transition import build_admission
    return publish(tmp_path / "source-transition.json", build_admission(**inputs))


def test_transition_is_recomputed_from_artifacts(inputs, tmp_path):
    ref = source_admission(inputs, tmp_path)
    assert continuation.check_transition(ref) == bound(ref)
    path = inputs["execution_root"] / next(iter(continuation.rebuild_transition.__globals__["REVIEWED_TRANSPORT_PAIRS"]))
    path.write_text("changed after freeze")
    with pytest.raises(ValueError, match="manifest"):
        continuation.check_transition(ref)


def prefix_fixture(inputs, tmp_path):
    transition_ref = source_admission(inputs, tmp_path)
    transition = bound(transition_ref)
    old_plan = load_plan(transition["historical_coverage_plan"])
    new_config = bound(transition["execution_config"])
    root = tmp_path / "fresh-campaign"
    refs = {name: publish(tmp_path / f"dependency-{name}.json", {}) for name in continuation.REFS}
    refs["source_transition"] = transition_ref
    spec_ref = publish(tmp_path / "spec.json", dict(**refs, campaign_root=str(root)))
    markers = []
    for epoch, cohort in continuation._cohorts(old_plan)[:18]:
        index = cohort["cohort_index"]
        original = publish(tmp_path / "history" / f"e{epoch:02d}-c{index:03d}.json", {
            "source_sha256": old_plan["source_sha256"], "ordinal": len(markers)})
        markers.append(dict(epoch=epoch, index=index, completion=original))
        target = root / "round-1" / f"coverage/e{epoch:02d}-c{index:03d}/completion.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(original["path"]).read_bytes())
    admission = dict(schema=continuation.SCHEMA, spec=spec_ref, run_id=new_config["run_id"], round_id=1,
        campaign_root=str(root), source_transition=transition_ref, historical_source_sha256=old_plan["source_sha256"],
        execution_source_sha256=new_config["source_sha256"], completed_cohorts=18, completed_updates=36,
        next_iteration=37, historical_markers=markers, parent_checkpoint_sha256="e" * 64,
        maximum_round_id=1, automatic_next_round=False, native_replay_allowed=False, compute_release_requested=False)
    ref = publish(tmp_path / "admission.json", admission)
    return ref, new_config, root


def test_prefix_never_relabels_historical_source_or_plan(inputs, tmp_path):
    ref, config, root = prefix_fixture(inputs, tmp_path)
    prefix = continuation.HistoricalPrefix(ref, config, root)
    old = prefix.plan_for(root / "round-1/coverage/e02-c007", 2, 7)
    new = prefix.plan_for(root / "round-1/coverage/e02-c008", 2, 8)
    assert old["source_sha256"] != new["source_sha256"] == config["source_sha256"]
    assert digest(old) != digest(new)


@pytest.mark.parametrize("mutation", ["missing", "rewritten", "symlink"])
def test_prefix_refuses_missing_or_rewritten_marker(inputs, tmp_path, mutation):
    ref, config, root = prefix_fixture(inputs, tmp_path)
    marker = root / "round-1/coverage/e02-c007/completion.json"
    marker.unlink()
    if mutation == "rewritten":
        marker.write_text("{}")
    elif mutation == "symlink":
        marker.symlink_to(bound(ref)["historical_markers"][-1]["completion"]["path"])
    with pytest.raises(ValueError, match="historical prefix"):
        continuation.HistoricalPrefix(ref, config, root).plan_for(marker.parent, 2, 7)


def test_prefix_refuses_execution_source_or_root_substitution(inputs, tmp_path):
    ref, config, root = prefix_fixture(inputs, tmp_path)
    config["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="config/root"):
        continuation.HistoricalPrefix(ref, config, root)
    with pytest.raises(ValueError, match="config/root"):
        continuation.HistoricalPrefix(ref, bound(bound(bound(ref)["source_transition"])["execution_config"]), tmp_path / "other")


def test_uncertain_new_cohort_intent_still_cannot_replay(tmp_path):
    directory = tmp_path / "e02-c008"
    directory.mkdir(); (directory / "intent.json").write_text("{}")
    with pytest.raises(RuntimeError, match="never replay"):
        collector.completed_prefix(directory, {}, 1, 2, 8, "e" * 64, [])


@pytest.mark.parametrize("field,value", [("maximum_round_id", 2), ("automatic_next_round", True),
    ("native_replay_allowed", True), ("compute_release_requested", True),
    ("completed_updates", 37), ("completed_cohorts", 19), ("next_iteration", 38)])
def test_scope_cannot_expand(inputs, tmp_path, field, value):
    ref, _, _ = prefix_fixture(inputs, tmp_path)
    modified = bound(ref); modified[field] = value
    bad = publish(tmp_path / "bad-admission.json", modified)
    with pytest.raises(ValueError, match="invalid cohort"):
        continuation.validate_admission(bad)


def boundary(config, parent):
    return dict(coordinator=dict(run=dict(config_sha256=digest(config), run_id=config["run_id"],
        current_round=1, status="collecting", actor_ids=continuation.actor_ids(config), active_checkpoint_sha256=parent),
        iterations=[dict(iteration_id=i, round_id=1, status="advanced" if i < 37 else "collecting",
                         candidate_checkpoint_sha256=str(i) if i < 37 else None) for i in range(1, 38)]),
        mse=dict(status="active", iteration_id=37, checkpoint_sha256=parent, selection=dict(history=[dict(iteration=0), dict(iteration=20)])))


def test_live_boundary_accepts_only_exact36_prefix(inputs, tmp_path):
    tr = bound(source_admission(inputs, tmp_path)); config = bound(tr["execution_config"])
    value = boundary(config, tr["parent_checkpoint_sha256"])
    assert len(continuation.validate_boundary(value, config, tr)) == 37
    value["coordinator"]["iterations"][-1]["candidate_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="empty iteration37"):
        continuation.validate_boundary(value, config, tr)


@pytest.mark.parametrize("mutation", ["actor", "iteration", "mse_history", "checkpoint", "config", "round"])
def test_live_boundary_refuses_changed_owner_or_lineage(inputs, tmp_path, mutation):
    tr = bound(source_admission(inputs, tmp_path)); config = bound(tr["execution_config"])
    value = boundary(config, tr["parent_checkpoint_sha256"])
    if mutation == "actor": value["coordinator"]["run"]["actor_ids"].pop()
    if mutation == "iteration": value["coordinator"]["iterations"].pop()
    if mutation == "mse_history": value["mse"]["selection"]["history"].append(dict(iteration=40))
    if mutation == "checkpoint": value["mse"]["checkpoint_sha256"] = "f" * 64
    if mutation == "config": value["coordinator"]["run"]["config_sha256"] = "f" * 64
    if mutation == "round": value["coordinator"]["run"]["current_round"] = 2
    with pytest.raises(ValueError, match="empty iteration37"):
        continuation.validate_boundary(value, config, tr)


def test_rebind_requires_real_import_transaction(inputs, tmp_path):
    tr_ref = source_admission(inputs, tmp_path); tr = bound(tr_ref)
    value = dict(schema="roborender.dagger-cohort-runtime-rebind.v3", source_transition=tr_ref,
        import_receipt=tr["import_receipt"], old_config=tr["historical_config"], new_config=tr["execution_config"],
        database_update_complete=True, completed_updates=36, next_iteration=37, requests_submitted=0,
        native_work_started=False, allocation_retained=True, compute_release_requested=False)
    continuation._validate_rebind(value, tr_ref, tr)
    value["database_update_complete"] = False
    with pytest.raises(ValueError, match="owner-host"):
        continuation._validate_rebind(value, tr_ref, tr)


def test_stage_copies_only_verified_markers_and_preserves_failed_namespace(tmp_path, monkeypatch):
    oldroot = tmp_path / "historical"; root = tmp_path / "continuation"
    marker = publish(oldroot / "round-1/coverage/e01-c000/completion.json", {"history": "old-source"})
    failed = publish(oldroot / "round-1/coverage/e02-c008/intent.json", {"uncertain": True})
    old = dict(config={"old": True}, native_qualification_binding={"historical": True},
               native_qualification={"source": "old"}, phase="COLLECT_TRAIN")
    config = {"source_sha256": "new-source"}
    spec = dict(source_transition=publish(tmp_path / "transition.json", {
        "execution_config": publish(tmp_path / "config.json", config)}),
        historical_state=publish(oldroot / "state.json", old),
        service_binding=publish(tmp_path / "services.json", {"mse_worker_command": ["/python"]}),
        mse_plan=publish(tmp_path / "mse.json", {"fixed": True}))
    spec_ref = publish(tmp_path / "spec.json", spec)
    value = dict(campaign_root=str(root), source_transition=spec["source_transition"],
                 historical_markers=[dict(epoch=1, index=0, completion=marker)])
    monkeypatch.setattr(continuation, "inspect", lambda ref: value)
    before = Path(failed["path"]).read_bytes()
    continuation.stage(spec_ref)
    assert (root / "round-1/coverage/e01-c000/completion.json").read_bytes() == Path(marker["path"]).read_bytes()
    assert not (root / "round-1/coverage/e02-c008").exists()
    state = json.loads((root / "state.json").read_text())
    assert state["operation_id"] is None and state["native_qualification"] == old["native_qualification"]
    assert Path(failed["path"]).read_bytes() == before
    with pytest.raises(ValueError, match="already staged"):
        continuation.stage(spec_ref)


@pytest.mark.parametrize("phase", ["prepare", "abort", "control_only", "evaluate"])
def test_collect_phase_cannot_dispatch_other_operation(phase):
    campaign = continuation.ContinuationCampaign.__new__(continuation.ContinuationCampaign)
    campaign.state = {"phase": "COLLECT_TRAIN"}; campaign._guard = lambda: None
    with pytest.raises(ValueError, match="cannot dispatch"):
        campaign.operation(phase)


def test_round1_advance_finishes_instead_of_dispatching_round2(tmp_path):
    campaign = continuation.ContinuationCampaign.__new__(continuation.ContinuationCampaign)
    campaign.admission_ref = {"fixture": True}; campaign.root = tmp_path
    campaign.state = dict(phase="ADVANCE", final_checkpoint={"sha256": "a" * 64},
        logging_receipt={"online_metrics_verified": True}, mse_completion={"completed_iterations": 100},
        evaluation_report={"purpose": "test", "status": "complete", "completed": 295,
                           "checkpoint_sha256": "a" * 64, "successes": 60, "goal_reached_rate": 60 / 295})
    campaign._guard = lambda: None; campaign.validate_production_test_evidence = lambda: None
    campaign.get_controller = lambda: SimpleNamespace(mse=SimpleNamespace(context=lambda: {
        "status": "complete", "selection": {"best_checkpoint_sha256": "a" * 64}}))
    campaign.save = lambda phase, **kwargs: campaign.state.update(phase=phase, **kwargs)
    result = campaign.tick()
    assert result["phase"] == "COMPLETE" and len(result["completed_rounds"]) == 1
    receipt = json.loads((tmp_path / "round1-continuation-completion.json").read_text())
    assert receipt["automatic_next_round"] is False and receipt["compute_released"] is False


@pytest.mark.parametrize("phase", ["collect", "release", "evaluate"])
def test_every_phase_pins_same_pool_and_retained_owner(tmp_path, phase):
    site = publish(tmp_path / "site.json", {})
    retained = publish(tmp_path / "retained.json", {})
    pool = publish(tmp_path / "pool.json", {"native": "fresh"})
    argv = ["/python", "/retained_campaign_site.py", "--site", site["path"], "--site-sha256", site["raw_sha256"],
        "--retained", retained["path"], "--retained-sha256", retained["raw_sha256"],
        "--resident-pool", "/home/ldu/new-pool.json", "--resident-pool-sha256", digest(bound(pool))]
    commands = {name: {"argv": list(argv)} for name in ("collect", "release", "evaluate")}
    spec = dict(site=site, retained=retained, resident_pool=pool,
        operations=publish(tmp_path / "operations.json", {"commands": commands}))
    assert continuation.validate_operation_bindings(spec) == "/home/ldu/new-pool.json"
    changed = deepcopy(commands); changed[phase]["argv"][-1] = "f" * 64
    spec["operations"] = publish(tmp_path / "changed.json", {"commands": changed})
    with pytest.raises(ValueError, match="cannot substitute"):
        continuation.validate_operation_bindings(spec)


@pytest.mark.parametrize("reuse", [None, "jobs", "runtime", "container", "generation"])
def test_fresh_native_namespace_is_enforced(monkeypatch, reuse):
    from pipeline.deployment.rtx_actor import production_resident_pool
    monkeypatch.setattr(production_resident_pool, "validate_pool", lambda value: value)
    old = dict(source_sha256="old", jobs_root="/jobs/old", workers=[dict(
        worker_instance_id=f"old-{gpu}", runtime_root=f"/native/old-{gpu}", container_name=f"old-{gpu}") for gpu in (0, 1)])
    new = dict(source_sha256="new", jobs_root="/jobs/new", workers=[dict(
        worker_instance_id=f"new-{gpu}", runtime_root=f"/native/new-{gpu}", container_name=f"new-{gpu}") for gpu in (0, 1)])
    state = dict(config={"source_sha256": "old"}, service_binding={"native_phase_barrier": {"workers": old["workers"]}})
    if reuse == "jobs": new["jobs_root"] = old["jobs_root"] + "/descendant"
    if reuse == "runtime": new["workers"][0]["runtime_root"] = old["workers"][0]["runtime_root"]
    if reuse == "container": new["workers"][0]["container_name"] = old["workers"][0]["container_name"]
    if reuse == "generation": new["workers"][0]["worker_instance_id"] = old["workers"][0]["worker_instance_id"]
    if reuse:
        with pytest.raises(ValueError, match="fresh remote|new native"):
            continuation._validate_pool(new, state, {"source_sha256": "new"}, old)
    else:
        assert continuation._validate_pool(new, state, {"source_sha256": "new"}, old) == new
