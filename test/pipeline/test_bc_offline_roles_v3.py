"""CPU-only artifact scheduling tests; synthetic bytes are not production proof."""
from copy import deepcopy
import hashlib
import json
import importlib.util
from pathlib import Path

import pytest

from pipeline.legacy.bc_collection.artifact_roles import panel_reference
from pipeline.legacy.bc_collection.sampling_plan import build_plan, build_offline_plan, canonical, validate_selection_plan
from pipeline.legacy.bc_collection.ingest_native_cohorts import build_pool
from pipeline.legacy.bc_collection.plan_native_sessions import build_schedule
from pipeline.bc_training.create_bc_splits import create_splits
from pipeline.bc_training import data_admission as admission
from test.pipeline.test_bc_sampling_plan import inventory
from test.pipeline import test_bc_native_ingestion as legacy_ingestion


def put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(doc) + b"\n")
    return {"path": str(path), "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def make_panel(tmp_path, inv, *, seed=1071042203):
    name = "new-offline-v3-test"
    rows = inv["motions"]
    cohorts = []
    inv_sha = hashlib.sha256(canonical(inv)).hexdigest()
    for index, start in enumerate(range(0, len(rows), 2)):
        chosen = rows[start:start+2]
        cohorts.append(dict(assignment=dict(schema="roborender.native-motion-cohort.v1", run_id=name,
            cohort_id=f"cohort-{index}", purpose="test", simulator_seed=seed+index,
            motion_inventory_sha256=inv_sha, lanes=[dict(lane=i, motion_id=r["motion_id"],
                path="/motions/"+r["filename"], sha256=r["sha256"]) for i, r in enumerate(chosen)]),
            case_ids=[f"case-{start+i}" for i in range(len(chosen))]))
    panel = dict(schema="roborender.explicit-motion-success-panel.v1", panel_id=name, purpose="test",
        motion_inventory_sha256=inv_sha, observation_mode="raytraced_3view",
        action_mode="deterministic_action_mean_11_plus_homie_15", metric="native_goal_or_until_done",
        max_vector_steps_per_cohort=450, episode_count=len(rows), cohorts=cohorts)
    return panel, put(tmp_path / "success-panel.json", panel)


def v3_inputs(tmp_path, monkeypatch):
    def builder(inv, **kwargs):
        _, binding = make_panel(tmp_path, inv)
        return build_offline_plan(inv, seed=kwargs["seed"], num_envs=kwargs["num_envs"],
            success_panel=binding, train_chunks=16, validation_chunks=2)
    monkeypatch.setattr(legacy_ingestion, "build_plan", builder)
    return legacy_ingestion.inputs(tmp_path, "matched_exposure_v2")


def test_full295_plan_schedules_2880_offline_chunks_and_no_test_rgb(tmp_path):
    inv = inventory()
    _, binding = make_panel(tmp_path, inv)
    plan = build_offline_plan(inv, seed=41, num_envs=6, success_panel=binding)
    assert plan["schema"].endswith(".v3") and plan["total_target_chunks"] == 2880
    assert {role: value["target_chunks"] for role, value in plan["partitions"].items()} == {"train": 2560, "validation": 320}
    assert plan["artifact_roles"]["offline_test_chunks"] == 0
    assert plan["artifact_roles"]["closed_loop_test"]["episode_count"] == 295
    old = build_plan(inv, seed=41, num_envs=6, selection_policy="matched_exposure_v2", success_cases=295)
    assert all(plan["partitions"][role] == old["partitions"][role] for role in ("train", "validation"))
    assert plan["collection_cost"]["minimum_vector_steps_across_cohorts"] < old["collection_cost"]["minimum_vector_steps_across_cohorts"]
    schedule = build_schedule(plan, run_id="bc-leo-v3-test", motion_root="/motions")
    assert sum(row["target_chunks"] for row in schedule) == 2880
    requests = [json.loads(raw) for s in schedule for raw in s["requests"].values()]
    assert {r["split"] for r in requests} == {"train", "validation"}
    assert all(r["schema"].endswith(".v2") for r in requests)  # Native buffer/selection implementation unchanged.


def test_v3_ingest_splits_and_qualification_keep_final_test_external(tmp_path, monkeypatch):
    plan, cohorts, transfers, dataset = v3_inputs(tmp_path, monkeypatch)
    manifest, _ = build_pool(plan, cohorts, transfers, dataset_root=dataset, expected_total=18)
    assert manifest["partition_counts"] == {"train": 16, "validation": 2}
    assert manifest["artifact_roles"] == plan["artifact_roles"]
    assert all(r["split"] != "test" for r in manifest["chunks"])
    assert any(r["emitted_chunk_ordinal"] > 0 for r in manifest["chunks"] if r["split"] == "validation")
    for row in manifest["chunks"]:
        row.update(render_contract=admission.INDEPENDENT_RECIPE, render_chain_id=row["chunk_id"],
            render_chain_length=1, render_chain_index=0, render_inherits_previous_reference=False)
    source = dataset / "manifest.json"
    put(source, manifest)
    split_dir = dataset / "splits"
    created = create_splits(dataset, source_manifest_path=source, split_dir=split_dir)
    assert set(created["files"]) == {"train", "validation"}
    assert not (split_dir / "test.json").exists()
    assert (split_dir / "success_test.json").is_file()
    reference = json.loads((split_dir / "success_test.json").read_bytes())
    assert reference["offline_payload_count"] == 0 and reference["executed"] is False
    paths = {r: split_dir / (r + ".json") for r in ("train", "validation")}
    original_open = Path.open
    def no_payload(path, *args, **kwargs):
        assert path.suffix != ".npz"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_payload)
    report = admission.audit_split_files(source, paths, require_independent_render=True,
        purpose="qualification", run_name="smoke-v3-offline")
    assert report["schema"].endswith(".v3") and report["qualification_admitted"]
    assert not report["production_admitted"] and not report["closed_loop_test_executed"]
    report_path = dataset / "admission.json"
    put(report_path, report)
    assert admission.validate_training_admission(report_path, paths["train"], paths["validation"],
        allow_qualification=True, run_name="smoke-v3-offline") == report


@pytest.mark.parametrize("mutation", ["changed_panel", "missing_mode", "seed_collision", "prefix", "wrong_quota", "extra_test"])
def test_v3_rejects_panel_or_selection_drift(tmp_path, mutation):
    inv = inventory(2)
    panel, binding = make_panel(tmp_path, inv)
    plan = build_offline_plan(inv, seed=17, num_envs=2, success_panel=binding,
        train_chunks=16, validation_chunks=2)
    if mutation == "changed_panel":
        Path(binding["path"]).write_bytes(b"changed")
    elif mutation == "missing_mode":
        panel["cohorts"][0]["assignment"]["lanes"].pop()
        panel["cohorts"][0]["case_ids"].pop()
        panel["episode_count"] -= 1
        plan["artifact_roles"]["closed_loop_test"]["panel"] = put(Path(binding["path"]), panel)
    elif mutation == "seed_collision":
        panel["cohorts"][0]["assignment"]["simulator_seed"] = plan["partitions"]["train"]["cohorts"][0]["process_seed"]
        plan["artifact_roles"]["closed_loop_test"]["panel"] = put(Path(binding["path"]), panel)
    elif mutation == "prefix":
        slot = plan["partitions"]["validation"]["cohorts"][0]["slots"][0]
        slot["selected_chunk_ordinals"] = [0 if slot["selected_chunk_ordinals"] != [0] else 1]
    elif mutation == "wrong_quota": plan["artifact_roles"]["offline_chunk_counts"]["validation"] = 3
    else: plan["partitions"]["test"] = deepcopy(plan["partitions"]["validation"])
    with pytest.raises(ValueError): validate_selection_plan(plan)


def test_legacy_v2_still_requires_three_complete_offline_roles(tmp_path, monkeypatch):
    plan, cohorts, transfers, dataset = legacy_ingestion.inputs(tmp_path, "matched_exposure_v2")
    cohorts.pop()
    transfers.pop()
    with pytest.raises(ValueError, match="all and only"):
        build_pool(plan, cohorts, transfers, dataset_root=dataset, expected_total=20)


@pytest.mark.parametrize("mutation", ["missing_cohort", "wrong_ordinal", "role_seed_collision", "quota", "test_reentry"])
def test_v3_native_ingestion_rejects_incomplete_or_relabelled_population(tmp_path, monkeypatch, mutation):
    plan, cohorts, transfers, dataset = v3_inputs(tmp_path, monkeypatch)
    if mutation == "missing_cohort": cohorts.pop()
    elif mutation == "wrong_ordinal": cohorts[-1]["document"]["chunks"][0]["emitted_chunk_ordinal"] += 1
    elif mutation == "role_seed_collision": cohorts[-1]["document"]["request"]["process_seed"] = cohorts[0]["document"]["request"]["process_seed"]
    elif mutation == "quota": cohorts[-1]["document"]["request"]["chunks_per_env"][0] += 1
    else: cohorts[-1]["document"]["request"]["split"] = "test"
    with pytest.raises(ValueError):
        build_pool(plan, cohorts, transfers, dataset_root=dataset, expected_total=18)


def test_v3_panel_raw_hash_is_rechecked_on_resume(tmp_path, monkeypatch):
    plan, cohorts, transfers, dataset = v3_inputs(tmp_path, monkeypatch)
    manifest, _ = build_pool(plan, cohorts, transfers, dataset_root=dataset, expected_total=18)
    for row in manifest["chunks"]:
        row.update(render_contract=admission.INDEPENDENT_RECIPE, render_chain_id=row["chunk_id"],
            render_chain_length=1, render_chain_index=0, render_inherits_previous_reference=False)
    source = dataset / "manifest.json"
    put(source, manifest)
    split_dir = dataset / "splits"
    create_splits(dataset, source_manifest_path=source, split_dir=split_dir)
    paths = {r: split_dir / (r + ".json") for r in ("train", "validation")}
    report = admission.audit_split_files(source, paths, require_independent_render=True,
        purpose="qualification", run_name="smoke-v3-resume")
    report_path = dataset / "admission.json"
    put(report_path, report)
    Path(plan["artifact_roles"]["closed_loop_test"]["panel"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="panel bytes changed"):
        admission.validate_training_admission(report_path, paths["train"], paths["validation"],
            allow_qualification=True, run_name="smoke-v3-resume")


def test_full295_v3_production_metadata_chain_requires_external_test_binding(tmp_path, monkeypatch):
    """Synthetic complete receipts exercise gates; no real assets/runtime claimed."""
    from pipeline.bc_training import production_protocol as protocol
    legacy_path = Path(__file__).with_name("test_purpose_aware_admission.py")
    spec = importlib.util.spec_from_file_location("isolated_v2_fixture", legacy_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    def builder(inv, **kwargs):
        _, binding = make_panel(tmp_path, inv)
        return build_offline_plan(inv, seed=kwargs["seed"], num_envs=kwargs["num_envs"], success_panel=binding)
    monkeypatch.setattr(fixture, "build_plan", builder)
    original_pool = fixture.build_pool
    def pool(plan, *args, **kwargs):
        return original_pool(plan, *args, expected_total=plan["total_target_chunks"], **kwargs)
    monkeypatch.setattr(fixture, "build_pool", pool)
    complete = fixture.make_complete(tmp_path)
    p = complete.protocol
    p.update(schema=protocol.SCHEMA_V3, closed_loop_test_panel=complete.plan["artifact_roles"]["closed_loop_test"]["panel"])
    p["capacity_receipts"] = [b for b in p["capacity_receipts"] if protocol.bound(b)["collection_role"] != "test"]
    session = protocol.bound(p["session_execution_receipts"][0])
    session["scene_initializations"] = len(p["cohort_execution_receipts"])
    p["session_execution_receipts"] = [put(tmp_path / "correct-v3-session.json", session)]
    updated = []
    for i, binding in enumerate(p["cohort_execution_receipts"]):
        doc = protocol.bound(binding)
        doc["session_execution_canonical_sha256"] = protocol.sha(canonical(session))
        updated.append(put(tmp_path / f"correct-v3-cohort-{i}.json", doc))
    p["cohort_execution_receipts"] = updated
    binding = put(tmp_path / "v3-production.json", p)
    paths = {r: path for r, path in complete.splits.items() if r != "test"}
    report = admission.audit_split_files(complete.manifest, paths, require_independent_render=True,
        purpose="production", run_name="prod-v3-fixture", protocol_path=binding["path"], protocol_sha256=binding["raw_sha256"],
        render_receipt_path=complete.rb["path"], render_receipt_sha256=complete.rb["raw_sha256"])
    assert report["production_admitted"] and report["schema"] == admission.SCHEMA_V3
    assert report["protocol"]["exact_role_quotas"] == {"train": 2560, "validation": 320}
    assert report["artifact_roles"]["closed_loop_test"]["episode_count"] == 295
    assert report["closed_loop_test_executed"] is False
    p["closed_loop_test_panel"] = {**p["closed_loop_test_panel"], "raw_sha256": "0"*64}
    wrong = put(tmp_path / "wrong-external-test.json", p)
    with pytest.raises(ValueError, match="panel bytes changed"):
        protocol.validate_protocol_inputs(wrong["path"], wrong["raw_sha256"])
