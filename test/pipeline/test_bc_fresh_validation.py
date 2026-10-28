from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.legacy.bc_collection import fresh_validation as fresh
from pipeline.legacy.bc_collection.ingest_native_cohorts import build_pool
from pipeline.legacy.bc_collection.plan_native_sessions import build_schedule, write_schedule
from pipeline.deployment.rtx_actor.isaac6_bc_production import (
    NATIVE_COLLECTOR_SHA256, canonical, source_run_tag,
)


def write_bound(path, value):
    path.write_bytes(canonical(value) + b"\n")
    return fresh.binding(path)


def inputs(tmp_path):
    motions = [dict(motion_id=f"motion-{i:03d}", filename=f"motion-{i:03d}.npz", bytes=123,
                    sha256=hashlib.sha256(f"motion{i}".encode()).hexdigest()) for i in range(295)]
    inventory = dict(schema="roborender.motion-inventory.v1", motion_count=295, motions=motions,
                     motion_records_sha256=hashlib.sha256(canonical(motions)).hexdigest())
    inv = write_bound(tmp_path / "inventory.json", inventory)
    rows = [dict(chunk_id=f"old-{i}", frames=81, source_sha256=hashlib.sha256(f"old{i}".encode()).hexdigest(),
                 simulator_seed=(None if i < 490 else 2 + i % 7)) for i in range(2000)]
    old = write_bound(tmp_path / "legacy.json", dict(chunk_count=2000, chunks=rows))
    panel_id = "separate-native-success-test"
    inv_sha = hashlib.sha256(canonical(inventory)).hexdigest()
    cohorts = []
    for index, start in enumerate(range(0, 295, 2)):
        selected = motions[start:start+2]
        assignment = dict(schema="roborender.native-motion-cohort.v1", run_id=panel_id,
            cohort_id=f"test-{index}", purpose="test", simulator_seed=10000+index,
            motion_inventory_sha256=inv_sha,
            lanes=[dict(lane=i, motion_id=row["motion_id"], path="/motions/"+row["filename"],
                        sha256=row["sha256"]) for i, row in enumerate(selected)])
        cohorts.append(dict(assignment=assignment, case_ids=[f"test-{start+i}" for i in range(len(selected))]))
    panel = dict(schema="roborender.explicit-motion-success-panel.v1", panel_id=panel_id, purpose="test",
        motion_inventory_sha256=inv_sha, observation_mode="raytraced_3view",
        action_mode="deterministic_action_mean_11_plus_homie_15", metric="native_goal_or_until_done",
        max_vector_steps_per_cohort=450, episode_count=295, cohorts=cohorts)
    test = write_bound(tmp_path / "test-panel.json", panel)
    return dict(motion_inventory=inv, legacy_manifest=old, closed_loop_test_panel=test, seed=2026101519)


def test_validation_only_balanced_fresh_seed_and_nonprefix_selection(tmp_path):
    args = inputs(tmp_path)
    plan = fresh.build_plan(**args)
    assert fresh.validate_plan(plan) == plan
    assert fresh.build_plan(**args) == plan
    assert set(plan["partitions"]) == {"validation"}
    cohorts = plan["partitions"]["validation"]["cohorts"]
    slots = [s for c in cohorts for s in c["slots"]]
    assert len(cohorts) == 50 and {c["num_envs"] for c in cohorts} == {1, 6}
    assert len(slots) == 295 and sum(s["target_chunks"] for s in slots) == 320
    assert sum(s["target_chunks"] == 2 for s in slots) == 25
    assert {s["exposure_span_chunks"] for s in slots} == {9}
    assert sum(s["selected_chunk_ordinals"] != list(range(s["target_chunks"])) for s in slots) > 230
    seeds = {c["process_seed"] for c in cohorts}
    assert len(seeds) == 50
    assert not seeds.intersection(plan["legacy_training"]["known_process_seeds"])
    assert not seeds.intersection(plan["closed_loop_test"]["simulator_seeds"])
    assert plan["legacy_training"]["unknown_seed_chunks"] == 490
    assert plan["new_training_chunks"] == plan["offline_test_chunks"] == 0
    assert plan["production_admitted"] is plan["gpu_capacity_qualified"] is False
    assert "not proven matched" in plan["sampling_interpretation"]


@pytest.mark.parametrize("field,value", [("chunks", 2880), ("chunks", True), ("num_envs", 8),
    ("exposure_span_chunks", 1), ("seed", True), ("seed", -1)])
def test_parameters_do_not_silently_change_population_or_capacity(tmp_path, field, value):
    args = inputs(tmp_path)
    args[field] = value
    with pytest.raises(ValueError):
        fresh.build_plan(**args)


@pytest.mark.parametrize("mutation", [
    lambda p: p["partitions"].update(train={}),
    lambda p: p.update(new_training_chunks=1),
    lambda p: p["partitions"]["validation"]["cohorts"][0]["slots"][0].update(selected_chunk_ordinals=[0]),
    lambda p: p["partitions"]["validation"]["cohorts"][0].update(process_seed=2),
    lambda p: p["partitions"]["validation"]["cohorts"].pop(),
    lambda p: p["legacy_training"].update(unknown_seed_chunks=0),
    lambda p: p.update(production_admitted=True),
])
def test_frozen_recipe_rejects_role_seed_ordinal_and_claim_changes(tmp_path, mutation):
    plan = fresh.build_plan(**inputs(tmp_path))
    mutation(plan)
    with pytest.raises(ValueError, match="recipe"):
        fresh.validate_plan(plan)


def test_input_bytes_are_bound_not_just_declared_names(tmp_path):
    plan = fresh.build_plan(**inputs(tmp_path))
    path = Path(plan["legacy_manifest"]["path"])
    value = json.loads(path.read_bytes())
    value["chunks"][0]["simulator_seed"] = 99
    path.write_bytes(canonical(value))
    with pytest.raises(ValueError, match="metadata changed"):
        fresh.validate_plan(plan)


def test_known_seed_collision_is_rejected_not_repaired_by_guessing(tmp_path):
    args = inputs(tmp_path)
    plan = fresh.build_plan(**args)
    seed = plan["partitions"]["validation"]["cohorts"][0]["process_seed"]
    with pytest.raises(ValueError, match="seed collision"):
        fresh.build_plan(**args, excluded_process_seeds=[seed])


def test_schedule_has_only_native_validation_requests_and_exact_tail(tmp_path):
    plan = fresh.build_plan(**inputs(tmp_path))
    sessions = build_schedule(plan, run_id="bc-leo-reuse2k-val320-test", motion_root="/motions",
                              rtx_gpus=(0,), cohorts_per_session=4)
    assert len(sessions) == 14  # 49 B6 cohorts in13sessions, plus one B1.
    assert sum(s["target_chunks"] for s in sessions) == 320
    assert all(s["physical_gpu"] == 0 for s in sessions)
    by_id = {c["cohort_id"]: c for c in plan["partitions"]["validation"]["cohorts"]}
    for session in sessions:
        assert 1 <= len(session["requests"]) <= 4
        for raw in session["requests"].values():
            request = json.loads(raw)
            cohort = by_id[request["cohort_id"]]
            assert request["schema"] == "roborender.bc-native-collection-request.v2"
            assert request["split"] == "validation"
            assert request["sampling_plan_sha256"] == hashlib.sha256(canonical(plan)).hexdigest()
            assert request["selected_chunk_ordinals_per_env"] == [s["selected_chunk_ordinals"] for s in cohort["slots"]]
            assert request["max_windows"] == cohort["minimum_complete_chunk_windows"] * 3 + 4
    result = write_schedule(tmp_path / "schedule", plan, sessions)
    assert result["target_chunks"] == 320 and result["production_admitted"] is False


def native_envelopes(plan, root):
    """Synthetic metadata fixture only; never substituted for runtime evidence."""
    cohorts, transfers = [], []
    plan_sha = hashlib.sha256(canonical(plan)).hexdigest()
    for ci, cohort in enumerate(plan["partitions"]["validation"]["cohorts"]):
        request = fresh.request_from_cohort(plan, cohort, run_id="bc-leo-fixture-only", motion_root="/motions",
            output_dir=f"/native/{cohort['cohort_id']}", rank=0, global_env_offset=ci*6, max_windows=31)
        request_sha = hashlib.sha256(canonical(request)).hexdigest()
        chunks, moved, emissions = [], [], []
        for lane, slot in enumerate(cohort["slots"]):
            count = max(slot["selected_chunk_ordinals"]) + 1
            emissions.append(count)
            for ordinal in range(count):
                source = f"/native/{cohort['cohort_id']}/chunk-{ci}-{lane}-{ordinal}.npz"
                sha = hashlib.sha256(source.encode()).hexdigest()
                chosen = ordinal in slot["selected_chunk_ordinals"]
                row = dict(local_env_id=lane, global_env_id=ci*6+lane, source_run_tag=source_run_tag(request),
                    rank=0, seq=ordinal, source_path=source, source_sha256=sha, source_bytes=123, frames=81,
                    motion_id=slot["motion_id"], motion_sha256=slot["motion_sha256"], simulator_seed=request["process_seed"],
                    split="validation", sampling_plan_sha256=plan_sha, request_sha256=request_sha, teacher_only=True,
                    episode_id=f"{source_run_tag(request)}-lane{lane}-episode0", first_episode_step_index=ordinal*81,
                    last_episode_step_index=ordinal*81+80, emitted_chunk_ordinal=ordinal,
                    selection_policy="explicit_preselected_ordinals_v2", selected_for_split=chosen, overcollection=not chosen)
                chunks.append(row)
                if chosen:
                    moved.append(dict(source_path=source, source_sha256=sha, destination_sha256=sha,
                        source_bytes=123, destination_bytes=123, destination_path=str(root/"isaac"/Path(source).name)))
        doc = dict(schema="roborender.bc-native-cohort.v1", status="complete", request=request,
            request_sha256=request_sha, vector_steps=max(emissions)*81, optimizer_steps=0, teacher_only=True,
            native_payload_writer_unchanged=True, native_collector_sha256=NATIVE_COLLECTOR_SHA256,
            model_assets={key:dict(path="/models/"+key+".pt",sha256=str(i)*64)
                          for i,key in enumerate(("teacher","homie_walk","homie_stand"),start=1)},
            chunks=chunks, emitted_per_lane=emissions, selected_chunks=cohort["target_chunks"],
            overcollection_chunks=sum(emissions)-cohort["target_chunks"])
        envelope = dict(document=doc, sha256=hashlib.sha256(canonical(doc)).hexdigest())
        transfer = dict(schema="roborender.bc-cohort-transfer.v1", verified=True, cohort_id=cohort["cohort_id"],
                        cohort_manifest_sha256=envelope["sha256"],files=moved)
        cohorts.append(envelope)
        transfers.append(dict(document=transfer,sha256=hashlib.sha256(canonical(transfer)).hexdigest()))
    return cohorts, transfers


def test_ingestion_reuses_native_checks_without_fabricating_train_or_test(tmp_path):
    plan = fresh.build_plan(**inputs(tmp_path))
    root = tmp_path / "data"
    cohorts, transfers = native_envelopes(plan, root)
    manifest, seal = build_pool(plan, cohorts, transfers, dataset_root=root,
                                expected_total=320, verify_destination_stats=False)
    assert manifest["partition_counts"] == {"validation":320}
    assert manifest["chunk_count"] == 320 and manifest["motion_count"] == 295
    assert manifest["collection_purpose"] == "fresh_validation_only"
    assert manifest["sampling_plan_schema"] == fresh.SCHEMA
    assert manifest["legacy_training_manifest"] == plan["legacy_manifest"]
    assert manifest["production_admitted"] is False and seal["payloads_read"] is False
    assert len(manifest["overcollection_excluded"]) > 320


@pytest.mark.parametrize("mutation", [
    lambda c,t: c.pop(), lambda c,t: t.pop(),
    lambda c,t: c[0]["document"]["request"].update(split="train"),
    lambda c,t: c[0]["document"]["chunks"][0].update(emitted_chunk_ordinal=99),
    lambda c,t: t[0]["document"]["files"][0].update(destination_sha256="f"*64),
    lambda c,t: c[0]["document"].update(teacher_only=False),
])
def test_validation_only_ingestion_rejects_missing_or_changed_native_proof(tmp_path, mutation):
    plan = fresh.build_plan(**inputs(tmp_path))
    root = tmp_path / "data"
    cohorts, transfers = native_envelopes(plan, root)
    mutation(cohorts, transfers)
    with pytest.raises(ValueError):
        build_pool(plan, cohorts, transfers, dataset_root=root, expected_total=320, verify_destination_stats=False)


def test_default_ingestion_never_adopts_new_validation_only_plan(tmp_path):
    plan = fresh.build_plan(**inputs(tmp_path))
    with pytest.raises(ValueError, match="sampling-plan size"):
        build_pool(plan, [], [], dataset_root=tmp_path)
