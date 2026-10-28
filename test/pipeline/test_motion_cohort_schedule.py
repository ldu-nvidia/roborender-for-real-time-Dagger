from copy import deepcopy
import hashlib

import pytest

from pipeline.legacy.bc_collection.sampling_plan import build_plan, build_offline_plan, canonical
from pipeline.dagger.motion_cohort_schedule import build_schedule, build_grouped_success_panel


def fixture(count=2):
    rows = [{"motion_id": f"motion-{i}", "filename": f"motion-{i}.npz", "bytes": i + 1,
             "sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(count)]
    inv = {"schema": "roborender.motion-inventory.v1", "motion_count": count, "motions": rows,
           "motion_records_sha256": hashlib.sha256(canonical(rows)).hexdigest()}
    return inv, build_plan(inv, seed=1710, num_envs=2 if count == 2 else 28,
                          chunks=20 if count == 2 else 3200, success_cases=2 if count == 2 else 320)


def schedule(inv, plan, **kwargs):
    return build_schedule(inv, plan, run_prefix="smoke-matched", seed=1711,
        num_envs=kwargs.pop("num_envs", 2), windows_per_training_cohort=kwargs.pop("windows", 1),
        motion_root="/native/motions", **kwargs)


def test_two_motion_smoke_common_population_disjoint_native_seeds():
    inv, plan = fixture()
    result = schedule(inv, plan)
    assert result["plan_only"] is True and result["production_admitted"] is False
    assert result["bc_partition_chunks"] == {"train": 16, "validation": 2, "test": 2}
    seeds = {c["process_seed"] for p in plan["partitions"].values() for c in p["cohorts"]}
    testseeds = {c["process_seed"] for c in plan["success_test"]["cases"]}
    seeds |= testseeds
    for i, round in enumerate(result["rounds"], 1):
        assert round["teacher_beta"] == 1 - i * .25
        assert round["planned_optimizer_steps"] == 1
        assert round["planned_training_transitions"] == 162
        for role in ("training", "heldout"):
            cohort, = round[role]
            assert cohort["vector_steps"] == 81 and cohort["num_envs"] == 2
            assert cohort["reset_at_window_boundary"] is False
            assert {x["motion_id"] for x in cohort["assignment"]["lanes"]} == set(result["motion_support"])
            seed = cohort["assignment"]["simulator_seed"]
            assert seed not in seeds
            seeds.add(seed)
            assert cohort["optimizer_steps"] == (1 if role == "training" else 0)
    assert {c["process_seed"] for c in result["success_test"]["cases"]} == testseeds
    assert all(c["num_envs"] == 1 and c["completed_episodes"] == 1 for c in result["success_test"]["cases"])
    assert schedule(inv, plan) == result


def test_full_295_coverage_does_not_mislabel_50_per_cohort_as_50_total():
    inv, plan = fixture(295)
    result = schedule(inv, plan, num_envs=28, windows=50)
    for round in result["rounds"]:
        assert len(round["training"]) == len(round["heldout"]) == 11
        assert round["planned_optimizer_steps"] == 550
        assert round["planned_training_transitions"] == 295 * 50 * 81
        assert len({c["assignment"]["run_id"] for c in round["training"]}) == 11
        assert all(c["initial_checkpoint_binding"] == "previous_completed_cohort_student_and_optimizer"
                   for c in round["training"][1:])
        for role in ("training", "heldout"):
            lanes = [lane for c in round[role] for lane in c["assignment"]["lanes"]]
            assert len(lanes) == len({r["motion_id"] for r in lanes}) == 295
        assert round["training"][-1]["num_envs"] == 15
        assert all(c["vector_steps"] == 50 * 81 for c in round["training"])
    assert result["success_test"]["completed_episodes"] == 320
    assert len({c["motion_id"] for c in result["success_test"]["cases"]}) == 295


@pytest.mark.parametrize("mutation", [
    lambda i, p: p["partitions"]["test"]["cohorts"][0]["slots"][0].update(motion_sha256="0" * 64),
    lambda i, p: p["partitions"]["validation"]["cohorts"][0].update(process_seed=p["partitions"]["train"]["cohorts"][0]["process_seed"]),
    lambda i, p: p["success_test"]["cases"][0].update(process_seed=p["partitions"]["train"]["cohorts"][0]["process_seed"]),
    lambda i, p: p["success_test"]["cases"][0].update(motion_sha256="0" * 64),
    lambda i, p: i["motions"][0].update(filename="../motion-0.npz"),
    lambda i, p: p.update(frames_per_chunk=80),
])
def test_schedule_rejects_distribution_or_seed_mismatch(mutation):
    inv, plan = fixture()
    mutation(inv, plan)
    with pytest.raises(ValueError):
        schedule(inv, plan)


@pytest.mark.parametrize("kwargs", [
    {"windows": 0}, {"windows": True}, {"num_envs": 41}, {"physical_gpus": [2]}, {"physical_gpus": [0, 0]},
])
def test_schedule_rejects_unsupported_implicit_topology(kwargs):
    inv, plan = fixture()
    with pytest.raises(ValueError):
        schedule(inv, plan, **kwargs)


def test_new_grouped_panel_preserves_old_cases_and_matches_explicit_evaluator():
    from pipeline.deployment.rtx_actor.explicit_motion_panel import validate_panel
    inv, plan = fixture(295)
    before = deepcopy(plan)
    scheduling = schedule(inv, plan, num_envs=28)
    excluded = [c["assignment"]["simulator_seed"] for r in scheduling["rounds"]
                for role in ("training", "heldout") for c in r[role]]
    panel = build_grouped_success_panel(inv, plan, panel_id="new-grouped-test", seed=1712,
        num_envs=28, motion_root="/native/motions", excluded_seeds=excluded)
    validate_panel(panel)
    assert plan == before and len(plan["success_test"]["cases"]) == 320
    assert panel["episode_count"] == 295 and len(panel["cohorts"]) == 11
    assert [len(c["case_ids"]) for c in panel["cohorts"]] == [28] * 10 + [15]
    assert len({c["assignment"]["simulator_seed"] for c in panel["cohorts"]}) == 11
    assert not set(excluded) & {c["assignment"]["simulator_seed"] for c in panel["cohorts"]}
    assert all(c["assignment"]["purpose"] == "test" for c in panel["cohorts"])


def offline_fixture(tmp_path, count=2):
    inv, legacy = fixture(count)
    panel = build_grouped_success_panel(inv, legacy, panel_id="v3-native-test", seed=1712,
        num_envs=2 if count == 2 else 6, motion_root="/native/motions")
    path = tmp_path / "native-test-panel.json"
    path.write_bytes(canonical(panel))
    plan = build_offline_plan(inv, seed=1710, num_envs=2 if count == 2 else 6,
        train_chunks=16 if count == 2 else 2560,
        validation_chunks=2 if count == 2 else 320,
        success_panel={"path": str(path), "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return inv, plan, panel, path


def test_v3_preserves_separate_native_test_panel_and_uses_only_two_offline_roles(tmp_path):
    inv, plan, panel, path = offline_fixture(tmp_path)
    before = deepcopy(plan)
    panel_bytes = path.read_bytes()
    result = schedule(inv, plan)
    assert result["plan_only"] and not result["production_admitted"]
    assert result["bc_partition_chunks"] == {"train": 16, "validation": 2}
    success = result["success_test"]
    assert success["schema"] == "roborender.matched-motion-success-plan.v2"
    assert not success["native_executor_qualified"]
    assert "cases" not in success
    assert success["completed_episodes"] == 2
    assert success["panel_reference"] == plan["artifact_roles"]["closed_loop_test"]["panel"]
    assert success["panel_canonical_sha256"] == hashlib.sha256(canonical(panel)).hexdigest()
    assert [{k: v for k, v in c.items() if k != "physical_gpu_index"}
            for c in success["cohorts"]] == panel["cohorts"]
    used = {c["process_seed"] for p in plan["partitions"].values() for c in p["cohorts"]}
    used.update(c["assignment"]["simulator_seed"] for c in panel["cohorts"])
    for round in result["rounds"]:
        for role in ("training", "heldout"):
            cohort, = round[role]
            process_seed = cohort["assignment"]["simulator_seed"]
            assert process_seed not in used
            used.add(process_seed)
            assert {x["motion_id"] for x in cohort["assignment"]["lanes"]} == set(result["motion_support"])
    assert plan == before and path.read_bytes() == panel_bytes
    assert schedule(inv, plan) == result


def test_v3_full_population_keeps_grouped_test_seeds_and_exact_update_count(tmp_path):
    inv, plan, panel, _ = offline_fixture(tmp_path, 295)
    result = schedule(inv, plan, num_envs=6)
    assert result["bc_partition_chunks"] == {"train": 2560, "validation": 320}
    assert result["success_test"]["completed_episodes"] == 295
    assert len(result["success_test"]["cohorts"]) == 50
    for round in result["rounds"]:
        assert round["planned_optimizer_steps"] == 50
        assert round["planned_training_transitions"] == 295 * 81
        assert [c["num_envs"] for c in round["training"]] == [6] * 49 + [1]
        for role in ("training", "heldout"):
            assert {lane["motion_id"] for c in round[role]
                    for lane in c["assignment"]["lanes"]} == set(result["motion_support"])
    assert [c["assignment"]["simulator_seed"] for c in result["success_test"]["cohorts"]] == [
        c["assignment"]["simulator_seed"] for c in panel["cohorts"]]


def test_v3_changed_native_panel_bytes_fail_closed(tmp_path):
    inv, plan, _, path = offline_fixture(tmp_path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        schedule(inv, plan)


@pytest.mark.parametrize("mutation", [
    lambda p: p["partitions"].update(test=deepcopy(p["partitions"]["validation"])),
    lambda p: p["artifact_roles"]["closed_loop_test"].update(panel_canonical_sha256="0" * 64),
    lambda p: p["partitions"]["validation"]["cohorts"][0].update(
        process_seed=p["partitions"]["train"]["cohorts"][0]["process_seed"]),
])
def test_v3_rejects_legacy_split_unbound_panel_or_overlapping_offline_seeds(tmp_path, mutation):
    inv, plan, _, _ = offline_fixture(tmp_path)
    mutation(plan)
    with pytest.raises(ValueError):
        schedule(inv, plan)


def test_v3_cannot_replace_pinned_native_panel_as_planner_side_effect(tmp_path):
    inv, plan, _, _ = offline_fixture(tmp_path)
    with pytest.raises(ValueError, match="already pins"):
        build_grouped_success_panel(inv, plan, panel_id="replacement", seed=1713,
            num_envs=2, motion_root="/native/motions")
