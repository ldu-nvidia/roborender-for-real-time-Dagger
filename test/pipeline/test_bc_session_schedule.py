import copy
import hashlib
import json

import pytest

from pipeline.legacy.bc_collection.sampling_plan import build_plan, canonical
from pipeline.legacy.bc_collection.plan_native_sessions import build_schedule, write_schedule


def plan_fixture(selection_policy="prefix_v1"):
    motions = [{"motion_id": f"motion-{i}", "filename": f"motion-{i}.npz", "sha256": str(i) * 64,
                "bytes": 100} for i in range(3)]
    inventory = {"schema": "roborender.motion-inventory.v1", "motion_count": 3, "motions": motions,
                 "motion_records_sha256": hashlib.sha256(canonical(motions)).hexdigest()}
    return build_plan(inventory, seed=77, num_envs=2, chunks=40, success_cases=4, selection_policy=selection_policy)


def test_scheduler_preserves_every_cohort_role_seed_and_quota():
    plan = plan_fixture()
    result = build_schedule(plan, run_id="bc-leo-test", motion_root="/native/motions")
    expected = {c["cohort_id"]: c for part in plan["partitions"].values() for c in part["cohorts"]}
    seen = []
    for session in result:
        requests = [json.loads(raw) for raw in session["requests"].values()]
        assert len({len(r["motions"]) for r in requests}) == 1
        assert requests[0]["output_dir"] == "/dagger-runtime/bc_isaac"
        for request in requests:
            cohort = expected[request["cohort_id"]]
            seen.append(request["cohort_id"])
            assert request["process_seed"] == cohort["process_seed"]
            assert request["split"] == cohort["split"]
            assert request["chunks_per_env"] == [s["target_chunks"] for s in cohort["slots"]]
    assert sorted(seen) == sorted(expected)
    assert sum(s["target_chunks"] for s in result) == 40
    assert {s["num_envs"] for s in result} == {1, 2}


def test_schedule_is_immutable_and_does_not_claim_native_admission(tmp_path):
    plan = plan_fixture()
    schedule = build_schedule(plan, run_id="bc-leo-test", motion_root="/native/motions", rtx_gpus=[1])
    result = write_schedule(tmp_path / "release", plan, schedule)
    assert not result["production_admitted"] and not result["gpu_capacity_qualified"]
    assert {r["physical_gpu"] for r in result["sessions"]} == {1}
    with pytest.raises(FileExistsError):
        write_schedule(tmp_path / "release", plan, schedule)


@pytest.mark.parametrize("field,value", [("rtx_gpus", [2]), ("rtx_gpus", [0, 0]),
    ("motion_root", "/native/../secret"), ("window_multiplier", 0)])
def test_invalid_host_or_bounds_rejected(field, value):
    args = {"run_id": "bc-leo-test", "motion_root": "/native/motions"}
    args[field] = value
    with pytest.raises(ValueError):
        build_schedule(plan_fixture(), **args)


def test_reused_seed_rejected_before_publication():
    plan = copy.deepcopy(plan_fixture())
    plan["partitions"]["test"]["cohorts"][0]["process_seed"] = plan["partitions"]["train"]["cohorts"][0]["process_seed"]
    with pytest.raises(ValueError, match="duplicate identities/seeds"):
        build_schedule(plan, run_id="bc-leo-test", motion_root="/native/motions")


def test_v2_window_guard_uses_highest_ordinal_not_selected_quota():
    plan = plan_fixture("matched_exposure_v2")
    schedule = build_schedule(plan, run_id="bc-leo-ordinal", motion_root="/native/motions",
                              window_multiplier=2, extra_windows=3)
    found_late = False
    for session in schedule:
        for raw in session["requests"].values():
            request = json.loads(raw)
            maximum_stop = max(values[-1] + 1 for values in request["selected_chunk_ordinals_per_env"])
            assert request["max_windows"] == maximum_stop * 2 + 3
            if request["split"] != "train" and maximum_stop > max(request["chunks_per_env"]):
                found_late = True
    assert found_late


def test_guarded_session_limit_preserves_order_seeds_ordinals_and_exact_tail():
    plan = plan_fixture("matched_exposure_v2")
    common = dict(run_id="bc-leo-bounded", motion_root="/native/motions", rtx_gpus=[0])
    legacy = build_schedule(plan, **common)
    bounded = build_schedule(plan, cohorts_per_session=2, **common)
    assert len(bounded) > len(legacy)
    assert all(len(row["requests"]) <= 2 for row in bounded)
    assert {row["num_envs"] for row in bounded} == {1, 2}
    def semantic(rows):
        result = []
        for session in rows:
            requests = [json.loads(raw) for raw in session["requests"].values()]
            assert {len(row["motions"]) for row in requests} == {session["num_envs"]}
            for request in requests:
                # The output root is scoped to its new session, not a semantic
                # sampling/trajectory change. Everything else is identical.
                request.pop("output_dir")
                result.append(request)
        return result
    assert semantic(legacy) == semantic(bounded)
    assert sum(row["target_chunks"] for row in bounded) == plan["total_target_chunks"]


@pytest.mark.parametrize("value", [0, -1, 301, True, 1.5])
def test_session_grouping_limit_is_explicit_and_bounded(value):
    with pytest.raises(ValueError):
        build_schedule(plan_fixture(), run_id="bc-leo-bad", motion_root="/native/motions", cohorts_per_session=value)
