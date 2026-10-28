import hashlib
import pytest

from pipeline.legacy.bc_collection.sampling_plan import build_plan, canonical, validate_selection_plan


def inventory(count=295):
    rows = [{"motion_id": f"motion-{i}", "filename": f"motion-{i}.npz", "bytes": 123,
             "sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(count)]
    return {"schema": "roborender.motion-inventory.v1", "motion_count": count,
            "ordering": "explicit_filename_lexical_not_filesystem_order", "motions": rows,
            "motion_records_sha256": hashlib.sha256(canonical(rows)).hexdigest()}


def test_all_modes_shared_population_seeds_disjoint():
    inv = inventory()
    plan = build_plan(inv, seed=1234, num_envs=8)
    expected = {row["motion_id"] for row in inv["motions"]}
    seeds = []
    assert plan["total_target_chunks"] == 3200
    for role, part in plan["partitions"].items():
        slots = [s for c in part["cohorts"] for s in c["slots"]]
        assert {s["motion_id"] for s in slots} == expected
        assert len(slots) == 295
        assert sum(s["target_chunks"] for s in slots) == part["target_chunks"]
        assert max(s["target_chunks"] for s in slots) - min(s["target_chunks"] for s in slots) <= 1
        assert all(1 <= c["num_envs"] <= 8 for c in part["cohorts"])
        seeds.extend(c["process_seed"] for c in part["cohorts"])
    seeds.extend(c["process_seed"] for c in plan["success_test"]["cases"])
    assert len(seeds) == len(set(seeds))
    assert not plan["production_admitted"] and not plan["gpu_capacity_qualified"]


def test_reproducible_and_new_master_seed_changes_assignments():
    assert build_plan(inventory(), seed=1, num_envs=12) == build_plan(inventory(), seed=1, num_envs=12)
    assert build_plan(inventory(), seed=1, num_envs=12) != build_plan(inventory(), seed=2, num_envs=12)


def test_too_few_validation_modes_rejected():
    with pytest.raises(ValueError, match="every motion"):
        build_plan(inventory(), seed=1, num_envs=8, chunks=2000)


def test_partial_full_batch_not_silently_dropped():
    with pytest.raises(ValueError, match="16-chunk"):
        build_plan(inventory(), seed=1, num_envs=8, chunks=2950)


def test_changed_inventory_rejected():
    inv = inventory()
    inv["motions"][0]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="inventory"):
        build_plan(inv, seed=1, num_envs=8)


def test_v2_preselects_complete_chunk_ordinals_over_shared_train_exposure():
    plan = build_plan(inventory(), seed=1234, num_envs=8, selection_policy="matched_exposure_v2")
    assert plan["schema"].endswith(".v2") and plan["selection_uses_labels_rewards_or_loss"] is False
    validate_selection_plan(plan)
    spans = {s["motion_id"]: s["target_chunks"] for c in plan["partitions"]["train"]["cohorts"] for s in c["slots"]}
    assert set(spans.values()) == {8, 9}
    later = 0
    for role, partition in plan["partitions"].items():
        for cohort in partition["cohorts"]:
            for slot in cohort["slots"]:
                ordinals = slot["selected_chunk_ordinals"]
                assert len(ordinals) == slot["target_chunks"]
                assert slot["exposure_span_chunks"] == spans[slot["motion_id"]]
                assert ordinals == sorted(set(ordinals)) and max(ordinals) < slot["exposure_span_chunks"]
                if role == "train":
                    assert ordinals == list(range(slot["target_chunks"]))
                else:
                    later += int(max(ordinals) > 0)
    assert later > 450  # Held-out selection is not systematically prefix-only.
    assert plan["collection_cost"]["minimum_native_complete_chunks_emitted"] > 3200
    assert plan["collection_cost"]["minimum_unselected_complete_chunks_retained"] > 0
    assert build_plan(inventory(), seed=1234, num_envs=8, selection_policy="matched_exposure_v2") == plan


def test_v2_rejects_posthoc_ordinals_or_shortened_heldout_exposure():
    import copy
    plan = build_plan(inventory(), seed=1234, num_envs=8, selection_policy="matched_exposure_v2")
    changed = copy.deepcopy(plan)
    slot = changed["partitions"]["validation"]["cohorts"][0]["slots"][0]
    slot["exposure_span_chunks"] = slot["target_chunks"]
    with pytest.raises(ValueError, match="exposure span"):
        validate_selection_plan(changed)
    changed = copy.deepcopy(plan)
    slot = changed["partitions"]["validation"]["cohorts"][0]["slots"][0]
    slot["selected_chunk_ordinals"] = list(range(slot["target_chunks"])) if slot["selected_chunk_ordinals"] != list(range(slot["target_chunks"])) else list(range(1, slot["target_chunks"] + 1))
    with pytest.raises(ValueError, match="predeclared"):
        validate_selection_plan(changed)
