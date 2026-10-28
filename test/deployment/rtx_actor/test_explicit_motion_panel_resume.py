"""CPU-only completed-cohort recovery; no Isaac, GPU, or policy changes."""
from test.deployment.source_paths import ACTOR_ROOT
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(ACTOR_ROOT))
from explicit_motion_panel import (
    aggregate_partitions, canonical, completed_progress, make_cohort_receipt,
    make_partition, panel_sha256, partition_panel, run_panel, run_partition,
    validate_cohort_receipt, validate_result,
)
from test.deployment.rtx_actor.test_explicit_motion_panel import panel, result

CHECKPOINT = "b" * 64


def receipt(p, index):
    return make_cohort_receipt(p, index, result(p["cohorts"][index]), checkpoint_sha256=CHECKPOINT)


def test_resume_skips_completed_native_cohorts_and_binds_original_panel():
    p, executed, saved, progress = panel(), [], [], []
    old = receipt(p, 0)
    def execute(spec, index):
        executed.append(index)
        return result(spec)
    actual = run_panel(p, execute_cohort=execute, completed_receipts=[old],
        expected_checkpoint_sha256=CHECKPOINT, save_receipt=saved.append, report=progress.append)
    assert executed == [1]
    assert [r["cohort_index"] for r in saved] == [1]
    assert actual["panel_sha256"] == panel_sha256(p)
    assert actual["successes"] == 2 and actual["completed_episodes"] == 4
    assert validate_result(p, actual, CHECKPOINT) == actual
    assert all("success_rate" not in event and event.get("complete") is not True for event in progress)


def test_all_completed_resume_does_not_execute_again():
    p = panel()
    actual = run_panel(p, execute_cohort=lambda *_: pytest.fail("must not rerun"),
        completed_receipts=[receipt(p, i) for i in range(2)], expected_checkpoint_sha256=CHECKPOINT)
    assert actual["completed_episodes"] == 4 and actual["success_rate"] == .5


def test_resume_cannot_begin_without_exact_checkpoint_pin():
    p = panel()
    with pytest.raises(ValueError, match="checkpoint"):
        run_panel(p, execute_cohort=lambda *_: pytest.fail("must fail before execution"),
                  completed_receipts=[receipt(p, 0)])


@pytest.mark.parametrize("indices", [[1], [0, 0], [1, 0], [0, 1, 1]])
def test_missing_reordered_duplicate_or_extra_completed_cohorts_rejected(indices):
    p = panel()
    with pytest.raises(ValueError):
        run_panel(p, execute_cohort=lambda *_: pytest.fail("must not execute"),
            completed_receipts=[receipt(p, i) for i in indices], expected_checkpoint_sha256=CHECKPOINT)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(panel_sha256="d" * 64),
    lambda r: r.update(checkpoint_sha256="d" * 64),
    lambda r: r.update(cohort_index=1),
    lambda r: r.update(cohort_index=False),
    lambda r: r.update(case_ids=["wrong-case", "wrong-case-2"]),
    lambda r: r.update(result_sha256="d" * 64),
    lambda r: r.update(extra="unexpected"),
    lambda r: r["result"].update(complete=False),
    lambda r: r["result"].update(completed=1),
    lambda r: r["result"].update(seed=999),
    lambda r: r["result"].update(checkpoint_sha256="d" * 64),
    lambda r: r["result"]["outcomes"][0].update(timeout="false"),
])
def test_changed_receipt_is_not_reusable(mutation):
    p = panel()
    value = receipt(p, 0)
    mutation(value)
    with pytest.raises(ValueError):
        validate_cohort_receipt(p, value, checkpoint_sha256=CHECKPOINT)


def test_receipt_result_digest_cannot_bypass_native_accounting_checks():
    p = panel()
    value = receipt(p, 0)
    value["result"]["original_raven_metric_successes"] = 0
    value["result_sha256"] = hashlib.sha256(canonical(value["result"])).hexdigest()
    with pytest.raises(ValueError):
        validate_cohort_receipt(p, value, checkpoint_sha256=CHECKPOINT)


def test_incomplete_next_cohort_does_not_get_a_durable_receipt():
    p, saved = panel(), []
    def failed(spec, index):
        value = result(spec)
        value["complete"], value["completed"] = False, 1
        return value
    with pytest.raises(ValueError):
        run_panel(p, execute_cohort=failed, completed_receipts=[receipt(p, 0)],
            expected_checkpoint_sha256=CHECKPOINT, save_receipt=saved.append)
    assert saved == []
    progress = completed_progress(p, [receipt(p, 0)], checkpoint_sha256=CHECKPOINT)
    assert progress["completed_episodes"] == 2 and progress["expected_episodes"] == 4
    assert progress["complete"] is False and "success_rate" not in progress


def test_each_partition_binds_original_full_panel_and_original_case_indices():
    p, seen = panel(), []
    partitions = partition_panel(p, max_cohorts=1)
    results = []
    for partition in partitions:
        assert partition["panel_sha256"] == panel_sha256(p)
        assert partition["full_expected_episodes"] == 4
        value = run_partition(p, partition, execute_cohort=lambda spec, i: (seen.append(i), result(spec))[1],
                              checkpoint_sha256=CHECKPOINT)
        assert "success_rate" not in value
        results.append(value)
    assert seen == [0, 1]
    # Hosts may complete partition lifecycles in any order; result order is canonical.
    actual = aggregate_partitions(p, list(reversed(results)), checkpoint_sha256=CHECKPOINT)
    assert [row["case_id"] for row in actual["outcomes"]] == ["case-0-0", "case-0-1", "case-1-0", "case-1-1"]
    assert actual["panel_sha256"] == panel_sha256(p) and actual["success_rate"] == .5


def test_partition_resume_skips_only_its_own_complete_prefix():
    p = panel()
    part = make_partition(p, [1])
    actual = run_partition(p, part, execute_cohort=lambda *_: pytest.fail("must not rerun"),
        checkpoint_sha256=CHECKPOINT, completed_receipts=[receipt(p, 1)])
    assert actual["completed_episodes"] == 2
    with pytest.raises(ValueError):
        run_partition(p, part, execute_cohort=lambda *_: pytest.fail("must not execute"),
            checkpoint_sha256=CHECKPOINT, completed_receipts=[receipt(p, 0)])


@pytest.mark.parametrize("indices", [[], [1, 0], [0, 0], [0, 2], [-1], [True]])
def test_ambiguous_partition_ranges_rejected(indices):
    with pytest.raises(ValueError):
        make_partition(panel(), indices)


def test_aggregate_rejects_holes_duplicates_changed_denominator_or_checkpoint():
    p = panel()
    parts = [run_partition(p, part, execute_cohort=lambda spec, i: result(spec), checkpoint_sha256=CHECKPOINT)
             for part in partition_panel(p, max_cohorts=1)]
    for bad in [parts[:1], parts + parts[:1]]:
        with pytest.raises(ValueError):
            aggregate_partitions(p, bad, checkpoint_sha256=CHECKPOINT)
    for field, value in [("full_expected_episodes", 2), ("complete", False),
                         ("checkpoint_sha256", "d" * 64), ("successes", 2)]:
        bad = deepcopy(parts)
        bad[0][field] = value
        with pytest.raises(ValueError):
            aggregate_partitions(p, bad, checkpoint_sha256=CHECKPOINT)


def test_partition_cannot_relabel_a_subset_as_the_full_original_panel():
    p = panel()
    part = partition_panel(p, max_cohorts=1)[0]
    subset = deepcopy(p)
    subset["cohorts"] = subset["cohorts"][:1]
    subset["episode_count"] = 2
    with pytest.raises(ValueError):
        run_partition(subset, part, execute_cohort=lambda *_: pytest.fail("must not run"),
                      checkpoint_sha256=CHECKPOINT)


def test_persistence_failure_stops_before_running_the_next_cohort():
    p, seen = panel(), []
    def save(_):
        raise OSError("durability failed")
    with pytest.raises(OSError):
        run_panel(p, execute_cohort=lambda spec, i: (seen.append(i), result(spec))[1],
            expected_checkpoint_sha256=CHECKPOINT, save_receipt=save)
    assert seen == [0]


def test_295_case_recovery_preserves_102_completed_cases_and_original_denominator():
    """Mirror the B2 panel failure boundary without invoking any native runtime."""
    p = panel()
    template = deepcopy(p["cohorts"][0])
    p["episode_count"], p["cohorts"] = 295, []
    for index in range(148):
        spec = deepcopy(template)
        spec["assignment"].update(cohort_id=f"cohort-{index}", simulator_seed=100 + index)
        if index == 147:
            spec["assignment"]["lanes"] = spec["assignment"]["lanes"][:1]
        spec["case_ids"] = [f"case-{index}-{lane}" for lane in range(len(spec["assignment"]["lanes"]))]
        p["cohorts"].append(spec)
    def native(spec):
        count = len(spec["case_ids"])
        value = result(spec, successes=(True, False)[:count])
        value["expected"] = value["completed"] = value["original_raven_metric_completed"] = count
        return value
    prefix = [make_cohort_receipt(p, i, native(p["cohorts"][i]), checkpoint_sha256=CHECKPOINT)
              for i in range(51)]
    progress = completed_progress(p, prefix, checkpoint_sha256=CHECKPOINT)
    assert progress["completed_episodes"] == 102 and progress["expected_episodes"] == 295
    seen = []
    final = run_panel(p, execute_cohort=lambda spec, i: (seen.append(i), native(spec))[1],
        completed_receipts=prefix, expected_checkpoint_sha256=CHECKPOINT)
    assert seen == list(range(51, 148))
    assert final["completed_episodes"] == final["expected_episodes"] == 295
    assert final["panel_sha256"] == panel_sha256(p)
    assert len({row["case_id"] for row in final["outcomes"]}) == 295
