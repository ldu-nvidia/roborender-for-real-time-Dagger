import copy

import pytest

from pipeline.evaluation.beta_small_batch_panel import make_panel, summarize, validate_comparison, validate_panel


def results(panel):
    return [{"seed": c["seed"], "complete": True, "completed": 5, "optimizer_steps": 0,
             "policy_unchanged": True, "accounting_disagreements": 0,
             "outcomes": [{"lane": i, "exported_success": i == 4, "steps": 12, "timeout": False}
                          for i in range(5)]} for c in panel["cohorts"]]


def test_exact_256_unique_cases_and_unscored_tail():
    panel = validate_panel(make_panel("a" * 64))
    rows = results(panel)
    summary = summarize(panel, rows)
    assert summary["completed_episodes"] == 256
    assert summary["native_completed_episodes"] == 260
    assert summary["successes"] == 51  # last cohort lane4 is deliberately not scored
    assert len({r["case_id"] for r in summary["outcomes"]}) == 256
    assert summary["legacy_b64_panel_equivalent"] is False


def test_smoke_is_complete_only_under_distinct_ten_case_panel():
    panel = validate_panel(make_panel("a" * 64, smoke=True))
    summary = summarize(panel, results(panel))
    assert panel["panel_id"] == "beta-isaac6-b5-heldout-10-smoke-v1"
    assert summary["completed_episodes"] == summary["native_completed_episodes"] == 10
    assert summary["unscored_episodes"] == 0
    assert summary["successes"] == 2 and summary["goal_reached_rate"] == .2
    with pytest.raises(ValueError):
        summarize(make_panel("a" * 64), results(panel))
    changed = copy.deepcopy(panel)
    changed["panel_id"] = "beta-isaac6-b5-heldout-256-v4"
    with pytest.raises(ValueError):
        validate_panel(changed)


@pytest.mark.parametrize("change", ["duplicate", "seed", "missing", "trained", "weights", "count"])
def test_reject_invalid_complete_result(change):
    panel = make_panel("a" * 64)
    rows = results(panel)
    if change == "duplicate": rows[0]["outcomes"][4]["lane"] = 3
    if change == "seed": rows[0]["seed"] += 1
    if change == "missing": rows.pop()
    if change == "trained": rows[0]["optimizer_steps"] = 1
    if change == "weights": rows[0]["policy_unchanged"] = False
    if change == "count": rows[0]["completed"] = 4
    with pytest.raises(ValueError): summarize(panel, rows)


def test_seed_or_assignment_changes_require_new_version():
    panel = copy.deepcopy(make_panel("a" * 64))
    panel["cohorts"][0]["cases"][0]["lane"] = 4
    with pytest.raises(ValueError): validate_panel(panel)


def comparison():
    panel = make_panel("a" * 64)
    rows = {}
    hashes = {"bc": "b" * 64, "dagger": "c" * 64}
    for arm, digest in hashes.items():
        rows[arm] = {**summarize(panel, results(panel)), "checkpoint_sha256": digest,
                     "runtime_physics_sha256": "d" * 64, "policy_unchanged": True,
                     "goal_accounting_version": panel["goal_accounting_version"],
                     "paired_initial_states": arm == "dagger"}
    return panel, rows, hashes


def test_final_pair_receipt_checks_exact_denominator_and_runtime():
    panel, rows, hashes = comparison()
    result = validate_comparison(panel, rows, hashes)
    assert result["paired_outcomes"] == dict(both_success=51, bc_only=0, candidate_only=0, neither=205)
    assert result["paired_complete"] and not result["quality_improvement_certified"]


@pytest.mark.parametrize("field,value", [
    ("successes", 52), ("goal_reached_rate", .99), ("checkpoint_sha256", "e" * 64),
    ("paired_initial_states", False), ("runtime_physics_sha256", "e" * 64),
    ("goal_accounting_version", "wrong"), ("policy_unchanged", False), ("completed_episodes", 255),
])
def test_reject_mismatched_pair_receipts(field, value):
    panel, rows, hashes = comparison()
    rows["dagger"][field] = value
    with pytest.raises(ValueError): validate_comparison(panel, rows, hashes)
