from __future__ import annotations

import pytest

from pipeline.evaluation.compare_paired_success import (
    PairedEvaluationError,
    compare_reports,
)


def report(name: str, values: list[bool], *, manifest: str = "a" * 64) -> dict:
    return {
        "evaluation": {
            "evaluation_id": name,
            "checkpoint_sha256": ("b" if name == "base" else "c") * 64,
            "scenario_manifest_sha256": manifest,
            "observation_mode": "raytraced",
            "metric": "raven_goal_reached_rate",
            "status": "complete",
            "completed": len(values),
            "total": len(values),
            "outcomes": [
                {"scenario_id": f"case-{index}", "goal_reached": value}
                for index, value in enumerate(values)
            ],
        }
    }


def test_exact_paired_cells_and_rate_change():
    result = compare_reports(
        report("base", [False, False, True, True]),
        report("candidate", [False, True, False, True]),
        expected_cases=4,
    )
    assert result["paired_cells"] == {
        "both_failed": 1,
        "baseline_failed_candidate_succeeded": 1,
        "baseline_succeeded_candidate_failed": 1,
        "both_succeeded": 1,
    }
    assert result["absolute_rate_change"] == 0.0
    assert result["exact_two_sided_mcnemar_p"] == 1.0


def test_rejects_different_panels():
    with pytest.raises(PairedEvaluationError, match="manifest"):
        compare_reports(
            report("base", [False], manifest="a" * 64),
            report("candidate", [True], manifest="d" * 64),
            expected_cases=1,
        )


def test_rejects_missing_or_duplicated_cases():
    candidate = report("candidate", [False, True])
    candidate["evaluation"]["outcomes"][1]["scenario_id"] = "case-0"
    with pytest.raises(PairedEvaluationError, match="repeats scenario"):
        compare_reports(
            report("base", [False, True]), candidate, expected_cases=2
        )


@pytest.mark.parametrize("field", ["goal_accounting_version", "runtime_physics_sha256",
                                  "reset_identity_sha256", "camera_calibration_sha256"])
def test_rejects_mismatched_or_one_sided_runtime_identity(field):
    baseline, candidate = report("base", [False]), report("candidate", [True])
    candidate["evaluation"][field] = "new"
    with pytest.raises(PairedEvaluationError, match=field):
        compare_reports(baseline, candidate, expected_cases=1)
    baseline["evaluation"][field] = "old"
    with pytest.raises(PairedEvaluationError, match=field):
        compare_reports(baseline, candidate, expected_cases=1)


def test_unversioned_historical_reports_are_not_certified_runtime_matches():
    result = compare_reports(report("base", [False]), report("candidate", [True]), expected_cases=1)
    assert result["goal_accounting_version"] == "legacy_unversioned"
    assert result["runtime_identity_fields_complete"] is False


def test_exact_corrected_runtime_identity_is_reported():
    baseline, candidate = report("base", [False]), report("candidate", [True])
    identity = {"goal_accounting_version": "roborender.raven.pre_reset_goal.v1",
                "runtime_physics_sha256": "d" * 64, "reset_identity_sha256": "e" * 64,
                "camera_calibration_sha256": "f" * 64}
    baseline["evaluation"].update(identity)
    candidate["evaluation"].update(identity)
    result = compare_reports(baseline, candidate, expected_cases=1)
    assert result["runtime_identity_fields_complete"] is True
    assert result["runtime_identity_fields"] == identity
