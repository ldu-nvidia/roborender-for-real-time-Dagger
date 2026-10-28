from __future__ import annotations
from test.deployment.source_paths import REPO_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = REPO_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "pipeline/evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline/dagger/rtx/code/src"))

from dagger_rtx.fixed_panel_validation import (  # noqa: E402
    FixedPanelValidationError,
    WristCameraCalibration,
    file_sha256,
)
from three_view_fixed_panel_suite import (  # noqa: E402
    SuiteError,
    build_plan,
    validate_results,
)


def calibration_document(status: str = "certified") -> dict:
    evidence = {
        "native_capture_sha256": "1" * 64,
        "visual_review_sha256": "2" * 64,
        "projection_check_sha256": "3" * 64,
        "certified_at_utc": "2026-09-13T00:00:00Z",
    }
    if status == "provisional":
        evidence = {key: None for key in evidence}
    return {
        "schema": "roborender.three_view_camera_calibration.v1",
        "calibration_id": "test-calibration",
        "status": status,
        "policy_views": ["head_color", "hand_left", "hand_right"],
        "camera_resolution": [108, 192],
        "cameras": {
            "head_color": {
                "link": "d435_link", "position": [0, 0.035, 0],
                "rotation_wxyz": [1, 0, 0, 0],
            },
            "hand_left": {
                "link": "left_wrist_yaw_link", "position": [-0.08, 0, -0.06],
                "rotation_wxyz": [1, 0, 0, 0],
            },
            "hand_right": {
                "link": "right_wrist_yaw_link", "position": [-0.08, 0, -0.06],
                "rotation_wxyz": [1, 0, 0, 0],
            },
        },
        "evidence": evidence,
    }


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_calibration_requires_certification_and_exact_runtime(tmp_path):
    path = write_json(tmp_path / "calibration.json", calibration_document("provisional"))
    with pytest.raises(FixedPanelValidationError, match="has not passed"):
        WristCameraCalibration.load(
            path, expected_sha256=file_sha256(path), require_certified=True
        )

    path = write_json(tmp_path / "calibration.json", calibration_document())
    calibration = WristCameraCalibration.load(
        path, expected_sha256=file_sha256(path), require_certified=True
    )
    rows = [
        {"name": name, **dict(calibration.cameras[name])}
        for name in calibration.policy_views
    ]
    report = calibration.attest_runtime(rows)
    assert len(report["runtime_camera_sha256"]) == 64
    rows[1]["position"] = [-0.07, 0, -0.06]
    with pytest.raises(FixedPanelValidationError, match="differs"):
        calibration.attest_runtime(rows)


def make_plan(tmp_path: Path) -> dict:
    calibration = write_json(tmp_path / "calibration.json", calibration_document())
    bc = tmp_path / "bc.pt"
    teacher = tmp_path / "teacher.pt"
    dagger = tmp_path / "dagger.pt"
    bc.write_bytes(b"bc")
    teacher.write_bytes(b"teacher")
    dagger.write_bytes(b"dagger")
    args = argparse.Namespace(
        run_id="paired-eval",
        panel=PROJECT_ROOT / "pipeline/dagger/config/fixed_validation_panel.v1.json",
        calibration=calibration,
        calibration_sha256=file_sha256(calibration),
        allow_provisional_calibration=False,
        bc_checkpoint=bc,
        teacher_checkpoint=teacher,
        dagger_checkpoint=[("iter-50", dagger)],
    )
    return build_plan(args)


def test_plan_is_full_three_policy_by_two_observation_matrix(tmp_path):
    plan = make_plan(tmp_path)
    assert plan["scenario_count"] == 256
    assert plan["policy_views"] == ["head_color", "hand_left", "hand_right"]
    assert len(plan["evaluations"]) == 5
    assert {
        (row["policy_role"], row["observation_mode"])
        for row in plan["evaluations"]
    } == {
        ("bc_student", "raytraced"),
        ("bc_student", "generated"),
        ("privileged_teacher", "raytraced"),
        ("dagger_student", "raytraced"),
        ("dagger_student", "generated"),
    }
    teacher = [row for row in plan["evaluations"] if row["policy_role"] == "privileged_teacher"]
    assert all(row["teacher_checkpoint_sha256"] for row in teacher)
    assert all(row["action_policy"] == "deterministic_privileged_teacher_action_mean" for row in teacher)


def test_result_validation_requires_complete_exact_matrix(tmp_path):
    plan = make_plan(tmp_path)
    paths = []
    for index, contract in enumerate(plan["evaluations"]):
        path = tmp_path / f"result-{index}.json"
        write_json(path, {
            "evaluation_id": contract["evaluation_id"],
            "scenario_manifest_sha256": plan["scenario_manifest_sha256"],
            "completed_episodes": 256,
            "metric": plan["metric"],
            "views": plan["policy_views"],
            "observation_mode": contract["observation_mode"],
            "action_policy": contract["action_policy"],
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "teacher_checkpoint_sha256": contract["teacher_checkpoint_sha256"],
            "wrist_calibration_sha256": contract["wrist_calibration_sha256"],
            "successes": 128,
            "goal_reached_rate": 0.5,
            "wilson_95_interval": [0.439, 0.561],
        })
        paths.append(path)
    result = validate_results(plan, paths)
    assert len(result["results"]) == 5
    with pytest.raises(SuiteError, match="incomplete"):
        validate_results(plan, paths[:-1])


def test_plan_digest_covers_every_immutable_input(tmp_path):
    plan = make_plan(tmp_path)
    digest = plan.pop("suite_sha256")
    from dagger_rtx.contracts import canonical_json

    assert hashlib.sha256(canonical_json(plan)).hexdigest() == digest
