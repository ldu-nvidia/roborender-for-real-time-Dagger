#!/usr/bin/env python3
"""Plan and validate one immutable three-view 256-case evaluation suite.

The suite uses the same scenario manifest, camera calibration and completed
scenario identities for a BC checkpoint, the privileged teacher oracle and
one or more DAgger checkpoints.  Every visual student is evaluated independently
with ray-traced observations and with Image3F-only observations.  This module does
not launch Isaac or a renderer; it creates and verifies the identities that
the RTX fixed-panel launcher consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RTX_SRC = PROJECT_ROOT / "pipeline/dagger/rtx/code/src"
if str(RTX_SRC) not in sys.path:
    sys.path.insert(0, str(RTX_SRC))

from dagger_rtx.contracts import canonical_json, require_identifier  # noqa: E402
from dagger_rtx.fixed_panel_validation import (  # noqa: E402
    EXPECTED_EPISODES,
    EXPECTED_SCENARIO_MANIFEST_SHA256,
    EXPECTED_VIEWS,
    METRIC,
    WristCameraCalibration,
    file_sha256,
)


SUITE_SCHEMA = "roborender.three_view_fixed_panel_suite.v1"
RESULT_SCHEMA = "roborender.three_view_fixed_panel_suite_result.v1"
MODES = ("raytraced", "generated")
_LABEL = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class SuiteError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


def _checkpoint(label: str, role: str, path: Path) -> dict[str, Any]:
    if not _LABEL.fullmatch(label):
        raise SuiteError(f"invalid checkpoint label: {label!r}")
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise SuiteError(f"checkpoint is not a regular non-symlink file: {path}")
    return {
        "label": label,
        "policy_role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _parse_dagger(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("DAgger checkpoint must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not _LABEL.fullmatch(label):
        raise argparse.ArgumentTypeError("DAgger label must be a lowercase identifier")
    return label, Path(path)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    panel = args.panel.resolve(strict=True)
    if file_sha256(panel) != EXPECTED_SCENARIO_MANIFEST_SHA256:
        raise SuiteError("fixed 256-case scenario manifest identity changed")
    panel_value = json.loads(panel.read_text(encoding="utf-8"))
    if panel_value.get("case_count") != EXPECTED_EPISODES:
        raise SuiteError("scenario manifest is not the sealed 256-case panel")
    calibration = WristCameraCalibration.load(
        args.calibration,
        expected_sha256=args.calibration_sha256 or file_sha256(args.calibration),
        require_certified=not args.allow_provisional_calibration,
    )
    bc = _checkpoint("bc", "bc_student", args.bc_checkpoint)
    teacher = _checkpoint("teacher-oracle", "privileged_teacher", args.teacher_checkpoint)
    dagger = [
        _checkpoint(label, "dagger_student", path)
        for label, path in args.dagger_checkpoint
    ]
    labels = [bc["label"], teacher["label"], *(row["label"] for row in dagger)]
    if len(labels) != len(set(labels)):
        raise SuiteError("checkpoint labels must be unique")
    run_id = require_identifier(args.run_id, "run_id")
    evaluations = []
    for checkpoint in (bc, teacher, *dagger):
        # The privileged teacher consumes teacher_obs, not policy RGB. Running
        # it through Image3F would be a duplicate result and could falsely
        # suggest the teacher is a visual policy. Observation ablation applies
        # to every visual student checkpoint.
        modes = ("raytraced",) if checkpoint["policy_role"] == "privileged_teacher" else MODES
        for mode in modes:
            evaluation_id = f"{run_id}-{checkpoint['label']}-{mode}"
            if len(evaluation_id) > 128:
                raise SuiteError("derived evaluation_id is too long")
            action_policy = (
                "deterministic_privileged_teacher_action_mean"
                if checkpoint["policy_role"] == "privileged_teacher"
                else "deterministic_student_action_mean_beta_0"
            )
            # Raven still instantiates the student model for teacher-oracle
            # evaluation. BC is its inert, exactly attested scaffold.
            scaffold = bc if checkpoint["policy_role"] == "privileged_teacher" else checkpoint
            evaluations.append(
                {
                    "evaluation_id": evaluation_id,
                    "policy_label": checkpoint["label"],
                    "policy_role": checkpoint["policy_role"],
                    "observation_mode": mode,
                    "action_policy": action_policy,
                    "checkpoint_path": scaffold["path"],
                    "checkpoint_sha256": scaffold["sha256"],
                    "teacher_checkpoint_path": (
                        teacher["path"] if checkpoint["policy_role"] == "privileged_teacher" else None
                    ),
                    "teacher_checkpoint_sha256": (
                        teacher["sha256"] if checkpoint["policy_role"] == "privileged_teacher" else None
                    ),
                    "scenario_manifest_sha256": EXPECTED_SCENARIO_MANIFEST_SHA256,
                    "wrist_calibration_sha256": calibration.sha256,
                    "status": (
                        "ready" if calibration.status == "certified"
                        else "blocked_pending_camera_calibration"
                    ),
                }
            )
    preimage = {
        "schema": SUITE_SCHEMA,
        "run_id": run_id,
        "scenario_manifest": str(panel),
        "scenario_manifest_sha256": EXPECTED_SCENARIO_MANIFEST_SHA256,
        "scenario_count": EXPECTED_EPISODES,
        "policy_views": list(EXPECTED_VIEWS),
        "metric": METRIC,
        "camera_calibration": {
            "path": str(calibration.path),
            "sha256": calibration.sha256,
            "calibration_id": calibration.calibration_id,
            "status": calibration.status,
        },
        "checkpoints": [bc, teacher, *dagger],
        "evaluations": evaluations,
    }
    return {
        **preimage,
        "suite_sha256": hashlib.sha256(canonical_json(preimage)).hexdigest(),
    }


def validate_results(plan: Mapping[str, Any], result_paths: list[Path]) -> dict[str, Any]:
    if plan.get("schema") != SUITE_SCHEMA:
        raise SuiteError("unsupported suite plan")
    expected = {row["evaluation_id"]: row for row in plan["evaluations"]}
    observed: dict[str, Mapping[str, Any]] = {}
    for path in result_paths:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        evaluation_id = value.get("evaluation_id")
        if evaluation_id not in expected or evaluation_id in observed:
            raise SuiteError("result evaluation identity is unknown or duplicated")
        contract = expected[evaluation_id]
        if (
            value.get("scenario_manifest_sha256") != plan["scenario_manifest_sha256"]
            or value.get("completed_episodes") != EXPECTED_EPISODES
            or value.get("metric") != METRIC
            or value.get("views") != list(EXPECTED_VIEWS)
            or value.get("observation_mode") != contract["observation_mode"]
            or value.get("action_policy") != contract["action_policy"]
            or value.get("checkpoint_sha256") != contract["checkpoint_sha256"]
            or value.get("teacher_checkpoint_sha256") != contract["teacher_checkpoint_sha256"]
            or value.get("wrist_calibration_sha256") != contract["wrist_calibration_sha256"]
        ):
            raise SuiteError(f"result contract differs: {evaluation_id}")
        observed[evaluation_id] = value
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise SuiteError(f"evaluation suite is incomplete: {missing}")
    rows = []
    for evaluation_id in expected:
        value = observed[evaluation_id]
        rows.append(
            {
                "evaluation_id": evaluation_id,
                "policy_label": expected[evaluation_id]["policy_label"],
                "policy_role": expected[evaluation_id]["policy_role"],
                "observation_mode": expected[evaluation_id]["observation_mode"],
                "successes": value["successes"],
                "completed_episodes": value["completed_episodes"],
                "goal_reached_rate": value["goal_reached_rate"],
                "wilson_95_interval": value["wilson_95_interval"],
            }
        )
    return {
        "schema": RESULT_SCHEMA,
        "suite_sha256": plan["suite_sha256"],
        "scenario_manifest_sha256": plan["scenario_manifest_sha256"],
        "scenario_count": EXPECTED_EPISODES,
        "policy_views": list(EXPECTED_VIEWS),
        "metric": METRIC,
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--run-id", required=True)
    plan.add_argument(
        "--panel", type=Path,
        default=PROJECT_ROOT / "pipeline/dagger/config/fixed_validation_panel.v1.json",
    )
    plan.add_argument("--calibration", type=Path, required=True)
    plan.add_argument("--calibration-sha256")
    plan.add_argument("--allow-provisional-calibration", action="store_true")
    plan.add_argument("--bc-checkpoint", type=Path, required=True)
    plan.add_argument("--teacher-checkpoint", type=Path, required=True)
    plan.add_argument("--dagger-checkpoint", type=_parse_dagger, action="append", default=[])
    plan.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("validate-results")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--result", type=Path, action="append", required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        value = build_plan(args)
    else:
        plan_value = json.loads(args.plan.read_text(encoding="utf-8"))
        expected_digest = plan_value.pop("suite_sha256", None)
        actual_digest = hashlib.sha256(canonical_json(plan_value)).hexdigest()
        plan_value["suite_sha256"] = expected_digest
        if expected_digest != actual_digest:
            raise SuiteError("suite plan digest mismatch")
        value = validate_results(plan_value, args.result)
    _atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
