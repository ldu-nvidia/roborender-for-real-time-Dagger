#!/usr/bin/env python3
"""Register and finalize fixed-panel Raven success validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dagger.raven_success import (
    RAVEN_SUCCESS_METRIC,
    expand_validation_panel,
)
from pipeline.dagger.services.client import ServiceClient
from pipeline.dagger.services.contracts import file_sha256


DEFAULT_PANEL = (
    PROJECT_ROOT / "pipeline/dagger/config/fixed_validation_panel.v1.json"
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_panel(path: Path) -> tuple[list[dict], str]:
    panel = json.loads(path.read_text(encoding="utf-8"))
    episodes = expand_validation_panel(panel)
    return episodes, file_sha256(path)


def client(args: argparse.Namespace, url: str | None = None) -> ServiceClient:
    token = args.token_file.read_text(encoding="utf-8").strip()
    return ServiceClient(url or args.evaluation_url, token, timeout=30)


def register(args: argparse.Namespace) -> int:
    episodes, digest = load_panel(args.panel)
    if args.contract_version == "v2":
        if args.wrist_calibration is None:
            raise ValueError("v2 requires --wrist-calibration")
        calibration_digest = file_sha256(args.wrist_calibration)
        if args.wrist_calibration_sha256 and calibration_digest != args.wrist_calibration_sha256:
            raise ValueError("wrist calibration SHA-256 mismatch")
        calibration = json.loads(args.wrist_calibration.read_text(encoding="utf-8"))
        calibration_id = calibration.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id:
            raise ValueError("wrist calibration lacks calibration_id")
        if args.policy_role == "privileged_teacher":
            if not args.teacher_checkpoint_sha256:
                raise ValueError("teacher oracle requires --teacher-checkpoint-sha256")
            action_policy = "deterministic_privileged_teacher_action_mean"
        else:
            if args.teacher_checkpoint_sha256:
                raise ValueError("student evaluation cannot bind a teacher checkpoint")
            action_policy = "deterministic_student_action_mean_beta_0"
    else:
        if (
            args.policy_role != "dagger_student"
            or args.teacher_checkpoint_sha256
            or args.wrist_calibration is not None
        ):
            raise ValueError("legacy v1 only supports the existing DAgger student contract")
        calibration_digest = None
        calibration_id = None
        action_policy = "deterministic_student_action_mean_beta_0"
    response = client(args).post("/v2/evaluations", {
        "evaluation_id": args.evaluation_id,
        "run_id": args.run_id,
        "round_id": args.round_id,
        "purpose": args.purpose,
        "checkpoint_sha256": args.checkpoint_sha256,
        "scenario_manifest_sha256": digest,
        "scenario_ids": [episode["scenario_id"] for episode in episodes],
        "observation_mode": args.observation_mode,
        "metric": RAVEN_SUCCESS_METRIC,
    })
    assignment = {
        "schema": (
            "roborender.three_view_success_validation_assignment.v2"
            if args.contract_version == "v2"
            else "roborender.raven_success_validation_assignment.v1"
        ),
        "evaluation_id": args.evaluation_id,
        "run_id": args.run_id,
        "round_id": args.round_id,
        "checkpoint_sha256": args.checkpoint_sha256,
        "scenario_manifest": str(args.panel.resolve()),
        "scenario_manifest_sha256": digest,
        "observation_mode": args.observation_mode,
        "action_policy": action_policy,
        "metric": RAVEN_SUCCESS_METRIC,
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    if args.contract_version == "v2":
        assignment.update({
            "policy_role": args.policy_role,
            "teacher_checkpoint_sha256": args.teacher_checkpoint_sha256,
            "wrist_calibration_sha256": calibration_digest,
            "wrist_calibration_id": calibration_id,
        })
    atomic_json(args.assignment_output, assignment)
    print(json.dumps(response, sort_keys=True))
    return 0


def finalize(args: argparse.Namespace) -> int:
    if args.iteration_id < 1:
        raise ValueError("iteration_id must be positive")
    report = client(args).get(f"/v2/evaluations/{args.evaluation_id}")["evaluation"]
    if report["purpose"] != "validation" or report["metric"] != RAVEN_SUCCESS_METRIC:
        raise RuntimeError("evaluation is not Raven success validation")
    if report["status"] != "complete" or report["completed"] != report["total"]:
        raise RuntimeError(
            f"validation incomplete: {report['completed']}/{report['total']} episodes"
        )
    advance = client(args, args.coordinator_url).post("/v1/iterations/advance", {
        "run_id": report["run_id"],
        "round_id": report["round_id"],
        "evaluation_id": args.evaluation_id,
    })
    if advance["completed_iteration_id"] != args.iteration_id:
        raise RuntimeError("coordinator advanced a different DAgger iteration")
    result = {
        "schema": "roborender.raven_success_validation_result.v2",
        "run_id": report["run_id"],
        "round_id": report["round_id"],
        "iteration_id": args.iteration_id,
        "evaluation_id": args.evaluation_id,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "completed_episodes": report["completed"],
        "successes": report["successes"],
        "goal_reached_rate": report["goal_reached_rate"],
        "wilson_95_interval": report["wilson_95_interval"],
        "safety_violations": report["safety_violations"],
        "best_goal_reached_rate": advance["best_goal_reached_rate"],
        "best_iteration_id": advance["best_iteration_id"],
        "bad_validations": advance["bad_validations"],
        "saturated": advance["saturated"],
        "stop_reason": advance["stop_reason"],
        "next_iteration_id": advance["next_iteration_id"],
    }
    atomic_json(args.result_output, result)
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id or report["run_id"],
            resume="must",
            mode=args.wandb_mode,
        )
        run.log({
            "iteration_id": args.iteration_id,
            "validation/goal_reached_rate": report["goal_reached_rate"],
            "validation/successes": report["successes"],
            "validation/completed_episodes": report["completed"],
            "validation/wilson95_lower": report["wilson_95_interval"][0],
            "validation/wilson95_upper": report["wilson_95_interval"][1],
            "validation/safety_violations": report["safety_violations"],
            "validation/best_goal_reached_rate": advance["best_goal_reached_rate"],
            "validation/bad_checks": advance["bad_validations"],
            "validation/saturated": int(advance["saturated"]),
        })
        run.summary.update({
            "best_validation_goal_reached_rate": advance["best_goal_reached_rate"],
            "best_validation_iteration": advance["best_iteration_id"],
        })
        run.finish()
    print(json.dumps(result, sort_keys=True))
    return 0


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluation-url", default="http://127.0.0.1:18614")
    parser.add_argument(
        "--token-file", type=Path,
        default=PROJECT_ROOT / "runtime/secrets/dagger_services.token",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("register")
    common(create)
    create.add_argument("--evaluation-id", required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument("--round-id", type=int, choices=range(5), required=True)
    create.add_argument(
        "--purpose", choices=("validation", "test"), default="validation",
        help=(
            "validation advances an awaiting coordinator iteration; test is a "
            "standalone checkpoint audit and never promotes training state"
        ),
    )
    create.add_argument("--checkpoint-sha256", required=True)
    create.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    create.add_argument("--observation-mode", choices=("raytraced", "generated"), default="raytraced")
    create.add_argument("--contract-version", choices=("v1", "v2"), default="v1")
    create.add_argument(
        "--policy-role",
        choices=("bc_student", "dagger_student", "privileged_teacher"),
        default="dagger_student",
    )
    create.add_argument("--teacher-checkpoint-sha256")
    create.add_argument("--wrist-calibration", type=Path)
    create.add_argument("--wrist-calibration-sha256")
    create.add_argument("--assignment-output", type=Path, required=True)
    create.set_defaults(handler=register)

    finish = commands.add_parser("finalize")
    common(finish)
    finish.add_argument("--evaluation-id", required=True)
    finish.add_argument("--iteration-id", type=int, required=True)
    finish.add_argument("--result-output", type=Path, required=True)
    finish.add_argument("--coordinator-url", default="http://127.0.0.1:18610")
    finish.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    finish.add_argument("--wandb-project", default="dagger")
    finish.add_argument("--wandb-run-id")
    finish.set_defaults(handler=finalize)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
