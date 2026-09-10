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
    response = client(args).post("/v2/evaluations", {
        "evaluation_id": args.evaluation_id,
        "run_id": args.run_id,
        "round_id": args.round_id,
        "purpose": "validation",
        "checkpoint_sha256": args.checkpoint_sha256,
        "scenario_manifest_sha256": digest,
        "scenario_ids": [episode["scenario_id"] for episode in episodes],
        "observation_mode": args.observation_mode,
        "metric": RAVEN_SUCCESS_METRIC,
    })
    assignment = {
        "schema": "roborender.raven_success_validation_assignment.v1",
        "evaluation_id": args.evaluation_id,
        "run_id": args.run_id,
        "round_id": args.round_id,
        "checkpoint_sha256": args.checkpoint_sha256,
        "scenario_manifest": str(args.panel.resolve()),
        "scenario_manifest_sha256": digest,
        "observation_mode": args.observation_mode,
        "action_policy": "deterministic_student_action_mean_beta_0",
        "metric": RAVEN_SUCCESS_METRIC,
        "episode_count": len(episodes),
        "episodes": episodes,
    }
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
    create.add_argument("--checkpoint-sha256", required=True)
    create.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    create.add_argument("--observation-mode", choices=("raytraced", "generated"), default="raytraced")
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
