#!/usr/bin/env python3
"""Plan or finalize one H200-registered fixed-panel Raven evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

from dagger_rtx.durable import atomic_write_json  # noqa: E402
from dagger_rtx.fixed_panel_validation import (  # noqa: E402
    AuthenticatedEvaluationClient,
    DurableOutcomeJournal,
    FixedPanelAssignment,
)


def add_assignment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--scenario-manifest-sha256", required=True)


def load(args: argparse.Namespace) -> FixedPanelAssignment:
    return FixedPanelAssignment.load(
        args.assignment,
        expected_assignment_sha256=args.assignment_sha256,
        expected_evaluation_id=args.evaluation_id,
        expected_run_id=args.run_id,
        expected_round_id=args.round_id,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        expected_scenario_manifest_sha256=args.scenario_manifest_sha256,
    )


def plan(args: argparse.Namespace) -> int:
    assignment = load(args)
    value = {
        "schema": "roborender.raven_success_validation_execution_plan.v1",
        "assignment_path": str(assignment.path),
        "assignment_sha256": assignment.assignment_sha256,
        "evaluation_id": assignment.evaluation_id,
        "run_id": assignment.run_id,
        "round_id": assignment.round_id,
        "checkpoint_sha256": assignment.checkpoint_sha256,
        "policy_role": assignment.policy_role,
        "observation_mode": assignment.observation_mode,
        "action_policy": assignment.action_policy,
        "teacher_checkpoint_sha256": assignment.teacher_checkpoint_sha256,
        "wrist_calibration_sha256": assignment.wrist_calibration_sha256,
        "wrist_calibration_id": assignment.wrist_calibration_id,
        "scenario_manifest_sha256": assignment.scenario_manifest_sha256,
        "episode_count": len(assignment.episodes),
        "motion_groups": list(assignment.motion_groups),
        "seeds": list(assignment.seeds),
        "execution": "four_sequential_gpu1_cohorts_of_64",
        "cohorts": [
            {
                "cohort_index": cohort.cohort_index,
                "cohort_id": cohort.cohort_id,
                "paired_seed": cohort.paired_seed,
                "environment_count": len(cohort.cases),
                "scenario_ids": [case.scenario_id for case in cohort.cases],
            }
            for cohort in assignment.cohorts()
        ],
    }
    atomic_write_json(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


def summarize(args: argparse.Namespace) -> int:
    assignment = load(args)
    client = AuthenticatedEvaluationClient(args.evaluation_url, args.token_file)
    journal = DurableOutcomeJournal(args.ledger_root, assignment)
    path = journal.write_summary(client)
    value = json.loads(path.read_bytes())
    if args.output != path:
        atomic_write_json(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


def attest(args: argparse.Namespace) -> int:
    assignment = load(args)
    client = AuthenticatedEvaluationClient(args.evaluation_url, args.token_file)
    journal = DurableOutcomeJournal(args.ledger_root, assignment)
    report = journal.verify_remote(client, require_complete=False)
    print(json.dumps(report, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make_plan = commands.add_parser("plan")
    add_assignment_arguments(make_plan)
    make_plan.add_argument("--output", type=Path, required=True)
    make_plan.set_defaults(handler=plan)

    check = commands.add_parser("attest")
    add_assignment_arguments(check)
    check.add_argument("--ledger-root", type=Path, required=True)
    check.add_argument("--token-file", type=Path, required=True)
    check.add_argument("--evaluation-url", default="http://127.0.0.1:18614")
    check.set_defaults(handler=attest)

    finish = commands.add_parser("summarize")
    add_assignment_arguments(finish)
    finish.add_argument("--ledger-root", type=Path, required=True)
    finish.add_argument("--token-file", type=Path, required=True)
    finish.add_argument("--evaluation-url", default="http://127.0.0.1:18614")
    finish.add_argument("--output", type=Path, required=True)
    finish.set_defaults(handler=summarize)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
