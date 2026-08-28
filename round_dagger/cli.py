"""Command-line control plane for round-based RoboRender DAgger."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from round_dagger.artifacts import load_strict_json, sha256_file
from round_dagger.gr00t_policy import REPLAY_ACTION_DIM
from round_dagger.promotion import (
    promote_candidate,
    stage_bootstrap_policy,
    stage_candidate,
    write_evaluation,
)
from round_dagger.replay import ReplayLedger
from round_dagger.state import (
    _check_path_chain,
    _internal_path,
    initialize_campaign,
    load_campaign,
    load_collection_manifest,
    load_state,
    open_macro_round,
    seal_collection,
    transition,
)

_SHA_RE = re.compile(r"[0-9a-f]{64}")


def _digest(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file() and not path.is_symlink():
        return sha256_file(path)
    if _SHA_RE.fullmatch(value):
        return value
    raise argparse.ArgumentTypeError(
        f"{value!r} must be a regular file or lowercase SHA-256"
    )


def _root(value: str) -> Path:
    path = Path(os.path.abspath(Path(value).expanduser()))
    try:
        _check_path_chain(path)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"unsafe campaign root: {exc}") from exc
    # ``init`` is allowed to name a not-yet-created directory.
    return path


def _json_metrics(value: str) -> dict[str, float]:
    path = Path(value).expanduser()
    try:
        document = load_strict_json(path) if path.is_file() else json.loads(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid metrics JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise argparse.ArgumentTypeError("metrics must be a JSON object")
    return {str(key): float(number) for key, number in document.items()}


def _json_object_file(value: str) -> dict[str, Any]:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise argparse.ArgumentTypeError(
            "data lineage must be an absolute non-symlink JSON file"
        )
    try:
        document = load_strict_json(path)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid data-lineage JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise argparse.ArgumentTypeError("data lineage must be a JSON object")
    return document


def _print(document: Any) -> None:
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialize a campaign and version-zero policy")
    init.add_argument("--root", type=_root, required=True)
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--bootstrap-checkpoint", required=True)
    init.add_argument("--code", type=_digest, required=True)
    init.add_argument("--policy-config", type=_digest, required=True)
    init.add_argument("--teacher", type=_digest, required=True)
    init.add_argument("--renderer", type=_digest, required=True)
    init.add_argument("--camera-mapping", type=_digest, required=True)
    init.add_argument("--training-config", type=_digest, required=True)
    init.add_argument("--target-environments", type=int, default=256)
    init.add_argument("--active-environments", type=int, default=64)
    init.add_argument("--steps-per-environment", type=int, default=512)
    init.add_argument(
        "--macro-round-count",
        type=int,
        help=(
            "persist a finite campaign horizon; omitted preserves the legacy "
            "unbounded non-production campaign contract"
        ),
    )
    init.add_argument(
        "--data-lineage",
        type=_json_object_file,
        help=(
            "absolute immutable production_data_lineage.v1 JSON; required for "
            "a production campaign and omitted for legacy experiments"
        ),
    )
    # Round replay stores the 11-D learned student/teacher action followed by
    # the fixed 15-D Homie action.  The 31-D value in the Isaac configuration
    # is the downstream whole-body command after HomieBase expands this input,
    # not the actor/replay width.
    init.add_argument("--action-dim", type=int, default=REPLAY_ACTION_DIM)
    init.add_argument(
        "--beta-kind", choices=("constant", "linear", "exponential"), default="linear"
    )
    init.add_argument("--beta-start", type=float, default=1.0)
    init.add_argument("--beta-end", type=float, default=0.0)
    init.add_argument("--beta-decay-rounds", type=int, default=4)
    init.add_argument("--beta-rate", type=float, default=0.99)

    status = sub.add_parser("status", help="show the campaign and coordinator state")
    status.add_argument("--root", type=_root, required=True)

    open_parser = sub.add_parser(
        "open-round", help="freeze policy and beta for one macro-round"
    )
    open_parser.add_argument("--root", type=_root, required=True)
    open_parser.add_argument("--expected-revision", type=int)

    seal = sub.add_parser(
        "seal-collection",
        help="verify cohorts and publish a macro-round collection seal",
    )
    seal.add_argument("--root", type=_root, required=True)
    seal.add_argument("--macro-round", type=int, required=True)
    seal.add_argument("--cohort-manifest", action="append", required=True)

    collected = sub.add_parser(
        "mark-collected", help="bind a verified macro-round collection seal"
    )
    collected.add_argument("--root", type=_root, required=True)
    collected.add_argument("--macro-round", type=int, required=True)
    collected.add_argument("--round-manifest", required=True)

    ingest = sub.add_parser("ingest", help="transactionally ingest one sealed cohort")
    ingest.add_argument("--root", type=_root, required=True)
    ingest.add_argument("--cohort-manifest", required=True)

    snapshot = sub.add_parser(
        "snapshot", help="freeze cumulative replay through one round"
    )
    snapshot.add_argument("--root", type=_root, required=True)
    snapshot.add_argument("--campaign-id")
    snapshot.add_argument("--through-macro-round", type=int, required=True)
    snapshot.add_argument("--expected-cohorts", type=int)
    snapshot.add_argument("--output")

    candidate = sub.add_parser("stage-candidate", help="stage a learner checkpoint")
    candidate.add_argument("--root", type=_root, required=True)
    candidate.add_argument("--macro-round", type=int, required=True)
    candidate.add_argument("--checkpoint", required=True)
    candidate.add_argument("--code", type=_digest, required=True)
    candidate.add_argument("--learner-config", type=_digest, required=True)
    candidate.add_argument("--optimizer-steps", type=int, required=True)

    evaluate = sub.add_parser(
        "record-evaluation", help="bind fixed-seed evaluation evidence"
    )
    evaluate.add_argument("--root", type=_root, required=True)
    evaluate.add_argument("--macro-round", type=int, required=True)
    evaluate.add_argument("--candidate-manifest", required=True)
    evaluate.add_argument(
        "--passed", action=argparse.BooleanOptionalAction, required=True
    )
    evaluate.add_argument("--metrics", type=_json_metrics, required=True)
    evaluate.add_argument("--evaluator-config", type=_digest, required=True)
    evaluate.add_argument("--evidence-bundle", required=True)
    evaluate.add_argument("--evidence-bundle-sha256", type=_digest, required=True)

    promote = sub.add_parser("promote", help="atomically promote a passed candidate")
    promote.add_argument("--root", type=_root, required=True)
    promote.add_argument("--macro-round", type=int, required=True)
    promote.add_argument("--candidate-manifest", required=True)
    promote.add_argument("--evaluation-manifest", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root
    if args.command == "init":
        bootstrap = stage_bootstrap_policy(
            root,
            checkpoint_source=args.bootstrap_checkpoint,
            code_sha256=args.code,
            policy_config_sha256=args.policy_config,
        )
        result = initialize_campaign(
            root,
            campaign_id=args.campaign_id,
            bootstrap_policy_sha256=bootstrap["checkpoint_sha256"],
            bootstrap_policy_path=bootstrap["checkpoint_path"],
            target_environments=args.target_environments,
            active_environments=args.active_environments,
            steps_per_environment=args.steps_per_environment,
            action_dim=args.action_dim,
            teacher_sha256=args.teacher,
            renderer_sha256=args.renderer,
            camera_mapping_sha256=args.camera_mapping,
            training_config_sha256=args.training_config,
            macro_round_count=args.macro_round_count,
            data_lineage=args.data_lineage,
            beta_schedule={
                "kind": args.beta_kind,
                "start": args.beta_start,
                "end": args.beta_end,
                "decay_rounds": args.beta_decay_rounds,
                "rate": args.beta_rate,
            },
        )
        _print({"bootstrap": bootstrap, **result})
    elif args.command == "status":
        campaign = load_campaign(root)
        coordinator = load_state(root)
        result: dict[str, Any] = {
            "campaign": campaign,
            "state": coordinator,
        }
        macro_round_count = campaign.get("macro_round_count")
        if macro_round_count is not None:
            completed = coordinator["next_macro_round"]
            result["execution"] = {
                "macro_round_count": macro_round_count,
                "completed_macro_rounds": completed,
                "remaining_macro_rounds": macro_round_count - completed,
                "complete": (
                    coordinator["phase"] == "READY"
                    and completed == macro_round_count
                ),
            }
        _print(result)
    elif args.command == "open-round":
        _print(open_macro_round(root, expected_revision=args.expected_revision))
    elif args.command == "seal-collection":
        _print(
            seal_collection(
                root,
                macro_round=args.macro_round,
                cohort_manifests=args.cohort_manifest,
            )
        )
    elif args.command == "mark-collected":
        report = load_collection_manifest(
            root, args.round_manifest, expected_macro_round=args.macro_round
        )
        _print(
            transition(
                root,
                expected_phase="COLLECTING",
                next_phase="COLLECTED",
                macro_round=args.macro_round,
                evidence_path=report["manifest_path"],
                evidence_sha256=report["manifest_sha256"],
            )
        )
    elif args.command == "ingest":
        ledger_path = root / "replay" / "ledger.sqlite3"
        with ReplayLedger(ledger_path) as ledger:
            _print(ledger.ingest_cohort(args.cohort_manifest))
    elif args.command == "snapshot":
        campaign = load_campaign(root)
        campaign_id = args.campaign_id or campaign["campaign_id"]
        if campaign_id != campaign["campaign_id"]:
            raise ValueError("--campaign-id does not match the campaign root")
        expected_cohorts = args.expected_cohorts or campaign["cohort_count"]
        if expected_cohorts != campaign["cohort_count"]:
            raise ValueError("--expected-cohorts does not match campaign.cohort_count")
        output = (
            _internal_path(root, args.output, "replay snapshot output")[0]
            if args.output
            else root
            / "replay"
            / f"snapshot_through_round_{args.through_macro_round:06d}.json"
        )
        with ReplayLedger(root / "replay" / "ledger.sqlite3") as ledger:
            report = ledger.build_snapshot(
                campaign_id,
                through_macro_round=args.through_macro_round,
                output=output,
                expected_train_cohorts=expected_cohorts,
            )
        transition(
            root,
            expected_phase="COLLECTED",
            next_phase="INGESTED",
            macro_round=args.through_macro_round,
            evidence_path=report["path"],
            evidence_sha256=report["file_sha256"],
        )
        _print(report)
    elif args.command == "stage-candidate":
        state = load_state(root)
        candidate = stage_candidate(
            root,
            macro_round=args.macro_round,
            checkpoint_source=args.checkpoint,
            replay_snapshot_sha256=state["replay_snapshot_sha256"],
            code_sha256=args.code,
            learner_config_sha256=args.learner_config,
            optimizer_steps=args.optimizer_steps,
        )
        transition(
            root,
            expected_phase="INGESTED",
            next_phase="TRAINED",
            macro_round=args.macro_round,
            evidence_path=candidate["manifest_path"],
            evidence_sha256=candidate["manifest_sha256"],
            candidate_policy={
                "version": candidate["policy_version"],
                "macro_round": candidate["macro_round"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "checkpoint_path": candidate["checkpoint_path"],
                "runtime_policy_sha256": candidate["runtime_policy_sha256"],
                "policy_spec_sha256": candidate["policy_spec_sha256"],
                "policy_bundle_manifest_sha256": candidate[
                    "policy_bundle_manifest_sha256"
                ],
                "manifest_sha256": candidate["manifest_sha256"],
                "manifest_path": candidate["manifest_path"],
                "parent_policy_version": candidate["parent_policy_version"],
                "parent_checkpoint_sha256": candidate["parent_checkpoint_sha256"],
                "replay_snapshot_sha256": candidate["replay_snapshot_sha256"],
                "code_sha256": candidate["code_sha256"],
                "policy_config_sha256": candidate["learner_config_sha256"],
            },
        )
        _print(candidate)
    elif args.command == "record-evaluation":
        evidence = write_evaluation(
            root,
            candidate_manifest=args.candidate_manifest,
            passed=args.passed,
            metrics=args.metrics,
            evaluator_config_sha256=args.evaluator_config,
            evidence_bundle_path=args.evidence_bundle,
            evidence_bundle_sha256=args.evidence_bundle_sha256,
        )
        transition(
            root,
            expected_phase="TRAINED",
            next_phase="EVALUATED",
            macro_round=args.macro_round,
            evidence_path=evidence["manifest_path"],
            evidence_sha256=evidence["evidence_sha256"],
        )
        _print(evidence)
    elif args.command == "promote":
        _print(
            promote_candidate(
                root,
                macro_round=args.macro_round,
                candidate_manifest=args.candidate_manifest,
                evaluation_manifest=args.evaluation_manifest,
            )
        )
    else:  # pragma: no cover - argparse prevents this
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
