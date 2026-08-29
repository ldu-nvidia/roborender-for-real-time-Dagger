#!/usr/bin/env python3
"""Evaluate active policy-v0 on fixed256 before rollout collection completes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger import artifacts  # noqa: E402
from round_dagger.heldout_baseline import (  # noqa: E402
    BASELINE_WANDB_RECEIPT_SCHEMA,
    CANONICAL_WANDB_RECEIPT_RELATIVE,
    HeldoutBaselineError,
    _publish_idempotent,
    _symbol,
    baseline_wandb_run_id,
    evaluate_active_policy_v0,
    wandb_metrics,
)
from scripts.scaling.wandb_preflight import (  # noqa: E402
    WandbPreflightError,
    _load_wandb,
    query_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--heldout-snapshot", type=Path, required=True)
    parser.add_argument("--heldout-snapshot-file-sha256", required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gr00t-root", type=Path)
    parser.add_argument(
        "--model-factory",
        default="round_dagger.gr00t_policy:create_vision_actor_obj_pred",
    )
    parser.add_argument(
        "--adapter-factory",
        default="round_dagger.gr00t_policy:vision_actor_obj_pred_adapter_factory",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-expected-email")
    parser.add_argument("--wandb-authorization-scope", choices=("team",))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    wandb_values = (
        arguments.wandb_project,
        arguments.wandb_entity,
        arguments.wandb_expected_email,
        arguments.wandb_authorization_scope,
    )
    if any(value is not None for value in wandb_values) and any(
        value is None for value in wandb_values
    ):
        print("policy-v0 baseline failed: W&B arguments must be supplied together", file=sys.stderr)
        return 2
    try:
        if arguments.gr00t_root is not None:
            gr00t_root = Path(os.path.abspath(arguments.gr00t_root.expanduser()))
            if gr00t_root.is_symlink() or not gr00t_root.is_dir():
                raise HeldoutBaselineError("GR00T root is missing or unsafe")
            sys.path.insert(0, str(gr00t_root))
        result = evaluate_active_policy_v0(
            campaign_root=arguments.campaign_root,
            heldout_snapshot_path=arguments.heldout_snapshot,
            heldout_snapshot_file_sha256=arguments.heldout_snapshot_file_sha256,
            code_sha256=arguments.code_sha256,
            output=arguments.output,
            device=arguments.device,
            batch_size=arguments.batch_size,
            model_factory=_symbol(arguments.model_factory),
            adapter_factory=_symbol(arguments.adapter_factory),
        )
        if arguments.wandb_project is not None:
            wandb = _load_wandb()
            query_identity(
                wandb,
                expected_email=arguments.wandb_expected_email,
                requested_entity=arguments.wandb_entity,
                authorization_scope=arguments.wandb_authorization_scope,
                timeout_s=15.0,
            )
            run_id = baseline_wandb_run_id(
                campaign_id=result["campaign_id"],
                campaign_sha256=result["campaign_sha256"],
                checkpoint_sha256=result["checkpoint_sha256"],
                heldout_snapshot_sha256=result["heldout_snapshot_sha256"],
            )
            run = wandb.init(
                project=arguments.wandb_project,
                entity=arguments.wandb_entity,
                id=run_id,
                name=f"{result['campaign_id']}-policy-v0-fixed256-baseline",
                resume="allow",
                config={
                    "round_dagger/campaign_id": result["campaign_id"],
                    "round_dagger/policy_version": 0,
                    "round_dagger/checkpoint_sha256": result["checkpoint_sha256"],
                    "round_dagger/heldout_snapshot_sha256": result[
                        "heldout_snapshot_sha256"
                    ],
                    "round_dagger/result_sha256": result["result_sha256"],
                    "round_dagger/evaluation_scope": "pre-collection-fixed256-action-quality",
                },
                tags=("round-dagger", "baseline", "fixed256", "policy-v0"),
            )
            exit_code = 1
            try:
                run.define_metric("baseline/optimizer_step")
                run.define_metric("baseline/*", step_metric="baseline/optimizer_step")
                run.define_metric("quality/*", step_metric="baseline/optimizer_step")
                run.log(wandb_metrics(result), step=0)
                exit_code = 0
            finally:
                run.finish(exit_code=exit_code)
            if exit_code != 0:
                raise HeldoutBaselineError("W&B baseline sender did not finish cleanly")
            receipt = {
                "schema": BASELINE_WANDB_RECEIPT_SCHEMA,
                "campaign_id": result["campaign_id"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "heldout_snapshot_sha256": result["heldout_snapshot_sha256"],
                "result_sha256": result["result_sha256"],
                "run_id": run_id,
                "entity": arguments.wandb_entity,
                "project": arguments.wandb_project,
                "optimizer_step": 0,
            }
            receipt_path = (
                Path(os.path.abspath(arguments.campaign_root.expanduser()))
                / CANONICAL_WANDB_RECEIPT_RELATIVE
            )
            receipt_sha = _publish_idempotent(receipt_path, receipt)
            result["wandb_receipt_path"] = str(receipt_path)
            result["wandb_receipt_sha256"] = receipt_sha
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except WandbPreflightError:
        print("policy-v0 baseline failed: W&B identity/preflight rejected", file=sys.stderr)
        return 2
    except Exception as exc:
        if isinstance(exc, HeldoutBaselineError):
            detail = str(exc)
        else:
            detail = f"unexpected {type(exc).__name__}"
        print(f"policy-v0 baseline failed: {detail}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
