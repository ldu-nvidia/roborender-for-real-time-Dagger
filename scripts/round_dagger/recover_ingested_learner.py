#!/usr/bin/env python3
"""Recover exactly one INGESTED production round through learner TRAINED CAS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.artifacts import ArtifactError  # noqa: E402
from round_dagger.learner_recovery import (  # noqa: E402
    LearnerRecoveryError,
    recover_ingested_learner,
)
from round_dagger.orchestration import OrchestrationError  # noqa: E402
from round_dagger.policy_bundle import PolicyBundleError  # noqa: E402
from round_dagger.promotion import PromotionError  # noqa: E402
from round_dagger.replay import ReplayError  # noqa: E402
from round_dagger.state import StateError  # noqa: E402
from scripts.round_dagger.production_launch import ProductionLaunchError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--macro-round", required=True, type=int)
    parser.add_argument("--expected-state-revision", required=True, type=int)
    parser.add_argument("--expected-campaign-sha256", required=True)
    parser.add_argument("--expected-replay-snapshot-file-sha256", required=True)
    parser.add_argument("--expected-replay-snapshot-sha256", required=True)
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-runtime-policy-sha256", required=True)
    parser.add_argument("--campaign-contract", required=True, type=Path)
    parser.add_argument("--expected-campaign-contract-sha256", required=True)
    parser.add_argument("--learner-contract", required=True, type=Path)
    parser.add_argument("--expected-learner-contract-sha256", required=True)
    parser.add_argument("--sealed-source-root", required=True)
    parser.add_argument("--sealed-source-inventory-sha256", required=True)
    parser.add_argument("--expected-bridge-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = recover_ingested_learner(
            campaign_root=arguments.campaign_root,
            macro_round=arguments.macro_round,
            expected_state_revision=arguments.expected_state_revision,
            expected_campaign_sha256=arguments.expected_campaign_sha256,
            expected_replay_snapshot_file_sha256=(
                arguments.expected_replay_snapshot_file_sha256
            ),
            expected_replay_snapshot_sha256=(
                arguments.expected_replay_snapshot_sha256
            ),
            expected_parent_checkpoint_sha256=(
                arguments.expected_parent_checkpoint_sha256
            ),
            expected_parent_runtime_policy_sha256=(
                arguments.expected_parent_runtime_policy_sha256
            ),
            campaign_contract_path=arguments.campaign_contract,
            expected_campaign_contract_sha256=(
                arguments.expected_campaign_contract_sha256
            ),
            learner_contract_path=arguments.learner_contract,
            expected_learner_contract_sha256=(
                arguments.expected_learner_contract_sha256
            ),
            sealed_source_root=arguments.sealed_source_root,
            sealed_source_inventory_sha256=(
                arguments.sealed_source_inventory_sha256
            ),
            expected_bridge_sha256=arguments.expected_bridge_sha256,
            dry_run=arguments.dry_run,
        )
    except (
        LearnerRecoveryError,
        ArtifactError,
        OrchestrationError,
        PolicyBundleError,
        PromotionError,
        ReplayError,
        StateError,
        ProductionLaunchError,
    ) as exc:
        print(f"INGESTED learner recovery failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
