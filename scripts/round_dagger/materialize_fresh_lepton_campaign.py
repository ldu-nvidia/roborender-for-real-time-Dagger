#!/usr/bin/env python3
"""Create and validate one previously absent Lepton production campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.fresh_lepton_campaign import (  # noqa: E402
    FreshLeptonCampaignError,
    materialize_fresh_lepton_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protected-oci-campaign-root", type=Path, required=True)
    parser.add_argument("--invariant-profile", type=Path, required=True)
    parser.add_argument("--invariant-profile-sha256", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--runtime-contract-sha256", required=True)
    parser.add_argument("--capture-aggregate", type=Path, required=True)
    parser.add_argument("--capture-aggregate-sha256", required=True)
    parser.add_argument("--heldout-plan", type=Path, required=True)
    parser.add_argument("--heldout-plan-sha256", required=True)
    parser.add_argument("--motion-split", type=Path, required=True)
    parser.add_argument("--motion-split-sha256", required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--scenarios-sha256", required=True)
    parser.add_argument("--base-learner-contract", type=Path, required=True)
    parser.add_argument("--base-learner-contract-sha256", required=True)
    parser.add_argument("--learner-sealed-source-root", required=True)
    parser.add_argument("--learner-source-inventory-sha256", required=True)
    parser.add_argument("--evaluation-sealed-source-root", type=Path, required=True)
    parser.add_argument("--evaluation-source-inventory-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = materialize_fresh_lepton_campaign(
            campaign_root=arguments.campaign_root,
            protected_oci_campaign_root=arguments.protected_oci_campaign_root,
            invariant_profile_path=arguments.invariant_profile,
            invariant_profile_sha256=arguments.invariant_profile_sha256,
            runtime_contract_path=arguments.runtime_contract,
            runtime_contract_sha256=arguments.runtime_contract_sha256,
            capture_aggregate_path=arguments.capture_aggregate,
            capture_aggregate_sha256=arguments.capture_aggregate_sha256,
            heldout_plan_path=arguments.heldout_plan,
            heldout_plan_sha256=arguments.heldout_plan_sha256,
            motion_split_path=arguments.motion_split,
            motion_split_sha256=arguments.motion_split_sha256,
            scenarios_path=arguments.scenarios,
            scenarios_sha256=arguments.scenarios_sha256,
            base_learner_contract_path=arguments.base_learner_contract,
            base_learner_contract_sha256=(
                arguments.base_learner_contract_sha256
            ),
            learner_sealed_source_root=arguments.learner_sealed_source_root,
            learner_source_inventory_sha256=(
                arguments.learner_source_inventory_sha256
            ),
            evaluation_sealed_source_root=(
                arguments.evaluation_sealed_source_root
            ),
            evaluation_source_inventory_sha256=(
                arguments.evaluation_source_inventory_sha256
            ),
        )
    except FreshLeptonCampaignError as exc:
        print(f"fresh Lepton campaign materialization refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
