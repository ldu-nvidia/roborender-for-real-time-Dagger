#!/usr/bin/env python3
"""Create an exact production learner contract from verified fixed-256 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.heldout_learner_contract import (  # noqa: E402
    HeldoutLearnerContractError,
    materialize_heldout_learner_contract,
)


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-contract", required=True, type=Path)
    parser.add_argument("--conversion-attestation", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = materialize_heldout_learner_contract(
            base_contract_path=arguments.base_contract,
            conversion_attestation_path=arguments.conversion_attestation,
            campaign_root=arguments.campaign_root,
            campaign_id=arguments.campaign_id,
            output_path=arguments.output,
        )
    except HeldoutLearnerContractError as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
