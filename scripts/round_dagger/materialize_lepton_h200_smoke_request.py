#!/usr/bin/env python3
"""Create one immutable, fixed four-step Lepton H200 smoke request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.lepton_job import LeptonJobError, load_contract  # noqa: E402
from round_dagger.lepton_smoke_job import materialize_smoke_request  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    contract = load_contract(arguments.contract)
    if contract.sha256 != arguments.contract_sha256:
        raise LeptonJobError("backend contract SHA-256 mismatch")
    request = materialize_smoke_request(arguments.output, contract=contract)
    print(
        json.dumps(
            {
                "schema": request.document["schema"],
                "request_path": str(Path(arguments.output).resolve()),
                "request_sha256": request.sha256,
                "smoke_identity_sha256": request.identity_sha256,
                "max_steps": request.execution["max_steps"],
                "world_size": request.execution["world_size"],
                "heldout_cases": request.execution["heldout_cases"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeptonJobError as exc:
        print(f"Lepton bounded-smoke request failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
