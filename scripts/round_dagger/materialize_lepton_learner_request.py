#!/usr/bin/env python3
"""Materialize an immutable Lepton learner request from orchestrator env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.lepton_job import (  # noqa: E402
    LeptonJobError,
    load_contract,
    materialize_request,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--local-campaign-root", required=True)
    parser.add_argument("--remote-campaign-root", required=True)
    parser.add_argument("--remote-learner-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    if contract.sha256 != args.contract_sha256:
        raise LeptonJobError("backend contract SHA-256 mismatch")
    request = materialize_request(
        args.output,
        contract=contract,
        local_campaign_root=args.local_campaign_root,
        remote_campaign_root=args.remote_campaign_root,
        remote_learner_output=args.remote_learner_output,
        source_environment=os.environ,
    )
    print(
        json.dumps(
            {
                "schema": request.document["schema"],
                "request_path": str(Path(args.output).resolve()),
                "request_sha256": request.sha256,
                "learner_identity_sha256": request.identity_sha256,
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
        print(f"Lepton request materialization failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
