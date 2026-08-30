"""In-container command-line entrypoint for a staged Lepton learner request."""

from __future__ import annotations

import argparse
import json
import sys

from round_dagger.lepton_job import LeptonJobError, execute_request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--semantic-attempt", required=True, type=int)
    args = parser.parse_args()
    result = execute_request(
        contract_path=args.contract,
        contract_sha256=args.contract_sha256,
        request_path=args.request,
        request_sha256=args.request_sha256,
        semantic_attempt=args.semantic_attempt,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeptonJobError as exc:
        print(f"Lepton learner execution failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
