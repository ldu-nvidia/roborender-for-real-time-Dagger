#!/usr/bin/env python3
"""Materialize or verify the sealed native-H200 held-out contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from round_dagger.heldout_contracts import (  # noqa: E402
    HeldoutContractError,
    OUTPUT_PATHS,
    materialize,
    verify_commitment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("materialize")
    verify = subcommands.add_parser("verify-commitment")
    verify.add_argument(
        "--commitment", type=Path, default=OUTPUT_PATHS["test_commitment"]
    )
    verify.add_argument(
        "--test-registry", type=Path, default=OUTPUT_PATHS["test_registry"]
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "materialize":
        report = materialize()
    else:
        report = verify_commitment(arguments.commitment, arguments.test_registry)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HeldoutContractError as exc:
        print(f"held-out contract operation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
