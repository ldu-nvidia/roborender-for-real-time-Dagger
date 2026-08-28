#!/usr/bin/env python3
"""Build or recursively verify the fixed 256-case held-out replay bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from round_dagger.heldout_admission import (
    HeldoutAdmissionError,
    build_fixed_heldout_bundle,
    verify_fixed_heldout_bundle,
)


def _sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--motion-split", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    _sources(build)
    build.add_argument("--cohort", type=Path, action="append", required=True)
    build.add_argument("--snapshot-output", type=Path, required=True)
    build.add_argument("--manifest-output", type=Path, required=True)
    verify = commands.add_parser("verify")
    _sources(verify)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "plan": args.plan,
        "motion_split": args.motion_split,
        "scenarios": args.scenarios,
    }
    if args.command == "build":
        result = build_fixed_heldout_bundle(
            args.cohort,
            snapshot_output=args.snapshot_output,
            manifest_output=args.manifest_output,
            **common,
        )
        output = {
            "status": "published",
            "snapshot_path": result["snapshot"]["path"],
            "snapshot_file_sha256": result["snapshot"]["file_sha256"],
            "snapshot_sha256": result["snapshot"]["snapshot_sha256"],
            "manifest_path": result["case_manifest"]["path"],
            "manifest_file_sha256": result["case_manifest"]["file_sha256"],
            "manifest_sha256": result["case_manifest"]["manifest_sha256"],
            "case_count": result["case_manifest"]["case_count"],
            "shard_count": result["case_manifest"]["shard_count"],
        }
    else:
        output = verify_fixed_heldout_bundle(
            args.snapshot, args.manifest, **common
        )
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HeldoutAdmissionError as error:
        print(f"held-out admission failed: {error}", file=sys.stderr)
        raise SystemExit(2)
