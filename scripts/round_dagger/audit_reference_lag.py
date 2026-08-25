#!/usr/bin/env python3
"""Publish an immutable proof of the Image3F lag-reference transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from round_dagger.artifacts import publish_immutable_json
from round_dagger.reference_audit import audit_reference_telemetry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar-dir", required=True)
    parser.add_argument("--lag", required=True, type=int)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--rollout-shard")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_reference_telemetry(
        args.scalar_dir,
        lag=args.lag,
        expected_steps=args.expected_steps,
        rollout_shard=args.rollout_shard,
    )
    output = Path(args.output).expanduser().resolve()
    digest = publish_immutable_json(output, report)
    print(json.dumps({"output": str(output), "sha256": digest, **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
