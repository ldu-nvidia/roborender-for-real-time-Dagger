#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

from dagger_rtx.capacity_report import build_from_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a bounded capacity acceptance report")
    parser.add_argument("--native-result", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--resource-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_from_paths(
        native_result_path=args.native_result,
        telemetry_path=args.telemetry,
        resource_summary_path=args.resource_summary,
        output_path=args.output,
    )
    print(json.dumps({"result": report["result"], "report": str(args.output)}, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
