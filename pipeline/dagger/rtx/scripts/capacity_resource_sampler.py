#!/usr/bin/env python3
"""Sample one isolated capacity container until SIGINT/SIGTERM."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

from dagger_rtx.resource_monitor import sample_until_stopped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=1.0)
    args = parser.parse_args()
    if args.interval_s < 0.1 or args.interval_s > 10:
        raise SystemExit("--interval-s must be in [0.1,10]")
    stopped = False

    def stop(_signum, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    sample_until_stopped(
        physical_gpu=args.gpu,
        container_name=args.container_name,
        journal_path=args.journal,
        summary_path=args.summary,
        should_stop=lambda: stopped,
        interval_s=args.interval_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
