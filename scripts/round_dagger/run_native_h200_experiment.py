#!/usr/bin/env python3
"""Command-line entrypoint for the local native-H200 experiment controller."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from round_dagger.native_h200_experiment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
