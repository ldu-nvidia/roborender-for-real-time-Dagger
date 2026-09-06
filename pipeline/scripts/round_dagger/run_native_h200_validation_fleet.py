#!/usr/bin/env python3
"""Dry-run, launch, or verify the sealed native-H200 validation collector."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from round_dagger.native_h200_validation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
