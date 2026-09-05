#!/usr/bin/env python3
"""Run the bounded GPU-0 native-H200 BC saturation actor."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from round_dagger.native_actor_saturation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
