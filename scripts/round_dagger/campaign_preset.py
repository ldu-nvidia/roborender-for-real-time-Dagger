#!/usr/bin/env python3
"""Read-only CLI for allowlisted RoboRender DAgger campaign presets."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.campaign_presets import main


if __name__ == "__main__":
    raise SystemExit(main())
