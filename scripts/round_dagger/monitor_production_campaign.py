#!/usr/bin/env python3
"""Entrypoint for the read-only 30-minute production health monitor."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.production_campaign_monitor import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
