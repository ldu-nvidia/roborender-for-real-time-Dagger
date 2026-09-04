#!/usr/bin/env python3
"""Read-only entrypoint for the reserved Lepton H200 health projection."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.lepton_monitor import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
