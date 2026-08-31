#!/usr/bin/env python3
"""Repository-local entry point for :mod:`round_dagger.cli`."""

import sys
from pathlib import Path

# Direct execution sets ``sys.path[0]`` to ``scripts/round_dagger``.  Put the
# repository root first so that directory cannot shadow the actual package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
