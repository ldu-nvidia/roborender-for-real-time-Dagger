#!/usr/bin/env python3
"""Seal a deterministic train/validation/test split after native H200 BC collection."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from round_dagger.conditioning_split import (  # noqa: E402
    CANONICAL_CAMPAIGN_ROOT,
    ConditioningSplitError,
    EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV,
    split_collection,
)


def main() -> int:
    try:
        report = split_collection(
            CANONICAL_CAMPAIGN_ROOT,
            protocol_path=ROOT
            / "configs/round_dagger/native_h200_experiment_protocol.json",
            enforce_canonical_root=True,
            expected_transitions_per_group=EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV,
        )
    except ConditioningSplitError as exc:
        raise SystemExit(f"BC collection split failed: {exc}") from exc
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
