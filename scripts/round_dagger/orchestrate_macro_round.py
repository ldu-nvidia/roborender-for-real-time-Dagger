#!/usr/bin/env python3
"""Run or resume one fail-closed 4x64x512 DAgger macro-round."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.orchestration import (  # noqa: E402
    MacroRoundConfig,
    MacroRoundOrchestrator,
    OrchestrationError,
)
from round_dagger.artifacts import ArtifactError  # noqa: E402
from round_dagger.handoff import HandoffError  # noqa: E402
from round_dagger.promotion import PromotionError  # noqa: E402
from round_dagger.replay import ReplayError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="strict orchestration_config.v1 JSON; credentials belong in the environment",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = MacroRoundConfig.load(arguments.config)
    result = MacroRoundOrchestrator(config).run()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OrchestrationError,
        ArtifactError,
        HandoffError,
        PromotionError,
        ReplayError,
    ) as error:
        print(f"round DAgger orchestration failed: {error}", file=sys.stderr)
        raise SystemExit(2)
