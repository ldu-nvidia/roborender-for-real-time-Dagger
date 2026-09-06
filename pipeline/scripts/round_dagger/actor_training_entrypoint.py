#!/usr/bin/env python3
"""Run the immutable GR00T trainer with the actor warm-start bridge installed."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from round_dagger.actor_checkpoint_compat import (
    ActorCheckpointCompatibilityError,
    install_actor_warmstart_checkpoint_compat,
)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit("usage: actor_training_entrypoint.py ABSOLUTE_TRAINER [ARGS...]")
    trainer = Path(argv[1]).expanduser()
    if not trainer.is_absolute() or trainer.is_symlink() or not trainer.is_file():
        raise SystemExit("actor trainer path must be an absolute regular non-symlink file")
    policy = os.environ.get("ROUND_DAGGER_POLICY_CHECKPOINT", "")
    policy_sha256 = os.environ.get("ROUND_DAGGER_POLICY_SHA256", "")
    try:
        install_actor_warmstart_checkpoint_compat(
            policy_checkpoint=policy,
            policy_sha256=policy_sha256,
        )
    except ActorCheckpointCompatibilityError as exc:
        raise SystemExit(f"actor checkpoint compatibility preflight failed: {exc}") from exc
    # GR00T uses Hydra's normal __main__ execution semantics.  Preserve that
    # contract exactly after the in-process bridge has been bound.
    sys.argv = [os.fspath(trainer), *argv[2:]]
    runpy.run_path(os.fspath(trainer), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv)
