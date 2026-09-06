#!/usr/bin/env python3
"""Report the serialization-independent policy hash of a verified checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from round_dagger.actor import ActorExportError, runtime_state_dict_sha256  # noqa: E402
from round_dagger.artifacts import sha256_file  # noqa: E402
from round_dagger.policy_identity import (  # noqa: E402
    PolicyIdentityError,
    runtime_policy_identity_sha256,
    validate_homie_controller_sha256,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser.parse_args()


def _policy_state(document: Any) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise ActorExportError("checkpoint root must be a mapping")
    for key in ("policy_state_dict", "actor_model_state_dict", "model_state_dict"):
        value = document.get(key)
        if isinstance(value, Mapping) and value:
            return value
    # A plain state_dict is allowed only when every entry is tensor/array-like.
    if document and all(
        hasattr(value, "shape") and hasattr(value, "dtype")
        for value in document.values()
    ):
        return document
    raise ActorExportError("checkpoint has no supported non-empty policy state_dict")


def main() -> int:
    arguments = _arguments()
    checkpoint = arguments.checkpoint.resolve(strict=True)
    if arguments.checkpoint.is_symlink() or checkpoint.is_symlink():
        raise ActorExportError("checkpoint must not be a symlink")
    expected = arguments.expected_sha256.lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ActorExportError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise ActorExportError(f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}")

    import torch

    # Pickle loading is permitted only after the caller-provided content hash
    # proves that this is the exact trusted artifact selected for the round.
    document = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _policy_state(document)
    if not isinstance(document, Mapping):
        raise ActorExportError("checkpoint root must bind Homie controllers")
    try:
        controller_sha256 = validate_homie_controller_sha256(
            document.get("homie_controller_sha256")
        )
        policy_state_sha256 = runtime_state_dict_sha256(state)
        runtime_policy_sha256 = runtime_policy_identity_sha256(
            policy_state_sha256, controller_sha256
        )
    except PolicyIdentityError as exc:
        raise ActorExportError(str(exc)) from exc
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": actual,
                "policy_state_sha256": policy_state_sha256,
                "homie_controller_sha256": controller_sha256,
                "runtime_policy_sha256": runtime_policy_sha256,
                "state_entries": len(state),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
