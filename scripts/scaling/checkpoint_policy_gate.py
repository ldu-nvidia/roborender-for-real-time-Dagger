#!/usr/bin/env python3
"""Attest that training changed policy tensors, not only checkpoint metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .resume_continuity_gate import (
        GateError,
        _atomic_json,
        _load_torch_checkpoint,
        _policy_digest,
        _policy_schema_digest,
        _state_step,
    )
except ImportError:  # Direct script execution inside the sealed trainer image.
    from resume_continuity_gate import (
        GateError,
        _atomic_json,
        _load_torch_checkpoint,
        _policy_digest,
        _policy_schema_digest,
        _state_step,
    )


def inspect(
    torch, path: Path, expected_step: int, expected_sha256: str
) -> dict[str, object]:
    resolved = Path(os.path.abspath(path.expanduser()))
    payload, checkpoint_sha256 = _load_torch_checkpoint(
        torch, resolved, expected_sha256=expected_sha256
    )
    if payload.get("checkpoint_schema") != "gr00t.trl.full_state.v2":
        raise GateError("checkpoint is not an atomic full-state v2 checkpoint")
    step = _state_step(payload.get("state"))
    if step != expected_step:
        raise GateError(f"checkpoint step {step} != required {expected_step}")
    policy = payload.get("policy_state_dict")
    if not isinstance(policy, dict) or not policy:
        raise GateError("policy_state_dict must be a non-empty dictionary")
    names = payload.get("policy_parameter_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise GateError("policy_parameter_names must be a non-empty unique string list")
    missing = sorted(set(names) - set(policy))
    if missing:
        raise GateError(
            "policy_parameter_names references missing policy tensors: "
            + ", ".join(missing[:10])
        )
    parameter_state = {name: policy[name] for name in names}
    return {
        "path": str(resolved),
        "checkpoint_step": step,
        "checkpoint_sha256": checkpoint_sha256,
        # This digest is intentionally over named parameters only.  It exactly
        # reproduces the trainer's cross-rank DDP digest and excludes buffers.
        "policy_sha256": _policy_digest(torch, parameter_state),
        "policy_state_sha256": _policy_digest(torch, policy),
        "policy_parameter_schema_sha256": _policy_schema_digest(
            torch, policy, names
        ),
        "policy_tensor_count": len(parameter_state),
        "policy_state_tensor_count": len(policy),
    }


def validate_change(initial: dict[str, object], final: dict[str, object]) -> None:
    if (
        initial["policy_parameter_schema_sha256"]
        != final["policy_parameter_schema_sha256"]
    ):
        raise GateError("initial and final policy parameter schemas do not match exactly")
    if initial["policy_tensor_count"] != final["policy_tensor_count"]:
        raise GateError("initial and final policy parameter counts do not match")
    if initial["policy_sha256"] == final["policy_sha256"]:
        raise GateError("final policy tensors are identical to the fresh baseline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--expected-initial-sha256", required=True)
    parser.add_argument("--expected-final-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.final_step <= 0:
        raise SystemExit("--final-step must be positive")
    try:
        import torch

        initial = inspect(torch, args.initial, 0, args.expected_initial_sha256)
        final = inspect(
            torch, args.final, args.final_step, args.expected_final_sha256
        )
        validate_change(initial, final)
        report = {
            "schema": "fast_roborender.checkpoint_policy_change.v1",
            "status": "PASS",
            "initial": initial,
            "final": final,
        }
        _atomic_json(args.output, report)
    except GateError as exc:
        print(f"FATAL: {exc}")
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
