"""Durable delivery evidence for actor-cohort W&B telemetry.

The generic sidecar cursor is an acknowledgement only after ``run.finish``.
This module verifies that every currently published scalar/media fingerprint is
covered by that cursor and publishes a replayable scalar when a sidecar wrapper
fails after the immutable cohort has already been sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.scaling import wandb_run_watcher as watcher

from .actor_pair_terminal import (
    SCALAR_SCHEMA,
    ActorPairTerminalError,
    _absolute_directory,
    _immutable_bytes,
    _inside,
)


FAILURE_SCHEMA = "fast_roborender.round_dagger.wandb_delivery_failure_scalar.v1"


class WandbDeliveryError(RuntimeError):
    """Local delivery evidence is unsafe or contradictory."""


def _safe_run_id(path: Path) -> str | None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > 256:
        raise WandbDeliveryError("W&B sidecar run ID is unsafe")
    try:
        return watcher._validate_run_id(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, watcher.ArtifactError) as exc:
        raise WandbDeliveryError("W&B sidecar run ID is invalid") from exc


def verify_delivery(run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return coverage evidence, or ``None`` when a safe replay is required."""

    root = _absolute_directory(run_dir, "actor run directory")
    telemetry = root / "telemetry"
    if not telemetry.exists():
        return None
    _absolute_directory(telemetry, "actor telemetry directory")
    run_id = _safe_run_id(telemetry / "WANDB_SIDECAR_RUN_ID")
    state_path = telemetry / ".wandb_upload_state.json"
    if run_id is None or not state_path.exists():
        return None
    try:
        delivered = watcher.load_state(state_path, expected_run_id=run_id)
    except watcher.ArtifactError as exc:
        raise WandbDeliveryError("W&B upload acknowledgement is invalid") from exc

    errors: list[str] = []
    now = time.time() + 1.0
    artifacts = watcher.discover_scalars(
        root, min_age_s=0.0, now=now, errors=errors
    ) + watcher.discover_artifacts(
        root, min_age_s=0.0, now=now, errors=errors
    )
    if errors:
        raise WandbDeliveryError("actor telemetry contains invalid artifacts")
    terminal_count = 0
    delivery_failure_count = 0
    for artifact in artifacts:
        if delivered.get(artifact.relative_path) != artifact.fingerprint:
            return None
        if artifact.kind != "scalar":
            continue
        try:
            document = json.loads(artifact.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WandbDeliveryError("delivered scalar became unreadable") from exc
        if isinstance(document, dict) and document.get("schema") == SCALAR_SCHEMA:
            terminal_count += 1
        elif isinstance(document, dict) and document.get("schema") == FAILURE_SCHEMA:
            delivery_failure_count += 1
    if terminal_count == 0:
        return None
    receipt_path = telemetry / "WANDB_DELIVERY_RECEIPT.json"
    try:
        info = os.lstat(receipt_path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024:
        raise WandbDeliveryError("W&B delivery receipt is unsafe")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WandbDeliveryError("W&B delivery receipt is invalid") from exc
    state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
    if receipt != {
        "schema": watcher.DELIVERY_RECEIPT_SCHEMA,
        "wandb_run_id": run_id,
        "upload_state_sha256": state_sha256,
        "outcome": "SUCCEEDED",
    }:
        return None
    return {
        "status": "DELIVERED",
        "wandb_run_id": run_id,
        "artifact_count": len(artifacts),
        "terminal_scalar_count": terminal_count,
        "recovered_delivery_failure_count": delivery_failure_count,
    }


def publish_delivery_failure(
    *,
    attempt_dir: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    macro_round: int,
    cohort_index: int,
    wrapper_exit_code: int,
) -> Path:
    """Publish immutable telemetry that a sealed cohort needs W&B recovery."""

    root = _absolute_directory(run_dir, "actor run directory")
    attempt = _absolute_directory(attempt_dir, "actor attempt directory")
    _inside(attempt, root, "actor attempt directory")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (macro_round, cohort_index)):
        raise WandbDeliveryError("round and cohort indices must be non-negative")
    if (
        isinstance(wrapper_exit_code, bool)
        or not isinstance(wrapper_exit_code, int)
        or not 1 <= wrapper_exit_code <= 255
    ):
        raise WandbDeliveryError("W&B wrapper failure requires a nonzero exit code")
    step = macro_round * 4 + cohort_index
    document = {
        "schema": FAILURE_SCHEMA,
        "step": step,
        "metrics": {
            "collection/wandb_delivery_failed": 1,
            "collection/wandb_recovery_required": 1,
            "collection/wandb_wrapper_exit_code": wrapper_exit_code,
            "collection/macro_round": macro_round,
            "collection/cohort_index": cohort_index,
        },
    }
    path = (
        root
        / "telemetry"
        / "scalars"
        / f"step_{step:06d}_wandb_delivery_failure_{attempt.name}.json"
    )
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        _immutable_bytes(path, payload)
    except ActorPairTerminalError as exc:
        raise WandbDeliveryError("cannot publish W&B delivery failure scalar") from exc
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-dir", required=True)
    failure = subparsers.add_parser("failure")
    failure.add_argument("--attempt-dir", required=True)
    failure.add_argument("--run-dir", required=True)
    failure.add_argument("--macro-round", type=int, required=True)
    failure.add_argument("--cohort-index", type=int, required=True)
    failure.add_argument("--wrapper-exit-code", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.operation == "verify":
            report = verify_delivery(arguments.run_dir)
            if report is None:
                print("W&B delivery is incomplete", file=__import__("sys").stderr)
                return 2
            print(json.dumps(report, sort_keys=True))
            return 0
        path = publish_delivery_failure(
            attempt_dir=arguments.attempt_dir,
            run_dir=arguments.run_dir,
            macro_round=arguments.macro_round,
            cohort_index=arguments.cohort_index,
            wrapper_exit_code=arguments.wrapper_exit_code,
        )
        print(json.dumps({"failure_scalar_path": str(path)}, sort_keys=True))
        return 0
    except Exception as exc:
        detail = str(exc) if isinstance(exc, WandbDeliveryError) else type(exc).__name__
        print(f"W&B delivery evidence failed: {detail}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
