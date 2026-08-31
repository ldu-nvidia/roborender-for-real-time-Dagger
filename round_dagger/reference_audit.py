#!/usr/bin/env python3
"""Fail-closed audit of bootstrapped Image3F fixed-lag reference telemetry."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from round_dagger.artifacts import load_strict_json, sha256_file


REFERENCE_AUDIT_SCHEMA = "fast_roborender.reference_lag_validation.v2"


class ReferenceAuditError(RuntimeError):
    """Reference telemetry is missing, malformed, or violates fixed-lag policy."""


def audit_uninterrupted_rollout(
    rollout_shard: str | os.PathLike[str], *, expected_steps: int, env_index: int = 0
) -> dict[str, Any]:
    """Prove that one telemetry source environment did not reset during a lag gate.

    Fixed-lag references are intentionally episode-local. A global telemetry
    counter therefore cannot be compared with a renderer request step unless
    the corresponding environment stayed in one episode. This check binds the
    scalar audit to immutable rollout episode evidence.
    """

    supplied_path = Path(rollout_shard).expanduser()
    if expected_steps < 1:
        raise ReferenceAuditError("expected_steps must be positive")
    if env_index < 0:
        raise ReferenceAuditError("env_index must be non-negative")
    if supplied_path.is_symlink():
        raise ReferenceAuditError(f"unsafe rollout shard symlink: {supplied_path}")
    path = supplied_path.resolve()
    if not path.is_file():
        raise ReferenceAuditError(f"unsafe or missing rollout shard: {path}")

    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = {"dones", "episode_ids", "episode_steps"}.difference(
                payload.files
            )
            if missing:
                raise ReferenceAuditError(
                    f"rollout shard is missing episode evidence: {sorted(missing)}"
                )
            dones = np.asarray(payload["dones"])
            episode_ids = np.asarray(payload["episode_ids"])
            episode_steps = np.asarray(payload["episode_steps"])
    except ReferenceAuditError:
        raise
    except Exception as exc:
        raise ReferenceAuditError(f"cannot read rollout shard {path}: {exc}") from exc

    for name, array in (
        ("dones", dones),
        ("episode_ids", episode_ids),
        ("episode_steps", episode_steps),
    ):
        if array.ndim != 2 or array.shape[0] != expected_steps:
            raise ReferenceAuditError(
                f"rollout {name} must have shape [{expected_steps}, B] "
                f"(got {array.shape})"
            )
        if env_index >= array.shape[1]:
            raise ReferenceAuditError(
                f"rollout {name} has no environment index {env_index}"
            )

    selected_dones = dones[:, env_index]
    selected_ids = episode_ids[:, env_index]
    selected_steps = episode_steps[:, env_index]
    terminal_steps = np.flatnonzero(selected_dones).astype(int).tolist()
    if terminal_steps:
        raise ReferenceAuditError(
            f"lag validation environment terminated at rollout steps {terminal_steps}"
        )
    if not np.all(selected_ids == selected_ids[0]):
        changes = (
            np.flatnonzero(selected_ids[1:] != selected_ids[:-1]) + 1
        ).astype(int)
        raise ReferenceAuditError(
            f"lag validation environment changed episode at rollout steps {changes.tolist()}"
        )
    expected_episode_steps = np.arange(expected_steps, dtype=np.int64)
    if not np.array_equal(selected_steps.astype(np.int64), expected_episode_steps):
        mismatch = int(np.flatnonzero(selected_steps != expected_episode_steps)[0])
        raise ReferenceAuditError(
            "lag validation episode steps are not uninterrupted: "
            f"rollout step {mismatch} records episode step {int(selected_steps[mismatch])}"
        )

    return {
        "rollout_shard": path.name,
        "rollout_shard_sha256": sha256_file(path),
        "environment_index": env_index,
        "episode_id": int(selected_ids[0]),
        "first_episode_step": 0,
        "last_episode_step": expected_steps - 1,
        "terminal_step_count": 0,
    }


def _require_metric(metrics: dict[str, Any], key: str, expected: int) -> None:
    if metrics.get(key) != expected:
        raise ReferenceAuditError(
            f"telemetry metric {key!r} must be {expected} (got {metrics.get(key)!r})"
        )


def audit_reference_telemetry(
    scalar_dir: str | os.PathLike[str],
    *,
    lag: int,
    expected_steps: int,
    rollout_shard: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Prove bootstrap -> generated-frame-0 -> fixed-lag scalar telemetry."""

    directory = Path(scalar_dir).expanduser().resolve()
    if lag < 1:
        raise ReferenceAuditError("lag must be positive")
    if expected_steps <= lag:
        raise ReferenceAuditError(
            "expected_steps must cross the reference boundary (expected_steps > lag)"
        )
    if not directory.is_dir() or directory.is_symlink():
        raise ReferenceAuditError(f"unsafe or missing scalar directory: {directory}")

    chain = hashlib.sha256()
    bootstrap_steps = 0
    frame_zero_steps = 0
    generated_steps = 0
    for step in range(expected_steps):
        path = directory / f"step_{step:09d}.json"
        if not path.is_file() or path.is_symlink():
            raise ReferenceAuditError(f"missing scalar telemetry for step {step}: {path}")
        document = load_strict_json(path)
        if not isinstance(document, dict) or document.get("step") != step:
            raise ReferenceAuditError(f"scalar document step mismatch at {path}")
        metrics = document.get("metrics")
        if not isinstance(metrics, dict):
            raise ReferenceAuditError(f"scalar document metrics are malformed at {path}")

        generated = step >= lag
        bootstrap = step == 0
        frame_zero = 0 < step < lag
        expected_source_step = step - lag if generated else (-1 if bootstrap else 0)
        _require_metric(metrics, "renderer/reference_provenance_valid", 1)
        _require_metric(metrics, "renderer/reference_provenance/request_step", step)
        _require_metric(metrics, "renderer/reference_provenance/configured_lag", lag)
        _require_metric(
            metrics,
            "renderer/reference_provenance/source_step",
            expected_source_step,
        )
        _require_metric(metrics, "renderer/reference_source_anchor", int(bootstrap))
        _require_metric(
            metrics, "renderer/reference_source_bootstrap_anchor", int(bootstrap)
        )
        _require_metric(
            metrics, "renderer/reference_source_generated_frame0", int(frame_zero)
        )
        _require_metric(metrics, "renderer/reference_source_generated_lag", int(generated))
        _require_metric(metrics, "renderer/reference_source_client", 0)
        _require_metric(metrics, "renderer/reference_source_idempotent_replay", 0)
        _require_metric(metrics, "renderer/reference_source_unknown", 0)
        if generated:
            _require_metric(metrics, "renderer/reference_lag_frames", lag)
            generated_steps += 1
        elif frame_zero:
            frame_zero_steps += 1
        else:
            bootstrap_steps += 1

        chain.update(step.to_bytes(8, "big", signed=False))
        chain.update(bytes.fromhex(sha256_file(path)))

    report = {
        "schema": REFERENCE_AUDIT_SCHEMA,
        "lag": lag,
        "expected_steps": expected_steps,
        "boundary_step": lag,
        "bootstrap_anchor_step_count": bootstrap_steps,
        "generated_frame_zero_step_count": frame_zero_steps,
        "generated_step_count": generated_steps,
        "first_generated_request_step": lag,
        "first_generated_source_step": 0,
        "last_generated_request_step": expected_steps - 1,
        "last_generated_source_step": expected_steps - 1 - lag,
        "scalar_chain_sha256": chain.hexdigest(),
    }
    if rollout_shard is not None:
        report["uninterrupted_rollout"] = audit_uninterrupted_rollout(
            rollout_shard, expected_steps=expected_steps
        )
    return report
