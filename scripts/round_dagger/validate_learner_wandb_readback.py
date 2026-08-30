#!/usr/bin/env python3
"""Read back one learner W&B run and match durable scalar/eval telemetry.

W&B stores the initial held-out evaluation in a sparse row without a training
loss.  Asking ``scan_history`` for both keys therefore hides that row.  This
validator deliberately reads the loss and quality series independently,
collapses only byte-equivalent duplicate optimizer frontiers, and compares the
remote values with the fsynced JSONL histories before publishing a report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import publish_immutable_json, sha256_file  # noqa: E402


SCHEMA = "fast_roborender.round_dagger.learner_wandb_readback.v1"
TELEMETRY_SCHEMA = "fast_roborender.round_dagger.learner_telemetry.v2"
HELDOUT_TELEMETRY_SCHEMA = (
    "fast_roborender.round_dagger.learner_heldout_telemetry.v1"
)
QUALITY_KEYS = (
    "quality/heldout_action_mse",
    "quality/heldout_action_mae",
    "quality/heldout_action_rmse",
    "quality/heldout_action_cosine",
    "quality/heldout_transitions",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_PATH = re.compile(
    r"[A-Za-z0-9_.-]{1,128}/[A-Za-z0-9_.-]{1,128}/[A-Za-z0-9_.-]{1,128}\Z"
)


class ReadbackError(RuntimeError):
    """Local durable evidence and remote W&B history did not agree."""


def _regular(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    if absolute.is_symlink() or not absolute.is_file():
        raise ReadbackError(f"{label} must be a regular non-symlink file")
    return absolute.resolve(strict=True)


def _jsonl(path: Path, *, schema: str, label: str) -> list[dict[str, Any]]:
    path = _regular(path, label)
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ReadbackError(f"{label} is empty or torn")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReadbackError(f"{label} line {number} is invalid") from exc
        if not isinstance(row, dict) or row.get("schema") != schema:
            raise ReadbackError(f"{label} line {number} schema mismatch")
        step = row.get("optimizer_step")
        if type(step) is not int or step < 0:
            raise ReadbackError(f"{label} line {number} optimizer step is invalid")
        rows.append(row)
    steps = [int(row["optimizer_step"]) for row in rows]
    if steps != sorted(set(steps)):
        raise ReadbackError(f"{label} optimizer steps are duplicated or unordered")
    return rows


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReadbackError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReadbackError(f"{label} is non-finite")
    return result


def _collapse(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str], *, label: str
) -> dict[int, dict[str, float]]:
    collapsed: dict[int, dict[str, float]] = {}
    for row in rows:
        raw_step = row.get("optimizer_step")
        if raw_step is None:
            continue
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise ReadbackError(f"remote {label} optimizer step is malformed")
        step_float = float(raw_step)
        if not step_float.is_integer() or step_float < 0:
            raise ReadbackError(f"remote {label} optimizer step is malformed")
        step = int(step_float)
        values = collapsed.setdefault(step, {})
        for key in keys:
            raw = row.get(key)
            if raw is None:
                continue
            value = _finite(raw, f"remote {label} {key}")
            if key in values and values[key] != value:
                raise ReadbackError(
                    f"remote {label} has conflicting duplicate step {step}"
                )
            values[key] = value
    return collapsed


def _scan(remote: object, keys: Sequence[str], label: str) -> list[Mapping[str, Any]]:
    try:
        rows = list(remote.scan_history(keys=list(keys), page_size=1000))
    except Exception as exc:
        raise ReadbackError(
            f"remote W&B {label} history query failed ({type(exc).__name__})"
        ) from None
    if any(not isinstance(row, Mapping) for row in rows):
        raise ReadbackError(f"remote W&B {label} history returned a non-object")
    return rows


def validate(
    *,
    run_path: str,
    train_telemetry: Path,
    heldout_telemetry: Path,
    expected_config: Mapping[str, str],
    output: Path,
    api: object,
) -> dict[str, Any]:
    if _RUN_PATH.fullmatch(run_path) is None:
        raise ReadbackError("W&B run path is malformed")
    if not expected_config:
        raise ReadbackError("expected W&B config must not be empty")
    for key, value in expected_config.items():
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise ReadbackError("expected W&B config contains an invalid entry")
        if key.endswith("sha256") and _SHA256.fullmatch(value) is None:
            raise ReadbackError(f"expected config {key} is not a SHA-256")

    train_path = _regular(train_telemetry, "training telemetry")
    heldout_path = _regular(heldout_telemetry, "held-out telemetry")
    train = _jsonl(train_path, schema=TELEMETRY_SCHEMA, label="training telemetry")
    heldout = _jsonl(
        heldout_path,
        schema=HELDOUT_TELEMETRY_SCHEMA,
        label="held-out telemetry",
    )
    try:
        remote = api.run(run_path)
    except Exception as exc:
        raise ReadbackError(
            f"remote W&B run query failed ({type(exc).__name__})"
        ) from None
    if getattr(remote, "state", None) != "finished":
        raise ReadbackError("remote W&B learner run is not finished")
    expected_id = run_path.rsplit("/", 1)[1]
    if getattr(remote, "id", expected_id) != expected_id:
        raise ReadbackError("remote W&B run ID mismatch")
    remote_config = getattr(remote, "config", None)
    if not isinstance(remote_config, Mapping) or any(
        remote_config.get(key) != value for key, value in expected_config.items()
    ):
        raise ReadbackError("remote W&B config identity mismatch")

    loss_rows = _scan(remote, ("optimizer_step", "loss"), "loss")
    quality_rows = _scan(
        remote, ("optimizer_step", *QUALITY_KEYS), "quality"
    )
    remote_loss = _collapse(loss_rows, ("loss",), label="loss")
    remote_quality = _collapse(quality_rows, QUALITY_KEYS, label="quality")
    expected_train_steps = [int(row["optimizer_step"]) for row in train]
    expected_quality_steps = [int(row["optimizer_step"]) for row in heldout]
    if sorted(remote_loss) != expected_train_steps:
        raise ReadbackError("remote W&B loss frontiers differ from local telemetry")
    if sorted(remote_quality) != expected_quality_steps:
        raise ReadbackError("remote W&B quality frontiers differ from local telemetry")
    for row in train:
        step = int(row["optimizer_step"])
        if remote_loss[step].get("loss") != _finite(row.get("loss"), "local loss"):
            raise ReadbackError(f"remote W&B loss differs at step {step}")
    for row in heldout:
        step = int(row["optimizer_step"])
        expected = {
            "quality/heldout_action_mse": row.get("action_mse"),
            "quality/heldout_action_mae": row.get("action_mae"),
            "quality/heldout_action_rmse": row.get("action_rmse"),
            "quality/heldout_action_cosine": row.get("action_cosine"),
            "quality/heldout_transitions": row.get("heldout_transitions_global"),
        }
        for key, raw in expected.items():
            if remote_quality[step].get(key) != _finite(raw, f"local {key}"):
                raise ReadbackError(f"remote W&B quality differs at step {step}")

    output = Path(os.path.abspath(output.expanduser()))
    if output.exists() or output.is_symlink():
        raise ReadbackError("readback output already exists")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "run_path": run_path,
        "run_state": "finished",
        "training_optimizer_steps": expected_train_steps,
        "quality_optimizer_steps": expected_quality_steps,
        "training_history_rows": len(loss_rows),
        "quality_history_rows": len(quality_rows),
        "duplicate_quality_rows_are_identical": len(quality_rows)
        >= len(expected_quality_steps),
        "exact_training_match": True,
        "exact_heldout_match": True,
        "identity_config_match": True,
        "training_telemetry_sha256": sha256_file(train_path),
        "heldout_telemetry_sha256": sha256_file(heldout_path),
        "expected_config": dict(sorted(expected_config.items())),
    }
    published = publish_immutable_json(output, report)
    report["report_file_sha256"] = published
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--train-telemetry", required=True, type=Path)
    parser.add_argument("--heldout-telemetry", required=True, type=Path)
    parser.add_argument(
        "--expected-config",
        required=True,
        action="append",
        metavar="KEY=VALUE",
        help="repeat for every identity field that must match W&B config",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    expected: dict[str, str] = {}
    for item in arguments.expected_config:
        key, separator, value = item.partition("=")
        if not separator or key in expected:
            raise ReadbackError("--expected-config must use unique KEY=VALUE entries")
        expected[key] = value
    try:
        import wandb

        api = wandb.Api(timeout=60)
    except Exception as exc:
        raise ReadbackError(
            f"W&B API initialization failed ({type(exc).__name__})"
        ) from None
    report = validate(
        run_path=arguments.run_path,
        train_telemetry=arguments.train_telemetry,
        heldout_telemetry=arguments.heldout_telemetry,
        expected_config=expected,
        output=arguments.output,
        api=api,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReadbackError as exc:
        print(f"Learner W&B readback failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
