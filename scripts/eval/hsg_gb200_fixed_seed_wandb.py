#!/usr/bin/env python3
"""Upload an HSG fixed-seed report and verify its scalar telemetry by readback."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scaling"))
from wandb_preflight import _load_wandb, query_identity  # noqa: E402

import fixed_seed_gate  # noqa: E402


SCHEMA = "fast_roborender.hsg_fixed_seed_wandb_upload.v1"
ORDINALS = (0, 1, 2, 3)
HISTORY_KEYS = (
    "eval/checkpoint_ordinal",
    "eval/checkpoint_label",
    "eval/goal_reached_rate",
    "eval/reward",
    "eval/episodes",
)
SUMMARY_KEYS = (
    "policy_improvement_gate/status",
    "policy_improvement_gate/paired_success_delta",
    "policy_improvement_gate/paired_success_delta_ci95_low",
    "policy_improvement_gate/paired_success_delta_ci95_high",
)
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class UploadError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UploadError(f"cannot read {path}: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise UploadError(f"{path} must contain one JSON object")
    return value


def validate_readback(
    rows: Sequence[Mapping[str, Any]], remote_summary: Mapping[str, Any]
) -> dict[str, Any]:
    observed: dict[int, set[str]] = {}
    for row in rows:
        ordinal = row.get("eval/checkpoint_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in ORDINALS:
            continue
        observed.setdefault(ordinal, set()).update(key for key in HISTORY_KEYS if key in row)
    missing_rows = []
    for ordinal in ORDINALS:
        missing = sorted(set(HISTORY_KEYS) - observed.get(ordinal, set()))
        if missing:
            missing_rows.append({"ordinal": ordinal, "missing_keys": missing})
    missing_summary = sorted(key for key in SUMMARY_KEYS if key not in remote_summary)
    if remote_summary.get("policy_improvement_gate/status") != "PASS":
        raise UploadError("remote W&B summary does not attest a passing policy gate")
    if missing_rows or missing_summary:
        raise UploadError(
            f"remote W&B readback incomplete: rows={missing_rows}, summary={missing_summary}"
        )
    return {
        "observed_checkpoint_ordinals": list(ORDINALS),
        "observed_history_keys": list(HISTORY_KEYS),
        "observed_summary_keys": list(SUMMARY_KEYS),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def upload(args: argparse.Namespace) -> dict[str, Any]:
    if SAFE_ID.fullmatch(args.run_id) is None:
        raise UploadError("run ID is malformed")
    manifest = load_json(args.manifest)
    summary = load_json(args.summary)
    if summary.get("gate", {}).get("status") != "PASS":
        raise UploadError("refusing to upload a non-passing fixed-seed report")
    wandb = _load_wandb()
    identity = query_identity(
        wandb,
        expected_email=args.expected_email,
        requested_entity=args.entity,
        require_personal=True,
    )
    run = wandb.init(
        entity=identity.entity,
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        resume="allow",
        job_type="fixed-seed-simulator-rgb-policy-mechanics",
        config={
            "protocol": manifest["protocol"],
            "observation_mode": "simulator_rgb",
            "image3f_in_the_loop": False,
            "seeds": manifest["seeds"],
            "num_envs_per_seed": manifest["num_envs_per_seed"],
            "reference_config_sha256": manifest["reference_config_sha256"],
        },
    )
    try:
        for ordinal, label in enumerate(fixed_seed_gate.ROLE_ORDER):
            metrics = summary["checkpoint_summaries"][label]
            run.log(
                {
                    "eval/checkpoint_ordinal": ordinal,
                    "eval/checkpoint_label": label,
                    "eval/goal_reached_rate": metrics["goal_reached_rate"],
                    "eval/reward": metrics["reward"],
                    "eval/episodes": metrics["episodes"],
                },
                step=ordinal,
            )
        gate = summary["gate"]
        run.summary["policy_improvement_gate/status"] = gate["status"]
        run.summary["policy_improvement_gate/paired_success_delta"] = gate[
            "paired_success_delta"
        ]
        run.summary["policy_improvement_gate/paired_success_delta_ci95_low"] = gate[
            "paired_success_delta_seed_block_bootstrap95_low"
        ]
        run.summary["policy_improvement_gate/paired_success_delta_ci95_high"] = gate[
            "paired_success_delta_seed_block_bootstrap95_high"
        ]
    finally:
        run.finish(exit_code=0)

    run_path = f"{identity.entity}/{args.project}/{args.run_id}"
    deadline = time.monotonic() + args.timeout_s
    last_error = ""
    readback: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            remote = wandb.Api(timeout=30).run(run_path)
            rows = list(
                remote.scan_history(
                    keys=list(HISTORY_KEYS), min_step=0, max_step=4, page_size=10
                )
            )
            readback = validate_readback(rows, remote.summary)
            break
        except UploadError:
            last_error = "UploadError"
        except Exception as exc:  # Never persist third-party text or credentials.
            last_error = type(exc).__name__
        time.sleep(5)
    status = "PASS" if readback is not None else "FAIL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "account_email": identity.email,
        "entity": identity.entity,
        "personal": identity.personal,
        "project": args.project,
        "run_id": args.run_id,
        "run_path": run_path,
        "manifest_sha256": fixed_seed_gate.sha256_file(args.manifest),
        "summary_sha256": fixed_seed_gate.sha256_file(args.summary),
        "last_api_error_type": last_error,
        **(readback or {}),
    }
    atomic_json(args.output, report)
    if status != "PASS":
        raise UploadError("W&B fixed-seed telemetry readback did not complete")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--expected-email", required=True)
    result.add_argument("--entity", required=True)
    result.add_argument("--project", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--run-name", required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--timeout-s", type=float, default=180.0)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        report = upload(args)
    except Exception as exc:
        if not isinstance(exc, UploadError):
            exc = UploadError(f"W&B fixed-seed upload failed ({type(exc).__name__})")
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
