#!/usr/bin/env python3
"""Upload and read back the production Image3F fixed-seed evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scaling"))
from wandb_preflight import _load_wandb, query_identity  # noqa: E402

import fixed_seed_gate  # noqa: E402
import hsg_gb200_image3f_fixed_seed_gate as image3f_gate  # noqa: E402
import wandb_team_contract as team_contract  # noqa: E402


SCHEMA = image3f_gate.WANDB_SCHEMA
ORDINALS = (0, 1, 2, 3)
HISTORY_KEYS = (
    "eval/checkpoint_ordinal",
    "eval/checkpoint_label",
    "eval/goal_reached_rate",
    "eval/reward",
    "eval/episodes",
    "image3f/served_requests",
    "image3f/generated_frames",
)
SUMMARY_KEYS = (
    "policy_improvement_gate/status",
    "policy_improvement_gate/paired_success_delta",
    "policy_improvement_gate/paired_success_delta_ci95_low",
    "policy_improvement_gate/paired_success_delta_ci95_high",
    "image3f_gate/status",
    "image3f_gate/total_served_requests",
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
        raise UploadError("remote W&B policy-improvement summary is not PASS")
    if remote_summary.get("image3f_gate/status") != "PASS":
        raise UploadError("remote W&B Image3F runtime summary is not PASS")
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


def _runtime_by_role(results_dir: Path) -> dict[str, dict[str, int]]:
    totals = {
        label: {"served_requests": 0, "generated_frames": 0}
        for label in fixed_seed_gate.ROLE_ORDER
    }
    files = sorted(results_dir.glob("*/seed_*/image3f_attestation.json"))
    if not files:
        raise UploadError("no Image3F runtime attestations were found")
    for path in files:
        row = load_json(path)
        label = row.get("checkpoint_label")
        if label not in totals or row.get("status") != "PASS":
            raise UploadError(f"invalid Image3F runtime attestation: {path}")
        if row.get("renderer_errors") != 0:
            raise UploadError(f"renderer errors recorded in {path}")
        totals[label]["served_requests"] += int(row.get("served_requests", -1))
        totals[label]["generated_frames"] += int(row.get("generated_frames", -2))
    if any(
        value["served_requests"] <= 0
        or value["served_requests"] != value["generated_frames"]
        for value in totals.values()
    ):
        raise UploadError("Image3F served/generated runtime totals are invalid")
    return totals


def upload(args: argparse.Namespace) -> dict[str, Any]:
    if SAFE_ID.fullmatch(args.run_id) is None:
        raise UploadError("run ID is malformed")
    try:
        team_contract.validate_authorized_request(
            expected_email=args.expected_email,
            entity=args.entity,
            authorization_scope=team_contract.AUTHORIZED_SCOPE,
        )
        online_acceptance = team_contract.validate_online_acceptance(
            team_contract.load_json(args.wandb_acceptance)
        )
    except team_contract.TeamContractError as exc:
        raise UploadError(str(exc)) from None
    acceptance_hash = fixed_seed_gate.sha256_file(args.wandb_acceptance)
    if acceptance_hash != args.wandb_acceptance_sha256:
        raise UploadError("W&B online acceptance SHA-256 mismatch")
    manifest = load_json(args.manifest)
    summary = load_json(args.summary)
    if manifest.get("observation_mode") != image3f_gate.OBSERVATION_MODE:
        raise UploadError("manifest is not the production Image3F observation protocol")
    if summary.get("gate", {}).get("status") != "PASS":
        raise UploadError("refusing to upload a non-passing policy report")
    runtime = _runtime_by_role(args.results_dir)
    wandb = _load_wandb()
    identity = query_identity(
        wandb,
        expected_email=args.expected_email,
        requested_entity=args.entity,
        authorization_scope=team_contract.AUTHORIZED_SCOPE,
    )
    try:
        identity_binding = team_contract.validate_live_identity(identity)
    except team_contract.TeamContractError as exc:
        raise UploadError(str(exc)) from None
    run = wandb.init(
        entity=identity.entity,
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        resume="allow",
        job_type="fixed-seed-image3f-3view-production-evaluation",
        config={
            "protocol": manifest["protocol"],
            "observation_mode": image3f_gate.OBSERVATION_MODE,
            "image3f_in_the_loop": True,
            "production_observation_distribution": True,
            "seeds": manifest["seeds"],
            "num_envs_per_seed": manifest["num_envs_per_seed"],
            "reference_config_sha256": manifest["reference_config_sha256"],
            "effective_contract_sha256": manifest["effective_contract_sha256"],
            "runtime_contract": manifest["runtime_contract"],
            "topology": manifest["topology"],
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
                    "image3f/served_requests": runtime[label]["served_requests"],
                    "image3f/generated_frames": runtime[label]["generated_frames"],
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
        run.summary["image3f_gate/status"] = "PASS"
        run.summary["image3f_gate/total_served_requests"] = sum(
            row["served_requests"] for row in runtime.values()
        )
    finally:
        run.finish(exit_code=0)

    run_path = f"{identity.entity}/{args.project}/{args.run_id}"
    deadline = time.monotonic() + args.timeout_s
    readback: dict[str, Any] | None = None
    last_error = ""
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
        except Exception as exc:  # Do not persist third-party text or credentials.
            last_error = type(exc).__name__
        time.sleep(5)
    report = {
        "schema": SCHEMA,
        "status": "PASS" if readback is not None else "FAIL",
        **identity_binding,
        "project": args.project,
        "run_id": args.run_id,
        "run_path": run_path,
        "manifest_sha256": fixed_seed_gate.sha256_file(args.manifest),
        "summary_sha256": fixed_seed_gate.sha256_file(args.summary),
        "wandb_online_acceptance_sha256": acceptance_hash,
        "wandb_online_acceptance_observed_keys": online_acceptance["observed_keys"],
        "last_api_error_type": last_error,
        **(readback or {}),
    }
    atomic_json(args.output, report)
    if readback is None:
        raise UploadError("W&B production Image3F telemetry readback did not complete")
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
    result.add_argument("--results-dir", type=Path, required=True)
    result.add_argument("--wandb-acceptance", type=Path, required=True)
    result.add_argument("--wandb-acceptance-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--timeout-s", type=float, default=180.0)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        report = upload(args)
    except Exception as exc:
        if not isinstance(exc, UploadError):
            exc = UploadError(f"W&B production upload failed ({type(exc).__name__})")
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
