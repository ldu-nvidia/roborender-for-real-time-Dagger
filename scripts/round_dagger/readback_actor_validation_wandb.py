#!/usr/bin/env python3
"""Read back a bounded actor-validation W&B sidecar, without a trainer run.

The round actor is frozen and intentionally has no optimizer or trainer W&B
run.  This gate therefore binds the local rank-zero scalar, PNG, and MP4
artifacts directly to one durable sidecar run and verifies the downloaded
remote media byte-for-byte.  It is deliberately separate from the production
trainer readback so bounded validation cannot be reported as learner training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from round_dagger.artifacts import load_strict_json
from scripts.scaling.wandb_preflight import (
    _load_wandb,
    query_identity,
)
from scripts.scaling.wandb_run_watcher import load_initialized_marker
from scripts.scaling.wandb_training_readback import (
    HISTORY_PAGE_SIZE,
    ReadbackError,
    _atomic_json,
    _expected_sidecar_evidence,
    _fully_verified_keys,
    _run_id,
    _scalar_evidence_report,
    _scan_scalar_points,
    _verify_remote_media_attempt,
)


REPORT_SCHEMA = "fast_roborender.round_dagger.wandb_actor_validation_readback.v1"
EXPECTED_JOB_TYPE = "dagger-bounded-validation-sidecar"
EXPECTED_SCOPE = "bounded-validation"
READY_FIELDS = {
    "status",
    "account_email",
    "entity",
    "authorization_scope",
    "team_member",
    "entity_is_team",
    "personal",
    "project",
    "telemetry_scope",
    "pid",
    "ready_unix_s",
    "wandb_run_id",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument("--authorization-scope", required=True, choices=("team",))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    return parser


def _ready_marker(
    run_dir: Path,
    *,
    run_id: str,
    expected_email: str,
    entity: str,
    project: str,
) -> dict[str, object]:
    path = run_dir / "telemetry" / "WANDB_WATCHER_READY.json"
    if not path.is_file() or path.is_symlink():
        raise ReadbackError("bounded validation W&B ready marker is missing or unsafe")
    payload = load_strict_json(path)
    if set(payload) != READY_FIELDS:
        raise ReadbackError("bounded validation W&B ready marker fields differ")
    expected = {
        "status": "ready",
        "account_email": expected_email,
        "entity": entity,
        "authorization_scope": "team",
        "team_member": True,
        "entity_is_team": True,
        "personal": False,
        "project": project,
        "telemetry_scope": EXPECTED_SCOPE,
        "wandb_run_id": run_id,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ReadbackError(f"bounded validation W&B ready marker mismatch for {key}")
    pid = payload.get("pid")
    ready_time = payload.get("ready_unix_s")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise ReadbackError("bounded validation W&B ready marker pid is invalid")
    if isinstance(ready_time, bool) or not isinstance(ready_time, (int, float)):
        raise ReadbackError("bounded validation W&B ready marker time is invalid")
    return payload


def _config_value(config: object, key: str) -> object:
    if isinstance(config, Mapping):
        return config.get(key)
    try:
        return config[key]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def _verify_remote_run_contract(
    remote_run: object,
    *,
    expected_email: str,
    entity: str,
) -> None:
    if getattr(remote_run, "job_type", None) != EXPECTED_JOB_TYPE:
        raise ReadbackError("remote W&B run is not a bounded validation sidecar")
    config = getattr(remote_run, "config", None)
    expected = {
        "telemetry_scope": EXPECTED_SCOPE,
        "expected_account": expected_email,
        "account_email": expected_email,
        "entity": entity,
        "authorization_scope": "team",
        "team_member": True,
        "entity_is_team": True,
        "personal": False,
    }
    for key, value in expected.items():
        if _config_value(config, key) != value:
            raise ReadbackError(f"remote bounded validation contract mismatch for {key}")


def _report(
    *,
    status: str,
    identity: object,
    project: str,
    sidecar_run_id: str,
    local_counts: dict[str, int],
    required_scalars: tuple[object, ...],
    verified_scalars: tuple[object, ...],
    required_media: tuple[object, ...],
    observed_media: set[str],
    media_events: list[dict[str, Any]],
    last_error: str,
) -> dict[str, object]:
    required_scalar_keys = {item.key for item in required_scalars}
    observed_scalar_keys = _fully_verified_keys(required_scalars, verified_scalars)
    required_media_keys = {item.key for item in required_media}
    scalar_report = _scalar_evidence_report(
        required_scalars,
        verified_scalars,
        axis="trainer_step",
        resume_training=False,
        resume_checkpoint_step=0,
        rollout_steps=1,
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "telemetry_scope": EXPECTED_SCOPE,
        "account_email": identity.email,
        "entity": identity.entity,
        "authorization_scope": identity.authorization_scope,
        "team_member": identity.team_member,
        "entity_is_team": identity.entity_is_team,
        "personal": identity.personal,
        "project": project,
        "sidecar_run_id": sidecar_run_id,
        "trainer_run_required": False,
        "required_sidecar_scalar_keys": sorted(required_scalar_keys),
        "observed_sidecar_scalar_keys": sorted(observed_scalar_keys),
        "sidecar_scalar_evidence": scalar_report,
        "required_sidecar_media_keys": sorted(required_media_keys),
        "observed_sidecar_media_keys": sorted(observed_media),
        "required_remote_media_event_count": len(required_media),
        "verified_remote_media_event_count": len(media_events),
        "verified_remote_media_events": media_events,
        "remote_media_history_steps_unique": len(
            {item["remote_history_step"] for item in media_events}
        )
        == len(media_events),
        "local_artifact_counts": local_counts,
        "last_api_error_type": last_error,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout_s <= 0 or args.poll_interval_s <= 0:
        print("bounded W&B readback: timeout and poll interval must be positive", file=sys.stderr)
        return 2
    try:
        if not args.run_dir.is_absolute() or not args.output.is_absolute():
            raise ReadbackError("run directory and output must be absolute")
        run_dir = args.run_dir.resolve(strict=True)
        output = args.output.resolve(strict=False)
        output.relative_to(run_dir)
        if output.exists() and (not output.is_file() or output.is_symlink()):
            raise ReadbackError("readback output is unsafe")
        sidecar_run_id = _run_id(
            run_dir / "telemetry" / "WANDB_SIDECAR_RUN_ID",
            "sidecar W&B run ID",
        )
        if not load_initialized_marker(
            run_dir / "telemetry" / "WANDB_SIDECAR_INITIALIZED.json",
            expected_run_id=sidecar_run_id,
            telemetry_scope=EXPECTED_SCOPE,
            project=args.project,
            entity=args.entity,
        ):
            raise ReadbackError(
                "bounded validation W&B initialized marker is missing"
            )
        _ready_marker(
            run_dir,
            run_id=sidecar_run_id,
            expected_email=args.expected_email,
            entity=args.entity,
            project=args.project,
        )
        local_scalars, local_media, local_counts = _expected_sidecar_evidence(run_dir)
        wandb = _load_wandb()
        identity = query_identity(
            wandb,
            expected_email=args.expected_email,
            requested_entity=args.entity,
            authorization_scope=args.authorization_scope,
        )
    except Exception as exc:
        print(
            f"bounded W&B readback failed ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 2

    try:
        api = wandb.Api(timeout=30)
    except Exception as exc:
        print(
            f"bounded W&B readback failed ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 2
    verified_scalars: tuple[object, ...] = ()
    observed_media: set[str] = set()
    media_events: list[dict[str, Any]] = []
    last_error = ""
    deadline = time.monotonic() + args.timeout_s
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    attempt = 0
    required_scalar_keys = {item.key for item in local_scalars}
    required_media_keys = {item.key for item in local_media}
    while time.monotonic() < deadline:
        attempt += 1
        try:
            remote = api.run(f"{identity.entity}/{args.project}/{sidecar_run_id}")
            _verify_remote_run_contract(
                remote,
                expected_email=identity.email,
                entity=identity.entity,
            )
            verified_scalars = _scan_scalar_points(
                remote,
                local_scalars,
                axis="trainer_step",
                page_size=HISTORY_PAGE_SIZE,
            )
            observed_scalar_keys = _fully_verified_keys(
                local_scalars, verified_scalars
            )
            if (
                len(verified_scalars) == len(local_scalars)
                and required_media_keys != observed_media
            ):
                observed_media, media_events = _verify_remote_media_attempt(
                    remote,
                    local_media,
                    validation_dir=validation_dir,
                    attempt=attempt,
                )
            if (
                required_scalar_keys <= observed_scalar_keys
                and required_media_keys <= observed_media
                and len(media_events) == len(local_media)
            ):
                break
        except Exception as exc:  # Never persist third-party exception text.
            last_error = type(exc).__name__
        time.sleep(args.poll_interval_s)

    observed_scalar_keys = _fully_verified_keys(local_scalars, verified_scalars)
    scalar_report = _scalar_evidence_report(
        local_scalars,
        verified_scalars,
        axis="trainer_step",
        resume_training=False,
        resume_checkpoint_step=0,
        rollout_steps=1,
    )
    status = (
        "PASS"
        if required_scalar_keys <= observed_scalar_keys
        and required_media_keys <= observed_media
        and scalar_report["all_points_verified"] is True
        and len(media_events) == len(local_media)
        and len({item["remote_history_step"] for item in media_events})
        == len(local_media)
        else "FAIL"
    )
    report = _report(
        status=status,
        identity=identity,
        project=args.project,
        sidecar_run_id=sidecar_run_id,
        local_counts=local_counts,
        required_scalars=local_scalars,
        verified_scalars=verified_scalars,
        required_media=local_media,
        observed_media=observed_media,
        media_events=media_events,
        last_error=last_error,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
